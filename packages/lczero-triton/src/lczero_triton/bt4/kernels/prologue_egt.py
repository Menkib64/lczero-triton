"""EGT2 prologue (item E, K1; the packed-``E`` kernels rebuilt in round 22, K1b): the 34-channel attack graph, the pair
normalization and the edge-stream seed.

The lab's EGT2 512x15 nets (gcap, triplet_path, triplet_ag) read three per-position tensors, all built once from
the input planes (export node ids from the gcap export):

* ``E [34, 64, 64]`` (node 10279), indexed ``[channel, source i, target j]``: the 17 base channels of
  ``attack_graph.edge_channels`` with every flag on, then ``tdelta_channels``, then the transpose of all 17:

  ====  ===============  =====================================================================================
  bit   channel          definition (``p`` = piece plane > 0.5; a slider ray includes its first blocker)
  ====  ===============  =====================================================================================
  0/1   attacks stm/opp  leapers, plus the occlusion-aware slider flood; any target, friendly pieces included
  2/3   defends stm/opp  attacks whose target holds a piece of the same side
  4/5   xray stm/opp     slider rays strictly beyond the first blocker, up to and including the second
  6/7   pinray stm/opp   ``[b, t]``: a slider of that side hits blocker ``b`` first along direction ``d``, and ``t``
                         is the first occupied square beyond ``b`` along the same ``d``
  8     kingring1_opp    ``[q, k]``: ``k`` holds their king and ``q``'s attacks (stm) touch a square at Chebyshev 1
  9     kingring1_own    the same with the opponent's attacks and our king
  10/11 kingring2        as 8/9 with Chebyshev distance exactly 2
  12    blockescape      ``[s, k]``: ``s`` occupied, ``k`` a king (either side) at Chebyshev 1 from ``s``
  13-16 gained stm/opp,  ``attacks & ~previous``, ``previous & ~attacks``, where ``previous`` is channels 0/1
        lost stm/opp     rebuilt from planes 13-24 (the previous position, already in the current side's view)
  17-33 transposes       bit ``17 + c`` at ``(i, j)`` is bit ``c`` at ``(j, i)``
  ====  ===============  =====================================================================================

* ``S [64, 64]`` (node 10304): ``1 / sqrt(mean_t Z[t]^2 + 1e-6)`` with ``Z = sum_c D[c, t] E[c] + T[t]``, the pair
  stream's normalization. Every consumer of ``Z0 = S * Z`` folds to ``S * (G.E + K)`` (the static fold generalised,
  ``attention_static``), so ``S`` and ``E`` are all that ``Z0``'s folds need.
* ``e0 [16, 64, 64]`` (node 10331): ``rms_t(sum_c p_in[c, t] E[c] + t_off[relidx][t])``, the edge-stream seed.

E packing: the K1 -> K2 interface (unchanged by K1b)
----------------------------------------------------
``edges`` is ``uint64 [batch, 64, 64]``; cell ``(i, j)`` lives at ``sample * 4096 + 64 * i + j`` with ``i`` the query
(source) square, ``j`` the key (target) square and squares numbered ``8 * row + column`` in plane coordinates.
**Bit ``c`` of a cell is export channel ``c``** for all 34 channels (the table above); bits 34-63 are zero. This is
P5's static convention (bits 0/1 base, 2/3 transposes) widened: a consumer reads any channel as ``(code >> c) & 1``
from one load, with no index swap, and extracts a code group for a per-code table as ``(code >> first) & mask``.
8 bytes per cell is 2.1 MB at batch 64, against 17.8 MB for FP16 channels.

Three kernels
-------------
``_prologue_egt_rows_kernel`` (K1b) builds the 17 base channels as **rows**: one program per position, lane =
source square, each channel one ``uint64`` board of targets per lane, stored as ``uint64 [batch, 17, 64]``. Every
row is a function of the lane's square and the position's 24 piece boards alone, so there is no flood: the eight
ray masks come from the file / rank / diagonal formulas, the blockers are ``ray & occupied``, the first blocker is
the lowest set bit (``b ^ (b - 1)`` masks up to it) for the four positive directions and the highest for the four
negative ones (the same scan on the bit-reversed word, ``__nv_brevll``), the x-ray is a second scan past it, and a
pin at ``b`` reads as "the first blocker from ``b`` in the opposite direction is a compatible slider", so it needs
no whole-board pass. K1's kernel flooded seven steps per direction from every square, twice, for pins and again
for the previous frame; this does the same work in a few dozen bit operations per direction.

``_prologue_egt_edges_kernel`` (K1b) packs the codes: ``EDGE_SPLIT`` programs per position, each writing
``64 // EDGE_SPLIT`` source rows of the ``[64, 64]`` tile. A program reads its own rows (bits 0-16: bit ``j`` of
row ``c`` of source ``i``) and every row's bit ``i`` (bits 17-33, the transposes), in 32-bit halves so every extract
is one shift; there is no in-program transpose and no global-memory round trip. The split is a free parameter of
the specialization; measured, 2 programs per position (128 at batch 64, each a ``[32, 64]`` slice) was fastest.

``_prologue_egt_seed_kernel`` (K1, unchanged) reads the codes and builds ``S`` and ``e0`` in FP32, one ``[64, 64]``
tile per pair channel: four row gathers from ``/prologue/mix_codes [1536, 32]``. That table holds the build-time
sums of ``D`` (first 16 columns) and ``p_in`` (last 16) over the bits of each code of the groups ``CODE_GROUPS``
(channels 0-8, 9-16, 17-25, 26-33). ``e0``'s pre-norm is recomputed in a second pass rather than stored, so FP16
storage rounds only the final value.

Semantics follow the export on legal positions. On boards the lab never produces there is one known difference:
the export does not clip ``blockescape``, so a square holding both kings gives 2 there and 1 here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton
from triton.language.extra import libdevice

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_INPUT_PLANES = 112
_SQUARES = 64
BASE_CHANNELS = 17
EDGE_CHANNELS = 34
PAIR_CHANNELS = 16
STATE_CHANNELS = 16
# `const_1207` / `const_1215` in the gcap export, `_RMS_EPS` in the lab.
EPSILON = 1e-6
# (first channel, bit count) of each code group, and the row where its codes start in `/prologue/mix_codes`.
CODE_GROUPS = ((0, 9), (9, 8), (17, 9), (26, 8))
CODE_GROUP_BASES = (0, 512, 768, 1280)
CODE_ROWS = 1536
CODE_WIDTH = PAIR_CHANNELS + STATE_CHANNELS
# The plan names the seed kernel reads, in argument order (`lab._names.plan_prologue_egt`).
SEED_TABLES = ("/prologue/offsets", "/prologue/edge_offsets", "/prologue/mix_codes")
# K1b: the row kernel's scratch per position, `uint64 [17, 64]` (channel, source square) -> a board of targets.
ROW_BYTES = BASE_CHANNELS * _SQUARES * 8
# K1b: programs per position of the packing kernel; each writes 64 // EDGE_SPLIT source rows of the tile.
# Must divide 32 (a program's rows then sit in one 32-bit half of every row word). Graph-replayed on an RTX 5080,
# 2 was the fastest of 2/4/8/16/32 at every batch from 8 to 128 (by 0.2-0.6 us): the packing is cheap enough
# that a program's fixed cost (its 17 x 64 row-word loads) outweighs the extra parallelism.
EDGE_SPLIT = 2


def fold_mix_codes(channel_mix: torch.Tensor, edge_mix: torch.Tensor) -> torch.Tensor:
    """Fold ``D [34, 16]`` and ``p_in [34, 16]`` into the FP32 ``[1536, 32]`` code-row table, summed in FP64."""
    mix = torch.cat((channel_mix.double(), edge_mix.double()), dim=1).cpu()
    rows = []
    for first, width in CODE_GROUPS:
        codes = torch.arange(1 << width)
        bits = ((codes[:, None] >> torch.arange(width)[None, :]) & 1).double()
        rows.append(bits @ mix[first:first + width])
    return torch.cat(rows).float().contiguous()


def unpack_edges(codes: torch.Tensor) -> torch.Tensor:
    """``uint64 [batch, 64, 64]`` codes -> ``bool [batch, 34, 64, 64]`` channels, for tests and probes."""
    signed = codes.view(torch.int64)
    channels = torch.arange(EDGE_CHANNELS, device=codes.device, dtype=torch.int64)
    return ((signed[:, None] >> channels[None, :, None, None]) & 1).bool()


@triton.jit
def _step(board, row_step: tl.constexpr, column_step: tl.constexpr):
    """Move every set bit by one (row, column) step; bits leaving the board vanish."""
    if row_step * 8 + column_step >= 0:
        moved = board << tl.full((), row_step * 8 + column_step, tl.uint64)
    else:
        moved = board >> tl.full((), -(row_step * 8 + column_step), tl.uint64)
    if column_step == 1:
        moved = moved & ~tl.full((), 0x0101010101010101, tl.uint64)  # not column 0
    elif column_step == 2:  # noqa: PLR2004
        moved = moved & ~tl.full((), 0x0303030303030303, tl.uint64)  # not columns 0-1
    elif column_step == -1:
        moved = moved & tl.full((), 0x7F7F7F7F7F7F7F7F, tl.uint64)  # not column 7
    elif column_step == -2:  # noqa: PLR2004
        moved = moved & tl.full((), 0x3F3F3F3F3F3F3F3F, tl.uint64)  # not columns 6-7
    return moved


@triton.jit
def _king_moves(board):
    """Chebyshev distance exactly 1 (the lab's `_KING_M`, ring 1)."""
    moved = _step(board, 1, 0) | _step(board, -1, 0) | _step(board, 0, 1) | _step(board, 0, -1)
    return moved | _step(board, 1, 1) | _step(board, 1, -1) | _step(board, -1, 1) | _step(board, -1, -1)


@triton.jit
def _knight_moves(board):
    moved = _step(board, 1, 2) | _step(board, 2, 1) | _step(board, -1, 2) | _step(board, -2, 1)
    return moved | _step(board, 1, -2) | _step(board, 2, -1) | _step(board, -1, -2) | _step(board, -2, -1)


@triton.jit
def _ring2_moves(board):
    """Chebyshev distance exactly 2 (the lab's `_RING2_M`)."""
    moved = _step(board, 2, -2) | _step(board, 2, -1) | _step(board, 2, 0) | _step(board, 2, 1) | _step(board, 2, 2)
    moved = moved | _step(board, -2, -2) | _step(board, -2, -1) | _step(board, -2, 0) | _step(board, -2, 1)
    moved = moved | _step(board, -2, 2) | _step(board, -1, -2) | _step(board, 0, -2) | _step(board, 1, -2)
    return moved | _step(board, -1, 2) | _step(board, 0, 2) | _step(board, 1, 2)


@triton.jit
def _reverse(board):
    """Bit reversal of a ``uint64`` board (`__nv_brevll`): square ``s`` moves to ``63 - s``."""
    return libdevice.brev(board.to(tl.int64, bitcast=True)).to(tl.uint64, bitcast=True)


@triton.jit
def _rays(squares):
    """The eight ray masks from every lane's square, in `_SLIDER_DIRS` order: NE, NW, SE, SW, N, S, E, W.

    A ray excludes its own square and runs to the edge of the board. `above` / `below` split each line (file, rank,
    diagonal, anti-diagonal) through the square at the square itself: the four positive directions (+9, +7, +8, +1)
    are the higher bits of their line, the four negative ones (-7, -9, -8, -1) the lower bits.
    """
    shifts = squares.to(tl.uint64)
    bit = tl.full((), 1, tl.uint64) << shifts
    below = bit - 1
    above = ~(bit | below)
    column = squares & 7
    row = squares >> 3
    file = tl.full((), 0x0101010101010101, tl.uint64) << column.to(tl.uint64)
    rank = tl.full((), 0xFF, tl.uint64) << (squares & 56).to(tl.uint64)
    # a1-h8 (bits 0, 9, ..., 63) and h1-a8 (bits 7, 14, ..., 56), moved to the lane's square by row - column.
    main = (tl.full((), 0x80402010, tl.uint64) << 32) | tl.full((), 0x08040201, tl.uint64)
    anti = (tl.full((), 0x01020408, tl.uint64) << 32) | tl.full((), 0x10204080, tl.uint64)
    skew = 8 * column - 8 * row
    diagonal = (main >> tl.maximum(skew, 0).to(tl.uint64)) << tl.maximum(-skew, 0).to(tl.uint64)
    skew = 56 - 8 * column - 8 * row
    antidiagonal = (anti >> tl.maximum(skew, 0).to(tl.uint64)) << tl.maximum(-skew, 0).to(tl.uint64)
    return (diagonal & above, antidiagonal & above, antidiagonal & below, diagonal & below,
            file & above, file & below, rank & above, rank & below)


@triton.jit
def _scan(ray, occupied, positive: tl.constexpr):
    """One direction from every lane: ``(reach, first, xray)``.

    ``reach`` is the ray up to and including its first blocker (`_slider_flood`'s `reach`), ``first`` that blocker
    alone (zero if the ray is open), ``xray`` the squares strictly beyond it up to and including the second blocker
    (`_slider_flood`'s `xray`). For a positive direction the first blocker is the lowest set bit of
    ``ray & occupied`` and ``b ^ (b - 1)`` masks every bit up to it (all ones when there is none); a negative
    direction wants the highest bit, so it runs the same scan on the bit-reversed word.
    """
    blockers = ray & occupied
    if positive:
        upto = blockers ^ (blockers - 1)
        first = blockers & upto
        rest = blockers ^ first
        upto2 = rest ^ (rest - 1)
    else:
        reversed_blockers = _reverse(blockers)
        reversed_upto = reversed_blockers ^ (reversed_blockers - 1)
        upto = _reverse(reversed_upto)
        first = blockers & upto
        reversed_rest = reversed_blockers & ~reversed_upto
        upto2 = _reverse(reversed_rest ^ (reversed_rest - 1))
    return ray & upto, first, ray & upto2 & ~upto


@triton.jit
def _reach(ray, occupied, positive: tl.constexpr):
    """One direction from every lane, first blocker included (`_scan`'s `reach` alone)."""
    blockers = ray & occupied
    if positive:
        upto = blockers ^ (blockers - 1)
    else:
        reversed_blockers = _reverse(blockers)
        upto = _reverse(reversed_blockers ^ (reversed_blockers - 1))
    return ray & upto


@triton.jit
def _plane(masks, values, index):
    """One thresholded piece plane as a bitboard."""
    keep = tl.load(values + index) > 0.5
    return tl.where(keep, tl.load(masks + index), tl.zeros((), dtype=tl.uint64))


@triton.jit
def _attack_rows(sources, pawn, knight, bishop, rook, queen, king, diagonal, orthogonal, forward: tl.constexpr):
    """Attack rows of one side: pawns toward `forward`, knights, king, sliders (`attack_graph.side`)."""
    zero = tl.zeros_like(sources)
    pawn_targets = _step(sources, forward, -1) | _step(sources, forward, 1)
    rows = tl.where((sources & pawn) != 0, pawn_targets, zero)
    rows = rows | tl.where((sources & knight) != 0, _knight_moves(sources), zero)
    rows = rows | tl.where((sources & king) != 0, _king_moves(sources), zero)
    rows = rows | tl.where((sources & (bishop | queen)) != 0, diagonal, zero)
    return rows | tl.where((sources & (rook | queen)) != 0, orthogonal, zero)


@triton.jit
def _frame_rows(masks, values, base, sources, ne, nw, se, sw, n, s, e, w):  # noqa: PLR0913
    """Channels 0/1 of the frame whose planes start at `base` (the previous position, for tdelta)."""
    pawn_us = _plane(masks, values, base)
    knight_us = _plane(masks, values, base + 1)
    bishop_us = _plane(masks, values, base + 2)
    rook_us = _plane(masks, values, base + 3)
    queen_us = _plane(masks, values, base + 4)
    king_us = _plane(masks, values, base + 5)
    pawn_them = _plane(masks, values, base + 6)
    knight_them = _plane(masks, values, base + 7)
    bishop_them = _plane(masks, values, base + 8)
    rook_them = _plane(masks, values, base + 9)
    queen_them = _plane(masks, values, base + 10)
    king_them = _plane(masks, values, base + 11)
    occupied = pawn_us | knight_us | bishop_us | rook_us | queen_us | king_us
    occupied = occupied | pawn_them | knight_them | bishop_them | rook_them | queen_them | king_them
    diagonal = _reach(ne, occupied, True) | _reach(nw, occupied, True)
    diagonal = diagonal | _reach(se, occupied, False) | _reach(sw, occupied, False)
    orthogonal = _reach(n, occupied, True) | _reach(s, occupied, False)
    orthogonal = orthogonal | _reach(e, occupied, True) | _reach(w, occupied, False)
    ours = _attack_rows(sources, pawn_us, knight_us, bishop_us, rook_us, queen_us, king_us, diagonal, orthogonal, 1)
    theirs = _attack_rows(sources, pawn_them, knight_them, bishop_them, rook_them, queen_them, king_them,
                          diagonal, orthogonal, -1)
    return ours, theirs


@triton.jit
def _pin_row(here, first_toward, first_back, sliders):
    """Pinray row of one direction: the lane's first blocker along it, when the lane is occupied and the first
    blocker the other way is a slider that rides this direction (`attack_graph`'s `attacked[b] * first_d[b, :]`)."""
    zero = tl.zeros_like(first_toward)
    return tl.where(here & ((first_back & sliders) != 0), first_toward, zero)


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)],
    key=["batch_count"],
    cache_results=True,
)
@triton.jit
def _prologue_egt_rows_kernel(  # noqa: PLR0915
    rows,
    masks,
    values,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
) -> None:
    """Build the 17 base channels of one position as rows: lane = source square, one board of targets each."""
    sample = tl.program_id(0)
    squares = tl.arange(0, 64)
    sources = tl.full((), 1, tl.uint64) << squares.to(tl.uint64)
    zero = tl.zeros_like(sources)
    base = sample * 112

    pawn_us = _plane(masks, values, base)
    knight_us = _plane(masks, values, base + 1)
    bishop_us = _plane(masks, values, base + 2)
    rook_us = _plane(masks, values, base + 3)
    queen_us = _plane(masks, values, base + 4)
    king_us = _plane(masks, values, base + 5)
    pawn_them = _plane(masks, values, base + 6)
    knight_them = _plane(masks, values, base + 7)
    bishop_them = _plane(masks, values, base + 8)
    rook_them = _plane(masks, values, base + 9)
    queen_them = _plane(masks, values, base + 10)
    king_them = _plane(masks, values, base + 11)
    occupied_us = pawn_us | knight_us | bishop_us | rook_us | queen_us | king_us
    occupied_them = pawn_them | knight_them | bishop_them | rook_them | queen_them | king_them
    occupied = occupied_us | occupied_them
    diagonal_us = bishop_us | queen_us
    diagonal_them = bishop_them | queen_them
    orthogonal_us = rook_us | queen_us
    orthogonal_them = rook_them | queen_them
    here = (sources & occupied) != 0

    # Diagonal directions first, then orthogonal (`_SLIDER_DIRS`): NE, NW, SE, SW, N, S, E, W.
    ne, nw, se, sw, n, s, e, w = _rays(squares)
    reach0, first0, xray0 = _scan(ne, occupied, True)
    reach1, first1, xray1 = _scan(nw, occupied, True)
    reach2, first2, xray2 = _scan(se, occupied, False)
    reach3, first3, xray3 = _scan(sw, occupied, False)
    reach4, first4, xray4 = _scan(n, occupied, True)
    reach5, first5, xray5 = _scan(s, occupied, False)
    reach6, first6, xray6 = _scan(e, occupied, True)
    reach7, first7, xray7 = _scan(w, occupied, False)
    diagonal = reach0 | reach1 | reach2 | reach3
    orthogonal = reach4 | reach5 | reach6 | reach7
    xray_diagonal = xray0 | xray1 | xray2 | xray3
    xray_orthogonal = xray4 | xray5 | xray6 | xray7

    attacks_us = _attack_rows(sources, pawn_us, knight_us, bishop_us, rook_us, queen_us, king_us,
                              diagonal, orthogonal, 1)
    attacks_them = _attack_rows(sources, pawn_them, knight_them, bishop_them, rook_them, queen_them, king_them,
                                diagonal, orthogonal, -1)
    defends_us = attacks_us & occupied_us
    defends_them = attacks_them & occupied_them
    xray_us = tl.where((sources & diagonal_us) != 0, xray_diagonal, zero)
    xray_us = xray_us | tl.where((sources & orthogonal_us) != 0, xray_orthogonal, zero)
    xray_them = tl.where((sources & diagonal_them) != 0, xray_diagonal, zero)
    xray_them = xray_them | tl.where((sources & orthogonal_them) != 0, xray_orthogonal, zero)

    # A slider hits this lane first along d exactly when the lane's first blocker along -d is that slider.
    pin_us = _pin_row(here, first0, first3, diagonal_us) | _pin_row(here, first3, first0, diagonal_us)
    pin_us = pin_us | _pin_row(here, first1, first2, diagonal_us) | _pin_row(here, first2, first1, diagonal_us)
    pin_us = pin_us | _pin_row(here, first4, first5, orthogonal_us) | _pin_row(here, first5, first4, orthogonal_us)
    pin_us = pin_us | _pin_row(here, first6, first7, orthogonal_us) | _pin_row(here, first7, first6, orthogonal_us)
    pin_them = _pin_row(here, first0, first3, diagonal_them) | _pin_row(here, first3, first0, diagonal_them)
    pin_them = pin_them | _pin_row(here, first1, first2, diagonal_them) | _pin_row(here, first2, first1, diagonal_them)
    pin_them = pin_them | _pin_row(here, first4, first5, orthogonal_them)
    pin_them = pin_them | _pin_row(here, first5, first4, orthogonal_them)
    pin_them = pin_them | _pin_row(here, first6, first7, orthogonal_them)
    pin_them = pin_them | _pin_row(here, first7, first6, orthogonal_them)

    ring1_opp = _king_moves(attacks_us) & king_them
    ring1_own = _king_moves(attacks_them) & king_us
    ring2_opp = _ring2_moves(attacks_us) & king_them
    ring2_own = _ring2_moves(attacks_them) & king_us
    block_escape = tl.where(here, _king_moves(sources) & (king_us | king_them), zero)

    previous_us, previous_them = _frame_rows(masks, values, base + 13, sources, ne, nw, se, sw, n, s, e, w)
    gained_us = attacks_us & ~previous_us
    gained_them = attacks_them & ~previous_them
    lost_us = previous_us & ~attacks_us
    lost_them = previous_them & ~attacks_them

    out = rows + sample * (17 * 64) + squares
    tl.store(out, attacks_us)
    tl.store(out + 64, attacks_them)
    tl.store(out + 2 * 64, defends_us)
    tl.store(out + 3 * 64, defends_them)
    tl.store(out + 4 * 64, xray_us)
    tl.store(out + 5 * 64, xray_them)
    tl.store(out + 6 * 64, pin_us)
    tl.store(out + 7 * 64, pin_them)
    tl.store(out + 8 * 64, ring1_opp)
    tl.store(out + 9 * 64, ring1_own)
    tl.store(out + 10 * 64, ring2_opp)
    tl.store(out + 11 * 64, ring2_own)
    tl.store(out + 12 * 64, block_escape)
    tl.store(out + 13 * 64, gained_us)
    tl.store(out + 14 * 64, gained_them)
    tl.store(out + 15 * 64, lost_us)
    tl.store(out + 16 * 64, lost_them)


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4)],
    key=["batch_count", "split"],
    cache_results=True,
)
@triton.jit
def _prologue_egt_edges_kernel(
    edges,
    rows,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    split: tl.constexpr,
) -> None:
    """Pack ``64 // split`` source rows of one position's ``E`` tile from the row kernel's boards.

    Bits 0-16 of cell ``(i, j)`` are bit ``j`` of the program's own rows; bits 17-33 are bit ``i`` of every row
    (the transposes). Both are extracted from 32-bit halves: the program's rows all lie in one half of every row
    word (`split` divides 32), and the columns are handled as two half-tiles.
    """
    tl.static_assert(split >= 2 and 32 % (64 // split) == 0, "split must divide 32 twice over")
    rows_per: tl.constexpr = 64 // split
    program = tl.program_id(0)
    sample = program // split
    first_row = (program % split) * rows_per
    offsets = first_row + tl.arange(0, rows_per)
    columns = tl.arange(0, 32)
    row_shifts = (offsets & 31).to(tl.uint32)
    column_shifts = columns.to(tl.uint32)
    upper = first_row // 32
    words = rows.to(tl.pointer_type(tl.uint32))
    row_base = sample * (17 * 64)

    base_left = tl.zeros((rows_per, 32), dtype=tl.uint32)
    base_right = tl.zeros((rows_per, 32), dtype=tl.uint32)
    transposed_left = tl.zeros((rows_per, 32), dtype=tl.uint32)
    transposed_right = tl.zeros((rows_per, 32), dtype=tl.uint32)
    for channel in tl.static_range(17):
        own = tl.load(rows + row_base + channel * 64 + offsets)
        low = own.to(tl.uint32)
        high = (own >> 32).to(tl.uint32)
        # The half of rows_c[j] that holds bit i for the program's rows i, for the left and right column halves.
        left = tl.load(words + 2 * (row_base + channel * 64 + columns) + upper)
        right = tl.load(words + 2 * (row_base + channel * 64 + 32 + columns) + upper)
        base_left |= ((low[:, None] >> column_shifts[None, :]) & 1) << channel
        base_right |= ((high[:, None] >> column_shifts[None, :]) & 1) << channel
        transposed_left |= ((left[None, :] >> row_shifts[:, None]) & 1) << channel
        transposed_right |= ((right[None, :] >> row_shifts[:, None]) & 1) << channel

    cells = edges + sample * 4096 + offsets[:, None] * 64 + columns[None, :]
    tl.store(cells, base_left.to(tl.uint64) | (transposed_left.to(tl.uint64) << 17))
    tl.store(cells + 32, base_right.to(tl.uint64) | (transposed_right.to(tl.uint64) << 17))


@triton.jit
def _group_sum(mix_codes, row0, row1, row2, row3, column):
    """The four code-group rows' column `column`, summed: one mixed channel for every cell."""
    return (tl.load(mix_codes + row0 + column) + tl.load(mix_codes + row1 + column)
            + tl.load(mix_codes + row2 + column) + tl.load(mix_codes + row3 + column))


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
    key=["batch_count", "norm_f32", "state_f32"],
    cache_results=True,
)
@triton.jit
def _prologue_egt_seed_kernel(  # noqa: PLR0913
    edge_norm,
    edge_state,
    edges,
    offsets,
    edge_offsets,
    mix_codes,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    epsilon: tl.constexpr,
    norm_f32: tl.constexpr,
    state_f32: tl.constexpr,
) -> None:
    """Build ``S`` and ``e0`` for one position from its packed ``E``."""
    sample = tl.program_id(0)
    squares = tl.arange(0, 64)
    tile = 64 * 64
    direct = squares[:, None] * 64 + squares[None, :]
    code = tl.load(edges + sample * tile + direct)
    row0 = (code & 511).to(tl.int32) * 32
    row1 = (((code >> tl.full((), 9, tl.uint64)) & 255).to(tl.int32) + 512) * 32
    row2 = (((code >> tl.full((), 17, tl.uint64)) & 511).to(tl.int32) + 768) * 32
    row3 = (((code >> tl.full((), 26, tl.uint64)) & 255).to(tl.int32) + 1280) * 32

    pair_squares = tl.zeros((64, 64), dtype=tl.float32)
    state_squares = tl.zeros((64, 64), dtype=tl.float32)
    for channel in tl.static_range(16):
        pair = tl.load(offsets + channel * tile + direct) + _group_sum(mix_codes, row0, row1, row2, row3, channel)
        pair_squares += pair * pair
        state = tl.load(edge_offsets + channel * tile + direct) + _group_sum(
            mix_codes, row0, row1, row2, row3, channel + 16)
        state_squares += state * state
    normalization = 1.0 / tl.sqrt(pair_squares * (1.0 / 16.0) + epsilon)
    if norm_f32:
        tl.store(edge_norm + sample * tile + direct, normalization)
    else:
        tl.store(edge_norm + sample * tile + direct, normalization.to(tl.float16))

    scale = 1.0 / tl.sqrt(state_squares * (1.0 / 16.0) + epsilon)
    state_base = edge_state + sample * 16 * tile
    for channel in tl.static_range(16):
        state = tl.load(edge_offsets + channel * tile + direct) + _group_sum(
            mix_codes, row0, row1, row2, row3, channel + 16)
        if state_f32:
            tl.store(state_base + channel * tile + direct, state * scale)
        else:
            tl.store(state_base + channel * tile + direct, (state * scale).to(tl.float16))


@dataclass(frozen=True, slots=True)
class PrologueEgtEdgesSpecialization:
    """Immutable specialization of the packed-``E`` kernels (rows, then the packing kernel with `split` programs
    per position)."""

    batch_count: int
    architecture: int
    split: int = EDGE_SPLIT


@dataclass(frozen=True, slots=True)
class PrologueEgtSpecialization:
    """Immutable EGT2 prologue specialization.

    `norm_f32` stores ``S`` in FP32 instead of the static net's FP16; `state_f32` stores ``e0`` in FP32.
    """

    batch_count: int
    architecture: int
    norm_f32: bool = False
    state_f32: bool = True

    @property
    def edges(self) -> PrologueEgtEdgesSpecialization:
        """Return the packed-``E`` kernels' specialization, shared by every storage choice."""
        return PrologueEgtEdgesSpecialization(self.batch_count, self.architecture)


def buffer_bytes(specialization: PrologueEgtSpecialization) -> dict[str, int]:
    """Return the execution buffers one call writes, in bytes (`edge_rows` is the K1b scratch between kernels)."""
    cells = specialization.batch_count * _SQUARES * _SQUARES
    return {
        "edges": 8 * cells,
        "edge_rows": ROW_BYTES * specialization.batch_count,
        "edge_norm": (4 if specialization.norm_f32 else 2) * cells,
        "edge_state": (4 if specialization.state_f32 else 2) * STATE_CHANNELS * cells,
    }


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one program per position (the row and seed kernels)."""
    return (cast("int", configuration["batch_count"]),)


def _edges_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return `split` programs per position (the packing kernel)."""
    return (cast("int", configuration["batch_count"]) * cast("int", configuration["split"]),)


def launch_prologue_egt_edges(
    edges: torch.Tensor, masks: torch.Tensor, values: torch.Tensor, rows: torch.Tensor | None = None,
    split: int = EDGE_SPLIT,
) -> torch.Tensor:
    """Run the two packed-``E`` kernels on torch tensors (tests and benchmarks); returns the row scratch used."""
    batch = edges.shape[0]
    if rows is None:
        rows = torch.empty((batch, BASE_CHANNELS, _SQUARES), dtype=torch.uint64, device=edges.device)
    _prologue_egt_rows_kernel[_autotune_grid](rows, masks, values, batch)
    _prologue_egt_edges_kernel[_edges_grid](edges, rows, batch, split)
    return rows


def compile_prologue_egt_rows(specialization: PrologueEgtEdgesSpecialization) -> KernelArtifact:
    """Autotune and compile the row kernel (the 17 base channels as per-source boards)."""
    batch = specialization.batch_count
    rows = torch.empty((batch, BASE_CHANNELS, _SQUARES), dtype=torch.uint64, device="cuda")
    masks = torch.zeros((batch, _INPUT_PLANES), dtype=torch.uint64, device="cuda")
    values = torch.ones((batch, _INPUT_PLANES), dtype=torch.float32, device="cuda")
    compiled = _prologue_egt_rows_kernel[_autotune_grid](rows, masks, values, batch)
    return artifact_from_triton(
        compiled, grid=(batch, 1, 1), parameters=(_POINTER,) * 3, autotuner=_prologue_egt_rows_kernel,
    )


def compile_prologue_egt_edges(specialization: PrologueEgtEdgesSpecialization) -> KernelArtifact:
    """Autotune and compile the packing kernel (`split` programs per position)."""
    batch = specialization.batch_count
    edges = torch.empty((batch, _SQUARES, _SQUARES), dtype=torch.uint64, device="cuda")
    rows = torch.zeros((batch, BASE_CHANNELS, _SQUARES), dtype=torch.uint64, device="cuda")
    compiled = _prologue_egt_edges_kernel[_edges_grid](edges, rows, batch, specialization.split)
    return artifact_from_triton(
        compiled, grid=(batch * specialization.split, 1, 1), parameters=(_POINTER,) * 2,
        autotuner=_prologue_egt_edges_kernel,
    )


def compile_prologue_egt_seed(specialization: PrologueEgtSpecialization) -> KernelArtifact:
    """Autotune and compile the ``S`` / ``e0`` kernel."""
    batch = specialization.batch_count
    norm_type = torch.float32 if specialization.norm_f32 else torch.float16
    state_type = torch.float32 if specialization.state_f32 else torch.float16
    edge_norm = torch.empty((batch, _SQUARES, _SQUARES), dtype=norm_type, device="cuda")
    edge_state = torch.empty((batch, STATE_CHANNELS, _SQUARES, _SQUARES), dtype=state_type, device="cuda")
    edges = torch.zeros((batch, _SQUARES, _SQUARES), dtype=torch.uint64, device="cuda")
    offsets = torch.ones((PAIR_CHANNELS, _SQUARES, _SQUARES), dtype=torch.float32, device="cuda")
    edge_offsets = torch.ones((STATE_CHANNELS, _SQUARES, _SQUARES), dtype=torch.float32, device="cuda")
    mix_codes = torch.zeros((CODE_ROWS, CODE_WIDTH), dtype=torch.float32, device="cuda")
    compiled = _prologue_egt_seed_kernel[_autotune_grid](
        edge_norm, edge_state, edges, offsets, edge_offsets, mix_codes,
        batch, EPSILON, specialization.norm_f32, specialization.state_f32,
    )
    return artifact_from_triton(
        compiled, grid=(batch, 1, 1), parameters=(_POINTER,) * 6, autotuner=_prologue_egt_seed_kernel,
    )


def prologue_egt(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    edges: Buffer,
    edge_norm: Buffer,
    edge_state: Buffer,
    masks: Buffer,
    values: Buffer,
    tables: Mapping[str, Buffer],
    specialization: PrologueEgtSpecialization,
) -> None:
    """Append the prologue: the base rows, packed ``E``, then ``S`` and ``e0``, for every position of the batch.

    `tables` maps the plan names of `lab._names.plan_prologue_egt` to their persistent buffers; the seed kernel
    reads `SEED_TABLES` (``D`` and ``p_in`` arrive folded into `/prologue/mix_codes`). The row scratch between
    the two ``E`` kernels (`ROW_BYTES` per position) is a temporary of this program.
    """
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    rows = builder.temporary_buffer(size_bytes=ROW_BYTES * specialization.batch_count, alignment_bytes=256)
    rows_kernel = kernels.get(compile_prologue_egt_rows, specialization.edges)
    builder.call(rows_kernel, rows, masks, values, readonly=(masks, values))
    edge_kernel = kernels.get(compile_prologue_egt_edges, specialization.edges)
    builder.call(edge_kernel, edges, rows, readonly=(rows,))
    seed_kernel = kernels.get(compile_prologue_egt_seed, specialization)
    constants = tuple(tables[name] for name in SEED_TABLES)
    builder.call(seed_kernel, edge_norm, edge_state, edges, *constants, readonly=(edges, *constants))
