"""Static-net prologue: the attack-graph edge tensor and its pair normalization.

The lab's static 512x15 net reads two per-position tensors in every encoder
block and in the policy head, both built once from the input planes:

* ``E [4, 64, 64]`` -- ``[attacks_stm, attacks_opp, attacks_stm^T, attacks_opp^T]``,
  indexed ``[channel, source, target]`` (``attack_graph.edge_channels`` stacked
  with its transpose, ``model.py``).
* ``S [64, 64]`` -- ``1 / sqrt(mean_t Z[t]^2 + 1e-6)`` with
  ``Z = einsum("cij,cd->dij", E, D) + T`` over the 16 pair channels
  (``pair_stream._rmsnorm``; export nodes 2380-2390).

Semantics match the lab, including on boards that are not legal positions: a
piece plane bit counts when its plane value exceeds 0.5 (``const_626``), a
square may carry several pieces (their attacks are OR-ed, as the lab's
``minimum(sum, 1)``), and a slider's ray includes the first occupied square and
nothing behind it. No geometry table is uploaded: leaper patterns come from
rank and file differences, and slider rays from a seven-step flood inside the
kernel, so the only persistent inputs are the export's ``D [4, 16]`` and the
build-time ``T [16, 64, 64]``.
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

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_INPUT_PLANES = 112
_PIECE_PLANES = 12
_SQUARES = 64
EDGE_CHANNELS = 4
PAIR_CHANNELS = 16
# `const_729` in the export, `pair_stream._RMS_EPS` in the lab.
PAIR_EPSILON = 1e-6
# `const_626`: a plane bit counts when its value is above one half.
_PLANE_THRESHOLD = 0.5


@triton.jit
def _ray(occupancy, rows, columns, squares, row_step: tl.constexpr, column_step: tl.constexpr):
    """Return ``[64, 64]``: target reached from source along one direction.

    The flood follows ``attack_graph._slider_flood``: step, record the square,
    and stop once a recorded square is occupied -- so the first blocker is
    included and nothing behind it is.
    """
    reach = squares[:, None] < 0
    alive = squares >= 0
    for distance in tl.static_range(1, 8):
        target_row = rows + distance * row_step
        target_column = columns + distance * column_step
        on_board = (target_row >= 0) & (target_row < 8) & (target_column >= 0) & (target_column < 8)
        target = tl.where(on_board, target_row * 8 + target_column, 0)
        hit = alive & on_board
        reach = reach | ((squares[None, :] == target[:, None]) & hit[:, None])
        occupied = ((occupancy >> target.to(tl.uint64)) & 1) != 0
        alive = hit & (occupied == 0)
    return reach


@triton.jit
def _piece(masks, values, base, plane, squares):
    """Return the ``[64]`` indicator of one piece plane, thresholded."""
    mask = tl.load(masks + base + plane)
    keep = tl.load(values + base + plane) > 0.5
    bits = ((mask >> squares.to(tl.uint64)) & 1) != 0
    return bits & keep


@triton.jit
def _side_attacks(masks, values, base, first: tl.constexpr, pawn_row_step: tl.constexpr,  # noqa: PLR0913
                  squares, row_delta, column_delta, diagonal, orthogonal):
    """Return ``[64, 64]`` attacks of one side: pawns, knights, king, sliders."""
    pawn = _piece(masks, values, base, first, squares)
    knight = _piece(masks, values, base, first + 1, squares)
    bishop = _piece(masks, values, base, first + 2, squares)
    rook = _piece(masks, values, base, first + 3, squares)
    queen = _piece(masks, values, base, first + 4, squares)
    king = _piece(masks, values, base, first + 5, squares)
    absolute_row = tl.abs(row_delta)
    absolute_column = tl.abs(column_delta)
    pawn_pattern = (row_delta == pawn_row_step) & (absolute_column == 1)
    knight_pattern = ((absolute_row == 1) & (absolute_column == 2)) | ((absolute_row == 2) & (absolute_column == 1))
    king_pattern = (tl.maximum(absolute_row, absolute_column) == 1)
    attacks = pawn[:, None] & pawn_pattern
    attacks = attacks | (knight[:, None] & knight_pattern)
    attacks = attacks | (king[:, None] & king_pattern)
    attacks = attacks | ((bishop | queen)[:, None] & diagonal)
    attacks = attacks | ((rook | queen)[:, None] & orthogonal)
    return attacks


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
    key=["batch_count"],
    cache_results=True,
)
@triton.jit
def _prologue_static_kernel(  # noqa: PLR0913
    edges,
    edge_norm,
    masks,
    values,
    channel_mix,
    offsets,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    epsilon: tl.constexpr,
) -> None:
    """Build ``E`` and ``S`` for one position."""
    sample = tl.program_id(0)
    squares = tl.arange(0, 64)
    rows = squares // 8
    columns = squares % 8
    plane_base = sample * 112

    occupancy = tl.zeros((), dtype=tl.uint64)
    for plane in tl.static_range(12):
        mask = tl.load(masks + plane_base + plane)
        keep = tl.load(values + plane_base + plane) > 0.5
        occupancy = occupancy | tl.where(keep, mask, tl.zeros((), dtype=tl.uint64))

    diagonal = _ray(occupancy, rows, columns, squares, 1, 1)
    diagonal = diagonal | _ray(occupancy, rows, columns, squares, 1, -1)
    diagonal = diagonal | _ray(occupancy, rows, columns, squares, -1, 1)
    diagonal = diagonal | _ray(occupancy, rows, columns, squares, -1, -1)
    orthogonal = _ray(occupancy, rows, columns, squares, 1, 0)
    orthogonal = orthogonal | _ray(occupancy, rows, columns, squares, -1, 0)
    orthogonal = orthogonal | _ray(occupancy, rows, columns, squares, 0, 1)
    orthogonal = orthogonal | _ray(occupancy, rows, columns, squares, 0, -1)

    row_delta = rows[None, :] - rows[:, None]
    column_delta = columns[None, :] - columns[:, None]
    ours = _side_attacks(masks, values, plane_base, 0, 1, squares, row_delta, column_delta, diagonal, orthogonal)
    theirs = _side_attacks(masks, values, plane_base, 6, -1, squares, row_delta, column_delta, diagonal, orthogonal)

    tile = 64 * 64
    code_base = sample * tile
    direct = squares[:, None] * 64 + squares[None, :]
    swapped = squares[None, :] * 64 + squares[:, None]
    # P5: E packed one byte per cell -- bit 0 ours(r->c), bit 1 theirs(r->c), bits 2/3
    # their transposes. The low half is written first and read back transposed.
    low = (ours.to(tl.uint8) + (theirs.to(tl.uint8) << 1)).to(tl.uint8)
    tl.store(edges + code_base + direct, low)
    transposed = tl.load(edges + code_base + swapped)
    tl.store(edges + code_base + direct, (low + (transposed << 2)).to(tl.uint8))

    channel0 = ours.to(tl.float32)
    channel1 = theirs.to(tl.float32)
    channel2 = (transposed & 1).to(tl.float32)
    channel3 = ((transposed >> 1) & 1).to(tl.float32)
    squared = tl.zeros((64, 64), dtype=tl.float32)
    for pair_channel in tl.static_range(16):
        mixed = tl.load(channel_mix + 0 * 16 + pair_channel) * channel0
        mixed += tl.load(channel_mix + 1 * 16 + pair_channel) * channel1
        mixed += tl.load(channel_mix + 2 * 16 + pair_channel) * channel2
        mixed += tl.load(channel_mix + 3 * 16 + pair_channel) * channel3
        mixed += tl.load(offsets + pair_channel * tile + direct)
        squared += mixed * mixed
    normalization = 1.0 / tl.sqrt(squared * (1.0 / 16.0) + epsilon)
    tl.store(edge_norm + sample * tile + direct, normalization.to(tl.float16))


@dataclass(frozen=True, slots=True)
class PrologueStaticSpecialization:
    """Immutable static-net prologue specialization."""

    batch_count: int
    architecture: int


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one program per position."""
    return (cast("int", configuration["batch_count"]),)


def compile_prologue_static(specialization: PrologueStaticSpecialization) -> KernelArtifact:
    """Autotune and compile one prologue specialization."""
    batch = specialization.batch_count
    edges = torch.empty((batch, _SQUARES, _SQUARES), dtype=torch.uint8, device="cuda")  # P5: packed E
    edge_norm = torch.empty((batch, _SQUARES, _SQUARES), dtype=torch.float16, device="cuda")
    masks = torch.zeros((batch, _INPUT_PLANES), dtype=torch.uint64, device="cuda")
    values = torch.ones((batch, _INPUT_PLANES), dtype=torch.float32, device="cuda")
    channel_mix = torch.zeros((EDGE_CHANNELS, PAIR_CHANNELS), dtype=torch.float32, device="cuda")
    offsets = torch.ones((PAIR_CHANNELS, _SQUARES, _SQUARES), dtype=torch.float32, device="cuda")
    compiled = _prologue_static_kernel[_autotune_grid](
        edges, edge_norm, masks, values, channel_mix, offsets, batch, PAIR_EPSILON,
    )
    return artifact_from_triton(
        compiled,
        grid=(batch, 1, 1),
        parameters=(_POINTER,) * 6,
        autotuner=_prologue_static_kernel,
    )


def prologue_static(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    edges: Buffer,
    edge_norm: Buffer,
    masks: Buffer,
    values: Buffer,
    channel_mix: Buffer,
    offsets: Buffer,
    specialization: PrologueStaticSpecialization,
) -> None:
    """Append the prologue: ``E`` and ``S`` for every position of the batch."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_prologue_static, specialization)
    builder.call(
        kernel, edges, edge_norm, masks, values, channel_mix, offsets,
        readonly=(masks, values, channel_mix, offsets),
    )
