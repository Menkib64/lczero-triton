"""Fused attention for the lab's static-mapping nets.

The static net replaces smolgen with a board-derived edge tensor and adds four
terms to the pre-softmax logits. Read straight out of the export's graph, one
encoder block computes

    scores[z,h,r,c] = (Q[z,h,r,:] . K[z,h,c,:]) / sqrt(head_dim)
                    + sum_e C1[h,e]         * E[z,e,r,c]        # const_733  [1,32,4]
                    + sum_e Qk[z,h,c,e]     * E[z,e,r,c]        # key-side, board-fed
                    + sum_e Qq[z,h,r,e]     * E[z,e,r,c]        # query-side, board-fed
                    + sum_t C2[h,t]         * M[z,t,r,c]        # const_736  [1,32,16]

where ``E`` is the 4-channel edge tensor built once in the prologue and shared
by all 15 blocks, and ``M`` is a 16-channel tensor also derived from ``E``:

    M[z,t,r,c] = (sum_e D[e,t] * E[z,e,r,c] + T[t,r,c]) * S[z,r,c]

with ``D`` and ``T`` constants and ``S`` (`rsqrt_out_0`) a **head-independent**
`[batch,1,64,64]` normalization. Because ``S`` carries no ``t`` and no ``h``
index, the last term factors:

    sum_t C2[h,t] * M[z,t,r,c] = S[z,r,c] * ( sum_e G[h,e] * E[z,e,r,c]
                                            + K[h,r,c] )
    G = C2 @ D.T      -> [heads, edge_channels]   folded at build time
    K = einsum(C2, T) -> [heads, tokens, tokens]  folded at build time, and it
                         is batch-independent, so it is shared by every sample

All four bias terms therefore collapse into **one** contraction over the four
edge channels, with a per-element coefficient assembled from a scalar, a row
vector, a column vector and one normalization tile:

    coefficient[e][r,c] = C1[h,e] + G[h,e]*S[z,r,c] + Qq[z,h,r,e] + Qk[z,h,c,e]

``Qq[z,h,r,e] = sum_d Q[z,h,r,d] P_q[h,e,d]`` and ``Qk`` likewise, from the
block's ``[heads, edge_channels, head_dim]`` pair tables (export MatMul nodes
2454 and 2445).

P6 (round 20c): the channel contraction as three lookups
---------------------------------------------------------
``E`` is binary and arrives packed one byte per cell (P5), so a cell holds one of
``2**edge_channels`` codes (16 here) and every channel sum above is a function
of the code alone. Four small per-block tables fold them at build time
(`fold_code_table`, `fold_code_sums`; the carrier plans `/mha/edge/*_codes` and
`/pair/scaled_codes`):

    query_codes[h,d,code]     = sum_e bit_e(code) * P_q[h,e,d]
    key_codes[h,d,code]       = sum_e bit_e(code) * P_k[h,e,d]
    coefficient_codes[h,code] = sum_e bit_e(code) * C1[h,e]
    scaled_codes[h,code]      = sum_e bit_e(code) * G[h,e]

and one (sample, head) program computes

    row_table[r,code]    = Q[r,:] . query_codes[h,:,code] + coefficient_codes[h,code]
    column_table[c,code] = K[c,:] . key_codes[h,:,code]
    scores[r,c]         += row_table[r, code[r,c]] + column_table[c, code[r,c]]
                           + S[r,c] * (scaled_codes[h, code[r,c]] + K[h,r,c])

Two `[tokens, codes]` dot products and three gathers replace the four-channel
loop that built a coefficient tile, decoded a bit tile, multiplied and added
per channel -- about forty FP32 passes over `[tokens, tokens]`. P5 showed that
compute, not the memory traffic, was this kernel's cost (28.6 µs per call against
the smolgen kernel's 9.4 µs at 16 heads, `REPORT_lc0ex_round20b` §5.6). The tables
hold FP32 sums of the FP16 values the loop read, so the change reorders FP32
additions: in class with the loop, not bit-identical.

Unlike `attention_fused`, the score product is **not** rounded to FP16 before
the softmax. That rounding exists there only to stay bit-identical to the three
kernels it replaces; there is no such incumbent here, so the reference to match
is the export's own FP32 arithmetic.

O (round 22): the attention output gate
---------------------------------------
`prenorm_ogate` multiplies the merged attention output by `2 sigmoid(x W_g + b_g)`
per (square, channel) before the out-projection, `x` being what Q, K and V read.
With ``output_gate`` the head program multiplies its attended `[tokens, head_dim]`
tile by `gate_scale * sigmoid(g)` **in FP32** before the FP16 store -- the lab
computes the gate in float32 because a near-1 gate rounded to a 16-bit format
perturbs the whole branch -- reading `g` either from a fourth lane of the packed
projection (``gate_packed``: ``[q | k | v | g]``, stride 4 width, ``gate_offset``
= 3 width, through the `queries` pointer) or from the kernel's auxiliary pointer
(the separate-GEMM form; the same slot carries smolgen's logits on the smolgen
twin, and a block has one or the other). No new pointer: the artifact keeps its
thirteen, and with the flags off the kernel is what it was.
"""

import os
from collections.abc import Mapping, Sequence
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

# Static-net logit terms, combinable as a bit set so that a build can enable
# them one at a time and a test can gate each against its own reference.
STATIC_NONE = 0
STATIC_EDGE = 1  # the three terms that read E directly: C1, Qq, Qk
STATIC_SCALED = 2  # the folded 16-channel term: S * (G.E + K)
STATIC_FULL = STATIC_EDGE | STATIC_SCALED
STATIC_SMOLGEN = 4  # a generated per-(sample, head) [tokens, tokens] bias: the smolgen twin

# Round 23d: which static attention kernel a build compiles. "p6" (default) is `_static_attention_kernel`;
# "r23d" is `_static_attention_kernel_r23d` (the scaled term gathered from a [codes] tile).
_STATIC_KERNELS = ("p6", "r23d")
_STATIC_KERNEL = os.environ.get("LC0EX_STATIC_KERNEL", "p6")

# One byte per cell holds the packed edge tensor (P5).
MAXIMUM_EDGE_CHANNELS = 8

# `tl.dot` requires a reduction dimension of at least 16.
_MINIMUM_DOT_REDUCTION = 16


def padded_head_dim(head_dim: int) -> int:
    """Return the `tl.arange` extent used for the head-depth axis."""
    return max(_MINIMUM_DOT_REDUCTION, cast("int", triton.next_power_of_2(head_dim)))


def code_bits(edge_channels: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Return the FP64 `[codes, edge_channels]` matrix whose entry `(code, e)` is bit `e` of `code`."""
    codes = torch.arange(1 << edge_channels, device=device)
    channels = torch.arange(edge_channels, device=device)
    return ((codes[:, None] >> channels[None, :]) & 1).double()


def fold_code_table(pair: torch.Tensor) -> torch.Tensor:
    """Fold a `[heads, edge_channels, head_dim]` pair table into FP32 `[heads, head_dim, codes]`."""
    bits = code_bits(pair.shape[1], device=pair.device)
    return torch.einsum("hed,ce->hdc", pair.double(), bits).float().contiguous()


def fold_code_sums(coefficients: torch.Tensor) -> torch.Tensor:
    """Fold `[heads, edge_channels]` per-channel coefficients into FP32 `[heads, codes]`."""
    bits = code_bits(coefficients.shape[1], device=coefficients.device)
    return (coefficients.double() @ bits.T).float().contiguous()


# FP32 `[tokens, tokens]` tiles live at the peak: scores, codes, the gathered
# terms and the normalization, so the narrow warp counts that suit
# `attention_fused` spill here; the candidate set starts wider.
_WARP_CONFIGURATIONS = (
    (4, 1),
    (4, 2),
    (8, 1),
    (8, 2),
    (8, 3),
    (16, 1),
    (16, 2),
)


def _attention_configs() -> list[triton.Config]:
    """Return warp-count and pipeline-depth candidates for one head block."""
    return [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps, num_stages in _WARP_CONFIGURATIONS
    ]


@triton.autotune(
    configs=_attention_configs(),
    key=[
        "batch_count",
        "heads",
        "tokens",
        "head_dim",
        "code_count",
        "use_edge",
        "use_scaled",
        "use_smolgen",
        "qkv_stride",
        "output_gate",
        "gate_packed",
    ],
    cache_results=True,
)
@triton.jit
def _static_attention_kernel(
    output,
    queries,
    keys,
    values,
    edges,
    edge_norm,
    query_codes,
    key_codes,
    coefficient_codes,
    scaled_codes,
    constant_bias,
    scale,
    auxiliary,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    heads: tl.constexpr,
    tokens: tl.constexpr,
    head_dim: tl.constexpr,
    depth_extent: tl.constexpr,
    code_count: tl.constexpr,
    use_edge: tl.constexpr,
    use_scaled: tl.constexpr,
    use_smolgen: tl.constexpr,
    qkv_stride: tl.constexpr,
    key_offset: tl.constexpr,
    value_offset: tl.constexpr,
    output_gate: tl.constexpr = False,  # noqa: FBT002
    gate_packed: tl.constexpr = False,  # noqa: FBT002
    gate_offset: tl.constexpr = 0,
    gate_scale: tl.constexpr = 2.0,
) -> None:
    """Compute one head's static-mapping attention block end to end."""
    matrix = tl.program_id(0)
    sample = matrix // heads
    head = matrix % heads
    width = heads * head_dim

    rows = tl.arange(0, tokens)
    columns = tl.arange(0, tokens)
    depth = tl.arange(0, depth_extent)
    depth_mask = depth < head_dim
    code_range = tl.arange(0, code_count)

    head_base = sample * tokens * width + head * head_dim
    # P3: Q, K and V may arrive packed per row as [q | k | v] from one GEMM; the
    # separate layout is qkv_stride = width with zero offsets, the same indices as before.
    qkv_base = sample * tokens * qkv_stride + head * head_dim
    query_values = tl.load(
        queries + qkv_base + rows[:, None] * qkv_stride + depth[None, :],
        mask=depth_mask[None, :],
        other=0.0,
    )
    key_values = tl.load(
        keys + key_offset + qkv_base + depth[:, None] + columns[None, :] * qkv_stride,
        mask=depth_mask[:, None],
        other=0.0,
    )
    scores = tl.dot(query_values, key_values, out_dtype=tl.float32)
    scores *= tl.load(scale).to(tl.float32)

    # P5: E arrives packed, one byte per cell. P6: that byte indexes the per-code tables.
    codes = tl.load(
        edges + sample * tokens * tokens + rows[:, None] * tokens + columns[None, :]
    ).to(tl.int32)
    if use_edge:
        table_base = head * head_dim * code_count
        query_table = tl.load(
            query_codes + table_base + depth[:, None] * code_count + code_range[None, :],
            mask=depth_mask[:, None],
            other=0.0,
        )
        # [tokens, codes]: each row's query-side term for every code, plus the code's C1 sum.
        row_table = tl.dot(query_values.to(tl.float32), query_table, out_dtype=tl.float32)
        row_table += tl.load(coefficient_codes + head * code_count + code_range)[None, :]
        scores += tl.gather(row_table, codes, 1)
        key_table = tl.load(
            key_codes + table_base + depth[:, None] * code_count + code_range[None, :],
            mask=depth_mask[:, None],
            other=0.0,
        )
        # [tokens, codes] over key columns; gathered through the transposed codes.
        column_table = tl.dot(
            tl.trans(key_values).to(tl.float32), key_table, out_dtype=tl.float32
        )
        scores += tl.trans(tl.gather(column_table, tl.trans(codes), 1))

    if use_scaled:
        # One normalization tile per sample, shared by every head of that sample.
        normalization = tl.load(
            edge_norm
            + sample * tokens * tokens
            + rows[:, None] * tokens
            + columns[None, :]
        ).to(tl.float32)
        # `constant_bias` carries no sample index: one `[heads, tokens, tokens]`
        # table serves the whole batch, so it is L2-resident by construction.
        constant = tl.load(
            constant_bias
            + head * tokens * tokens
            + rows[:, None] * tokens
            + columns[None, :]
        ).to(tl.float32)
        scores += normalization * (tl.load(scaled_codes + head * code_count + codes) + constant)

    if use_smolgen:
        # Row `matrix` of the `[samples * heads, tokens * tokens]` weight_gen output,
        # added unscaled after QK^T is scaled -- the export's order (smolgen twin).
        scores += tl.load(
            auxiliary
            + matrix * tokens * tokens
            + rows[:, None] * tokens
            + columns[None, :]
        ).to(tl.float32)

    is_nan = scores != scores  # noqa: PLR0124  # Device-side NaN test.
    # The CUDA reference clamps after the FP32 addition without hiding NaNs.
    clamped = tl.minimum(tl.maximum(scores, -131008.0), 131008.0)
    scores = tl.where(is_nan, scores, clamped)

    maximum = tl.max(scores, axis=1)
    exponentials = tl.exp(scores - maximum[:, None])
    denominator = tl.sum(exponentials, axis=1)
    probabilities = (exponentials / denominator[:, None]).to(tl.float16)

    value_values = tl.load(
        values + value_offset + qkv_base + columns[:, None] * qkv_stride + depth[None, :],
        mask=depth_mask[None, :],
        other=0.0,
    )
    attended = tl.dot(probabilities, value_values, out_dtype=tl.float32)
    if output_gate:
        # O: `gate_scale * sigmoid(g)` per (square, channel) of this head, in FP32 on the FP32 accumulator, so the
        # only rounding is the store's. The lab computes the gate in float32 for the same reason.
        if gate_packed:
            gate_pointers = queries + gate_offset + qkv_base + rows[:, None] * qkv_stride + depth[None, :]
        else:
            gate_pointers = auxiliary + head_base + rows[:, None] * width + depth[None, :]
        pre_activation = tl.load(gate_pointers, mask=depth_mask[None, :], other=0.0).to(tl.float32)
        attended = attended * (gate_scale * tl.sigmoid(pre_activation))
    tl.store(
        output + head_base + rows[:, None] * width + depth[None, :],
        attended.to(tl.float16),
        mask=depth_mask[None, :],
    )


@triton.autotune(
    configs=_attention_configs(),
    key=[
        "batch_count",
        "heads",
        "tokens",
        "head_dim",
        "code_count",
        "use_edge",
        "use_scaled",
        "use_smolgen",
        "qkv_stride",
        "output_gate",
        "gate_packed",
    ],
    cache_results=True,
)
@triton.jit
def _static_attention_kernel_r23d(
    output,
    queries,
    keys,
    values,
    edges,
    edge_norm,
    query_codes,
    key_codes,
    coefficient_codes,
    scaled_codes,
    constant_bias,
    scale,
    auxiliary,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
    heads: tl.constexpr,
    tokens: tl.constexpr,
    head_dim: tl.constexpr,
    depth_extent: tl.constexpr,
    code_count: tl.constexpr,
    use_edge: tl.constexpr,
    use_scaled: tl.constexpr,
    use_smolgen: tl.constexpr,
    qkv_stride: tl.constexpr,
    key_offset: tl.constexpr,
    value_offset: tl.constexpr,
    output_gate: tl.constexpr = False,  # noqa: FBT002
    gate_packed: tl.constexpr = False,  # noqa: FBT002
    gate_offset: tl.constexpr = 0,
    gate_scale: tl.constexpr = 2.0,
) -> None:
    """P6 with the scaled term gathered from a [codes] tile (round 23d); see the module docstring."""
    matrix = tl.program_id(0)
    sample = matrix // heads
    head = matrix % heads
    width = heads * head_dim

    rows = tl.arange(0, tokens)
    columns = tl.arange(0, tokens)
    depth = tl.arange(0, depth_extent)
    depth_mask = depth < head_dim
    code_range = tl.arange(0, code_count)

    head_base = sample * tokens * width + head * head_dim
    # P3: Q, K and V may arrive packed per row as [q | k | v] from one GEMM; the
    # separate layout is qkv_stride = width with zero offsets, the same indices as before.
    qkv_base = sample * tokens * qkv_stride + head * head_dim
    query_values = tl.load(
        queries + qkv_base + rows[:, None] * qkv_stride + depth[None, :],
        mask=depth_mask[None, :],
        other=0.0,
    )
    key_values = tl.load(
        keys + key_offset + qkv_base + depth[:, None] + columns[None, :] * qkv_stride,
        mask=depth_mask[:, None],
        other=0.0,
    )
    scores = tl.dot(query_values, key_values, out_dtype=tl.float32)
    scores *= tl.load(scale).to(tl.float32)

    # P5: E arrives packed, one byte per cell. P6: that byte indexes the per-code tables.
    codes = tl.load(
        edges + sample * tokens * tokens + rows[:, None] * tokens + columns[None, :]
    ).to(tl.int32)
    if use_edge:
        table_base = head * head_dim * code_count
        query_table = tl.load(
            query_codes + table_base + depth[:, None] * code_count + code_range[None, :],
            mask=depth_mask[:, None],
            other=0.0,
        )
        # [tokens, codes]: each row's query-side term for every code, plus the code's C1 sum.
        row_table = tl.dot(query_values.to(tl.float32), query_table, out_dtype=tl.float32)
        row_table += tl.load(coefficient_codes + head * code_count + code_range)[None, :]
        scores += tl.gather(row_table, codes, 1)
        key_table = tl.load(
            key_codes + table_base + depth[:, None] * code_count + code_range[None, :],
            mask=depth_mask[:, None],
            other=0.0,
        )
        # [tokens, codes] over key columns; gathered through the transposed codes.
        column_table = tl.dot(
            tl.trans(key_values).to(tl.float32), key_table, out_dtype=tl.float32
        )
        scores += tl.trans(tl.gather(column_table, tl.trans(codes), 1))

    if use_scaled:
        # One normalization tile per sample, shared by every head of that sample.
        normalization = tl.load(
            edge_norm
            + sample * tokens * tokens
            + rows[:, None] * tokens
            + columns[None, :]
        ).to(tl.float32)
        # `constant_bias` carries no sample index: one `[heads, tokens, tokens]`
        # table serves the whole batch, so it is L2-resident by construction.
        constant = tl.load(
            constant_bias
            + head * tokens * tokens
            + rows[:, None] * tokens
            + columns[None, :]
        ).to(tl.float32)
        # r23d: the head's per-code coefficients as one [codes] tile, gathered through the flattened codes (a pointer
        # load over the [tokens, tokens] cells was 3-6 us of this kernel at batch 64 on the RTX 4090).
        scaled_row = tl.load(scaled_codes + head * code_count + code_range)
        scaled_cells = tl.reshape(tl.gather(scaled_row, tl.reshape(codes, (tokens * tokens,)), 0), (tokens, tokens))
        scores += normalization * (scaled_cells + constant)

    if use_smolgen:
        # Row `matrix` of the `[samples * heads, tokens * tokens]` weight_gen output,
        # added unscaled after QK^T is scaled -- the export's order (smolgen twin).
        scores += tl.load(
            auxiliary
            + matrix * tokens * tokens
            + rows[:, None] * tokens
            + columns[None, :]
        ).to(tl.float32)

    is_nan = scores != scores  # noqa: PLR0124  # Device-side NaN test.
    # The CUDA reference clamps after the FP32 addition without hiding NaNs.
    clamped = tl.minimum(tl.maximum(scores, -131008.0), 131008.0)
    scores = tl.where(is_nan, scores, clamped)

    maximum = tl.max(scores, axis=1)
    exponentials = tl.exp(scores - maximum[:, None])
    denominator = tl.sum(exponentials, axis=1)
    probabilities = (exponentials / denominator[:, None]).to(tl.float16)

    value_values = tl.load(
        values + value_offset + qkv_base + columns[:, None] * qkv_stride + depth[None, :],
        mask=depth_mask[None, :],
        other=0.0,
    )
    attended = tl.dot(probabilities, value_values, out_dtype=tl.float32)
    if output_gate:
        # O: `gate_scale * sigmoid(g)` per (square, channel) of this head, in FP32 on the FP32 accumulator, so the
        # only rounding is the store's. The lab computes the gate in float32 for the same reason.
        if gate_packed:
            gate_pointers = queries + gate_offset + qkv_base + rows[:, None] * qkv_stride + depth[None, :]
        else:
            gate_pointers = auxiliary + head_base + rows[:, None] * width + depth[None, :]
        pre_activation = tl.load(gate_pointers, mask=depth_mask[None, :], other=0.0).to(tl.float32)
        attended = attended * (gate_scale * tl.sigmoid(pre_activation))
    tl.store(
        output + head_base + rows[:, None] * width + depth[None, :],
        attended.to(tl.float16),
        mask=depth_mask[None, :],
    )


@dataclass(frozen=True, slots=True)
class StaticAttentionSpecialization:
    """Immutable static-mapping attention specialization."""

    batch_count: int
    heads: int
    tokens: int
    head_dim: int
    edge_channels: int
    logit_terms: int
    architecture: int
    # P3: queries, keys and values are one [samples, tokens, 3 * width] buffer.
    packed_qkv: bool = False
    # O: multiply the output by `gate_scale * sigmoid(g)`; `g` is the fourth lane of the packed buffer
    # ([q | k | v | g], requires packed_qkv) or a separate [samples, tokens, width] buffer.
    output_gate: bool = False
    gate_packed: bool = False
    gate_scale: float = 2.0


def fold_scaled_coefficients(
    channel_mix: torch.Tensor,
    head_mix: torch.Tensor,
) -> torch.Tensor:
    """Fold ``C2 @ D.T`` into per-head edge-channel weights ``G``.

    ``channel_mix`` is the export's ``[edge_channels, mix_channels]`` constant
    (`bcast_out_223`'s initializer) and ``head_mix`` its
    ``[heads, mix_channels]`` constant (`const_736`).
    """
    return head_mix.float() @ channel_mix.float().T


def fold_constant_bias(
    offset_table: torch.Tensor,
    head_mix: torch.Tensor,
) -> torch.Tensor:
    """Fold ``C2 @ T`` into a batch-independent ``[heads, tokens, tokens]`` table.

    ``offset_table`` is the export's ``[mix_channels, tokens, tokens]``
    relative-position constant (`transpose_gather_data_1`).
    """
    return torch.einsum("ht,trc->hrc", head_mix.float(), offset_table.float())


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the one-program-per-head launch grid for a tuning candidate."""
    return (cast("int", configuration["batch_count"]),)


def _artifact_grid(batch_count: int) -> tuple[int, int, int]:
    """Resolve the serialized grid; it does not depend on the configuration."""
    return (batch_count, 1, 1)


def compile_static_attention(
    specialization: StaticAttentionSpecialization,
) -> KernelArtifact:
    """Autotune and compile one static-mapping attention specialization."""
    if not STATIC_NONE < specialization.logit_terms <= STATIC_FULL | STATIC_SMOLGEN:
        message = (
            f"logit_terms={specialization.logit_terms} is out of range; use "
            "STATIC_EDGE, STATIC_SCALED or STATIC_FULL."
        )
        raise ValueError(message)
    channels = specialization.edge_channels
    if channels > MAXIMUM_EDGE_CHANNELS:
        message = f"packed E holds at most {MAXIMUM_EDGE_CHANNELS} edge channels, got {channels}"
        raise ValueError(message)

    if specialization.gate_packed and not (specialization.packed_qkv and specialization.output_gate):
        message = "gate_packed needs packed_qkv and output_gate: the gate is the fourth lane of [q | k | v | g]"
        raise ValueError(message)
    if specialization.output_gate and not specialization.gate_packed and specialization.logit_terms & STATIC_SMOLGEN:
        message = "a separate output gate and smolgen logits share the kernel's auxiliary pointer; use gate_packed"
        raise ValueError(message)
    samples = specialization.batch_count // specialization.heads
    tokens = specialization.tokens
    code_count = 1 << channels
    width = specialization.heads * specialization.head_dim
    merged = (samples, tokens, width)
    output = torch.empty(merged, dtype=torch.float16, device="cuda")
    lanes = 4 if specialization.gate_packed else (3 if specialization.packed_qkv else 1)
    queries = torch.zeros((samples, tokens, lanes * width), dtype=torch.float16, device="cuda")
    keys = queries
    values = queries
    edges = torch.zeros((samples, tokens, tokens), dtype=torch.uint8, device="cuda")  # P5: packed E
    edge_norm = torch.zeros(
        (samples, tokens, tokens),
        dtype=torch.float16,
        device="cuda",
    )
    # P6: per-code FP32 tables.
    code_table = torch.zeros(
        (specialization.heads, specialization.head_dim, code_count),
        dtype=torch.float32,
        device="cuda",
    )
    code_sums = torch.zeros(
        (specialization.heads, code_count),
        dtype=torch.float32,
        device="cuda",
    )
    constant_bias = torch.zeros(
        (specialization.heads, tokens, tokens),
        dtype=torch.float16,
        device="cuda",
    )
    scale = torch.ones(1, dtype=torch.float16, device="cuda")
    # smolgen logits [samples * heads, tokens, tokens] or a separate gate [samples, tokens, width], never both.
    auxiliary = torch.zeros(
        (max(specialization.batch_count * tokens * tokens, samples * tokens * width),),
        dtype=torch.float16,
        device="cuda",
    )

    if _STATIC_KERNEL not in _STATIC_KERNELS:
        message = f"LC0EX_STATIC_KERNEL={_STATIC_KERNEL!r}; expected one of {_STATIC_KERNELS}"
        raise ValueError(message)
    kernel = _static_attention_kernel_r23d if _STATIC_KERNEL == "r23d" else _static_attention_kernel
    compiled = kernel[_autotune_grid](
        output,
        queries,
        keys,
        values,
        edges,
        edge_norm,
        code_table,
        code_table,
        code_sums,
        code_sums,
        constant_bias,
        scale,
        auxiliary,
        specialization.batch_count,
        specialization.heads,
        tokens,
        specialization.head_dim,
        padded_head_dim(specialization.head_dim),
        code_count,
        bool(specialization.logit_terms & STATIC_EDGE),
        bool(specialization.logit_terms & STATIC_SCALED),
        bool(specialization.logit_terms & STATIC_SMOLGEN),
        lanes * width,
        width if specialization.packed_qkv else 0,
        2 * width if specialization.packed_qkv else 0,
        output_gate=specialization.output_gate,
        gate_packed=specialization.gate_packed,
        gate_offset=3 * width if specialization.gate_packed else 0,
        gate_scale=specialization.gate_scale,
    )
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(specialization.batch_count),
        parameters=(_POINTER,) * 13,
    )


def static_attention(  # One buffer per graph tensor.
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    edges: Buffer,
    edge_norm: Buffer,
    query_codes: Buffer,
    key_codes: Buffer,
    coefficient_codes: Buffer,
    scaled_codes: Buffer,
    constant_bias: Buffer,
    scale: Buffer,
    specialization: StaticAttentionSpecialization,
    smolgen_logits: Buffer | None = None,
    gates: Buffer | None = None,
) -> None:
    """Append one static-mapping attention block.

    O: `gates` is the separate gate projection's [samples, tokens, width] buffer, required exactly when
    `output_gate` is set without `gate_packed`; in the packed form the gate lane lives in `queries`.
    """
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    uses_smolgen = bool(specialization.logit_terms & STATIC_SMOLGEN)
    if uses_smolgen != (smolgen_logits is not None):
        message = "STATIC_SMOLGEN and a smolgen_logits buffer must come together"
        raise ValueError(message)
    separate_gate = specialization.output_gate and not specialization.gate_packed
    if separate_gate != (gates is not None):
        message = "a separate output gate (output_gate without gate_packed) and a gates buffer must come together"
        raise ValueError(message)
    if uses_smolgen and gates is not None:
        message = "smolgen logits and a separate output gate share the auxiliary pointer; use gate_packed"
        raise ValueError(message)
    auxiliary = smolgen_logits if smolgen_logits is not None else gates
    kernel = kernels.get(compile_static_attention, specialization)
    arguments: tuple[Buffer, ...] = (
        output,
        queries,
        keys,
        values,
        edges,
        edge_norm,
        query_codes,
        key_codes,
        coefficient_codes,
        scaled_codes,
        constant_bias,
        scale,
        # Without smolgen or a separate gate the pointer is never dereferenced (as `fused_attention`).
        auxiliary if auxiliary is not None else scale,
    )
    readonly: Sequence[Buffer] = [
        source for source in arguments[1:] if source is not output
    ]
    builder.call(kernel, *arguments, readonly=readonly)
