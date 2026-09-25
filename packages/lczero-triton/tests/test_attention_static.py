"""CUDA numerical and artifact tests for static-mapping attention.

The reference here is deliberately the **direct** form the export's graph
computes -- five separate contractions, with the 16-channel tensor `M` built in
full -- so that passing gates both the fusion and the algebraic collapse the
kernel relies on (see `attention_static`'s module docstring).
"""

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.attention_static import (
    STATIC_EDGE,
    STATIC_FULL,
    STATIC_SCALED,
    STATIC_SMOLGEN,
    StaticAttentionSpecialization,
    _artifact_grid,
    _autotune_grid,
    _static_attention_kernel,
    compile_static_attention,
    code_bits,
    fold_code_sums,
    fold_code_table,
    fold_constant_bias,
    fold_scaled_coefficients,
    padded_head_dim,
    static_attention,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

# The lab's static 512x15 net: 64 squares, 32 heads, head_dim 16, 4 edge
# channels, 16 mix channels.
_TOKENS = 64
_HEADS = 32
_HEAD_DIM = 16
_EDGE_CHANNELS = 4
_MIX_CHANNELS = 16
_ATOL = 3e-2
_RTOL = 1e-2


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _inputs(samples: int) -> dict[str, torch.Tensor]:
    """Build one random instance of every tensor the export's block consumes."""
    torch.manual_seed(0x5741C)

    def half(*shape: int, spread: float = 0.2) -> torch.Tensor:
        return (torch.randn(shape, device="cuda") * spread).half()

    return {
        "queries": half(samples, _TOKENS, _HEADS * _HEAD_DIM),
        "keys": half(samples, _TOKENS, _HEADS * _HEAD_DIM),
        "values": half(samples, _TOKENS, _HEADS * _HEAD_DIM),
        # E is exactly 0 or 1: the kernel reads it packed, one byte per cell (P5).
        "edges": (torch.rand((samples, _EDGE_CHANNELS, _TOKENS, _TOKENS), device="cuda") < 0.3).half(),
        # `rsqrt_out_0` is a normalization, so it is positive and O(1).
        "edge_norm": (
            torch.rand((samples, _TOKENS, _TOKENS), device="cuda") + 0.5
        ).half(),
        # The block's pair tables, [heads, edge_channels, head_dim] as the carrier
        # plans them from the export's [1, heads, edge_channels, head_dim].
        "pair_query": half(_HEADS, _EDGE_CHANNELS, _HEAD_DIM),
        "pair_key": half(_HEADS, _EDGE_CHANNELS, _HEAD_DIM),
        "edge_coefficients": half(_HEADS, _EDGE_CHANNELS),
        "channel_mix": half(_EDGE_CHANNELS, _MIX_CHANNELS),
        "head_mix": half(_HEADS, _MIX_CHANNELS),
        "offset_table": half(_MIX_CHANNELS, _TOKENS, _TOKENS),
        # weight_gen's output, one [tokens, tokens] tile per (sample, head).
        "smolgen": half(samples * _HEADS, _TOKENS, _TOKENS, spread=1.0),
    }


def _direct_reference(
    tensors: dict[str, torch.Tensor],
    samples: int,
    logit_terms: int,
) -> torch.Tensor:
    """Compute the export's own five-term form in FP32, unfused."""
    heads_view = (samples, _TOKENS, _HEADS, _HEAD_DIM)
    queries = tensors["queries"].float().view(heads_view).permute(0, 2, 1, 3)
    keys = tensors["keys"].float().view(heads_view).permute(0, 2, 1, 3)
    values = tensors["values"].float().view(heads_view).permute(0, 2, 1, 3)
    edges = tensors["edges"].float()

    scores = (queries @ keys.transpose(-1, -2)) / (_HEAD_DIM**0.5)
    if logit_terms & STATIC_EDGE:
        scores = scores + torch.einsum(
            "he,zerc->zhrc", tensors["edge_coefficients"].float(), edges
        )
        # The export's MatMul nodes: Qk = K_heads . P_k^T, Qq = Q_heads . P_q^T.
        key_pair = torch.einsum("zhcd,hed->zhce", keys, tensors["pair_key"].float())
        query_pair = torch.einsum("zhrd,hed->zhre", queries, tensors["pair_query"].float())
        scores = scores + torch.einsum("zhce,zerc->zhrc", key_pair, edges)
        scores = scores + torch.einsum("zhre,zerc->zhrc", query_pair, edges)
    if logit_terms & STATIC_SCALED:
        # The 16-channel tensor, built in full exactly as the graph does.
        mixed = torch.einsum("et,zerc->ztrc", tensors["channel_mix"].float(), edges)
        mixed = mixed + tensors["offset_table"].float()
        mixed = mixed * tensors["edge_norm"].float().unsqueeze(1)
        scores = scores + torch.einsum(
            "ht,ztrc->zhrc", tensors["head_mix"].float(), mixed
        )

    if logit_terms & STATIC_SMOLGEN:
        scores = scores + tensors["smolgen"].float().view(samples, _HEADS, _TOKENS, _TOKENS)

    probabilities = torch.softmax(scores, dim=-1).half().float()
    attended = probabilities @ values
    return attended.permute(0, 2, 1, 3).reshape(
        samples, _TOKENS, _HEADS * _HEAD_DIM
    )


def _run_kernel(
    tensors: dict[str, torch.Tensor],
    samples: int,
    logit_terms: int,
    packed: bool = False,
) -> torch.Tensor:
    """Launch the fused kernel with the build-time folded constants."""
    output = torch.empty_like(tensors["queries"])
    width = _HEADS * _HEAD_DIM
    codes = sum(
        (tensors["edges"][:, channel].to(torch.uint8) << channel) for channel in range(_EDGE_CHANNELS)
    ).contiguous()
    if packed:
        qkv = torch.cat((tensors["queries"], tensors["keys"], tensors["values"]), dim=2).contiguous()
        queries = keys = values = qkv
        stride, key_offset, value_offset = 3 * width, width, 2 * width
    else:
        queries, keys, values = tensors["queries"], tensors["keys"], tensors["values"]
        stride, key_offset, value_offset = width, 0, 0
    scaled_coefficients = fold_scaled_coefficients(
        tensors["channel_mix"], tensors["head_mix"]
    ).half()
    constant_bias = fold_constant_bias(
        tensors["offset_table"], tensors["head_mix"]
    ).half()
    # P6: the kernel reads per-code FP32 tables folded from the FP16 per-channel ones.
    query_codes = fold_code_table(tensors["pair_query"])
    key_codes = fold_code_table(tensors["pair_key"])
    coefficient_codes = fold_code_sums(tensors["edge_coefficients"])
    scaled_codes = fold_code_sums(scaled_coefficients)
    scale = torch.full((1,), _HEAD_DIM**-0.5, dtype=torch.float16, device="cuda")
    _static_attention_kernel[_autotune_grid](
        output,
        queries,
        keys,
        values,
        codes,
        tensors["edge_norm"],
        query_codes,
        key_codes,
        coefficient_codes,
        scaled_codes,
        constant_bias,
        scale,
        tensors["smolgen"],
        samples * _HEADS,
        _HEADS,
        _TOKENS,
        _HEAD_DIM,
        padded_head_dim(_HEAD_DIM),
        1 << _EDGE_CHANNELS,
        bool(logit_terms & STATIC_EDGE),
        bool(logit_terms & STATIC_SCALED),
        bool(logit_terms & STATIC_SMOLGEN),
        stride,
        key_offset,
        value_offset,
    )
    return output.float()


@pytest.mark.parametrize("samples", [1, 4])
def test_static_attention_packed_qkv_matches_separate(samples: int) -> None:
    """P3: reading Q, K, V from one packed [q | k | v] buffer changes nothing."""
    tensors = _inputs(samples)
    terms = STATIC_FULL | STATIC_SMOLGEN
    separate = _run_kernel(tensors, samples, terms)
    packed = _run_kernel(tensors, samples, terms, packed=True)
    assert torch.equal(packed, separate)


@pytest.mark.parametrize("samples", [1, 4])
@pytest.mark.parametrize(
    "logit_terms",
    [STATIC_EDGE, STATIC_SCALED, STATIC_FULL, STATIC_EDGE | STATIC_SMOLGEN, STATIC_FULL | STATIC_SMOLGEN],
    ids=["edge", "scaled", "full", "edge+smolgen", "full+smolgen"],
)
def test_static_attention_matches_the_unfused_graph(
    samples: int,
    logit_terms: int,
) -> None:
    """The fused kernel reproduces the export's five-term score assembly."""
    tensors = _inputs(samples)
    produced = _run_kernel(tensors, samples, logit_terms)
    expected = _direct_reference(tensors, samples, logit_terms)
    torch.testing.assert_close(produced, expected, atol=_ATOL, rtol=_RTOL)


def test_folded_constants_reproduce_the_sixteen_channel_term() -> None:
    """`G` and `K` alone reconstruct the mix term, independently of attention.

    This is the algebraic claim the kernel rests on, isolated from the rest of
    the block so that a failure points at the fold and not at the softmax.
    """
    samples = 2
    tensors = _inputs(samples)
    edges = tensors["edges"].float()
    channel_mix = tensors["channel_mix"].float()
    head_mix = tensors["head_mix"].float()
    offset_table = tensors["offset_table"].float()
    normalization = tensors["edge_norm"].float()

    mixed = torch.einsum("et,zerc->ztrc", channel_mix, edges) + offset_table
    direct = torch.einsum("ht,ztrc->zhrc", head_mix, mixed * normalization.unsqueeze(1))

    scaled_coefficients = fold_scaled_coefficients(
        tensors["channel_mix"], tensors["head_mix"]
    )
    constant_bias = fold_constant_bias(tensors["offset_table"], tensors["head_mix"])
    folded = normalization.unsqueeze(1) * (
        torch.einsum("he,zerc->zhrc", scaled_coefficients, edges)
        + constant_bias.unsqueeze(0)
    )
    torch.testing.assert_close(folded, direct, atol=1e-3, rtol=1e-3)


def test_code_tables_reproduce_the_channel_contraction() -> None:
    """P6: gathering the per-code tables by the packed code equals the per-channel sums, in FP64."""
    samples = 2
    tensors = _inputs(samples)
    edges = tensors["edges"].double()
    heads_view = (samples, _TOKENS, _HEADS, _HEAD_DIM)
    queries = tensors["queries"].double().view(heads_view).permute(0, 2, 1, 3)
    keys = tensors["keys"].double().view(heads_view).permute(0, 2, 1, 3)
    pair_query, pair_key = tensors["pair_query"].double(), tensors["pair_key"].double()
    coefficients = tensors["edge_coefficients"].double()
    direct = torch.einsum("he,zerc->zhrc", coefficients, edges)
    direct = direct + torch.einsum("zhre,zerc->zhrc", torch.einsum("zhrd,hed->zhre", queries, pair_query), edges)
    direct = direct + torch.einsum("zhce,zerc->zhrc", torch.einsum("zhcd,hed->zhce", keys, pair_key), edges)

    bits = code_bits(_EDGE_CHANNELS, device=edges.device)
    weights = (2 ** torch.arange(_EDGE_CHANNELS, device=edges.device)).double()
    codes = torch.einsum("zerc,e->zrc", edges, weights).long()
    # x indexes the 2**edge_channels codes; r, c and k index tokens.
    query_table = torch.einsum("hed,xe->hdx", pair_query, bits)
    key_table = torch.einsum("hed,xe->hdx", pair_key, bits)
    row = torch.einsum("zhrd,hdx->zhrx", queries, query_table) + (coefficients @ bits.T)[None, :, None, :]
    column = torch.einsum("zhkd,hdx->zhkx", keys, key_table)
    index = codes[:, None].expand(samples, _HEADS, _TOKENS, _TOKENS)
    gathered = row.gather(3, index) + column.gather(3, index.transpose(2, 3)).transpose(2, 3)
    torch.testing.assert_close(gathered, direct, atol=1e-9, rtol=1e-9)
    # The folds the kernel reads are the same tables, in FP32.
    torch.testing.assert_close(fold_code_table(tensors["pair_query"]).double(), torch.einsum(
        "hed,ce->hdc", tensors["pair_query"].double(), bits), atol=1e-6, rtol=1e-6)


def test_static_attention_compiles_to_an_lc0ex_artifact() -> None:
    """The kernel serializes into an lc0ex program with thirteen pointers."""
    samples = 1
    specialization = StaticAttentionSpecialization(
        batch_count=samples * _HEADS,
        heads=_HEADS,
        tokens=_TOKENS,
        head_dim=_HEAD_DIM,
        edge_channels=_EDGE_CHANNELS,
        logit_terms=STATIC_FULL | STATIC_SMOLGEN,
        architecture=_architecture(),
    )
    artifact = compile_static_attention(specialization)
    assert artifact.grid == _artifact_grid(samples * _HEADS)
    # Triton appends two null parameters of its own; `test_attention_fused`
    # asserts the same shape.
    assert artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 13 + (
        lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,
    ) * 2

    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)

    def buffer(name: str, shape: tuple[int, ...], dtype: int = lc0ex_pb2.Buffer.DATA_TYPE_F16) -> object:
        return builder.persistent_buffer(
            name=name,
            shape=shape,
            dtype=dtype,
            alignment_bytes=256,
        )

    width = _HEADS * _HEAD_DIM
    static_attention(
        builder,
        kernels,
        buffer("sa/out", (samples, _TOKENS, width)),
        buffer("sa/q", (samples, _TOKENS, width)),
        buffer("sa/k", (samples, _TOKENS, width)),
        buffer("sa/v", (samples, _TOKENS, width)),
        buffer("sa/e", (samples, _EDGE_CHANNELS, _TOKENS, _TOKENS)),
        buffer("sa/norm", (samples, _TOKENS, _TOKENS)),
        # P6: per-code FP32 tables.
        buffer("sa/pq", (_HEADS, _HEAD_DIM, 1 << _EDGE_CHANNELS), lc0ex_pb2.Buffer.DATA_TYPE_F32),
        buffer("sa/pk", (_HEADS, _HEAD_DIM, 1 << _EDGE_CHANNELS), lc0ex_pb2.Buffer.DATA_TYPE_F32),
        buffer("sa/c1", (_HEADS, 1 << _EDGE_CHANNELS), lc0ex_pb2.Buffer.DATA_TYPE_F32),
        buffer("sa/g", (_HEADS, 1 << _EDGE_CHANNELS), lc0ex_pb2.Buffer.DATA_TYPE_F32),
        buffer("sa/kbias", (_HEADS, _TOKENS, _TOKENS)),
        buffer("sa/scale", (1,)),
        specialization,
        smolgen_logits=buffer("sa/smolgen", (samples * _HEADS, _TOKENS, _TOKENS)),
    )
