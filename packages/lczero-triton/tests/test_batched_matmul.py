"""Tests for indexed batched attention matrix multiplication."""

from collections.abc import Sequence

import pytest
import torch
from lc0ex import Buffer, ExecutableBuilder, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.batched_matmul import (
    _OPERATIONS,
    _POLICY_RECORD_SIZE,
    _TILE_CONFIGS,
    BatchedMatmulOperation,
    BatchedMatmulSpecialization,
    _artifact_grid,
    _attention_v_kernel,
    _autotune_grid,
    _qk_kernel,
    batched_matmul,
    compile_batched_matmul,
)

_FP16_ATOL = 3e-2
_FP16_RTOL = 2e-2
_TEST_HEADS = 2
_NULL_POINTERS = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2
_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare FP16 kernel output against an FP32 matrix reference."""
    torch.testing.assert_close(
        actual,
        expected.to(torch.float16),
        rtol=_FP16_RTOL,
        atol=_FP16_ATOL,
    )


def test_autotune_keys_cover_every_indexing_decision() -> None:
    """Static shapes and physical layouts participate in tuning lookup."""
    assert _qk_kernel.keys == [
        "operation",
        "batch_count",
        "heads_per_sample",
        "m",
        "n",
        "k",
        "output_batch_stride",
    ]
    assert _attention_v_kernel.keys == [
        "batch_count",
        "heads_per_sample",
        "m",
        "n",
        "k",
    ]
    assert _qk_kernel.cache_results
    assert _attention_v_kernel.cache_results

    qk_configs = {
        (
            config.kwargs["block_m"],
            config.kwargs["block_n"],
            config.kwargs["block_k"],
            config.num_warps,
            config.num_stages,
        )
        for config in _qk_kernel.configs
    }
    attention_v_configs = {
        (
            config.kwargs["block_m"],
            config.kwargs["block_n"],
            config.kwargs["block_k"],
            config.num_warps,
            config.num_stages,
        )
        for config in _attention_v_kernel.configs
    }
    assert qk_configs == set(_TILE_CONFIGS)
    assert attention_v_configs == set(_TILE_CONFIGS)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_body_qk_matches_explicit_head_reshape() -> None:
    """Body QK reads interleaved heads and emits contiguous head matrices."""
    torch.manual_seed(1)
    samples, heads, tokens, depth = 169, 32, 64, 32
    queries = (
        torch.randn(
            (samples, tokens, heads * depth), dtype=torch.float16, device="cuda"
        )
        * 0.05
    )
    keys = torch.randn_like(queries) * 0.05
    scale = torch.tensor([0.173], dtype=torch.float16, device="cuda")
    result = torch.empty(
        (samples, heads, tokens, tokens), dtype=torch.float16, device="cuda"
    )

    _qk_kernel[_autotune_grid](
        result,
        queries,
        keys,
        scale,
        _OPERATIONS["body_qk"],
        samples * heads,
        heads,
        tokens,
        tokens,
        depth,
        tokens * tokens,
    )

    query_heads = queries.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    key_heads = keys.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    expected = (
        torch.matmul(query_heads.float(), key_heads.float().transpose(-1, -2))
        * scale.float()
    )
    _assert_close(result, expected)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_attention_times_v_writes_directly_to_merged_heads() -> None:
    """Attention output lands in physical token-major body storage."""
    torch.manual_seed(2)
    samples, heads, tokens, depth = 169, 32, 64, 32
    attention = (
        torch.randn(
            (samples, heads, tokens, tokens), dtype=torch.float16, device="cuda"
        )
        * 0.05
    )
    values = (
        torch.randn(
            (samples, tokens, heads * depth), dtype=torch.float16, device="cuda"
        )
        * 0.05
    )
    result = torch.empty_like(values)

    _attention_v_kernel[_autotune_grid](
        result,
        attention,
        values,
        samples * heads,
        heads,
        tokens,
        depth,
        tokens,
    )

    value_heads = values.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    expected_heads = torch.matmul(attention.float(), value_heads.float())
    expected = expected_heads.permute(0, 2, 1, 3).reshape_as(values)
    _assert_close(result, expected)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_policy_qk_preserves_promotion_slots() -> None:
    """Policy QK scales the matrix prefix and leaves all 192 promotions intact."""
    torch.manual_seed(3)
    samples, tokens, width = 169, 64, 1024
    queries = (
        torch.randn((samples, tokens, width), dtype=torch.float16, device="cuda") * 0.02
    )
    keys = torch.randn_like(queries) * 0.02
    scale = torch.tensor([0.037], dtype=torch.float16, device="cuda")
    sentinel = -123.0
    result = torch.full(
        (samples, _POLICY_RECORD_SIZE), sentinel, dtype=torch.float16, device="cuda"
    )

    _qk_kernel[_autotune_grid](
        result,
        queries,
        keys,
        scale,
        _OPERATIONS["policy_qk"],
        samples,
        1,
        tokens,
        tokens,
        width,
        _POLICY_RECORD_SIZE,
    )

    expected = torch.matmul(queries.float(), keys.float().transpose(-1, -2))
    expected *= scale.float()
    _assert_close(result[:, : tokens * tokens].reshape_as(expected), expected)
    assert torch.equal(
        result[:, tokens * tokens :],
        torch.full(
            (samples, _POLICY_RECORD_SIZE - tokens * tokens),
            sentinel,
            dtype=torch.float16,
            device="cuda",
        ),
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_body_qk_crosses_from_last_head_to_next_sample() -> None:
    """Logical matrices 31 and 32 use different samples without overlap."""
    torch.manual_seed(4)
    samples, heads, tokens, depth = 2, 32, 16, 16
    queries = torch.randn(
        (samples, tokens, heads * depth), dtype=torch.float16, device="cuda"
    )
    keys = torch.randn_like(queries)
    scale = torch.ones(1, dtype=torch.float16, device="cuda")
    result = torch.empty(
        (samples * heads, tokens, tokens), dtype=torch.float16, device="cuda"
    )

    _qk_kernel[_autotune_grid](
        result,
        queries,
        keys,
        scale,
        _OPERATIONS["body_qk"],
        samples * heads,
        heads,
        tokens,
        tokens,
        depth,
        tokens * tokens,
    )

    query_heads = queries.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    key_heads = keys.view(samples, tokens, heads, depth).permute(0, 2, 1, 3)
    expected = torch.matmul(query_heads.float(), key_heads.float().transpose(-1, -2))
    expected = expected.reshape(samples * heads, tokens, tokens)
    _assert_close(result[30:34], expected[30:34])


_ARTIFACT_SPECIALIZATIONS = (
    BatchedMatmulSpecialization("body_qk", 4, 16, 16, 16, 2, 1),
    BatchedMatmulSpecialization("body_attention_v", 4, 16, 16, 16, 2, 1),
    BatchedMatmulSpecialization("policy_qk", 2, 16, 16, 16, 1, 1),
)


@pytest.mark.gpu
@_CUDA_REQUIRED
@pytest.mark.parametrize("template", _ARTIFACT_SPECIALIZATIONS)
def test_compilation_captures_selected_launch_and_abi(
    template: BatchedMatmulSpecialization,
) -> None:
    """Each physical layout serializes its tuned grid and pointer ABI."""
    specialization = BatchedMatmulSpecialization(
        template.operation,
        template.batch_count,
        template.m,
        template.n,
        template.k,
        template.heads_per_sample,
        _architecture(),
    )

    artifact = compile_batched_matmul(specialization)
    kernel = (
        _attention_v_kernel
        if specialization.operation == "body_attention_v"
        else _qk_kernel
    )
    selected = kernel.best_config
    pointer_count = 3 if specialization.operation == "body_attention_v" else 4

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (
        *((lc0ex_pb2.PARAMETER_TYPE_POINTER,) * pointer_count),
        *_NULL_POINTERS,
    )
    assert artifact.grid == _artifact_grid(
        selected.kwargs,
        specialization.m,
        specialization.n,
        specialization.batch_count,
    )
    assert artifact.block == (selected.num_warps * 32, 1, 1)


def _external_buffer(
    program: ProgramBuilder,
    name: str,
    shape: Sequence[int],
    *,
    writable: bool = False,
    persistent: bool = False,
) -> Buffer:
    """Declare one test execution buffer."""
    if persistent:
        return program.persistent_buffer(
            name=name,
            shape=shape,
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
            writable=writable,
        )
    return program.buffer(
        name=name,
        shape=shape,
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=writable,
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
@pytest.mark.parametrize("operation", ["body_qk", "body_attention_v", "policy_qk"])
def test_graph_call_has_only_physical_operands(
    operation: BatchedMatmulOperation,
) -> None:
    """The graph ABI allocates no pointer arrays or head-layout temporaries."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    heads = _TEST_HEADS if operation != "policy_qk" else 1
    batch_count = 2 * heads
    specialization = BatchedMatmulSpecialization(
        operation, batch_count, 16, 16, 16, heads, _architecture()
    )
    output_shape = (
        (2, _POLICY_RECORD_SIZE)
        if operation == "policy_qk"
        else (2, 16, heads * 16)
        if operation == "body_attention_v"
        else (batch_count, 16, 16)
    )
    left_shape = (
        (batch_count, 16, 16)
        if operation == "body_attention_v"
        else (2, 16, heads * 16)
    )
    right_shape = (2, 16, heads * 16)
    output = _external_buffer(program, "output", output_shape, writable=True)
    left = _external_buffer(program, "left", left_shape)
    right = _external_buffer(program, "right", right_shape)
    scale_name = (
        "/policy/scale/w" if operation == "policy_qk" else "/encoder0/mha/QK/scale/w"
    )
    scale = (
        None
        if operation == "body_attention_v"
        else _external_buffer(
            program,
            scale_name,
            (1,),
            persistent=True,
        )
    )

    batched_matmul(
        program,
        KernelCache(builder),
        output,
        left,
        right,
        specialization,
        scale=scale,
    )

    executable = builder.build()
    node = executable.programs[0].nodes[0]
    locations = {
        buffer.name: (buffer.offset,)
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }
    arguments = [(argument.allocation.offset,) for argument in node.arguments]
    expected_names = ["output", "left", "right"]
    if scale is not None:
        expected_names.append(scale_name)

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert set(locations) == set(expected_names)
    assert len(locations) == len(expected_names)
    assert arguments == [locations[name] for name in expected_names]


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_graph_dependencies_follow_physical_buffer_flow() -> None:
    """A subsequent QK read depends on the producer of its query buffer."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    specialization = BatchedMatmulSpecialization(
        "body_qk", 4, 16, 16, 16, _TEST_HEADS, _architecture()
    )
    kernels = KernelCache(builder)
    first_output = _external_buffer(program, "first_output", (4, 16, 16), writable=True)
    final_output = _external_buffer(program, "final_output", (4, 16, 16), writable=True)
    queries = _external_buffer(program, "queries", (2, 16, 32))
    keys = _external_buffer(program, "keys", (2, 16, 32))
    scale = _external_buffer(
        program,
        "/encoder0/mha/QK/scale/w",
        (1,),
        persistent=True,
    )

    batched_matmul(
        program,
        kernels,
        first_output,
        queries,
        keys,
        specialization,
        scale=scale,
    )
    batched_matmul(
        program,
        kernels,
        final_output,
        first_output,
        keys,
        specialization,
        scale=scale,
    )

    nodes = builder.build().programs[0].nodes
    assert list(nodes[0].dependencies) == []
    assert list(nodes[1].dependencies) == [0]
