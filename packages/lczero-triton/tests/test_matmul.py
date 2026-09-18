"""CUDA numerical and artifact tests for contiguous dense GEMM."""

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.matmul import (
    _GROUP_SIZES_M,
    _MATMUL_CONFIGS,
    _TILE_CONFIGS,
    MatmulSpecialization,
    _artifact_grid,
    _autotune_grid,
    _matmul_kernel,
    _matmul_skip_kernel,
    compile_matmul,
    matmul,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_BT4_DENSE_SHAPES = (
    (169, 768, 32768),
    (10816, 624, 1024),
    (10816, 1024, 1536),
    (10816, 1536, 1024),
    (10816, 1024, 1024),
    (10816, 1024, 32),
    (169, 2048, 256),
    (169, 256, 8192),
    (5408, 256, 4096),
    (10816, 1024, 128),
    (169, 8192, 128),
    (169, 128, 3),
    (169, 2048, 128),
    (169, 128, 1),
)
_NULL_POINTERS = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2
_FP16_ATOL = 2e-2
_FP16_RTOL = 1e-2


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def test_group_size_m_is_autotuned_for_every_tile_configuration() -> None:
    """Every tile shape competes with each grouped-M ordering candidate."""
    groups_by_tile: dict[tuple[int, int, int, int, int], set[int]] = {}
    for configuration in _MATMUL_CONFIGS:
        tile = (
            configuration.kwargs["block_m"],
            configuration.kwargs["block_n"],
            configuration.kwargs["block_k"],
            configuration.num_warps,
            configuration.num_stages,
        )
        groups_by_tile.setdefault(tile, set()).add(configuration.kwargs["group_size_m"])

    assert _matmul_kernel.keys == ["m", "n", "k", "has_bias", "activation"]
    assert _matmul_kernel.cache_results
    assert set(groups_by_tile) == set(_TILE_CONFIGS)
    assert all(groups == set(_GROUP_SIZES_M) for groups in groups_by_tile.values())


@pytest.mark.parametrize(("m", "k", "n"), _BT4_DENSE_SHAPES)
def test_matmul_matches_torch_for_every_bt4_specialization(
    m: int,
    k: int,
    n: int,
) -> None:
    """Every fixed dense shape preserves the ONNX `[K, N]` weight layout."""
    torch.manual_seed(m + k + n)
    activations = (
        torch.randn(
            (m, k),
            dtype=torch.float16,
            device="cuda",
        )
        * 0.05
    )
    weights = (
        torch.randn(
            (k, n),
            dtype=torch.float16,
            device="cuda",
        )
        * 0.05
    )
    bias = torch.zeros(n, dtype=torch.float16, device="cuda")
    result = torch.empty((m, n), dtype=torch.float16, device="cuda")

    _matmul_kernel[_autotune_grid](
        result,
        activations,
        weights,
        bias,
        m,
        n,
        k,
        has_bias=False,
        activation=0,
    )

    expected = torch.matmul(activations, weights)
    torch.testing.assert_close(
        result,
        expected,
        rtol=_FP16_RTOL,
        atol=_FP16_ATOL,
    )


def test_matmul_masks_non_divisible_m_n_and_k() -> None:
    """Tail tiles do not wrap around into valid rows or columns."""
    torch.manual_seed(37)
    m, k, n = 7, 37, 11
    activations = torch.randn((m, k), dtype=torch.float16, device="cuda") * 0.05
    weights = torch.randn((k, n), dtype=torch.float16, device="cuda") * 0.05
    bias = torch.zeros(n, dtype=torch.float16, device="cuda")
    result = torch.empty((m, n), dtype=torch.float16, device="cuda")

    _matmul_kernel[_autotune_grid](
        result,
        activations,
        weights,
        bias,
        m,
        n,
        k,
        has_bias=False,
        activation=0,
    )

    expected = torch.matmul(activations, weights)
    torch.testing.assert_close(
        result,
        expected,
        rtol=_FP16_RTOL,
        atol=_FP16_ATOL,
    )


def test_compilation_captures_selected_configuration_and_static_launch() -> None:
    """The selected tile determines the serialized grid and CUDA block size."""
    specialization = MatmulSpecialization(169, 3, 128, _architecture())

    artifact = compile_matmul(specialization)
    selected = _matmul_kernel.best_config

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        *_NULL_POINTERS,
    )
    assert artifact.grid == _artifact_grid(selected.kwargs, 169, 3)
    assert artifact.block == (selected.num_warps * 32, 1, 1)
    assert selected.kwargs["group_size_m"] in _GROUP_SIZES_M


def test_graph_call_preserves_output_activation_weight_order() -> None:
    """The graph ABI places the destination before both readonly operands."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    output = program.buffer(
        name="output",
        shape=(169, 3),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=True,
    )
    activations = program.buffer(
        name="activations",
        shape=(169, 128),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    weights = builder.persistent_buffer(
        name="weights",
        shape=(128, 3),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )

    matmul(
        program,
        KernelCache(builder),
        output,
        activations,
        weights,
        MatmulSpecialization(169, 3, 128, _architecture()),
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

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert arguments == [
        locations["output"],
        locations["activations"],
        locations["weights"],
        locations["output"],
    ]


def test_matmul_fused_skip_numerical_accuracy() -> None:
    """Fused skip kernel computes alpha * (X @ W + b) + skip accurately."""
    torch.manual_seed(42)
    m, k, n = 32, 64, 48
    activations = torch.randn((m, k), dtype=torch.float16, device="cuda") * 0.05
    weights = torch.randn((k, n), dtype=torch.float16, device="cuda") * 0.05
    bias = torch.randn(n, dtype=torch.float16, device="cuda") * 0.05
    skip = torch.randn((m, n), dtype=torch.float16, device="cuda") * 0.05
    alpha = torch.tensor([0.75], dtype=torch.float16, device="cuda")
    result = torch.empty((m, n), dtype=torch.float16, device="cuda")

    specialization = MatmulSpecialization(
        m, n, k, _architecture(), has_bias=True, has_skip=True
    )
    artifact = compile_matmul(specialization)
    assert artifact.parameters == (
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        *_NULL_POINTERS,
    )

    _matmul_skip_kernel[_autotune_grid](
        result,
        activations,
        weights,
        bias,
        skip,
        alpha,
        m,
        n,
        k,
        has_bias=True,
        activation=0,
    )

    expected = (
        torch.matmul(activations.float(), weights.float()) + bias.float()
    ) * alpha.float() + skip.float()
    torch.testing.assert_close(
        result,
        expected.half(),
        rtol=_FP16_RTOL,
        atol=_FP16_ATOL,
    )


def test_matmul_fused_skip_graph_call() -> None:
    """The graph ABI for fused skip passes 6 buffers in expected order."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    output = program.buffer(
        name="output",
        shape=(16, 32),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=True,
    )
    activations = program.buffer(
        name="activations",
        shape=(16, 64),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    weights = builder.persistent_buffer(
        name="weights",
        shape=(64, 32),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    bias = builder.persistent_buffer(
        name="bias",
        shape=(32,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    skip = program.buffer(
        name="skip",
        shape=(16, 32),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    alpha = builder.persistent_buffer(
        name="alpha",
        shape=(1,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )

    matmul(
        program,
        KernelCache(builder),
        output,
        activations,
        weights,
        MatmulSpecialization(16, 32, 64, _architecture(), has_bias=True, has_skip=True),
        bias=bias,
        skip=skip,
        alpha=alpha,
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

    assert arguments == [
        locations["output"],
        locations["activations"],
        locations["weights"],
        locations["bias"],
        locations["skip"],
        locations["alpha"],
    ]
