"""Tests for fused Smolgen addition and 64-way attention softmax."""

from collections.abc import Sequence
from dataclasses import fields

import pytest
import torch
from lc0ex import Buffer, ExecutableBuilder, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.softmax_64 import (
    _SOFTMAX_WIDTH,
    _TILE_CONFIGURATIONS,
    _TWICE_HALF_MAX,
    Softmax64Specialization,
    _artifact_grid,
    _autotune_grid,
    _softmax_64_kernel,
    compile_softmax_64,
    softmax_64,
)

_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
_FIXED_ROW_COUNT = 169 * 32 * 64
_NULL_POINTERS = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def _reference(scaled_qk: torch.Tensor, smolgen: torch.Tensor) -> torch.Tensor:
    """Evaluate LC0's FP16-input softmax edge behavior in FP32."""
    values = scaled_qk.float() + smolgen.float()
    values = torch.clamp(values, -_TWICE_HALF_MAX, _TWICE_HALF_MAX)
    return torch.softmax(values, dim=1).half()


def _launch(
    output: torch.Tensor,
    scaled_qk: torch.Tensor,
    smolgen: torch.Tensor,
) -> None:
    """Launch the autotuned kernel for contiguous 64-value rows."""
    _softmax_64_kernel[_autotune_grid](
        output,
        scaled_qk,
        smolgen,
        scaled_qk.shape[0],
    )


def test_autotune_contract_covers_shape_and_row_packing() -> None:
    """Persistent tuning owns row packing and warp count, not specialization."""
    assert _softmax_64_kernel.keys == ["row_count"]
    assert _softmax_64_kernel.cache_results
    assert (
        tuple(
            (config.kwargs["rows_per_program"], config.num_warps)
            for config in _softmax_64_kernel.configs
        )
        == _TILE_CONFIGURATIONS
    )
    assert {field.name for field in fields(Softmax64Specialization)} == {
        "row_count",
        "architecture",
    }


def test_candidate_and_artifact_grids_cover_tail_rows() -> None:
    """Selected row tiles round up both tuning and serialized launch grids."""
    configuration: dict[str, object] = {
        "row_count": 17,
        "rows_per_program": 8,
    }
    assert _autotune_grid(configuration) == (3,)
    assert _artifact_grid(configuration, 17) == (3, 1, 1)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_fixed_bt4_workload_matches_fp32_reference_and_sums_to_one() -> None:
    """The complete BT4 attention workload adds Smolgen before softmax."""
    torch.manual_seed(64)
    scaled_qk = torch.randn(
        (_FIXED_ROW_COUNT, _SOFTMAX_WIDTH),
        dtype=torch.float16,
        device="cuda",
    )
    smolgen = torch.randn_like(scaled_qk) * 0.7
    result = torch.empty_like(scaled_qk)

    _launch(result, scaled_qk, smolgen)

    expected = _reference(scaled_qk, smolgen)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        result.float().sum(dim=1),
        torch.ones(_FIXED_ROW_COUNT, device="cuda"),
        rtol=0.0,
        atol=2e-3,
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_extremes_infinities_and_nans_match_cuda_clamp() -> None:
    """Infinity clamping remains finite while any source NaN poisons its row."""
    scaled_qk = torch.zeros((8, _SOFTMAX_WIDTH), dtype=torch.float16)
    smolgen = torch.zeros_like(scaled_qk)
    scaled_qk[0, 0] = torch.inf
    scaled_qk[1, :2] = torch.inf
    scaled_qk[2] = -torch.inf
    scaled_qk[3, 0] = torch.inf
    smolgen[3, 0] = -torch.inf
    scaled_qk[4, 7] = torch.nan
    scaled_qk[5, 0] = 65504.0
    smolgen[5, 0] = 65504.0
    scaled_qk[6, 0] = -65504.0
    smolgen[6, 0] = -65504.0
    scaled_qk[7] = torch.linspace(-20.0, 20.0, _SOFTMAX_WIDTH).half()
    smolgen[7] = torch.linspace(3.0, -4.0, _SOFTMAX_WIDTH).half()
    scaled_qk_cuda = scaled_qk.cuda()
    smolgen_cuda = smolgen.cuda()
    result = torch.empty_like(scaled_qk_cuda)

    _launch(result, scaled_qk_cuda, smolgen_cuda)

    expected = _reference(scaled_qk_cuda, smolgen_cuda)
    torch.testing.assert_close(
        result,
        expected,
        rtol=2e-3,
        atol=2e-3,
        equal_nan=True,
    )
    assert result[0, 0] == 1
    assert torch.equal(result[1, :2], torch.tensor([0.5, 0.5], device="cuda"))
    assert torch.equal(
        result[2],
        torch.full((_SOFTMAX_WIDTH,), 1 / _SOFTMAX_WIDTH, device="cuda"),
    )
    assert torch.isnan(result[3]).all()
    assert torch.isnan(result[4]).all()


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_large_common_offset_is_stable() -> None:
    """Subtracting a large representable offset does not destabilize softmax."""
    logits = (torch.arange(_SOFTMAX_WIDTH, dtype=torch.float16) * 0.5).repeat(2, 1)
    shifted = logits - 1024.0
    zeros = torch.zeros_like(logits, device="cuda")
    baseline = torch.empty_like(zeros)
    shifted_result = torch.empty_like(zeros)

    _launch(baseline, logits.cuda(), zeros)
    _launch(shifted_result, shifted.cuda(), zeros)

    torch.testing.assert_close(shifted_result, baseline, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_scaled_qk_may_be_overwritten_in_place() -> None:
    """Production may replace QK logits after tuning with separate output."""
    torch.manual_seed(9)
    scaled_qk = torch.randn((7, _SOFTMAX_WIDTH), dtype=torch.float16, device="cuda")
    smolgen = torch.randn_like(scaled_qk)
    expected = _reference(scaled_qk, smolgen)

    _launch(torch.empty_like(scaled_qk), scaled_qk, smolgen)
    _launch(scaled_qk, scaled_qk, smolgen)

    torch.testing.assert_close(scaled_qk, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_compilation_captures_selected_launch_and_pointer_abi() -> None:
    """The artifact records the autotuned row grid and three physical pointers."""
    specialization = Softmax64Specialization(17, _architecture())

    artifact = compile_softmax_64(specialization)
    selected = _softmax_64_kernel.best_config

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (
        *(lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 3,
        *_NULL_POINTERS,
    )
    assert artifact.grid == _artifact_grid(selected.kwargs, 17)
    assert artifact.block == (selected.num_warps * 32, 1, 1)


def _external_buffer(
    program: ProgramBuilder,
    name: str,
    shape: Sequence[int],
    *,
    writable: bool = False,
) -> Buffer:
    """Declare one FP16 execution buffer for graph tests."""
    return program.buffer(
        name=name,
        shape=shape,
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=writable,
    )


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_graph_call_preserves_in_place_abi_and_dependencies() -> None:
    """An in-place QK write remains visible to the following attention consumer."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    shape = (4, _SOFTMAX_WIDTH)
    scaled_qk = _external_buffer(program, "scaled_qk", shape, writable=True)
    smolgen = _external_buffer(program, "smolgen", shape)
    output = _external_buffer(program, "output", shape, writable=True)
    specialization = Softmax64Specialization(4, _architecture())
    kernels = KernelCache(builder)

    softmax_64(
        program,
        kernels,
        scaled_qk,
        scaled_qk,
        smolgen,
        specialization,
    )
    softmax_64(
        program,
        kernels,
        output,
        scaled_qk,
        smolgen,
        specialization,
    )

    executable = builder.build()
    nodes = executable.programs[0].nodes
    locations = {
        buffer.name: (buffer.offset,)
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }
    first_arguments = [(argument.allocation.offset,) for argument in nodes[0].arguments]

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert first_arguments == [
        locations["scaled_qk"],
        locations["scaled_qk"],
        locations["smolgen"],
    ]
    assert list(nodes[0].dependencies) == []
    assert list(nodes[1].dependencies) == [0]
