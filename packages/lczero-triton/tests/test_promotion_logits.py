"""Tests for fused attention-policy promotion logits."""

from collections.abc import Sequence
from dataclasses import fields

import pytest
import torch
from lc0ex import Buffer, ExecutableBuilder, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.promotion_logits import (
    _POLICY_RECORD_SIZE,
    _PROMOTION_OUTPUT_START,
    _TILE_CONFIGURATIONS,
    PromotionLogitsSpecialization,
    _artifact_grid,
    _autotune_grid,
    _promotion_logits_kernel,
    compile_promotion_logits,
    promotion_logits,
)

_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
_FIXED_BATCH_SIZE = 169
_FIXED_WIDTH = 1024
_FP16_ATOL = 3e-2
_FP16_RTOL = 2e-2
_NULL_POINTERS = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def _reference(
    policy_records: torch.Tensor,
    policy_keys: torch.Tensor,
    promotion_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the fused CUDA promotion formula in FP32."""
    projections = torch.matmul(
        policy_keys[:, 56:64].float(),
        promotion_weights.float(),
    )
    offsets = projections[:, :, :3] + projections[:, :, 3:4]
    qk = policy_records[:, :4096].reshape(-1, 64, 64)[:, 48:56, 56:64].float()
    return (qk[:, :, :, None] + offsets[:, None, :, :]).half().reshape(-1, 192)


def _launch(
    policy_records: torch.Tensor,
    policy_keys: torch.Tensor,
    promotion_weights: torch.Tensor,
) -> None:
    """Launch promotion assembly for contiguous policy tensors."""
    _promotion_logits_kernel[_autotune_grid](
        policy_records,
        policy_keys,
        promotion_weights,
        policy_records.shape[0],
        policy_keys.shape[2],
    )


def test_autotune_contract_covers_static_shape_and_projection_tiles() -> None:
    """Persistent tuning owns row packing, projection tiles, and occupancy."""
    assert _promotion_logits_kernel.keys == ["batch_size", "width"]
    assert _promotion_logits_kernel.cache_results
    assert (
        tuple(
            (
                config.kwargs["rows_per_program"],
                config.kwargs["block_m"],
                config.kwargs["block_k"],
                config.num_warps,
                config.num_stages,
            )
            for config in _promotion_logits_kernel.configs
        )
        == _TILE_CONFIGURATIONS
    )
    assert {field.name for field in fields(PromotionLogitsSpecialization)} == {
        "batch_size",
        "width",
        "architecture",
    }


def test_candidate_and_artifact_grids_cover_tail_rows() -> None:
    """Selected row packing rounds up the tuning and serialized grids."""
    configuration: dict[str, object] = {
        "batch_size": 3,
        "width": 16,
        "rows_per_program": 16,
    }
    assert _autotune_grid(configuration) == (2,)
    assert _artifact_grid(configuration, 3) == (2, 1, 1)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_fixed_bt4_workload_matches_all_fp32_reference_logits() -> None:
    """All 192 promotion slots match the fused FP32 projection and assembly."""
    torch.manual_seed(10)
    policy_records = torch.randn(
        (_FIXED_BATCH_SIZE, _POLICY_RECORD_SIZE),
        dtype=torch.float16,
        device="cuda",
    )
    policy_keys = (
        torch.randn(
            (_FIXED_BATCH_SIZE, 64, _FIXED_WIDTH),
            dtype=torch.float16,
            device="cuda",
        )
        * 0.05
    )
    promotion_weights = (
        torch.randn(
            (_FIXED_WIDTH, 4),
            dtype=torch.float16,
            device="cuda",
        )
        * 0.05
    )
    expected = _reference(policy_records, policy_keys, promotion_weights)
    qk_prefix = policy_records[:, :_PROMOTION_OUTPUT_START].clone()

    _launch(policy_records, policy_keys, promotion_weights)

    torch.testing.assert_close(
        policy_records[:, _PROMOTION_OUTPUT_START:],
        expected,
        rtol=_FP16_RTOL,
        atol=_FP16_ATOL,
    )
    assert torch.equal(policy_records[:, :_PROMOTION_OUTPUT_START], qk_prefix)


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_onnx_weight_and_adjacent_record_indexing() -> None:
    """ONNX columns and y/w/c ordering remain separate across batch records."""
    batch_size, width = 2, 32
    policy_records = torch.full(
        (batch_size, _POLICY_RECORD_SIZE),
        -123.0,
        dtype=torch.float16,
        device="cuda",
    )
    policy_keys = torch.zeros(
        (batch_size, 64, width),
        dtype=torch.float16,
        device="cuda",
    )
    promotion_weights = torch.zeros(
        (width, 4),
        dtype=torch.float16,
        device="cuda",
    )
    promotion_weights[0] = torch.tensor(
        [1.0, 10.0, 100.0, 1000.0],
        dtype=torch.float16,
        device="cuda",
    )
    for batch in range(batch_size):
        for source_file in range(8):
            for destination_file in range(8):
                policy_records[
                    batch,
                    (48 + source_file) * 64 + 56 + destination_file,
                ] = batch * 100 + source_file * 10 + destination_file
        for destination_file in range(8):
            policy_keys[batch, 56 + destination_file, 0] = (
                batch * 10 + destination_file + 1
            )
    expected = _reference(policy_records, policy_keys, promotion_weights)

    _launch(policy_records, policy_keys, promotion_weights)

    assert torch.equal(policy_records[:, _PROMOTION_OUTPUT_START:], expected)
    assert policy_records[0, -1] != policy_records[1, _PROMOTION_OUTPUT_START]


@pytest.mark.gpu
@_CUDA_REQUIRED
def test_compilation_captures_selected_launch_and_pointer_abi() -> None:
    """The artifact records tuned row packing and three physical pointers."""
    specialization = PromotionLogitsSpecialization(2, 32, _architecture())

    artifact = compile_promotion_logits(specialization)
    selected = _promotion_logits_kernel.best_config

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (
        *(lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 3,
        *_NULL_POINTERS,
    )
    assert artifact.grid == _artifact_grid(selected.kwargs, 2)
    assert artifact.block == (selected.num_warps * 32, 1, 1)


def _external_buffer(
    program: ProgramBuilder,
    name: str,
    shape: Sequence[int],
    *,
    writable: bool = False,
    persistent: bool = False,
) -> Buffer:
    """Declare one FP16 external buffer for graph tests."""
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
def test_graph_call_uses_in_place_records_and_tracks_dependencies() -> None:
    """Promotion calls expose only physical tensors and serialize record writes."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    specialization = PromotionLogitsSpecialization(2, 32, _architecture())
    policy_records = _external_buffer(
        program,
        "policy_records",
        (2, _POLICY_RECORD_SIZE),
        writable=True,
    )
    policy_keys = _external_buffer(program, "policy_keys", (2, 64, 32))
    weights = _external_buffer(
        program,
        "/policy/promotion/matmul/w",
        (32, 4),
        persistent=True,
    )
    kernels = KernelCache(builder)

    promotion_logits(
        program,
        kernels,
        policy_records,
        policy_keys,
        weights,
        specialization,
    )
    promotion_logits(
        program,
        kernels,
        policy_records,
        policy_keys,
        weights,
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
    arguments = [(argument.allocation.offset,) for argument in nodes[0].arguments]

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert arguments == [
        locations["policy_records"],
        locations["policy_keys"],
        locations["/policy/promotion/matmul/w"],
    ]
    assert list(nodes[0].dependencies) == []
    assert list(nodes[1].dependencies) == [0]
