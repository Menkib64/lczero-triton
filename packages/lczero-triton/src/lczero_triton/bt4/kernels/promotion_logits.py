"""Autotuned fused attention-policy promotion-logit family."""

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
_POLICY_RECORD_SIZE = 4288
_POLICY_QK_WIDTH = 64
_PROMOTION_INPUT_START = 56
_PROMOTION_OUTPUT_START = 4096
_PROMOTION_ROW_COUNT = 8
_PROMOTION_WEIGHT_COUNT = 4
_PROMOTION_OUTPUT_COUNT = 3
_TRITON_POLICY_RECORD_SIZE = tl.constexpr(4288)
_TRITON_POLICY_QK_WIDTH = tl.constexpr(64)
_TRITON_PROMOTION_INPUT_START = tl.constexpr(56)
_TRITON_PROMOTION_OUTPUT_START = tl.constexpr(4096)
_TRITON_PROMOTION_ROW_COUNT = tl.constexpr(8)
_TRITON_PROMOTION_WEIGHT_COUNT = tl.constexpr(4)
_TRITON_PROMOTION_OUTPUT_COUNT = tl.constexpr(3)
_TRITON_KNIGHT_CHANNEL = tl.constexpr(3)
_TILE_CONFIGURATIONS = (
    (8, 16, 16, 2, 3),
    (8, 16, 32, 4, 3),
    (8, 16, 64, 4, 3),
    (16, 16, 32, 4, 3),
    (16, 16, 64, 4, 4),
    (32, 32, 32, 4, 3),
    (32, 32, 64, 8, 3),
)


def _promotion_configs() -> list[triton.Config]:
    """Return row-packing and projection-tile tuning candidates."""
    return [
        triton.Config(
            {
                "rows_per_program": rows_per_program,
                "block_m": block_m,
                "block_k": block_k,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for rows_per_program, block_m, block_k, num_warps, num_stages in (
            _TILE_CONFIGURATIONS
        )
    ]


@triton.autotune(
    configs=_promotion_configs(),
    key=["batch_size", "width"],
    cache_results=True,
)
@triton.jit
def _promotion_logits_kernel(
    policy_records,
    policy_keys,
    promotion_weights,
    batch_size: tl.constexpr,
    width: tl.constexpr,
    rows_per_program: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    """Project the final eight policy keys and assemble promotion logits."""
    row_lanes = tl.arange(0, block_m)
    rows = tl.program_id(0) * rows_per_program + row_lanes
    valid_rows = (row_lanes < rows_per_program) & (
        rows < batch_size * _TRITON_PROMOTION_ROW_COUNT
    )
    batches = rows // _TRITON_PROMOTION_ROW_COUNT
    destination_files = rows % _TRITON_PROMOTION_ROW_COUNT
    offsets_k = tl.arange(0, block_k)
    offsets_n = tl.arange(0, 16)

    key_pointers = (
        policy_keys
        + batches[:, None] * _TRITON_POLICY_QK_WIDTH * width
        + (_TRITON_PROMOTION_INPUT_START + destination_files[:, None]) * width
        + offsets_k[None, :]
    )
    weight_pointers = (
        promotion_weights
        + offsets_k[:, None] * _TRITON_PROMOTION_WEIGHT_COUNT
        + offsets_n[None, :]
    )
    accumulator = tl.zeros((block_m, 16), dtype=tl.float32)

    for k_block in range(tl.cdiv(width, block_k)):
        remaining_k = width - k_block * block_k
        key_values = tl.load(
            key_pointers,
            mask=valid_rows[:, None] & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        weight_values = tl.load(
            weight_pointers,
            mask=(offsets_k[:, None] < remaining_k)
            & (offsets_n[None, :] < _TRITON_PROMOTION_WEIGHT_COUNT),
            other=0.0,
        )
        accumulator = tl.dot(key_values, weight_values, accumulator)
        key_pointers += block_k
        weight_pointers += block_k * _TRITON_PROMOTION_WEIGHT_COUNT

    knight_offsets = tl.sum(
        tl.where(offsets_n[None, :] == _TRITON_KNIGHT_CHANNEL, accumulator, 0.0),
        axis=1,
    )
    for source_file in range(_TRITON_PROMOTION_ROW_COUNT):
        qk_pointers = (
            policy_records
            + batches * _TRITON_POLICY_RECORD_SIZE
            + (48 + source_file) * _TRITON_POLICY_QK_WIDTH
            + _TRITON_PROMOTION_INPUT_START
            + destination_files
        )
        qk_values = tl.load(qk_pointers, mask=valid_rows, other=0.0).to(tl.float32)
        for channel in range(_TRITON_PROMOTION_OUTPUT_COUNT):
            channel_offsets = tl.sum(
                tl.where(offsets_n[None, :] == channel, accumulator, 0.0),
                axis=1,
            )
            output_pointers = (
                policy_records
                + batches * _TRITON_POLICY_RECORD_SIZE
                + _TRITON_PROMOTION_OUTPUT_START
                + source_file * 24
                + destination_files * _TRITON_PROMOTION_OUTPUT_COUNT
                + channel
            )
            result = qk_values + channel_offsets + knight_offsets
            tl.store(output_pointers, result, mask=valid_rows)


@dataclass(frozen=True, slots=True)
class PromotionLogitsSpecialization:
    """Immutable fused promotion-logit specialization."""

    batch_size: int
    width: int
    architecture: int


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the selected-key-row grid for a tuning candidate."""
    batch_size = cast("int", configuration["batch_size"])
    rows_per_program = cast("int", configuration["rows_per_program"])
    row_count = batch_size * _PROMOTION_ROW_COUNT
    return ((row_count + rows_per_program - 1) // rows_per_program,)


def _artifact_grid(
    configuration: Mapping[str, object],
    batch_size: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected row packing."""
    rows_per_program = cast("int", configuration["rows_per_program"])
    #row_count = batch_size * _PROMOTION_ROW_COUNT
    return (
        "(batch_size * %d + %d) / %d" %
            (_PROMOTION_ROW_COUNT, rows_per_program - 1, rows_per_program),
        "1",
        "1",
    )


def compile_promotion_logits(
    specialization: PromotionLogitsSpecialization,
) -> KernelArtifact:
    """Autotune and compile one fused FP16 promotion-logit operation."""
    policy_records = torch.zeros(
        (specialization.batch_size, _POLICY_RECORD_SIZE),
        dtype=torch.float16,
        device="cuda",
    )
    policy_keys = torch.zeros(
        (specialization.batch_size, _POLICY_QK_WIDTH, specialization.width),
        dtype=torch.float16,
        device="cuda",
    )
    promotion_weights = torch.zeros(
        (specialization.width, _PROMOTION_WEIGHT_COUNT),
        dtype=torch.float16,
        device="cuda",
    )
    compiled = _promotion_logits_kernel[_autotune_grid](
        policy_records,
        policy_keys,
        promotion_weights,
        specialization.batch_size,
        specialization.width,
    )
    selected = _promotion_logits_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, specialization.batch_size),
        parameters=(_POINTER, _POINTER, _POINTER),
        autotuner=_promotion_logits_kernel,
    )


def promotion_logits(
    builder: ProgramBuilder,
    kernels: KernelCache,
    policy_records: Buffer,
    policy_keys: Buffer,
    promotion_weights: Buffer,
    specialization: PromotionLogitsSpecialization,
) -> None:
    """Append in-place promotion projection and policy-record assembly."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_promotion_logits, specialization)
    readonly = [
        source
        for source in (policy_keys, promotion_weights)
        if source is not policy_records
    ]
    builder.call(
        kernel,
        policy_records,
        policy_keys,
        promotion_weights,
        readonly=readonly,
    )
