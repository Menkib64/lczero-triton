"""Autotuned indexed FP16 batched matrix multiplication family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import cold_do_bench
from lczero_triton.bt4.kernels._cache import KernelCache

BatchedMatmulOperation = Literal["body_qk", "body_attention_v", "policy_qk"]

_BODY_QK = tl.constexpr(0)
_OPERATIONS: dict[BatchedMatmulOperation, int] = {
    "body_qk": 0,
    "body_attention_v": 1,
    "policy_qk": 2,
}
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_POLICY_RECORD_SIZE = 4288
_TILE_CONFIGS = (
    (16, 16, 16, 2, 3),
    (32, 32, 16, 4, 3),
    (64, 32, 16, 4, 3),
    (32, 64, 16, 4, 3),
    (64, 64, 16, 4, 3),
    (32, 32, 32, 4, 4),
    (64, 64, 32, 8, 3),
    (64, 64, 64, 8, 3),
)


def _matmul_configs() -> list[triton.Config]:
    """Return independent tuning configurations for one batched GEMM."""
    return [
        triton.Config(
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_m, block_n, block_k, num_warps, num_stages in _TILE_CONFIGS
    ]


@triton.autotune(
    configs=_matmul_configs(),
    key=[
        "operation",
        "batch_count",
        "heads_per_sample",
        "m",
        "n",
        "k",
        "output_batch_stride",
    ],
    do_bench=cold_do_bench,
    cache_results=True,
)
@triton.jit
def _qk_kernel(
    output,
    queries,
    keys,
    scale,
    operation: tl.constexpr,
    batch_count: tl.constexpr,
    heads_per_sample: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    output_batch_stride: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    """Compute scaled body or policy QK without materialized transposes."""
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    matrix = tl.program_id(2)
    matrix_valid = matrix < batch_count
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)

    sample = matrix // heads_per_sample
    head = matrix % heads_per_sample
    if operation == _BODY_QK:
        query_base = sample * m * heads_per_sample * k + head * k
        key_base = sample * n * heads_per_sample * k + head * k
        query_row_stride = heads_per_sample * k
        key_row_stride = heads_per_sample * k
    else:
        query_base = matrix * m * k
        key_base = matrix * n * k
        query_row_stride = k
        key_row_stride = k
    output_base = matrix * output_batch_stride

    query_pointers = (
        queries
        + query_base
        + offsets_m[:, None] * query_row_stride
        + offsets_k[None, :]
    )
    key_pointers = (
        keys + key_base + offsets_k[:, None] + offsets_n[None, :] * key_row_stride
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float16)

    for k_block in range(tl.cdiv(k, block_k)):
        remaining_k = k - k_block * block_k
        query_values = tl.load(
            query_pointers,
            mask=matrix_valid
            & (offsets_m[:, None] < m)
            & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        key_values = tl.load(
            key_pointers,
            mask=matrix_valid
            & (offsets_k[:, None] < remaining_k)
            & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(query_values, key_values, accumulator, out_dtype=tl.float16)
        query_pointers += block_k
        key_pointers += block_k

    output_pointers = output + output_base + offsets_m[:, None] * n + offsets_n[None, :]
    output_mask = matrix_valid & (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
    scale_value = tl.load(scale)
    tl.store(output_pointers, accumulator * scale_value, mask=output_mask)


@triton.autotune(
    configs=_matmul_configs(),
    key=["batch_count", "heads_per_sample", "m", "n", "k"],
    do_bench=cold_do_bench,
    cache_results=True,
)
@triton.jit
def _attention_v_kernel(
    output,
    attention,
    values,
    batch_count: tl.constexpr,
    heads_per_sample: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    """Multiply attention by interleaved V and write merged-head storage."""
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    matrix = tl.program_id(2)
    matrix_valid = matrix < batch_count
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)

    sample = matrix // heads_per_sample
    head = matrix % heads_per_sample
    attention_base = matrix * m * k
    value_width = heads_per_sample * n
    value_base = sample * k * value_width + head * n
    output_base = sample * m * value_width + head * n
    attention_pointers = (
        attention + attention_base + offsets_m[:, None] * k + offsets_k[None, :]
    )
    value_pointers = (
        values + value_base + offsets_k[:, None] * value_width + offsets_n[None, :]
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float16)

    for k_block in range(tl.cdiv(k, block_k)):
        remaining_k = k - k_block * block_k
        attention_values = tl.load(
            attention_pointers,
            mask=matrix_valid
            & (offsets_m[:, None] < m)
            & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        value_values = tl.load(
            value_pointers,
            mask=matrix_valid
            & (offsets_k[:, None] < remaining_k)
            & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(attention_values, value_values, accumulator)
        attention_pointers += block_k
        value_pointers += block_k * value_width

    output_pointers = (
        output + output_base + offsets_m[:, None] * value_width + offsets_n[None, :]
    )
    output_mask = matrix_valid & (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
    tl.store(output_pointers, accumulator, mask=output_mask)


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int, int, int]:
    """Return the tiled three-dimensional launch grid for tuning."""
    m = cast("int", configuration["m"])
    n = cast("int", configuration["n"])
    batch_count = cast("int", configuration["batch_count"])
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    return (
        (m + block_m - 1) // block_m,
        (n + block_n - 1) // block_n,
        batch_count,
    )


def _artifact_grid(
    configuration: Mapping[str, object],
    m: int,
    n: int,
    batch_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized launch grid from the selected configuration."""
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    return (
        (m + block_m - 1) // block_m,
        (n + block_n - 1) // block_n,
        batch_count,
    )


@dataclass(frozen=True, slots=True)
class BatchedMatmulSpecialization:
    """Immutable indexed batched GEMM specialization."""

    operation: BatchedMatmulOperation
    batch_count: int
    m: int
    n: int
    k: int
    heads_per_sample: int
    architecture: int


def compile_batched_matmul(
    specialization: BatchedMatmulSpecialization,
) -> KernelArtifact:
    """Autotune and compile one indexed FP16 batched GEMM specialization."""
    operation = _OPERATIONS[specialization.operation]
    output_batch_stride = (
        _POLICY_RECORD_SIZE
        if specialization.operation == "policy_qk"
        else specialization.m * specialization.n
    )
    output = torch.empty(
        specialization.batch_count * output_batch_stride,
        dtype=torch.float16,
        device="cuda",
    )
    left = torch.zeros(
        specialization.batch_count * specialization.m * specialization.k,
        dtype=torch.float16,
        device="cuda",
    )
    right = torch.zeros(
        specialization.batch_count * specialization.n * specialization.k,
        dtype=torch.float16,
        device="cuda",
    )
    parameters: tuple[int, ...]

    if specialization.operation == "body_attention_v":
        compiled = _attention_v_kernel[_autotune_grid](
            output,
            left,
            right,
            specialization.batch_count,
            specialization.heads_per_sample,
            specialization.m,
            specialization.n,
            specialization.k,
        )
        selected = _attention_v_kernel.best_config
        autotuner = _attention_v_kernel
        parameters = (_POINTER, _POINTER, _POINTER)
    else:
        scale = torch.ones(1, dtype=torch.float16, device="cuda")
        compiled = _qk_kernel[_autotune_grid](
            output,
            left,
            right,
            scale,
            operation,
            specialization.batch_count,
            specialization.heads_per_sample,
            specialization.m,
            specialization.n,
            specialization.k,
            output_batch_stride,
        )
        selected = _qk_kernel.best_config
        autotuner = _qk_kernel
        parameters = (_POINTER, _POINTER, _POINTER, _POINTER)

    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(
            selected.kwargs,
            specialization.m,
            specialization.n,
            specialization.batch_count,
        ),
        parameters=parameters,
        autotuner=autotuner,
    )


def batched_matmul(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    left: Buffer,
    right: Buffer,
    specialization: BatchedMatmulSpecialization,
    *,
    scale: Buffer | None = None,
) -> None:
    """Append one indexed batched matrix multiplication."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_batched_matmul, specialization)
    arguments: tuple[Buffer, ...] = (output, left, right)
    if scale is not None:
        arguments += (scale,)
    builder.call(kernel, *arguments, readonly=arguments[1:])
