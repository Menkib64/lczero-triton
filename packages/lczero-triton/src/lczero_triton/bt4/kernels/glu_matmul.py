"""Autotuned fused gated-linear-unit projection.

One launch computes ``glu(x @ W_gate + b_gate, x @ W_linear + b_linear)`` where
the two weight halves are stored as a single concatenated ``[k, 2n]`` matrix,
gate columns first. This is the shape the lab's JAX nets export -- the network
description concatenates the two ``[512, 683]`` matrices into one ``[512, 1366]``
buffer -- and it is also the form the arithmetic prefers:

* the ``[block_m, block_k]`` activation tile is loaded **once** and fed to both
  ``tl.dot`` accumulators, halving the A-operand traffic of the two-GEMM form;
* the ``[m, 2n]`` FP16 intermediate never exists. The two-GEMM form writes
  ``2 * m * n`` FP16 elements and reads them back for the elementwise multiply;
  this kernel writes ``m * n`` once. For the lab's 512x15 encoder FFN at
  ``m = 64 * batch``, ``n = 683`` that is ``2.7 KB`` of round trip removed per
  token, per block.

Layouts match `matmul.py` exactly: row-major contiguous FP16 throughout, and a
per-column FP16 bias vector of length ``2n`` (gate biases first) applied in FP32
before the gate.
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

from lczero_triton.bt4.kernels._activation import (
    GLU_GATES,
    SOFTCAPPED_GLU_GATES,
    SUPPORTED_GLU_GATES,
    GluGate,
    apply_glu,
)
from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_NULL_POINTER = lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER

BIAS_NONE = 0
BIAS_VECTOR = 1
_BIAS_VECTOR = tl.constexpr(BIAS_VECTOR)

# Two accumulators live at once, so the register-heavy 256-row tiles that
# `matmul.py` offers are dropped; everything else is the same candidate set.
_GROUP_SIZES_M = (1, 4, 8)
_TILE_CONFIGS = (
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 2, 5),
    (32, 64, 32, 2, 5),
    (128, 128, 32, 8, 4),
    (128, 64, 64, 8, 3),
    (64, 128, 64, 8, 3),
    (64, 64, 64, 4, 4),
)
_GLU_MATMUL_CONFIGS = tuple(
    triton.Config(
        {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "group_size_m": group_size_m,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_m, block_n, block_k, num_warps, num_stages in _TILE_CONFIGS
    for group_size_m in _GROUP_SIZES_M
)


@triton.autotune(
    configs=list(_GLU_MATMUL_CONFIGS),
    key=["m", "n", "k", "bias_mode", "gate", "softcap"],
    cache_results=True,
)
@triton.jit
def _glu_matmul_kernel(
    output,
    activations,
    weights,
    bias,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    bias_mode: tl.constexpr,
    gate: tl.constexpr,
    softcap: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_size_m: tl.constexpr,
) -> None:
    """Compute one grouped-M tile of the gated projection."""
    program_id = tl.program_id(0)
    program_count_m = tl.cdiv(m, block_m)
    program_count_n = tl.cdiv(n, block_n)
    programs_per_group = group_size_m * program_count_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * group_size_m
    group_program_count_m = min(program_count_m - first_program_m, group_size_m)
    program_m = first_program_m + (
        (program_id % programs_per_group) % group_program_count_m
    )
    program_n = (program_id % programs_per_group) // group_program_count_m

    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)

    # The concatenated matrix has row stride 2n; the linear half sits n columns
    # to the right of the gate half, so one column offset drives both operands.
    weight_stride = 2 * n
    activation_pointers = activations + offsets_m[:, None] * k + offsets_k[None, :]
    gate_pointers = weights + offsets_k[:, None] * weight_stride + offsets_n[None, :]
    linear_pointers = gate_pointers + n

    gate_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    linear_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_block in range(tl.cdiv(k, block_k)):
        remaining_k = k - k_block * block_k
        activation_values = tl.load(
            activation_pointers,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        column_mask = offsets_n[None, :] < n
        depth_mask = offsets_k[:, None] < remaining_k
        gate_values = tl.load(
            gate_pointers,
            mask=depth_mask & column_mask,
            other=0.0,
        )
        linear_values = tl.load(
            linear_pointers,
            mask=depth_mask & column_mask,
            other=0.0,
        )
        gate_accumulator = tl.dot(
            activation_values,
            gate_values,
            gate_accumulator,
            out_dtype=tl.float32,
        )
        linear_accumulator = tl.dot(
            activation_values,
            linear_values,
            linear_accumulator,
            out_dtype=tl.float32,
        )
        activation_pointers += block_k
        gate_pointers += block_k * weight_stride
        linear_pointers += block_k * weight_stride

    if bias_mode == _BIAS_VECTOR:
        column_mask = offsets_n < n
        gate_accumulator += tl.load(
            bias + offsets_n,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)[None, :]
        linear_accumulator += tl.load(
            bias + n + offsets_n,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)[None, :]

    result = apply_glu(gate_accumulator, linear_accumulator, gate, softcap)
    output_pointers = output + offsets_m[:, None] * n + offsets_n[None, :]
    output_mask = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
    tl.store(output_pointers, result.to(tl.float16), mask=output_mask)


@dataclass(frozen=True, slots=True)
class GluMatmulSpecialization:
    """Immutable fused gated-linear-unit projection specialization."""

    m: int
    n: int
    k: int
    architecture: int
    # No default: the lab's `ACTIVATION_SWIGLU` enum means the *sigmoid* gate,
    # so a silently defaulted gate is exactly the trap this package has to make
    # impossible. Resolve it with `_activation.gate_for_lab_config`.
    gate: GluGate
    has_bias: bool = False
    softcap: float = 0.0


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the grouped one-dimensional launch grid for a tuning candidate."""
    m = cast("int", configuration["m"])
    n = cast("int", configuration["n"])
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    return (((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),)


def _artifact_grid(
    configuration: Mapping[str, object],
    m: int,
    n: int,
) -> tuple[int, int, int]:
    """Resolve the serialized launch grid from the selected configuration."""
    block_m = cast("int", configuration["block_m"])
    block_n = cast("int", configuration["block_n"])
    return (
        ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n),
        1,
        1,
    )


def compile_glu_matmul(specialization: GluMatmulSpecialization) -> KernelArtifact:
    """Autotune and compile one fused GLU projection specialization."""
    gate_code = GLU_GATES[specialization.gate]
    if gate_code not in SUPPORTED_GLU_GATES:
        message = (
            f"gate={specialization.gate!r} is reserved but not implemented; "
            "supply its formula and add the branch in `_activation.apply_glu` "
            "plus an entry in `SUPPORTED_GLU_GATES`."
        )
        raise NotImplementedError(message)
    softcapped = gate_code in SOFTCAPPED_GLU_GATES
    if softcapped and specialization.softcap <= 0.0:
        # A non-positive cap turns SiTU-GLU silently back into plain SwiGLU.
        message = (
            f"gate={specialization.gate!r} requires softcap > 0; "
            f"got {specialization.softcap}."
        )
        raise ValueError(message)
    if not softcapped and specialization.softcap != 0.0:
        message = (
            f"gate={specialization.gate!r} ignores softcap, but "
            f"{specialization.softcap} was supplied."
        )
        raise ValueError(message)
    output = torch.empty(
        (specialization.m, specialization.n),
        dtype=torch.float16,
        device="cuda",
    )
    activations = torch.empty(
        (specialization.m, specialization.k),
        dtype=torch.float16,
        device="cuda",
    )
    weights = torch.empty(
        (specialization.k, 2 * specialization.n),
        dtype=torch.float16,
        device="cuda",
    )
    bias = torch.zeros(2 * specialization.n, dtype=torch.float16, device="cuda")
    compiled = _glu_matmul_kernel[_autotune_grid](
        output,
        activations,
        weights,
        bias,
        specialization.m,
        specialization.n,
        specialization.k,
        BIAS_VECTOR if specialization.has_bias else BIAS_NONE,
        gate_code,
        specialization.softcap,
    )
    selected = _glu_matmul_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, specialization.m, specialization.n),
        parameters=(
            _POINTER,
            _POINTER,
            _POINTER,
            _POINTER if specialization.has_bias else _NULL_POINTER,
        ),
    )


def glu_matmul(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    activations: Buffer,
    weights: Buffer,
    specialization: GluMatmulSpecialization,
    *,
    bias: Buffer | None = None,
) -> None:
    """Append one fused gated-linear-unit projection."""
    if (bias is not None) != specialization.has_bias:
        message = "GluMatmulSpecialization.has_bias must match the bias argument."
        raise ValueError(message)
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_glu_matmul, specialization)
    arguments: tuple[Buffer, ...] = (output, activations, weights)
    if bias is not None:
        arguments += (bias,)
    builder.call(kernel, *arguments, readonly=list(arguments[1:]))
