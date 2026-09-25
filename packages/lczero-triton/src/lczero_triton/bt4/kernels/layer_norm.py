"""Fused FP16 activation, residual, and layer-normalization family.

Q1 (2026-09-21): `quantise="int8"` (or `"e4m3"`) also emits the next GEMM's operand beside this norm's
FP16 output. Under post-norm the
norm's output is the FFN's input *and* the residual stream, so a SmoothQuant pre-scale cannot be folded
into `gamma` / `beta` -- that would rescale the stream too. It goes into the conversion instead, which
this kernel does in the same pass over the row, from the FP32 value it has already computed:

    q = clamp(floor((result - m) * r + 0.5), -127, 127)

`r` and `m` are `quantise_operand`'s per-channel vectors, the same two the artifact carries, read here as a
second pair of [width] FP32 buffers. Doing it here rather than in a second pass costs one FMA and two
vector loads a row, saves re-reading the output (2 bytes an element to write 1), and rounds better: a
separate pass only ever sees the FP16 copy. ⚠ It is not bit-identical to the plain kernel at every warp
count -- measured: identical at 4 and 8 warps, 6e-5 of outputs one ulp apart at 1 and 2 -- and the warp
count is the autotuner's to pick, so a quantised artifact is gated, not diffed. See `quantise_operand` for
the contract and `lab/_quant.py` for where `r` and `m` come from.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

Activation = Literal["none", "mish", "swish"]
# Q1: which operand format the int8/e4m3 side output writes; "" is off, and then this is the old kernel.
Quantise = Literal["", "int8", "e4m3"]
_QUANTISE: dict[str, int] = {"": 0, "int8": 1, "e4m3": 2}
_QUANTISE_DTYPES = {"int8": torch.int8, "e4m3": torch.float8_e4m3fn}

_ACTIVATION_NONE = 0
_ACTIVATION_MISH = tl.constexpr(1)
_ACTIVATION_SWISH = tl.constexpr(2)
_ACTIVATIONS: dict[Activation, int] = {
    "none": _ACTIVATION_NONE,
    "mish": 1,
    "swish": 2,
}
_MISH_BRANCH = tl.constexpr(-0.6)
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_WARP_COUNTS = (1, 2, 4, 8, 16)


def _layer_norm_configs() -> list[triton.Config]:
    """Return independent warp-count candidates for one normalized row."""
    return [triton.Config({}, num_warps=num_warps) for num_warps in _WARP_COUNTS]


@triton.jit
def _layer_norm_row(  # noqa: PLR0913, PLR0915
    output,
    input_,
    bias,
    skip,
    gammas,
    betas,
    alpha,
    quant_output,
    quant_prescale,
    quant_offset,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    has_skip: tl.constexpr,
    quantise: tl.constexpr,
    block_size: tl.constexpr,
    has_offset: tl.constexpr = True,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    pointers = row * width + offsets

    if width == block_size:
        values = tl.load(input_ + pointers).to(tl.float32)
        if has_bias:
            biases = tl.load(bias + offsets).to(tl.float32)
            values += biases

        if activation == _ACTIVATION_MISH:
            exponential = tl.exp(values)
            numerator = exponential * exponential + 2.0 * exponential
            division = values / (numerator + 2.0)
            values = tl.where(
                values <= _MISH_BRANCH,
                numerator * division,
                values - 2.0 * division,
            )
        elif activation == _ACTIVATION_SWISH:
            values /= 1.0 + tl.exp(-values)

        if has_skip:
            alpha_value = tl.load(alpha).to(tl.float32)
            skip_values = tl.load(skip + pointers).to(tl.float32)
            values = alpha_value * values + skip_values

        mean = tl.sum(values, axis=0) / width
        centered = values - mean
        variance = tl.sum(centered * centered, axis=0) / width
        normalized = centered / tl.sqrt(variance + epsilon)

        gamma_values = tl.load(gammas + offsets).to(tl.float32)
        beta_values = tl.load(betas + offsets).to(tl.float32)
        result = normalized * gamma_values + beta_values
        tl.store(output + pointers, result.to(tl.float16))
        if quantise != 0:
            scale = tl.load(quant_prescale + offsets).to(tl.float32)
            if has_offset:
                shift = tl.load(quant_offset + offsets).to(tl.float32) * scale
                operand = result * scale - shift
            else:
                # Round 26: no offset in the file (the family refuses `m`) -- 32 registers a thread and 4 KB a row
                # of L1 reads fewer; `result * scale` is the same arithmetic as `result * scale - 0 * scale`.
                operand = result * scale
            if quantise == 1:
                tl.store(quant_output + pointers, tl.clamp(tl.floor(operand + 0.5), -127.0, 127.0).to(tl.int8))
            else:
                tl.store(quant_output + pointers, operand.to(tl.float8e4nv))
    else:
        valid = (row < row_count) & (offsets < width)
        values = tl.load(input_ + pointers, mask=valid, other=0.0).to(tl.float32)
        if has_bias:
            biases = tl.load(bias + offsets, mask=valid, other=0.0).to(tl.float32)
            values += biases

        if activation == _ACTIVATION_MISH:
            exponential = tl.exp(values)
            numerator = exponential * exponential + 2.0 * exponential
            division = values / (numerator + 2.0)
            values = tl.where(
                values <= _MISH_BRANCH,
                numerator * division,
                values - 2.0 * division,
            )
        elif activation == _ACTIVATION_SWISH:
            values /= 1.0 + tl.exp(-values)

        if has_skip:
            alpha_value = tl.load(alpha).to(tl.float32)
            skip_values = tl.load(skip + pointers, mask=valid, other=0.0).to(tl.float32)
            values = alpha_value * values + skip_values

        values = tl.where(valid, values, 0.0)
        mean = tl.sum(values, axis=0) / width
        centered = tl.where(valid, values - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / width
        normalized = centered / tl.sqrt(variance + epsilon)

        gamma_values = tl.load(gammas + offsets, mask=valid, other=0.0).to(tl.float32)
        beta_values = tl.load(betas + offsets, mask=valid, other=0.0).to(tl.float32)
        result = normalized * gamma_values + beta_values
        tl.store(output + pointers, result.to(tl.float16), mask=valid)
        if quantise != 0:
            scale = tl.load(quant_prescale + offsets, mask=valid, other=0.0).to(tl.float32)
            if has_offset:
                shift = tl.load(quant_offset + offsets, mask=valid, other=0.0).to(tl.float32) * scale
                operand = result * scale - shift
            else:
                operand = result * scale
            if quantise == 1:
                tl.store(quant_output + pointers, tl.clamp(tl.floor(operand + 0.5), -127.0, 127.0).to(tl.int8),
                         mask=valid)
            else:
                tl.store(quant_output + pointers, operand.to(tl.float8e4nv), mask=valid)


@triton.autotune(
    configs=_layer_norm_configs(),
    key=["row_count", "width", "epsilon", "activation", "has_bias"],
    cache_results=True,
)
@triton.jit
def _layer_norm_kernel(
    output,
    input_,
    bias,
    gammas,
    betas,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    """Normalize rows without a residual connection or runtime alpha."""
    _layer_norm_row(
        output,
        input_,
        bias,
        input_,
        gammas,
        betas,
        gammas,
        output,
        gammas,
        betas,
        row_count,
        width,
        epsilon,
        activation,
        has_bias,
        0,
        0,
        block_size,
    )


@triton.autotune(
    configs=_layer_norm_configs(),
    key=["row_count", "width", "epsilon", "activation", "has_bias"],
    cache_results=True,
)
@triton.jit
def _layer_norm_skip_kernel(
    output,
    input_,
    bias,
    skip,
    gammas,
    betas,
    alpha,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    """Normalize rows after alpha scaling and a residual connection."""
    _layer_norm_row(
        output,
        input_,
        bias,
        skip,
        gammas,
        betas,
        alpha,
        output,
        gammas,
        betas,
        row_count,
        width,
        epsilon,
        activation,
        has_bias,
        1,
        0,
        block_size,
    )


# Q1: the same two kernels with the int8 side output. They are separate entry points, not a runtime flag,
# because the graph ABI counts pointer parameters: a net that does not quantise must not carry three.
@triton.autotune(
    configs=_layer_norm_configs(),
    key=["row_count", "width", "epsilon", "activation", "has_bias", "quantise", "has_offset"],
    cache_results=True,
)
@triton.jit
def _layer_norm_quant_kernel(  # noqa: PLR0913
    output,
    input_,
    bias,
    gammas,
    betas,
    quant_output,
    quant_prescale,
    quant_offset,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    quantise: tl.constexpr,
    block_size: tl.constexpr,
    has_offset: tl.constexpr = True,
) -> None:
    """Normalize rows and emit the int8 copy of the same output."""
    _layer_norm_row(
        output,
        input_,
        bias,
        input_,
        gammas,
        betas,
        gammas,
        quant_output,
        quant_prescale,
        quant_offset,
        row_count,
        width,
        epsilon,
        activation,
        has_bias,
        0,
        quantise,
        block_size,
        has_offset,
    )


@triton.autotune(
    configs=_layer_norm_configs(),
    key=["row_count", "width", "epsilon", "activation", "has_bias", "quantise", "has_offset"],
    cache_results=True,
)
@triton.jit
def _layer_norm_skip_quant_kernel(  # noqa: PLR0913
    output,
    input_,
    bias,
    skip,
    gammas,
    betas,
    alpha,
    quant_output,
    quant_prescale,
    quant_offset,
    row_count: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
    quantise: tl.constexpr,
    block_size: tl.constexpr,
    has_offset: tl.constexpr = True,
) -> None:
    """Normalize rows after the residual and emit the int8 copy of the same output."""
    _layer_norm_row(
        output,
        input_,
        bias,
        skip,
        gammas,
        betas,
        alpha,
        quant_output,
        quant_prescale,
        quant_offset,
        row_count,
        width,
        epsilon,
        activation,
        has_bias,
        1,
        quantise,
        block_size,
        has_offset,
    )


@dataclass(frozen=True, slots=True)
class LayerNormSpecialization:
    """Immutable contiguous FP16 layer-normalization specialization."""

    row_count: int
    width: int
    activation: Activation
    has_skip: bool
    architecture: int
    has_bias: bool = True
    epsilon: float = 1e-3
    # Q1: also write the GEMM's operand format beside the FP16 output, through the
    # per-channel `r` and `m`. "" is off, and then the compiled kernel is the one
    # every earlier artifact carries. int8 is SPEC v2's default, e4m3 its fallback.
    quantise: Quantise = ""
    # Round 26: whether the conversion subtracts the offset `m` (False when the vector file carries none -- the
    # flagship's recipe). The `m` buffer is still passed, so the call's ABI does not depend on it.
    quant_offset: bool = True


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one Triton program per normalized row."""
    return (cast("int", configuration["row_count"]),)


def _artifact_grid(row_count: int) -> tuple[int, int, int]:
    """Return the serialized one-program-per-row launch grid."""
    return (row_count, 1, 1)


def compile_layer_norm(
    specialization: LayerNormSpecialization,
) -> KernelArtifact:
    """Autotune and compile one fused FP16 layer-normalization variant."""
    shape = (specialization.row_count, specialization.width)
    output = torch.empty(shape, dtype=torch.float16, device="cuda")
    input_ = torch.zeros(shape, dtype=torch.float16, device="cuda")
    bias = torch.zeros(specialization.width, dtype=torch.float16, device="cuda")
    gammas = torch.ones(specialization.width, dtype=torch.float16, device="cuda")
    betas = torch.zeros(specialization.width, dtype=torch.float16, device="cuda")
    block_size = triton.next_power_of_2(specialization.width)
    activation = _ACTIVATIONS[specialization.activation]
    parameters: tuple[int, ...]
    # Q1: the folded pre-scale pair is FP32. `gamma * r` is of order 127 / amax, which FP16 holds but
    # with 11 bits of mantissa -- quantisation noise in the scale itself, for 4 KB a row out of L1.
    quant = (
        (
            torch.empty(shape, dtype=_QUANTISE_DTYPES[specialization.quantise], device="cuda"),
            torch.ones(specialization.width, dtype=torch.float32, device="cuda"),
            torch.zeros(specialization.width, dtype=torch.float32, device="cuda"),
        )
        if specialization.quantise
        else ()
    )

    if specialization.has_skip:
        skip = torch.zeros(shape, dtype=torch.float16, device="cuda")
        alpha = torch.ones(1, dtype=torch.float16, device="cuda")
        kernel = _layer_norm_skip_quant_kernel if specialization.quantise else _layer_norm_skip_kernel
        compiled = kernel[_autotune_grid](
            output,
            input_,
            bias,
            skip,
            gammas,
            betas,
            alpha,
            *quant,
            specialization.row_count,
            specialization.width,
            specialization.epsilon,
            activation,
            specialization.has_bias,
            *((_QUANTISE[specialization.quantise],) if specialization.quantise else ()),
            block_size,
            **({"has_offset": specialization.quant_offset} if specialization.quantise else {}),
        )
        autotuner = kernel
        parameters = (_POINTER,) * (10 if specialization.quantise else 7)
    else:
        kernel = _layer_norm_quant_kernel if specialization.quantise else _layer_norm_kernel
        compiled = kernel[_autotune_grid](
            output,
            input_,
            bias,
            gammas,
            betas,
            *quant,
            specialization.row_count,
            specialization.width,
            specialization.epsilon,
            activation,
            specialization.has_bias,
            *((_QUANTISE[specialization.quantise],) if specialization.quantise else ()),
            block_size,
            **({"has_offset": specialization.quant_offset} if specialization.quantise else {}),
        )
        autotuner = kernel
        parameters = (_POINTER,) * (8 if specialization.quantise else 5)

    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(specialization.row_count),
        parameters=parameters,
        autotuner=autotuner,
    )


def layer_norm(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    bias: Buffer | None,
    gammas: Buffer,
    betas: Buffer,
    specialization: LayerNormSpecialization,
    *,
    skip: Buffer | None = None,
    alpha: Buffer | None = None,
    quant_output: Buffer | None = None,
    quant_prescale: Buffer | None = None,
    quant_offset: Buffer | None = None,
) -> None:
    """Append fused activation, residual addition, and layer normalization."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_layer_norm, specialization)
    bias_buffer = bias if (bias is not None and specialization.has_bias) else output
    arguments = [output, input_, bias_buffer]
    if specialization.has_skip:
        if skip is None or alpha is None:
            message = (
                "Skip and alpha buffers are required when "
                "specialization.has_skip is True"
            )
            raise ValueError(message)
        arguments.append(skip)
        arguments.extend((gammas, betas))
        arguments.append(alpha)
    else:
        arguments.extend((gammas, betas))
    quant = (quant_output, quant_prescale, quant_offset)
    if bool(specialization.quantise) != all(buffer is not None for buffer in quant):
        message = (
            "The operand output and its two vectors are required exactly when "
            "specialization.quantise names a format"
        )
        raise ValueError(message)
    if specialization.quantise:
        arguments.extend(cast("tuple[Buffer, ...]", quant))
    readonly = [
        source
        for source in arguments[1:]
        if source is not output and source is not quant_output
    ]
    builder.call(kernel, *arguments, readonly=readonly)
