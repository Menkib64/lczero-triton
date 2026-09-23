"""Periodic FP16 vector-addition kernel family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import elementwise_configs
from lczero_triton.bt4.kernels._cache import KernelCache

Activation = Literal["none", "mish", "relu"]

_ACTIVATION_NONE = 0
_ACTIVATION_MISH = tl.constexpr(1)
_ACTIVATION_RELU = tl.constexpr(2)
_ACTIVATIONS: dict[Activation, int] = {
    "none": _ACTIVATION_NONE,
    "mish": 1,
    "relu": 2,
}
_MISH_BRANCH = tl.constexpr(-0.6)
_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@triton.autotune(
    configs=elementwise_configs(),
    key=["element_count", "bias_element_count", "activation"],
    cache_results=True,
)
@triton.jit
def _add_vectors_kernel(
    output,
    input_,
    bias,
    element_count: tl.constexpr,
    bias_element_count: tl.constexpr,
    activation: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < element_count
    values = tl.load(input_ + offsets, mask=valid, other=0.0)
    biases = tl.load(
        bias + offsets % bias_element_count,
        mask=valid,
        other=0.0,
    )
    values += biases
    if activation == _ACTIVATION_MISH:
        exponential = tl.exp(values.to(tl.float32))
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        values = tl.where(
            values <= _MISH_BRANCH,
            numerator * division,
            values - 2.0 * division,
        )
    elif activation == _ACTIVATION_RELU:
        values = tl.maximum(values, 0.0)
    tl.store(output + offsets, values, mask=valid)


@dataclass(frozen=True, slots=True)
class AddVectorsSpecialization:
    """Immutable periodic FP16 vector-addition specialization."""

    element_count: int
    bias_element_count: int
    activation: Activation
    architecture: int


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat vector-addition grid for a tuning candidate."""
    element_count = cast("int", configuration["element_count"])
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object],
    element_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size, 1, 1)


def compile_add_vectors(
    specialization: AddVectorsSpecialization,
) -> KernelArtifact:
    """Autotune and compile one periodic FP16 vector-addition specialization."""
    output = torch.empty(
        specialization.element_count,
        dtype=torch.float16,
        device="cuda",
    )
    input_ = torch.zeros(
        specialization.element_count,
        dtype=torch.float16,
        device="cuda",
    )
    bias = torch.zeros(
        specialization.bias_element_count,
        dtype=torch.float16,
        device="cuda",
    )
    compiled = _add_vectors_kernel[_autotune_grid](
        output,
        input_,
        bias,
        specialization.element_count,
        specialization.bias_element_count,
        _ACTIVATIONS[specialization.activation],
    )
    selected = _add_vectors_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, specialization.element_count),
        parameters=(_POINTER, _POINTER, _POINTER),
        autotuner=_add_vectors_kernel,
    )


def add_vectors(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    bias: Buffer,
    specialization: AddVectorsSpecialization,
) -> None:
    """Append periodic vector addition, retaining valid in-place writes."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_add_vectors, specialization)
    readonly = [source for source in (input_, bias) if source is not output]
    builder.call(kernel, output, input_, bias, readonly=readonly)
