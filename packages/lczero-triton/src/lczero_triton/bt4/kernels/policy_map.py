"""Attention-policy gather kernel family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder, SymbolHandle
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import elementwise_configs
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.mapping_table import values as mapping_values

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_STANDARD_INPUT_ELEMENT_COUNT = 4288


@triton.autotune(
    configs=elementwise_configs(),
    key=["batch_size", "input_element_count", "output_element_count"],
    cache_results=True,
)
@triton.jit
def _policy_map_kernel(
    output,
    input_,
    mapping,
    batch_size: tl.constexpr,
    input_element_count: tl.constexpr,
    output_element_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    element_count = batch_size * output_element_count
    valid = offsets < element_count
    output_index = offsets % output_element_count
    batch = offsets // output_element_count
    source_index = tl.load(mapping + output_index, mask=valid, other=-1)
    valid_source = valid & (source_index >= 0) & (source_index < input_element_count)
    values = tl.load(
        input_ + batch * input_element_count + source_index,
        mask=valid_source,
        other=0.0,
    )
    out_dtype = output.dtype.element_ty
    tl.store(output + offsets, values.to(out_dtype), mask=valid)


@dataclass(frozen=True, slots=True)
class PolicyMapSpecialization:
    """Immutable attention-policy gather specialization."""

    output_type: Literal[
        lc0ex_pb2.Buffer.DATA_TYPE_F32,
        lc0ex_pb2.Buffer.DATA_TYPE_F16,
    ]
    batch_size: int
    architecture: int
    input_element_count: int = 4288
    output_element_count: int = 1858


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat policy-gather grid for a tuning candidate."""
    element_count = cast("int", configuration["batch_size"]) * cast(
        "int", configuration["output_element_count"]
    )
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object],
    element_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size, 1, 1)


def _benchmark_mapping(specialization: PolicyMapSpecialization) -> torch.Tensor:
    """Create valid representative gather indices for autotuning."""
    standard_mapping = mapping_values()
    if (
        specialization.input_element_count == _STANDARD_INPUT_ELEMENT_COUNT
        and specialization.output_element_count == len(standard_mapping)
    ):
        return torch.tensor(standard_mapping, dtype=torch.int32, device="cuda")
    return torch.arange(
        specialization.output_element_count,
        dtype=torch.int32,
        device="cuda",
    ).remainder_(specialization.input_element_count)


def compile_policy_map(
    specialization: PolicyMapSpecialization,
) -> KernelArtifact:
    """Autotune and compile one FP32 attention-policy gather specialization."""
    element_count = specialization.batch_size * specialization.output_element_count
    output = torch.empty(
        element_count,
        dtype=torch.float16 if specialization.output_type == lc0ex_pb2.Buffer.DATA_TYPE_F16 else torch.float32,
        device="cuda"
    )
    input_ = torch.zeros(
        specialization.batch_size * specialization.input_element_count,
        dtype=torch.float16,
        device="cuda",
    )
    mapping = _benchmark_mapping(specialization)
    compiled = _policy_map_kernel[_autotune_grid](
        output,
        input_,
        mapping,
        specialization.batch_size,
        specialization.input_element_count,
        specialization.output_element_count,
    )
    selected = _policy_map_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs, element_count),
        parameters=(_POINTER, _POINTER, _POINTER),
        autotuner=_policy_map_kernel,
    )


def policy_map(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    mapping: SymbolHandle,
    specialization: PolicyMapSpecialization,
) -> None:
    """Append symbol-backed attention-policy gathering to an executable graph."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_policy_map, specialization)
    builder.call(kernel, output, input_, mapping, readonly=(input_,))
