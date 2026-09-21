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
_INT = lc0ex_pb2.PARAMETER_TYPE_U32
_STANDARD_INPUT_ELEMENT_COUNT = 4288
_AVERAGE_LEGAL_MOVES_PER_POSITION = 30


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
    total_legal_moves,
    batch_size: tl.constexpr,
    input_element_count: tl.constexpr,
    output_element_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < total_legal_moves
    source_index = tl.load(mapping + offsets, mask=valid)
    values = tl.load(
        input_ + source_index,
        mask=valid,
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
    output_element_count: int = 218


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat policy-gather grid for a tuning candidate."""
    element_count = cast("int", configuration["batch_size"]) * _AVERAGE_LEGAL_MOVES_PER_POSITION
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object]
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    return ("(total_legal_moves + %d) / %d" % (block_size - 1, block_size), "1", "1")


def _benchmark_mapping(specialization: PolicyMapSpecialization) -> torch.Tensor:
    """Create valid representative gather indices for autotuning."""
    return torch.arange(0, specialization.input_element_count,
        (specialization.input_element_count + _AVERAGE_LEGAL_MOVES_PER_POSITION - 1) // _AVERAGE_LEGAL_MOVES_PER_POSITION,
        dtype=torch.int32,
        device="cuda",
    )


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
    total_legal_moves = int(specialization.batch_size * _AVERAGE_LEGAL_MOVES_PER_POSITION)
    compiled = _policy_map_kernel[_autotune_grid](
        output,
        input_,
        mapping,
        total_legal_moves,
        specialization.batch_size,
        specialization.input_element_count,
        specialization.output_element_count,
    )
    selected = _policy_map_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(selected.kwargs),
        parameters=(_POINTER, _POINTER, _POINTER, _INT),
        autotuner=_policy_map_kernel,
    )


def policy_map(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    mapping: Buffer,
    specialization: PolicyMapSpecialization,
) -> None:
    """Append symbol-backed attention-policy gathering to an executable graph."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_policy_map, specialization)
    total_legal_moves = builder.add_int_parameter("total_legal_moves")
    builder.call(kernel, output, input_, mapping, total_legal_moves, readonly=(input_, mapping))
