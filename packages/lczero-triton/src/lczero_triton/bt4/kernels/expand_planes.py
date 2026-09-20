"""Packed chess-plane expansion kernel family."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._autotune import elementwise_configs
from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@triton.autotune(
    configs=elementwise_configs(),
    key=["plane_count", "square_count"],
    cache_results=True,
)
@triton.jit
def _expand_planes_kernel(
    output,
    masks,
    values,
    plane_count: tl.constexpr,
    square_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < plane_count * square_count
    plane = offsets // square_count
    square = offsets % square_count
    mask_values = tl.load(masks + plane, mask=valid, other=0)
    plane_values = tl.load(values + plane, mask=valid, other=0.0).to(tl.float16)
    is_set = ((mask_values >> square) & 1) != 0
    expanded = tl.where(is_set, plane_values, 0.0)
    tl.store(output + offsets, expanded, mask=valid)


@dataclass(frozen=True, slots=True)
class ExpandPlanesSpecialization:
    """Immutable U64/F32-to-FP16 plane-expansion specialization."""

    input_type: Litteral[
        lc0ex_pb2.Buffer.DATA_TYPE_f32,
        lc0ex_pb2.Buffer.DATA_TYPE_f16,
    ]
    plane_count: int
    architecture: int
    square_count: int = 64


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return the flat plane-expansion grid for a tuning candidate."""
    element_count = cast("int", configuration["plane_count"]) * cast(
        "int", configuration["square_count"]
    )
    block_size = cast("int", configuration["block_size"])
    return ((element_count + block_size - 1) // block_size,)


def _artifact_grid(
    configuration: Mapping[str, object],
    plane_count: int,
    square_count: int,
) -> tuple[int, int, int]:
    """Resolve the serialized grid from the selected configuration."""
    block_size = cast("int", configuration["block_size"])
    element_count = plane_count * square_count
    return ((element_count + block_size - 1) // block_size, 1, 1)


def compile_expand_planes(
    specialization: ExpandPlanesSpecialization,
) -> KernelArtifact:
    """Autotune and compile one packed-plane expansion specialization."""
    output = torch.empty(
        (specialization.plane_count, specialization.square_count),
        dtype=torch.float16,
        device="cuda",
    )
    masks = torch.zeros(
        specialization.plane_count,
        dtype=torch.uint64,
        device="cuda",
    )
    values = torch.zeros(
        specialization.plane_count,
        dtype=torch.float16 if specialization.input_type == lc0ex_pb2.Buffer.DATA_TYPE_F16 else torch.float32,
        device="cuda",
    )
    compiled = _expand_planes_kernel[_autotune_grid](
        output,
        masks,
        values,
        specialization.plane_count,
        specialization.square_count,
    )
    selected = _expand_planes_kernel.best_config
    return artifact_from_triton(
        compiled,
        grid=_artifact_grid(
            selected.kwargs,
            specialization.plane_count,
            specialization.square_count,
        ),
        parameters=(_POINTER, _POINTER, _POINTER),
        autotuner=_expand_planes_kernel,
    )


def expand_planes(
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    masks: Buffer,
    values: Buffer,
    specialization: ExpandPlanesSpecialization,
) -> None:
    """Append a packed-plane expansion call to an executable graph."""
    builder.set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        f"sm_{specialization.architecture}",
    )
    kernel = kernels.get(compile_expand_planes, specialization)
    builder.call(kernel, output, masks, values, readonly=(masks, values))
