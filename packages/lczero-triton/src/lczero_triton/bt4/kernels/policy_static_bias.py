"""Static-net policy head: the pair-stream bias added to the policy logits.

The export adds one `[batch, 64, 64]` term to `Q K^T` before the promotion
logits are derived (nodes 4165-4167): `sum_t c[t] M[t]` with the same shared
`M` every encoder block reads. Folded at build time exactly as the blocks' mix
term is, it is `S (sum_e g[e] E[e] + k)`, with `g = c @ D^T` `[1, 4]` and
`k = einsum(c, T)` `[1, 64, 64]` (`lab/_names.py`, `/policy/pair/*`).

It is applied in place to the first 4,096 entries of each sample's 4,288-entry
policy record, so `promotion_logits` -- which reads the rank-7 to rank-8 logits
back out of the same record -- sees the biased values, as the export's does.
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

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_RECORD = 4288
_EDGE_CHANNELS = 4


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
    key=["batch_count"],
    cache_results=True,
    # The kernel works IN PLACE: without restoring, every autotune candidate
    # would add the bias to the records again.
    restore_value=["records"],
)
@triton.jit
def _policy_static_bias_kernel(  # noqa: PLR0913
    records,
    edges,
    edge_norm,
    scaled_coefficients,
    constant_bias,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
) -> None:
    """Add `S (g.E + k)` to one sample's 64x64 policy logits."""
    sample = tl.program_id(0)
    squares = tl.arange(0, 64)
    direct = squares[:, None] * 64 + squares[None, :]
    tile = 64 * 64
    bias = tl.load(constant_bias + direct).to(tl.float32)
    codes = tl.load(edges + sample * tile + direct)  # P5: packed E, channel c = bit c
    for channel in tl.static_range(4):
        bias += tl.load(scaled_coefficients + channel).to(tl.float32) * ((codes >> channel) & 1).to(tl.float32)
    normalization = tl.load(edge_norm + sample * tile + direct).to(tl.float32)
    pointer = records + sample * 4288 + direct
    logits = tl.load(pointer).to(tl.float32) + normalization * bias
    tl.store(pointer, logits.to(tl.float16))


@dataclass(frozen=True, slots=True)
class PolicyStaticBiasSpecialization:
    """Immutable static-net policy-bias specialization."""

    batch_count: int
    architecture: int


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one program per sample."""
    return (cast("int", configuration["batch_count"]),)


def compile_policy_static_bias(specialization: PolicyStaticBiasSpecialization) -> KernelArtifact:
    """Autotune and compile one policy-bias specialization."""
    batch = specialization.batch_count
    records = torch.zeros((batch, _RECORD), dtype=torch.float16, device="cuda")
    edges = torch.zeros((batch, 64, 64), dtype=torch.uint8, device="cuda")  # P5: packed E
    edge_norm = torch.ones((batch, 64, 64), dtype=torch.float16, device="cuda")
    scaled = torch.zeros((1, _EDGE_CHANNELS), dtype=torch.float16, device="cuda")
    constant = torch.zeros((1, 64, 64), dtype=torch.float16, device="cuda")
    compiled = _policy_static_bias_kernel[_autotune_grid](records, edges, edge_norm, scaled, constant, batch)
    return artifact_from_triton(
        compiled,
        grid=(batch, 1, 1),
        parameters=(_POINTER,) * 5,
        autotuner=_policy_static_bias_kernel,
    )


def policy_static_bias(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    records: Buffer,
    edges: Buffer,
    edge_norm: Buffer,
    scaled_coefficients: Buffer,
    constant_bias: Buffer,
    specialization: PolicyStaticBiasSpecialization,
) -> None:
    """Append the in-place policy pair bias."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_policy_static_bias, specialization)
    builder.call(
        kernel, records, edges, edge_norm, scaled_coefficients, constant_bias,
        readonly=(edges, edge_norm, scaled_coefficients, constant_bias),
    )
