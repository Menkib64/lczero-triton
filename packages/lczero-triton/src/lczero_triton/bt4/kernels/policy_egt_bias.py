"""EGT2 policy head: the pair-stream bias added to the policy logits, at 34 edge channels (item E, K2).

The EGT2 exports add the static net's pair term to `Q K^T` before the promotion logits are derived (gcap nodes
12601-12602, divided by 22.6274 at 12665): `sum_t c[t] Z0[t]`, with `Z0 = S (D.E + T)` the pair stream. Folded at build
time as on the static path, it is `S (sum_c g[c] E[c] + k)` with `g = c @ D^T / divisor` `[1, 34]` and
`k = einsum(c, T) / divisor` `[1, 64, 64]` (R0's `/policy/pair/scaled_coefficients` and `/policy/pair/constant_bias`,
FP16). `policy_static_bias` reads 4 channels from a one-byte E; this variant reads K1's packed `uint64 [batch, 64, 64]`
E (bit c = export channel c, `prologue_egt`), so the static kernel and its test stay untouched.

The 34-channel sum splits each code into its two 17-bit halves (base channels, transposes) as int32 words and decodes
them bit by bit: K1 measured per-channel decoding directly on uint64 at 6x the cost of 32-bit words. There is one program
per sample and no head axis, so the loop costs one `[64, 64]` tile pass per channel per sample.

It is applied in place to the first 4,096 entries of each sample's 4,288-entry policy record, so `promotion_logits`
sees the biased values, as the export's does. The sum is FP32 and the stored logit FP16, as `policy_static_bias`.
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
EDGE_CHANNELS = 34
BASE_CHANNELS = 17
_BASE_MASK = (1 << BASE_CHANNELS) - 1


@triton.autotune(
    configs=[triton.Config({}, num_warps=warps) for warps in (1, 2, 4, 8)],
    key=["batch_count"],
    cache_results=True,
    # In place: without restoring, every autotune candidate would add the bias again.
    restore_value=["records"],
)
@triton.jit
def _policy_egt_bias_kernel(  # noqa: PLR0913
    records,
    edges,
    edge_norm,
    scaled_coefficients,
    constant_bias,
    batch_count: tl.constexpr,  # noqa: ARG001  # Autotune key and launch grid.
) -> None:
    """Add `S (g.E + k)` to one sample's 64x64 policy logits, g over all 34 channels."""
    sample = tl.program_id(0)
    squares = tl.arange(0, 64)
    direct = squares[:, None] * 64 + squares[None, :]
    tile = 64 * 64
    code = tl.load(edges + sample * tile + direct)
    low = (code & 0x1FFFF).to(tl.int32)  # jit code reads no module globals: 17 = BASE_CHANNELS
    high = (code >> tl.full((), 17, tl.uint64)).to(tl.int32)
    bias = tl.load(constant_bias + direct).to(tl.float32)
    for channel in tl.static_range(17):
        bias += tl.load(scaled_coefficients + channel).to(tl.float32) * ((low >> channel) & 1).to(tl.float32)
        bias += tl.load(scaled_coefficients + 17 + channel).to(tl.float32) * (
            (high >> channel) & 1).to(tl.float32)
    normalization = tl.load(edge_norm + sample * tile + direct).to(tl.float32)
    pointer = records + sample * 4288 + direct
    logits = tl.load(pointer).to(tl.float32) + normalization * bias
    tl.store(pointer, logits.to(tl.float16))


@dataclass(frozen=True, slots=True)
class PolicyEgtBiasSpecialization:
    """Immutable EGT2 policy-bias specialization."""

    batch_count: int
    architecture: int


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """Return one program per sample."""
    return (cast("int", configuration["batch_count"]),)


def compile_policy_egt_bias(specialization: PolicyEgtBiasSpecialization) -> KernelArtifact:
    """Autotune and compile one EGT2 policy-bias specialization."""
    batch = specialization.batch_count
    records = torch.zeros((batch, _RECORD), dtype=torch.float16, device="cuda")
    edges = torch.zeros((batch, 64, 64), dtype=torch.uint64, device="cuda")  # K1: packed E, bit c = channel c
    edge_norm = torch.ones((batch, 64, 64), dtype=torch.float16, device="cuda")
    scaled = torch.zeros((1, EDGE_CHANNELS), dtype=torch.float16, device="cuda")
    constant = torch.zeros((1, 64, 64), dtype=torch.float16, device="cuda")
    compiled = _policy_egt_bias_kernel[_autotune_grid](records, edges, edge_norm, scaled, constant, batch)
    return artifact_from_triton(
        compiled,
        grid=(batch, 1, 1),
        parameters=(_POINTER,) * 5,
        autotuner=_policy_egt_bias_kernel,
    )


def policy_egt_bias(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    records: Buffer,
    edges: Buffer,
    edge_norm: Buffer,
    scaled_coefficients: Buffer,
    constant_bias: Buffer,
    specialization: PolicyEgtBiasSpecialization,
) -> None:
    """Append the in-place 34-channel policy pair bias."""
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_policy_egt_bias, specialization)
    builder.call(
        kernel, records, edges, edge_norm, scaled_coefficients, constant_bias,
        readonly=(edges, edge_norm, scaled_coefficients, constant_bias),
    )
