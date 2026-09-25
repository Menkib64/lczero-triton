"""The activation quantiser: float to the GEMM's operand format through a PER-CHANNEL pre-scale (Q1).

Why the vector is here and not in the norm
------------------------------------------
Post-training int8 fails on our nets at one kind of site: a norm output that is wide per CHANNEL
(`ffn_in` first, then `attn_in`; the 09-20 PTQ read puts all of the -2.2..-4.6 pp there). The relief is
SmoothQuant -- divide channel j by `s_j` before quantising and multiply weight column j by `s_j` offline,
which is exact for a linear layer.

Under **pre-norm** that division folds into the LayerNorm affine and nothing at serving time changes.
Our nets, the flagship included, are **post-norm**: the norm's output is the FFN's input *and* the
residual stream, so the norm cannot absorb `s` without rescaling the stream as well. The vector therefore
lives in the conversion:

    q_j = clamp(floor((x_j - m_j) * r_j + 0.5), -127, 127)        r = 1 / (s * D), m = the channel mean
    W'  = W[:, j] * s_j, quantised per OUTPUT channel as today

`r` and `m` are [channels] fp32 vectors, static, shipped with the artifact (`lab/_quant.py` builds them
from the analyser's `.npz` and folds them where a producer allows it). `m` is optional: an offset fold
needs `W . m` added to the GEMM bias, which is an offline change to a tensor the artifact already carries.

⚠ The integer GEMM still sees ONE activation step `D`, so SPEC v2 §2.2 holds unchanged -- "the channel
axis of an activation is the GEMM reduction axis, so a per-channel activation scale is not expressible in
the mma at all". This vector acts **before** the mma, not inside it.

Two forms, same arithmetic
--------------------------
* **this kernel** -- a standalone pass over a float tensor. It is the general form: any producer, at the
  cost of re-reading the activation (2 bytes per element) to write 1.
* **the fold** -- a producer that already applies a per-channel affine absorbs the vector for free:
  `layer_norm(quantise=True)` emits the int8 copy beside its FP16 output from a second affine pair,
  `gamma' = gamma * r` and `beta' = (beta - m) * r`, so the pre-scale costs no load and no arithmetic,
  only the int8 store. That is the served form at `ffn_in` and `attn_in`. At `ffn_mid` and `attn_out` the
  producer is a GEMM whose epilogue must apply a per-column constant anyway (the contract's per-column
  output scale, R81); there the vector multiplies into that constant and lands with Q1's epilogue.

The rounding is the lane's int8 convention, the one `egt_state_tiles.cast_state_i8` writes: round half up,
clamp symmetrically to +-127 (-128 is never written), saturating silently past the calibrated range.

**The format is a parameter** (ruling 09-22, ask 2). int8 is SPEC v2's default and e4m3 its fallback, and the
09-21 accuracy read makes the choice a real one rather than a contingency: int8 costs 1.0-2.0 pp of DECISIVE
value accuracy on every net measured, e4m3 costs none (-0.34..+0.26 pp), against 3-13 % of the GEMM gain. So both
must be measurable on the real net. The two differ here only in the cast: e4m3 spends its bits on the exponent,
so its scale buys RANGE and the analyser ships no migration vector for it -- pass a flat `r` (`identity_prescale`)
and per-tensor weight scales, and the same kernel serves it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from lc0ex import Buffer, KernelArtifact, ProgramBuilder
from lc0ex.proto import lc0ex_pb2
from lc0ex.triton_module_compiler import artifact_from_triton

from lczero_triton.bt4.kernels._cache import KernelCache

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
INT8_MAX = 127.0
# The operand formats a quantised site can produce, and the codes the kernel branches on.
OperandFormat = Literal["int8", "e4m3"]
FORMAT_INT8 = 0
FORMAT_E4M3 = 1
_FORMATS: dict[str, int] = {"int8": FORMAT_INT8, "e4m3": FORMAT_E4M3}
_TORCH_FORMATS = {"int8": torch.int8, "e4m3": torch.float8_e4m3fn}

# (rows per program, num_warps). One program covers a run of rows and the whole channel axis, so the
# [channels] vectors are loaded once per program and every row of the run reuses them from registers.
_CONFIGURATIONS = ((1, 4), (1, 8), (2, 4), (2, 8), (4, 4), (4, 8), (8, 8), (16, 8))


@triton.jit
def _quantise_operand_body(  # noqa: PLR0913
    output,
    input_,
    prescale,
    offset,
    rows: tl.constexpr,
    channels: tl.constexpr,
    has_offset: tl.constexpr,
    operand: tl.constexpr,
    lanes: tl.constexpr,
    rows_per_program: tl.constexpr,
) -> None:
    """Quantise `rows_per_program` rows of a [rows, channels] activation into the GEMM's operand format."""
    first = tl.program_id(0) * rows_per_program
    channel = tl.arange(0, lanes)
    live = channel < channels
    scale = tl.load(prescale + channel, mask=live, other=0.0).to(tl.float32)
    if has_offset:
        # (x - m) * r == x * r - (m * r): the product is per channel, so it is hoisted out of the rows.
        shift = tl.load(offset + channel, mask=live, other=0.0).to(tl.float32) * scale
    for step in tl.static_range(rows_per_program):
        row = first + step
        if row < rows:
            pointers = row * channels + channel
            values = tl.load(input_ + pointers, mask=live, other=0.0).to(tl.float32) * scale
            if has_offset:
                values -= shift
            if operand == 0:  # FORMAT_INT8: round half up, clamp symmetrically (-128 is never written).
                code = tl.clamp(tl.floor(values + 0.5), -127.0, 127.0)
                tl.store(output + pointers, code.to(tl.int8), mask=live)
            else:  # FORMAT_E4M3: the cast rounds and saturates; the scale buys range, not precision.
                tl.store(output + pointers, values.to(tl.float8e4nv), mask=live)


_quantise_operand_kernel = triton.autotune(
    configs=[triton.Config({"rows_per_program": rows}, num_warps=warps) for rows, warps in _CONFIGURATIONS],
    key=["rows", "channels", "has_offset", "operand", "lanes"],
    cache_results=True,
)(_quantise_operand_body)


@dataclass(frozen=True, slots=True)
class QuantiseOperandSpecialization:
    """One float -> operand conversion over a [rows, channels] activation, pre-scaled per channel."""

    rows: int
    channels: int
    architecture: int
    # int8 (SPEC v2's default) or e4m3 (its fallback, and the one with no value-accuracy loss).
    operand: OperandFormat = "int8"
    # The offset fold (`m`). Off by default: it needs `W . m` folded into the GEMM bias offline, so a
    # net whose calibration never asked for one must not carry the load.
    has_offset: bool = False
    # The producer's format. FP16 is what every encoder site produces today; FP32 exists for the edge
    # stream, whose accumulation chain is FP32.
    input_f32: bool = False

    def __post_init__(self) -> None:
        """Refuse a shape the kernel cannot address or a channel axis a `tl.arange` cannot cover."""
        if self.rows <= 0 or self.channels <= 0:
            message = (f"QuantiseOperandSpecialization needs positive rows and channels; "
                       f"got {self.rows}x{self.channels}")
            raise ValueError(message)
        if self.channels > 8192:  # noqa: PLR2004  # one program covers the channel axis in registers.
            message = f"QuantiseOperandSpecialization.channels={self.channels} exceeds the 8192-lane channel axis"
            raise ValueError(message)
        if self.operand not in _FORMATS:
            message = f"QuantiseOperandSpecialization.operand={self.operand!r}; expected one of {tuple(_FORMATS)}"
            raise ValueError(message)


def _lanes(channels: int) -> int:
    """The `tl.arange` width covering the channel axis (a power of two, at least one warp)."""
    return max(32, triton.next_power_of_2(channels))


def _autotune_grid(configuration: Mapping[str, object]) -> tuple[int]:
    """One program per run of rows."""
    rows = cast("int", configuration["rows"])
    rows_per_program = cast("int", configuration["rows_per_program"])
    return ((rows + rows_per_program - 1) // rows_per_program,)


def launch_quantise_operand(
    output: torch.Tensor,
    input_: torch.Tensor,
    prescale: torch.Tensor,
    offset: torch.Tensor | None,
    specialization: QuantiseOperandSpecialization,
) -> object:
    """Launch one conversion on torch tensors (tests and benchmarks)."""
    return _quantise_operand_kernel[_autotune_grid](
        output,
        input_,
        prescale,
        offset if offset is not None else prescale,
        specialization.rows,
        specialization.channels,
        specialization.has_offset,
        _FORMATS[specialization.operand],
        _lanes(specialization.channels),
    )


def compile_quantise_operand(specialization: QuantiseOperandSpecialization) -> KernelArtifact:
    """Autotune and compile one conversion specialization."""
    rows, channels = specialization.rows, specialization.channels
    dtype = torch.float32 if specialization.input_f32 else torch.float16
    output = torch.empty((rows, channels), dtype=_TORCH_FORMATS[specialization.operand], device="cuda")
    input_ = torch.zeros((rows, channels), dtype=dtype, device="cuda")
    prescale = torch.ones(channels, dtype=torch.float32, device="cuda")
    offset = torch.zeros(channels, dtype=torch.float32, device="cuda") if specialization.has_offset else None
    compiled = launch_quantise_operand(output, input_, prescale, offset, specialization)
    rows_per_program = cast("int", _quantise_operand_kernel.best_config.kwargs["rows_per_program"])
    # Four pointers ALWAYS: Triton keeps the unused `offset` in the compiled signature, so a 3-pointer declaration
    # launches the kernel with one argument too few -- the runtime then reads past its parameter array (round 26:
    # the first served graph to use this builder segfaulted in the host at its first launch).
    parameters: Sequence[int] = (_POINTER,) * 4
    return artifact_from_triton(
        compiled,
        grid=((rows + rows_per_program - 1) // rows_per_program, 1, 1),
        parameters=tuple(parameters),
        autotuner=_quantise_operand_kernel,
    )


def quantise_operand(  # noqa: PLR0913
    builder: ProgramBuilder,
    kernels: KernelCache,
    output: Buffer,
    input_: Buffer,
    prescale: Buffer,
    offset: Buffer | None,
    specialization: QuantiseOperandSpecialization,
) -> None:
    """Append one float -> operand conversion to an executable graph."""
    if (offset is not None) != specialization.has_offset:
        message = "quantise_operand: the offset buffer and QuantiseOperandSpecialization.has_offset must agree"
        raise ValueError(message)
    builder.set_target(lc0ex_pb2.Target.VENDOR_NVIDIA, f"sm_{specialization.architecture}")
    kernel = kernels.get(compile_quantise_operand, specialization)
    # The offset slot takes the prescale buffer when there is no offset (never read: `has_offset` is a constexpr).
    reads = (input_, prescale, prescale if offset is None else offset)
    builder.call(kernel, output, *reads, readonly=tuple(dict.fromkeys(reads)))
