"""CUDA tests for Q1's activation quantiser: the per-channel pre-scale, standalone and inside the norm.

The three things that would make a served net wrong and still look healthy:
* a vector applied along the wrong axis (a transposed `r` still produces codes in range),
* an int8 side output that perturbs the FP16 stream the residual carries,
* a rounding rule that differs from the reference the analyser calibrated against.
Each has a test here.
"""

import pytest
import torch
from lczero_triton.bt4.kernels.layer_norm import (
    _autotune_grid as _norm_grid,
)
from lczero_triton.bt4.kernels.layer_norm import (
    _layer_norm_kernel,
    _layer_norm_quant_kernel,
)
from lczero_triton.bt4.kernels.quantise_operand import (
    QuantiseOperandSpecialization,
    launch_quantise_operand,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

# The BT6-test trunk: 64 squares of a batch-64 slot, d_model 1024.
_ROWS, _WIDTH = 4096, 1024


def _codes(values: torch.Tensor, prescale: torch.Tensor, offset: torch.Tensor | None) -> torch.Tensor:
    """The reference conversion, in the kernel's op order and precision."""
    scaled = values.float() * prescale
    if offset is not None:
        scaled = scaled - offset * prescale
    return torch.clamp(torch.floor(scaled + 0.5), -127.0, 127.0).to(torch.int8)


def _quantise(values: torch.Tensor, prescale: torch.Tensor, offset: torch.Tensor | None = None,
              operand: str = "int8") -> torch.Tensor:
    rows, channels = values.shape
    dtype = torch.int8 if operand == "int8" else torch.float8_e4m3fn
    output = torch.empty((rows, channels), dtype=dtype, device="cuda")
    specialization = QuantiseOperandSpecialization(rows, channels, 89, operand=operand,
                                                   has_offset=offset is not None)
    launch_quantise_operand(output, values, prescale, offset, specialization)
    return output


@pytest.mark.parametrize("channels", [1024, 683])
def test_prescale_is_per_channel(channels: int) -> None:
    """Every channel takes its own scale, including on a width no power of two covers."""
    torch.manual_seed(20260921)
    rows = 257
    values = (torch.randn(rows, channels, device="cuda") * 3.0).half()
    # Four orders of magnitude between the narrowest and the widest channel: this is the outlier
    # spread SmoothQuant exists for, and a per-tensor scale cannot serve both ends of it.
    prescale = torch.exp(torch.linspace(-4.0, 5.0, channels, device="cuda"))

    codes = _quantise(values, prescale)

    torch.testing.assert_close(codes, _codes(values, prescale, None), rtol=0.0, atol=0.0)
    # A transposed or broadcast vector would pass a range check; this does not.
    assert not torch.equal(codes, _codes(values, prescale.flip(0), None))


def test_offset_fold_shifts_before_rounding() -> None:
    """`(x - m) * r`, with `m` per channel, is the form the GEMM bias absorbs.

    A norm output carries the norm's beta as a per-channel MEAN. A symmetric grid spends its range on
    that mean unless the offset comes out first: here the channels sit at +-6 with a spread of 3, and a
    step fitted to the SPREAD saturates everything until `m` is subtracted.
    """
    torch.manual_seed(20260921)
    channels = 512
    values = (torch.randn(129, channels, device="cuda") + 6.0).half()
    prescale = torch.full((channels,), 127.0 / 3.0, device="cuda")
    offset = torch.linspace(5.5, 6.5, channels, device="cuda")

    codes = _quantise(values, prescale, offset)

    torch.testing.assert_close(codes, _codes(values, prescale, offset), rtol=0.0, atol=0.0)
    assert (_quantise(values, prescale).to(torch.int32).abs() == 127).float().mean() > 0.9  # noqa: PLR2004
    assert (codes.to(torch.int32).abs() == 127).float().mean() < 0.05  # noqa: PLR2004


def test_saturation_is_symmetric_and_silent() -> None:
    """Past the calibrated range the code clamps to +-127; -128 is never written."""
    channels = 64
    values = torch.linspace(-1e4, 1e4, 8 * channels, device="cuda").reshape(8, channels).half()
    prescale = torch.ones(channels, device="cuda")

    codes = _quantise(values, prescale)

    assert int(codes.to(torch.int32).min()) == -127
    assert int(codes.to(torch.int32).max()) == 127


def _e4m3_step(values: torch.Tensor) -> torch.Tensor:
    """One e4m3 step at each value's magnitude: 2^-9 below 2^-6, and 2^(exponent - 3) above it."""
    exponent = torch.floor(torch.log2(values.abs().clamp(min=2.0 ** -9)))
    return torch.ldexp(torch.ones_like(values), (exponent - 3).int()).clamp(min=2.0 ** -9)


def _norm_inputs(rows: int, width: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260921)
    return (
        torch.randn(rows, width, device="cuda").half(),
        torch.zeros(width, device="cuda").half(),
        (torch.rand(width, device="cuda") + 0.5).half(),
        (torch.rand(width, device="cuda") - 0.5).half(),
    )


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
def test_the_int8_side_output_leaves_the_fp16_stream_alone(warps: int) -> None:
    """The residual stream must read what it read before the site was quantised.

    Measured on this build at 1024 wide: bit-identical at 4 and 8 warps, and at 1 and 2 warps 6e-5 of
    the outputs move by exactly ONE FP16 ulp -- the same arithmetic, scheduled differently once the row
    carries the extra work. The warp count is the autotuner's to pick and it already changes this output
    on its own (a norm at 1 warp and the same norm at 2 do not agree to the last bit either), so a
    quantised artifact is gated, not diffed against its FP16 twin -- as a different tile choice is.
    """
    rows, width = 512, _WIDTH
    values, bias, gammas, betas = _norm_inputs(rows, width)
    prescale = torch.exp(torch.linspace(-1.0, 3.0, width, device="cuda"))
    offset = torch.linspace(-0.2, 0.2, width, device="cuda")

    plain = torch.empty(rows, width, dtype=torch.float16, device="cuda")
    folded = torch.empty(rows, width, dtype=torch.float16, device="cuda")
    codes = torch.empty(rows, width, dtype=torch.int8, device="cuda")
    _layer_norm_kernel.fn[(rows,)](
        plain, values, bias, gammas, betas, rows, width, 1e-3, 0, False, width, num_warps=warps,
    )
    _layer_norm_quant_kernel.fn[(rows,)](
        folded, values, bias, gammas, betas, codes, prescale, offset,
        rows, width, 1e-3, 0, False, 1, width, num_warps=warps,
    )

    difference = (folded.float() - plain.float()).abs()
    magnitude = plain.float().abs().clamp(min=2.0 ** -14)
    ulp = torch.ldexp(torch.ones_like(magnitude), torch.floor(torch.log2(magnitude)).int() - 10)
    assert float((difference / ulp).max()) <= 1.0
    assert float((difference > 0).float().mean()) < 1e-4


def test_the_fold_quantises_more_accurately_than_a_second_pass() -> None:
    """The fold is not only free, it is the better conversion -- and this says by how much.

    A standalone pass has to read what the norm already wrote, which is FP16: 11 bits of mantissa, so a
    value near a half-integer code boundary can round to either side. The in-norm conversion reads the
    FP32 output the row still holds in registers and never sees FP16 at all. Both are compared here
    against the same arithmetic in FP64.
    """
    rows, width = 1024, _WIDTH
    values, bias, gammas, betas = _norm_inputs(rows, width)
    # r of order 127 / amax with a 30x channel spread, the shape a real calibration produces.
    prescale = 40.0 * torch.exp(torch.linspace(-1.7, 1.7, width, device="cuda"))
    offset = torch.zeros(width, device="cuda")

    folded = torch.empty(rows, width, dtype=torch.float16, device="cuda")
    codes = torch.empty(rows, width, dtype=torch.int8, device="cuda")
    _layer_norm_quant_kernel[_norm_grid](
        folded, values, bias, gammas, betas, codes, prescale, offset,
        rows, width, 1e-3, 0, False, 1, width,
    )
    two_pass = _quantise(folded, prescale, offset)

    wide = values.double()
    centred = wide - wide.mean(dim=1, keepdim=True)
    normalized = centred / torch.sqrt((centred * centred).mean(dim=1, keepdim=True) + 1e-3)
    exact = normalized * gammas.double() + betas.double()
    reference = torch.clamp(
        torch.floor((exact - offset.double()) * prescale.double() + 0.5), -127.0, 127.0,
    ).to(torch.int8)

    fold_wrong = (codes.to(torch.int32) - reference.to(torch.int32)).abs()
    pass_wrong = (two_pass.to(torch.int32) - reference.to(torch.int32)).abs()
    assert int(fold_wrong.max()) <= 1
    assert int(pass_wrong.max()) <= 1
    fold_rate = float((fold_wrong > 0).float().mean())
    pass_rate = float((pass_wrong > 0).float().mean())
    assert fold_rate < pass_rate, f"fold {fold_rate:.2e} vs second pass {pass_rate:.2e}"


def test_the_e4m3_operand_is_the_same_pass_with_another_cast() -> None:
    """Ask 2 of the 09-22 ruling: the format is a parameter, not another kernel.

    e4m3 spends its bits on the exponent, so the pre-scale buys RANGE there rather than precision and the
    analyser ships no migration vector for it. What must hold is that the same pass produces it and that the
    scale still acts per channel.
    """
    torch.manual_seed(20260922)
    channels = 256
    values = (torch.randn(129, channels, device="cuda") * 2.0).half()
    prescale = torch.exp(torch.linspace(-2.0, 2.0, channels, device="cuda"))

    codes = _quantise(values, prescale, operand="e4m3")
    reference = (values.float() * prescale).to(torch.float8_e4m3fn).float()

    assert codes.dtype is torch.float8_e4m3fn
    # ⚠ Triton's cast and PyTorch's do NOT agree to the bit. They differ on 4.3e-3 of elements, always by
    # exactly ONE e4m3 step and always where the FP32 product lands within an FP16 ulp of an e4m3 midpoint --
    # Triton's lowering rounds twice (FP32 -> FP16 -> e4m3), PyTorch's once. It is bounded and tiny for a
    # 3-mantissa-bit format, but it means an e4m3 accuracy number simulated in PyTorch is NOT the served one;
    # if e4m3 is ever the chosen operand, the cast wants explicit `cvt.rn.satfinite.e4m3x2.f32`.
    difference = (codes.float() - reference).abs()
    assert bool((difference <= _e4m3_step(reference)).all())
    assert float((difference > 0).float().mean()) < 1e-2  # noqa: PLR2004
    # Per channel, not per tensor: the flipped vector is a different tensor.
    assert not torch.equal(codes.float(), (values.float() * prescale.flip(0)).to(torch.float8_e4m3fn).float())


def test_the_norm_emits_e4m3_too() -> None:
    """The in-norm conversion carries the same parameter."""
    torch.manual_seed(20260922)
    rows, width = 256, _WIDTH
    values, bias, gammas, betas = _norm_inputs(rows, width)
    prescale = torch.exp(torch.linspace(-1.0, 1.0, width, device="cuda"))
    offset = torch.zeros(width, device="cuda")

    folded = torch.empty(rows, width, dtype=torch.float16, device="cuda")
    codes = torch.empty(rows, width, dtype=torch.float8_e4m3fn, device="cuda")
    _layer_norm_quant_kernel.fn[(rows,)](
        folded, values, bias, gammas, betas, codes, prescale, offset,
        rows, width, 1e-3, 0, False, 2, width, num_warps=4,
    )

    reference = (folded.float() * prescale).to(torch.float8_e4m3fn).float()
    assert bool(((codes.float() - reference).abs() <= _e4m3_step(reference)).all())
