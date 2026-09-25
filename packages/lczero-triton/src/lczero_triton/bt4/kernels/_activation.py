"""LC0's activation approximations, as one shared device-side definition.

Every kernel family that applies an activation calls `apply_activation` so that
there is exactly one formula in the package. LC0's Mish is a branch-selected
approximation, not `x * tanh(softplus(x))`; reproducing it exactly is what keeps
a fused epilogue in the same fidelity class as the standalone kernel it replaces.
"""

from typing import Literal

import triton
import triton.language as tl

Activation = Literal["none", "mish", "swish", "relu"]

ACTIVATION_NONE = 0
ACTIVATION_MISH = 1
ACTIVATION_SWISH = 2
ACTIVATION_RELU = 3

ACTIVATIONS: dict[Activation, int] = {
    "none": ACTIVATION_NONE,
    "mish": ACTIVATION_MISH,
    "swish": ACTIVATION_SWISH,
    "relu": ACTIVATION_RELU,
}

_MISH = tl.constexpr(ACTIVATION_MISH)
_SWISH = tl.constexpr(ACTIVATION_SWISH)
_RELU = tl.constexpr(ACTIVATION_RELU)
# LC0 switches Mish branches here to keep the exponential in range.
_MISH_BRANCH = tl.constexpr(-0.6)


@triton.jit
def apply_activation(values, activation: tl.constexpr) -> tl.tensor:
    """Apply one activation to FP32 values, matching LC0's formulas exactly."""
    if activation == _MISH:
        exponential = tl.exp(values)
        numerator = exponential * exponential + 2.0 * exponential
        division = values / (numerator + 2.0)
        values = tl.where(
            values <= _MISH_BRANCH,
            numerator * division,
            values - 2.0 * division,
        )
    elif activation == _SWISH:
        values = values / (1.0 + tl.exp(-values))
    elif activation == _RELU:
        values = tl.maximum(values, 0.0)
    return values


# --- Gated linear unit family ----------------------------------------------
#
# A GLU splits one wide projection into a gate half and a linear half and
# multiplies them. The halves come from a single concatenated weight matrix
# (`[k, 2n]`, gate columns first), which is how the lab's JAX nets export and
# is also the faster form -- one wide GEMM instead of two, activation tile read
# once.
#
# ⛔ THE NAMING TRAP. The lab's proto enum `ACTIVATION_SWIGLU` does **not**
# select SwiGLU as the literature defines it. In the training tree's
# `model/shared.py` that enum takes the legacy branch
#
#     gate = nnx.sigmoid(linear_gate(x)); out = gate * linear1(x)  # noqa: ERA001
#
# i.e. a **sigmoid** gate -- Dauphin's original GLU. Literature SwiGLU (SiLU
# gate) is reached by a different route: `ffn_glu: true` with
# `ffn_activation: SWISH`. The static 512x15 export confirms the enum's
# behaviour: node 2490 is `Sigmoid`, and node 2496 multiplies it by a *second*
# `[512,683]` Gemm's output. So the gates below are named for their **formula**,
# never for the enum, and `gate_for_lab_config` is the only place allowed to map
# a lab config onto one.
#
# `situglu` is the lab's "SiTU-GLU": a softcapped SwiGLU, from
# `training-features_2026-08-08/01_ffn_glu` --
#
#     h = linear1(x); g = swish(linear_gate(x))  # noqa: ERA001
#     if softcap > 0: g = c*tanh(g/c); h = c*tanh(h/c)  # noqa: ERA001
#     out = h * g  # noqa: ERA001
#
# Note the order: the activation is applied to the gate **before** the cap, and
# the cap is applied to **both** branches. `round3_situglu_cv2` ran c = 30.0.
# Bounding both branches to +-c is what makes it a quantization play: the FFN's
# activation range becomes a known constant instead of an outlier-driven
# measurement.
#
# `glu_capped` is the form the BT6-test sponsor net trains (`DefaultsConfig.ffn_softcap`
# with `ffn_activation: ACTIVATION_SWIGLU`, `model/shared.py` of the 09-21 tree): the
# SAME cap on both branches, but around the **sigmoid** gate of `glu` --
#
#     g = sigmoid(linear_gate(x)); h = linear1(x)  # noqa: ERA001
#     g = c*tanh(g/c); h = c*tanh(h/c)  # noqa: ERA001
#     out = g * h  # noqa: ERA001
#
# With c = 12 the gate's cap is nearly the identity (12*tanh(1/12) = 0.99769 at g = 1),
# but it is in the trained graph, so it is served rather than rounded away.

GluGate = Literal["glu", "swiglu", "reglu", "situglu", "glu_capped"]

GLU_SIGMOID = 0
GLU_SWIGLU = 1
GLU_REGLU = 2
GLU_SITUGLU = 3
GLU_SIGMOID_CAPPED = 4

GLU_GATES: dict[GluGate, int] = {
    "glu": GLU_SIGMOID,
    "swiglu": GLU_SWIGLU,
    "reglu": GLU_REGLU,
    "situglu": GLU_SITUGLU,
    "glu_capped": GLU_SIGMOID_CAPPED,
}

# Every gate here has a device-side branch below and a numerical test.
SUPPORTED_GLU_GATES: tuple[int, ...] = (
    GLU_SIGMOID,
    GLU_SWIGLU,
    GLU_REGLU,
    GLU_SITUGLU,
    GLU_SIGMOID_CAPPED,
)

# Gates that read the `softcap` argument. Others ignore it.
SOFTCAPPED_GLU_GATES: tuple[int, ...] = (GLU_SITUGLU, GLU_SIGMOID_CAPPED)

_GLU_SIGMOID = tl.constexpr(GLU_SIGMOID)
_GLU_SWIGLU = tl.constexpr(GLU_SWIGLU)
_GLU_REGLU = tl.constexpr(GLU_REGLU)
_GLU_SITUGLU = tl.constexpr(GLU_SITUGLU)
_GLU_SIGMOID_CAPPED = tl.constexpr(GLU_SIGMOID_CAPPED)


def gate_for_lab_config(
    ffn_activation: str,
    *,
    ffn_glu: bool = False,
    ffn_softcap: float = 0.0,
) -> GluGate:
    """Map a lab training config's FFN fields onto this package's gate name.

    The only sanctioned place to resolve `ACTIVATION_SWIGLU`, which selects the
    sigmoid gate and not SiLU. See the naming-trap note above.
    """
    if ffn_glu:
        if ffn_activation != "ACTIVATION_SWISH":
            message = (
                f"ffn_glu with {ffn_activation!r} is not a gate this package "
                "implements; only ACTIVATION_SWISH is."
            )
            raise ValueError(message)
        return "situglu" if ffn_softcap > 0.0 else "swiglu"
    if ffn_activation == "ACTIVATION_SWIGLU":
        return "glu_capped" if ffn_softcap > 0.0 else "glu"
    message = (
        f"{ffn_activation!r} without ffn_glu does not describe a gated FFN."
    )
    raise ValueError(message)


@triton.jit
def _softcap(values, cap: tl.constexpr) -> tl.tensor:
    """Return `cap * tanh(values / cap)`, evaluated without overflow.

    This Triton has no `tl.math.tanh`, and `from ... import libdevice` inside a
    jitted function is rejected by the AST walker. The two-sided exponential
    form below compiles, agrees with `torch.tanh` to 1.19e-07 over +-5, and
    costs **9 float ops per element** against the 14 that libdevice's `tanh`
    expands to (R86) -- measured, not assumed.
    """
    magnitude = tl.abs(values) / cap
    decay = tl.exp(-2.0 * magnitude)
    tangent = (1.0 - decay) / (1.0 + decay)
    return cap * tl.where(values < 0.0, -tangent, tangent)


# Below this cap the unit-interval series is no longer FP32-exact and `_softcap_unit` takes the general form.
SOFTCAP_UNIT_SERIES_MIN = 8.0
_SOFTCAP_UNIT_SERIES_MIN = tl.constexpr(SOFTCAP_UNIT_SERIES_MIN)


@triton.jit
def _softcap_unit(values, cap: tl.constexpr) -> tl.tensor:
    """Return `cap * tanh(values / cap)` for `values` in [0, 1] -- a sigmoid gate -- without the exponential.

    With u = (values / cap)^2 the series is `values * (1 - u/3 + 2u^2/15 - 17u^3/315 + ...)`. At cap >= 8, u <= 1/64
    and the first dropped term is <= 2.1e-7 of the value (1.8e-8 at the sponsor net's cap 12): FP32 rounding, four
    orders below the FP16 the GLU writes. It trades the two special-function ops of the general form (one exp, one
    reciprocal per element) for multiply-adds. A tighter cap takes `_softcap`.
    """
    if cap >= _SOFTCAP_UNIT_SERIES_MIN:
        ratio = values / cap
        square = ratio * ratio
        result = values * (1.0 - square * (1.0 / 3.0 - square * (2.0 / 15.0)))
    else:
        result = _softcap(values, cap)
    return result


@triton.jit
def apply_glu(
    gate_values,
    linear_values,
    gate: tl.constexpr,
    softcap: tl.constexpr,
) -> tl.tensor:
    """Combine a gate half and a linear half into one GLU output, in FP32.

    SwiGLU's gate is SiLU, the same expression as LC0's Swish, so it is taken
    from `apply_activation` rather than written a second time.
    """
    if gate == _GLU_SIGMOID:
        result = (1.0 / (1.0 + tl.exp(-gate_values))) * linear_values
    elif gate == _GLU_SWIGLU:
        result = apply_activation(gate_values, _SWISH) * linear_values
    elif gate == _GLU_REGLU:
        result = apply_activation(gate_values, _RELU) * linear_values
    elif gate == _GLU_SITUGLU:
        activated = apply_activation(gate_values, _SWISH)
        result = _softcap(activated, softcap) * _softcap(linear_values, softcap)
    elif gate == _GLU_SIGMOID_CAPPED:
        activated = 1.0 / (1.0 + tl.exp(-gate_values))
        result = _softcap_unit(activated, softcap) * _softcap(linear_values, softcap)
    else:
        result = gate_values * linear_values
    return result
