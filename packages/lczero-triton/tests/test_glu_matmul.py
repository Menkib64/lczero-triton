"""CUDA numerical and artifact tests for the fused GLU projection.

⛔ The gate that matters for the lab's static 512x15 nets is `glu` -- a
**sigmoid** gate -- not `swiglu`. The proto enum is called `ACTIVATION_SWIGLU`
but selects the sigmoid branch; the export confirms it (node 2490 `Sigmoid`,
node 2496 `Mul` against a second `[512,683]` Gemm). `test_lab_config_mapping`
locks that down so the naming cannot silently pick the wrong kernel again.
"""

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._activation import (
    GLU_GATES,
    SOFTCAP_UNIT_SERIES_MIN,
    SUPPORTED_GLU_GATES,
    gate_for_lab_config,
)
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.glu_matmul import (
    BIAS_NONE,
    BIAS_VECTOR,
    GluMatmulSpecialization,
    _glu_matmul_kernel,
    compile_glu_matmul,
    glu_matmul,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_LAB_GLU_SHAPES = (
    (64, 683, 512),
    (512, 683, 512),
    (64, 1024, 512),
    (512, 1024, 512),
    (128, 341, 256),
)
_ONE_SHAPE = (64, 683, 512)
_PRODUCTION_SOFTCAP = 30.0
# `ffn_softcap: 12.0` of the BT6-test sponsor net (sigmoid gate, both branches capped).
_SPONSOR_SOFTCAP = 12.0
_FP16_ATOL = 2e-2
_FP16_RTOL = 1e-2


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _reference(
    activations: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    gate: str,
    softcap: float,
) -> torch.Tensor:
    """Compute the unfused two-GEMM form in FP32, as the lab's JAX does it."""
    left = activations.float()
    gate_half = left @ weights[:, :n].float()
    linear_half = left @ weights[:, n:].float()
    if bias is not None:
        gate_half = gate_half + bias[:n].float()
        linear_half = linear_half + bias[n:].float()
    if gate == "glu":
        return torch.sigmoid(gate_half) * linear_half
    if gate == "swiglu":
        return torch.nn.functional.silu(gate_half) * linear_half
    if gate == "reglu":
        return torch.relu(gate_half) * linear_half
    if gate == "situglu":
        # model/shared.py: activate the gate, THEN cap both branches.
        activated = torch.nn.functional.silu(gate_half)
        capped_gate = softcap * torch.tanh(activated / softcap)
        capped_linear = softcap * torch.tanh(linear_half / softcap)
        return capped_gate * capped_linear
    if gate == "glu_capped":
        # model/shared.py of the BT6-test tree: the SIGMOID gate, then the same cap on both branches.
        capped_gate = softcap * torch.tanh(torch.sigmoid(gate_half) / softcap)
        capped_linear = softcap * torch.tanh(linear_half / softcap)
        return capped_gate * capped_linear
    raise AssertionError(gate)


def _grid(m: int, n: int):
    """Return the launch grid closure for one problem size."""

    def resolve(configuration) -> tuple[int]:
        rows = (m + configuration["block_m"] - 1) // configuration["block_m"]
        columns = (n + configuration["block_n"] - 1) // configuration["block_n"]
        return (rows * columns,)

    return resolve


def _run(
    m: int,
    n: int,
    k: int,
    gate: str,
    *,
    has_bias: bool,
    softcap: float = 0.0,
    spread: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the kernel and return (produced, expected) in FP32."""
    torch.manual_seed(0xC0FFEE)
    activations = (torch.randn((m, k), device="cuda") * spread).half()
    weights = (torch.randn((k, 2 * n), device="cuda") * spread).half()
    bias = (torch.randn(2 * n, device="cuda") * spread).half() if has_bias else None
    output = torch.empty((m, n), dtype=torch.float16, device="cuda")
    _glu_matmul_kernel[_grid(m, n)](
        output,
        activations,
        weights,
        bias if bias is not None else activations,
        m,
        n,
        k,
        BIAS_VECTOR if has_bias else BIAS_NONE,
        GLU_GATES[gate],
        softcap,
    )
    expected = _reference(activations, weights, bias, n, gate, softcap)
    return output.float(), expected


@pytest.mark.parametrize(("m", "n", "k"), _LAB_GLU_SHAPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_sigmoid_gate_matches_the_static_export_form(
    m: int,
    n: int,
    k: int,
    *,
    has_bias: bool,
) -> None:
    """`glu` -- the gate the static 512x15 net actually uses -- across shapes."""
    produced, expected = _run(m, n, k, "glu", has_bias=has_bias)
    torch.testing.assert_close(produced, expected, atol=_FP16_ATOL, rtol=_FP16_RTOL)


@pytest.mark.parametrize(
    ("gate", "softcap"),
    [
        ("glu", 0.0),
        ("swiglu", 0.0),
        ("reglu", 0.0),
        ("situglu", _PRODUCTION_SOFTCAP),
        ("situglu", 1.0),
        ("glu_capped", _SPONSOR_SOFTCAP),
        ("glu_capped", 0.5),
    ],
    ids=["glu", "swiglu", "reglu", "situglu-30", "situglu-1", "glu_capped-12", "glu_capped-0.5"],
)
def test_every_gate_matches_its_reference(gate: str, softcap: float) -> None:
    """Each implemented gate reproduces the lab's formula for that gate."""
    m, n, k = _ONE_SHAPE
    # A wide spread so SiTU-GLU's cap is exercised rather than sitting in
    # tanh's linear region, where it would pass trivially.
    produced, expected = _run(m, n, k, gate, has_bias=True, softcap=softcap, spread=0.5)
    torch.testing.assert_close(produced, expected, atol=_FP16_ATOL, rtol=_FP16_RTOL)


def test_the_unit_interval_series_is_fp32_exact_where_it_is_used() -> None:
    """`_softcap_unit`: c*tanh(g/c) = g*(1 - u/3 + 2u^2/15) on a sigmoid's range, to FP32 rounding for c >= 8."""
    gate = torch.linspace(0.0, 1.0, 100001, dtype=torch.float64)
    for cap, bound in ((12.0, 2.0e-8), (SOFTCAP_UNIT_SERIES_MIN, 2.2e-7)):
        square = (gate / cap) ** 2
        series = gate * (1.0 - square * (1.0 / 3.0 - square * (2.0 / 15.0)))
        exact = cap * torch.tanh(gate / cap)
        relative = ((series - exact).abs() / exact.clamp_min(1e-30)).max().item()
        assert relative <= bound, f"cap {cap}: the series is off by {relative:.3e} of the value"


def test_sigmoid_cap_actually_bites() -> None:
    """`glu_capped` with a tight cap differs from `glu` and is bounded by c^2; at c = 12 the gate cap is small but real."""
    m, n, k = _ONE_SHAPE
    capped, _ = _run(m, n, k, "glu_capped", has_bias=True, softcap=0.25, spread=0.5)
    uncapped, _ = _run(m, n, k, "glu", has_bias=True, spread=0.5)
    assert (capped - uncapped).abs().max().item() > 0.1
    assert capped.abs().max().item() <= 0.25 * 0.25 + 1e-3


def test_softcap_actually_bites() -> None:
    """A tight cap must change the result, or the cap is a silent no-op.

    Guards the failure mode where `softcap` is threaded but never applied: the
    kernel would still match a SwiGLU reference and every other test would pass.
    """
    m, n, k = _ONE_SHAPE
    capped, _ = _run(m, n, k, "situglu", has_bias=True, softcap=0.25, spread=0.5)
    uncapped, _ = _run(m, n, k, "swiglu", has_bias=True, spread=0.5)
    assert (capped - uncapped).abs().max().item() > 0.1
    # The cap bounds the output: |c*tanh(a)| * |c*tanh(b)| <= c^2.
    assert capped.abs().max().item() <= 0.25 * 0.25 + 1e-3


def test_lab_config_mapping() -> None:
    """The lab's FFN config fields resolve to the right gate, including the trap."""
    assert gate_for_lab_config("ACTIVATION_SWIGLU") == "glu"
    assert gate_for_lab_config("ACTIVATION_SWIGLU", ffn_softcap=12.0) == "glu_capped"
    assert gate_for_lab_config("ACTIVATION_SWISH", ffn_glu=True) == "swiglu"
    assert (
        gate_for_lab_config("ACTIVATION_SWISH", ffn_glu=True, ffn_softcap=30.0)
        == "situglu"
    )
    with pytest.raises(ValueError, match="does not describe a gated FFN"):
        gate_for_lab_config("ACTIVATION_MISH")
    with pytest.raises(ValueError, match="not a gate this package"):
        gate_for_lab_config("ACTIVATION_MISH", ffn_glu=True)


def test_softcap_arguments_are_validated() -> None:
    """A softcap that would be a no-op, or one supplied to a gate that ignores it."""
    common = {"m": 64, "n": 683, "k": 512, "architecture": _architecture()}
    with pytest.raises(ValueError, match="requires softcap > 0"):
        compile_glu_matmul(GluMatmulSpecialization(**common, gate="situglu"))
    with pytest.raises(ValueError, match="ignores softcap"):
        compile_glu_matmul(GluMatmulSpecialization(**common, gate="glu", softcap=30.0))


def test_all_declared_gates_are_implemented() -> None:
    """Nothing sits in `GLU_GATES` without a device branch behind it."""
    assert set(GLU_GATES.values()) == set(SUPPORTED_GLU_GATES)


def test_glu_matmul_compiles_to_an_lc0ex_artifact() -> None:
    """The kernel serializes into an lc0ex program with the right parameters."""
    specialization = GluMatmulSpecialization(
        m=64,
        n=683,
        k=512,
        architecture=_architecture(),
        gate="glu",
        has_bias=True,
    )
    artifact = compile_glu_matmul(specialization)
    assert artifact.grid[0] > 0
    assert artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 4 + (
        lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,
    ) * 2

    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)

    def buffer(name: str, shape: tuple[int, ...]) -> object:
        return builder.persistent_buffer(
            name=name,
            shape=shape,
            dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
            alignment_bytes=256,
        )

    glu_matmul(
        builder,
        kernels,
        buffer("glu/out", (64, 683)),
        buffer("glu/in", (64, 512)),
        buffer("glu/w", (512, 1366)),
        specialization,
        bias=buffer("glu/b", (1366,)),
    )
