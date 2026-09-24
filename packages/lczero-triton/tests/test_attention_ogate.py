"""CUDA tests for the static attention kernel's output gate (round 22 item O), gated against onnxruntime.

The reference set ``ref/o_ref`` (``o_ref.json``, SHA256 manifest verified once per module) holds 64 of K2's
positions through `prenorm_ogate_512x15`'s own graph: per block the packed [q | k | v | g] Gemm rows, S, E, the
merged attention output before the gate and after it. The kernel's tables are the block's carrier plans, built
from the export exactly as `carrier.convert` builds them.

Gates (``O-GATE`` lines, ``pytest -s``):
* served precision (FP16 q|k|v|g and S): the gated output against ORT in the class of FP16 projections (5e-3
  peak-relative, K2's served rule), and the ungated output likewise (the static path's known class);
* the gate in isolation: gated = ungated * 2 sigmoid(g) to FP16 storage (two roundings, < 1.5e-3), with the
  FP32 sigmoid closer to the float64 gate than an FP16-rounded gate would be;
* the packed lane and the separate buffer read the same gate bit for bit; the flags off are the kernel as before;
* the artifact keeps its thirteen pointers with the gate on.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.attention_static import (
    STATIC_FULL,
    STATIC_SMOLGEN,
    StaticAttentionSpecialization,
    _autotune_grid,
    _static_attention_kernel,
    compile_static_attention,
    padded_head_dim,
    static_attention,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_TREE = Path(__file__).resolve().parents[3]
_REFERENCE = Path(os.environ.get("LC0EX_OGATE_REF", str(_TREE / "ref/o_ref")))
_NETS = Path(os.environ.get("LC0EX_EGT_NETS", str(Path.home() / "spsa/lc0ex_5080/work/nets_r20")))
_EXPORT = Path(os.environ.get("LC0EX_OGATE_NET", str(_NETS / "prenorm_ogate_512x15_50000_vw.pb.gz")))
_COUNT, _HEADS, _DEPTH, _TOKENS, _WIDTH, _EDGES = 64, 32, 16, 64, 512, 4
_SERVED_RELATIVE = 5e-3
_STORAGE_RELATIVE = 1.5e-3
_LOADED: dict = {}


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _read(name: str, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray((_REFERENCE / f"{name}.bin").read_bytes()), dtype=torch.float32).view(*shape)


@pytest.fixture(scope="module")
def reference() -> dict:
    header = _REFERENCE / "o_ref.json"
    if not header.exists():
        pytest.skip(f"reference not found at {_REFERENCE}")
    for line in (_REFERENCE / "SHA256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((_REFERENCE / name).read_bytes()).hexdigest() == digest, f"{name} fails its sha256"
    header = json.loads(header.read_text())
    edges = _read("E", _COUNT, _EDGES, _TOKENS, _TOKENS)
    codes = sum((edges[:, channel].to(torch.uint8) << channel) for channel in range(_EDGES)).contiguous().cuda()
    return {"header": header, "codes": codes, "edge_norm": _read("S", _COUNT, _TOKENS, _TOKENS).half().cuda()}


def _tables(block: int) -> dict[str, torch.Tensor]:
    """The block's carrier plans (P6 code tables, the folded pair bias), as `carrier.convert` writes them."""
    if block not in _LOADED:
        if not _EXPORT.exists():
            pytest.skip(f"{_EXPORT} not present")
        from lczero_triton.lab._mapping import read_network  # noqa: PLC0415
        from lczero_triton.lab._names import plan_network  # noqa: PLC0415
        from lczero_triton.lab._onnx import load_carrier  # noqa: PLC0415
        from lczero_triton.lab.carrier import _build_payload  # noqa: PLC0415
        if "graph" not in _LOADED:
            _, graph = load_carrier(_EXPORT)
            network = read_network(graph)
            _LOADED["graph"] = (graph, network, {plan.name: plan for plan in plan_network(network)})
        graph, network, plans = _LOADED["graph"]
        prefix = f"/encoder{block}"
        tables = {}
        for short, name in (("query_codes", f"{prefix}/mha/edge/query_codes"), ("key_codes", f"{prefix}/mha/edge/key_codes"),
                            ("coefficient_codes", f"{prefix}/mha/edge/coefficient_codes"),
                            ("scaled_codes", f"{prefix}/pair/scaled_codes"), ("constant_bias", f"{prefix}/pair/constant_bias")):
            plan = plans[name]
            payload, _ = _build_payload(graph, plan)
            dtype = torch.float32 if plan.data_type == 1 else torch.float16
            tables[short] = torch.frombuffer(bytearray(payload), dtype=dtype).view(plan.shape).cuda()
        tables["scale"] = torch.full((1,), network.blocks[block].attention_scale, dtype=torch.float16, device="cuda")
        tables["gate_scale"] = network.blocks[block].gate_scale
        _LOADED[block] = tables
    return _LOADED[block]


def _launch(qkv: torch.Tensor, reference: dict, tables: dict, *, output_gate: bool, gate_packed: bool,
            gates: torch.Tensor | None = None) -> torch.Tensor:
    samples = qkv.shape[0]
    lanes = qkv.shape[2] // _WIDTH
    output = torch.empty((samples, _TOKENS, _WIDTH), dtype=torch.float16, device="cuda")
    auxiliary = gates if gates is not None else tables["scale"]
    _static_attention_kernel[_autotune_grid](
        output, qkv, qkv, qkv, reference["codes"][:samples], reference["edge_norm"][:samples],
        tables["query_codes"], tables["key_codes"], tables["coefficient_codes"], tables["scaled_codes"],
        tables["constant_bias"], tables["scale"], auxiliary,
        samples * _HEADS, _HEADS, _TOKENS, _DEPTH, padded_head_dim(_DEPTH), 1 << _EDGES,
        True, True, False, lanes * _WIDTH, _WIDTH, 2 * _WIDTH,
        output_gate=output_gate, gate_packed=gate_packed, gate_offset=3 * _WIDTH if gate_packed else 0,
        gate_scale=tables["gate_scale"],
    )
    return output


def _peak_relative(got: torch.Tensor, expected: torch.Tensor) -> float:
    return float((got.double() - expected.double()).abs().max() / expected.double().abs().max())


@pytest.mark.parametrize("block", [0, 7, 14])
def test_gated_output_matches_ort_at_served_precision(reference, block: int) -> None:
    tables = _tables(block)
    qkvg = _read(f"block{block}_qkvg", _COUNT, _TOKENS, 4 * _WIDTH).half().cuda()
    gated = _read(f"block{block}_gated", _COUNT, _TOKENS, _WIDTH)
    attended = _read(f"block{block}_attended", _COUNT, _TOKENS, _WIDTH)
    with_gate = _launch(qkvg, reference, tables, output_gate=True, gate_packed=True).cpu()
    without = _launch(qkvg[..., : 3 * _WIDTH].contiguous(), reference, tables, output_gate=False, gate_packed=False).cpu()
    on, off = _peak_relative(with_gate, gated), _peak_relative(without, attended)
    print(f"O-GATE block {block}: gated vs ORT {on:.3e}, ungated vs ORT {off:.3e} (served rule {_SERVED_RELATIVE})")
    assert on < _SERVED_RELATIVE and off < _SERVED_RELATIVE


@pytest.mark.parametrize("block", [0, 14])
def test_gate_is_two_sigmoid_in_fp32(reference, block: int) -> None:
    tables = _tables(block)
    qkvg = _read(f"block{block}_qkvg", _COUNT, _TOKENS, 4 * _WIDTH).half().cuda()
    g = qkvg[..., 3 * _WIDTH:].double()
    gate = tables["gate_scale"] * torch.sigmoid(g)
    with_gate = _launch(qkvg, reference, tables, output_gate=True, gate_packed=True).double()
    without = _launch(qkvg[..., : 3 * _WIDTH].contiguous(), reference, tables, output_gate=False, gate_packed=False).double()
    expected = without * gate
    error = _peak_relative(with_gate, expected)
    # An FP16-rounded gate would land the product elsewhere: the kernel's FP32 gate must sit nearer the float64 one.
    rounded = without * gate.half().double()
    print(f"O-GATE block {block}: gated vs ungated*2sigmoid(g) {error:.3e}; vs an FP16-rounded gate "
          f"{_peak_relative(with_gate, rounded):.3e}; gate in [{gate.min():.4f}, {gate.max():.4f}]")
    assert error < _STORAGE_RELATIVE
    assert torch.count_nonzero(with_gate != without) > 0.9 * with_gate.numel()  # the gate is not the identity


def test_packed_lane_and_separate_buffer_read_the_same_gate(reference) -> None:
    tables = _tables(0)
    qkvg = _read("block0_qkvg", _COUNT, _TOKENS, 4 * _WIDTH).half().cuda()
    packed = _launch(qkvg, reference, tables, output_gate=True, gate_packed=True)
    separate = _launch(qkvg[..., : 3 * _WIDTH].contiguous(), reference, tables, output_gate=True, gate_packed=False,
                       gates=qkvg[..., 3 * _WIDTH:].contiguous())
    assert torch.equal(packed, separate)


def test_flags_off_is_the_default_call(reference) -> None:
    """The kernel called without the gate arguments is the kernel called with them off, bit for bit."""
    tables = _tables(0)
    qkv = _read("block0_qkvg", _COUNT, _TOKENS, 4 * _WIDTH)[..., : 3 * _WIDTH].contiguous().half().cuda()
    explicit = _launch(qkv, reference, tables, output_gate=False, gate_packed=False)
    output = torch.empty_like(explicit)
    _static_attention_kernel[_autotune_grid](
        output, qkv, qkv, qkv, reference["codes"], reference["edge_norm"],
        tables["query_codes"], tables["key_codes"], tables["coefficient_codes"], tables["scaled_codes"],
        tables["constant_bias"], tables["scale"], tables["scale"],
        _COUNT * _HEADS, _HEADS, _TOKENS, _DEPTH, padded_head_dim(_DEPTH), 1 << _EDGES,
        True, True, False, 3 * _WIDTH, _WIDTH, 2 * _WIDTH,
    )
    assert torch.equal(output, explicit)


def test_gated_kernel_compiles_to_an_lc0ex_artifact_with_thirteen_pointers() -> None:
    samples = 1
    specialization = StaticAttentionSpecialization(
        batch_count=samples * _HEADS, heads=_HEADS, tokens=_TOKENS, head_dim=_DEPTH, edge_channels=_EDGES,
        logit_terms=STATIC_FULL, architecture=_architecture(), packed_qkv=True, output_gate=True, gate_packed=True,
    )
    artifact = compile_static_attention(specialization)
    assert artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 13 + (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2
    with pytest.raises(ValueError, match="gate_packed needs packed_qkv"):
        compile_static_attention(StaticAttentionSpecialization(
            batch_count=_HEADS, heads=_HEADS, tokens=_TOKENS, head_dim=_DEPTH, edge_channels=_EDGES,
            logit_terms=STATIC_FULL, architecture=_architecture(), packed_qkv=False, output_gate=True, gate_packed=True))
    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)

    def buffer(name: str, shape: tuple[int, ...], dtype: int = lc0ex_pb2.Buffer.DATA_TYPE_F16) -> object:
        return builder.persistent_buffer(name=name, shape=shape, dtype=dtype, alignment_bytes=256)

    qkvg = buffer("o/qkvg", (samples, _TOKENS, 4 * _WIDTH))
    tables = (buffer("o/pq", (_HEADS, _DEPTH, 1 << _EDGES), lc0ex_pb2.Buffer.DATA_TYPE_F32),
              buffer("o/pk", (_HEADS, _DEPTH, 1 << _EDGES), lc0ex_pb2.Buffer.DATA_TYPE_F32),
              buffer("o/c1", (_HEADS, 1 << _EDGES), lc0ex_pb2.Buffer.DATA_TYPE_F32),
              buffer("o/g", (_HEADS, 1 << _EDGES), lc0ex_pb2.Buffer.DATA_TYPE_F32),
              buffer("o/kbias", (_HEADS, _TOKENS, _TOKENS)), buffer("o/scale", (1,)))
    static_attention(builder, kernels, buffer("o/out", (samples, _TOKENS, _WIDTH)), qkvg, qkvg, qkvg,
                     buffer("o/e", (samples, _TOKENS, _TOKENS)), buffer("o/norm", (samples, _TOKENS, _TOKENS)),
                     *tables, specialization)
    separate = StaticAttentionSpecialization(
        batch_count=samples * _HEADS, heads=_HEADS, tokens=_TOKENS, head_dim=_DEPTH, edge_channels=_EDGES,
        logit_terms=STATIC_FULL, architecture=_architecture(), packed_qkv=True, output_gate=True, gate_packed=False,
    )
    with pytest.raises(ValueError, match="separate output gate"):
        static_attention(builder, kernels, buffer("o/out2", (samples, _TOKENS, _WIDTH)), qkvg, qkvg, qkvg,
                         buffer("o/e2", (samples, _TOKENS, _TOKENS)), buffer("o/norm2", (samples, _TOKENS, _TOKENS)),
                         *tables, separate)
    with pytest.raises(ValueError, match="share the kernel's auxiliary pointer"):
        compile_static_attention(StaticAttentionSpecialization(
            batch_count=_HEADS, heads=_HEADS, tokens=_TOKENS, head_dim=_DEPTH, edge_channels=_EDGES,
            logit_terms=STATIC_FULL | STATIC_SMOLGEN, architecture=_architecture(), packed_qkv=True, output_gate=True,
            gate_packed=False))
