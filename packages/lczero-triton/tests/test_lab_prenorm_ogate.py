"""The lab reader and plans on the pre-norm + output-gate export (round 22 item O; CPU).

`prenorm_ogate_512x15` is the static family plus `prenorm: true` and `mha_output_gate: true`. The reader must
recognise both by structure -- the block boundary is the FFN residual Add, not a norm; the gate is the square
projection behind a Sigmoid -- and the static family must read exactly as before. Node ids and the numerical
wiring audit come from the ORT reference set `ref/o_ref` (`make_ref_o.py`); the float64 checks here recompute the
gate, LN1 and the residual chain from ORT's tensors and the export's initializers.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from lczero_triton.lab import _onnx
from lczero_triton.lab._mapping import ArchitectureError, read_network
from lczero_triton.lab._names import output_gate_form, plan_network
from lczero_triton.lab._onnx import FLOAT16, FLOAT32
from lczero_triton.lab.carrier import _as_float32, _build_payload

_TREE = Path(__file__).resolve().parents[3]
_NETS = Path(os.environ.get("LC0EX_EGT_NETS", str(Path.home() / "spsa/lc0ex_5080/work/nets_r20")))
_EXPORT = Path(os.environ.get("LC0EX_OGATE_NET", str(_NETS / "prenorm_ogate_512x15_50000_vw.pb.gz")))
_STATIC = Path(os.environ.get(
    "LC0EX_STATIC_NET", str(Path.home() / "spsa/lc0ex_5080/work/static_net/static_bs4g_512x15_50000_vw.pb.gz")))
_REFERENCE = Path(os.environ.get("LC0EX_OGATE_REF", str(_TREE / "ref/o_ref")))
_COUNT = 64
_LOADED: dict = {}


def _loaded(path: Path) -> tuple:
    if path not in _LOADED:
        if not path.exists():
            pytest.skip(f"{path} not present")
        _, graph = _onnx.load_carrier(path)
        network = read_network(graph)
        _LOADED[path] = (graph, network, {plan.name: plan for plan in plan_network(network)})
    return _LOADED[path]


@pytest.fixture(scope="module")
def reference() -> dict:
    header = _REFERENCE / "o_ref.json"
    if not header.exists():
        pytest.skip(f"reference not found at {_REFERENCE}")
    for line in (_REFERENCE / "SHA256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((_REFERENCE / name).read_bytes()).hexdigest() == digest, f"{name} fails its sha256"
    return json.loads(header.read_text())


def _read(name: str, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray((_REFERENCE / f"{name}.bin").read_bytes()), dtype=torch.float32).view(*shape).double()


def _raw(graph, name: str) -> torch.Tensor:
    return _as_float32(graph, name).double().reshape(graph.initializers[name].dims)


def _rel(got: torch.Tensor, expected: torch.Tensor) -> float:
    return float((got - expected).abs().max() / max(expected.abs().max().item(), 1e-30))


def test_prenorm_ogate_structure() -> None:
    graph, network, plans = _loaded(_EXPORT)
    shape = network.architecture
    assert (shape.blocks, shape.d_model, shape.heads, shape.head_dim, shape.ffn_hidden) == (15, 512, 32, 16, 683)
    assert (shape.edge_channels, shape.mix_channels, shape.smolgen_gen) == (4, 16, 0)
    assert shape.block_style == "prenorm" and shape.output_gate
    assert network.egt is None and network.pair is not None
    assert network.final_norm is not None and network.final_norm.node == 4301  # noqa: PLR2004
    assert (network.final_norm.scale, network.final_norm.bias) == ("const_397", "const_398")
    first = network.blocks[0]
    assert (first.gate_weight, first.gate_bias, first.gate_scale, first.gate_node) == ("const_36", "const_37", 2.0, 2492)
    assert (first.query_weight, first.key_weight, first.value_weight, first.output_weight) == (
        "const_26", "const_28", "const_30", "const_38")
    assert (first.ln1_scale, first.ln2_scale) == ("const_24", "const_40")
    for index, block in enumerate(network.blocks):
        assert block.gate_weight is not None and block.gate_scale == 2.0, index  # noqa: PLR2004
        assert graph.initializers[block.gate_weight].dims == (512, 512), index
        # alpha = 1 on both residuals: the export still multiplies by the scalar, and the plan copies it.
        for alpha in (block.attention_alpha, block.ffn_alpha):
            assert _as_float32(graph, alpha).item() == 1.0, (index, alpha)
    assert abs(network.policy_divisor - 512**0.5) < 1e-4  # noqa: PLR2004
    # The embedding is the static family's: post-norm around its FFN, alpha 0.427.
    assert abs(_as_float32(graph, network.embedding.ffn_alpha).item() - 0.4272870123386383) < 1e-9  # noqa: PLR2004


def test_prenorm_ogate_plans() -> None:
    _, network, plans = _loaded(_EXPORT)
    form = output_gate_form()
    if form == "packed":
        assert plans["/encoder0/mha/qkvg/w"].shape == (512, 2048) and plans["/encoder0/mha/qkvg/b"].shape == (2048,)
        assert plans["/encoder0/mha/qkvg/w"].tensors == ("const_26", "const_28", "const_30", "const_36")
        assert "/encoder0/mha/qkv/w" not in plans and "/encoder0/mha/gate/w" not in plans
    else:
        assert plans["/encoder0/mha/qkv/w"].shape == (512, 1536) and plans["/encoder0/mha/gate/w"].shape == (512, 512)
        assert plans["/encoder0/mha/gate/b"].shape == (512,) and "/encoder0/mha/qkvg/w" not in plans
    assert plans["/encoder/final_norm/scale"].shape == (512,) and plans["/encoder/final_norm/bias"].shape == (512,)
    names = list(plans)
    assert names.index("/encoder/final_norm/scale") > names.index("/encoder14/ln2/bias")
    assert names.index("/encoder/final_norm/bias") < names.index("/value/embedding/w")
    assert sum(name.endswith("/mha/alpha/w") for name in names) == 15  # noqa: PLR2004


def test_every_plan_builds_a_payload_of_its_declared_size() -> None:
    graph, _, plans = _loaded(_EXPORT)
    worst = 0.0
    for plan in plans.values():
        payload, error = _build_payload(graph, plan)
        width = 2 if plan.data_type == FLOAT16 else 4
        assert len(payload) == plan.element_count * width, plan.name
        worst = max(worst, error)
    assert worst < 2e-3, worst  # noqa: PLR2004


def test_static_family_still_reads_post_norm_without_a_gate() -> None:
    _, network, plans = _loaded(_STATIC)
    shape = network.architecture
    assert shape.block_style == "postnorm" and not shape.output_gate and network.final_norm is None
    assert all(block.gate_weight is None and block.gate_scale == 0.0 for block in network.blocks)
    assert "/encoder0/mha/qkv/w" in plans and plans["/encoder0/mha/qkv/w"].shape == (512, 1536)
    assert not [name for name in plans if "qkvg" in name or "/mha/gate/" in name or "final_norm" in name]


def test_a_graph_with_neither_style_is_refused() -> None:
    """Cut the pre-norm block's LN2 out of block 0 and the reader must name what it saw."""
    graph, network, _ = _loaded(_EXPORT)
    ln2 = next(node for node in graph.nodes if node.index == 2501)  # noqa: PLR2004
    original = ln2.op_type
    ln2.op_type = "Identity"
    try:
        with pytest.raises(ArchitectureError, match="consumed by .*Add.*Identity.*reads the residual's skip"):
            read_network(graph)
    finally:
        ln2.op_type = original


def test_reference_node_ids_match_the_reader(reference) -> None:
    graph, network, _ = _loaded(_EXPORT)
    assert reference["export_sha256"][:8] == hashlib.sha256(_EXPORT.read_bytes()).hexdigest()[:8]
    for b, meta in reference["blocks"].items():
        block = network.blocks[int(b)]
        assert meta["style"] == "prenorm" and meta["gate_product"] == block.gate_node
        assert meta["weights"] == {"query": block.query_weight, "key": block.key_weight, "value": block.value_weight,
                                   "gate": block.gate_weight, "output": block.output_weight}
    assert reference["final_norm"]["node"] == network.final_norm.node
    assert reference["worst_relative_error"] < 1e-5  # noqa: PLR2004
    assert reference["checks"]["block7_input_is_block6_output"] and reference["checks"]["block0_input_is_embedding_norm"]


@pytest.mark.parametrize("block_index", [0, 14])
def test_gate_and_residual_wiring_in_float64(reference, block_index: int) -> None:
    """The gate is `attended * 2 sigmoid(g)`; LN1 reads the block input; the out-projection adds into it un-normed."""
    graph, network, _ = _loaded(_EXPORT)
    block = network.blocks[block_index]
    qkvg = _read(f"block{block_index}_qkvg", _COUNT, 64, 2048)
    g = qkvg[..., 1536:]
    attended = _read(f"block{block_index}_attended", _COUNT, 64, 512)
    gated = _read(f"block{block_index}_gated", _COUNT, 64, 512)
    gate = block.gate_scale * torch.sigmoid(g)
    assert _rel(attended * gate, gated) < 1e-6  # noqa: PLR2004
    print(f"O-WIRING block {block_index}: gate in [{gate.min():.4f}, {gate.max():.4f}], mean {gate.mean():.4f}")
    x_in = _read(f"block{block_index}_input", _COUNT, 64, 512)
    ln1 = _read(f"block{block_index}_ln1", _COUNT, 64, 512)
    normed = torch.nn.functional.layer_norm(x_in, (512,), _raw(graph, block.ln1_scale), _raw(graph, block.ln1_bias), 1e-3)
    assert _rel(normed, ln1) < 1e-5  # noqa: PLR2004
    assert _rel(ln1 @ _raw(graph, block.gate_weight) + _raw(graph, block.gate_bias), g) < 1e-5  # noqa: PLR2004
    residual = _read(f"block{block_index}_residual", _COUNT, 64, 512)
    branch = gated @ _raw(graph, block.output_weight) + _raw(graph, block.output_bias)
    assert _rel(x_in + branch, residual) < 1e-5  # noqa: PLR2004


def test_final_norm_feeds_the_heads(reference) -> None:
    graph, network, _ = _loaded(_EXPORT)
    out14 = _read("block14_output", _COUNT, 64, 512)
    final = _read("final_norm", _COUNT, 64, 512)
    normed = torch.nn.functional.layer_norm(out14, (512,), _raw(graph, network.final_norm.scale),
                                            _raw(graph, network.final_norm.bias), 1e-3)
    assert _rel(normed, final) < 1e-5  # noqa: PLR2004
    # the value head's per-square projection reads the final norm: Mish(final W + b) is its first Gemm
    heads = network.heads
    assert graph.initializers[heads.value_square_weight].dims == (512, 128)
    assert FLOAT32 == graph.initializers[heads.value_square_weight].data_type
