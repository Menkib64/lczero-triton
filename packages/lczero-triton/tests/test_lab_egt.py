"""The EGT2 reader and carrier plans on the real exports (item E, R0; CPU).

The reader recognises the EGT2 edge stream by structure (`_mapping.read_network`), in both export forms: Einsum
(the gate-capped export) and the Einsum->MatMul rewrite (the two triplet exports). These tests hold it to the
item E map (`check_*.json`: 15 blocks, update sites after blocks 3/7/11, the cap only on gcap, the triplet
contraction) and check every build-time fold the carrier writes by recomputing it from the export's initializers
in float64: table by table, then as whole pre-softmax logits `H` and gates against the export's own formula, and
the update-site maps against the export's Einsum orientation. Errors print as `R0-FOLD` lines (`pytest -s`).
"""

import math
import os
import struct
from pathlib import Path

import pytest
import torch
from lczero_triton.lab import _onnx
from lczero_triton.lab._mapping import read_network
from lczero_triton.lab._names import plan_network, plan_prologue_egt
from lczero_triton.lab._onnx import FLOAT16, FLOAT32
from lczero_triton.lab.carrier import _as_float32, _as_int32, _build_payload

_NETS = Path(os.environ.get("LC0EX_EGT_NETS", str(Path.home() / "spsa/lc0ex_5080/work/nets_r20")))
_STATIC = Path(os.environ.get(
    "LC0EX_STATIC_NET", str(Path.home() / "spsa/lc0ex_5080/work/static_net/static_bs4g_512x15_50000_vw.pb.gz")))
_EXPORTS = {
    "gcap": "egt2_gcap_512x15_50000_vw.pb.gz",
    "triplet_path": "egt2_triplet_path_512x15_50000_vw.pb.gz",
    "triplet_ag": "egt2_triplet_ag_512x15_50000_vw.pb.gz",
}
# K1's gcap prologue names (`test_prologue_egt.GCAP_TABLES`, from the map's `check_gcap.json`).
GCAP_TABLES = {
    "channel_mix": "const_1205",
    "offset_table": "const_22",
    "edge_mix": "const_1213",
    "edge_offset_table": "const_25",
    "relative_index": "const_23",
}
# FP32 storage of a float64 fold: a peak-relative error well inside the map's 1e-5 pass rule.
_STORED_RELATIVE = 1e-6
_ASSEMBLED_RELATIVE = 1e-5
_LOADED: dict[str, tuple] = {}


def _loaded(label: str) -> tuple:
    if label not in _LOADED:
        path = _NETS / _EXPORTS[label] if label in _EXPORTS else _STATIC
        if not path.exists():
            pytest.skip(f"{path} not present")
        _, graph = _onnx.load_carrier(path)
        network = read_network(graph)
        _LOADED[label] = (graph, network, {plan.name: plan for plan in plan_network(network)})
    return _LOADED[label]


def _decode(graph, plan) -> torch.Tensor:
    payload, _ = _build_payload(graph, plan)
    dtype = torch.float32 if plan.data_type == FLOAT32 else torch.float16
    return torch.frombuffer(bytearray(payload), dtype=dtype).double().reshape(plan.shape)


def _raw(graph, name: str) -> torch.Tensor:
    """An export initializer in float64 with its exported dims."""
    return _as_float32(graph, name).double().reshape(graph.initializers[name].dims)


def _error(label: str, got: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (got - expected).abs().max().item()
    relative = difference / max(expected.abs().max().item(), 1e-30)
    print(f"R0-FOLD {label}: max_abs {difference:.3e} rel {relative:.3e}", flush=True)
    return relative


def _field(number: int, wire: int, payload: bytes) -> bytes:
    if wire == 0:
        return _onnx._tag(number, 0) + payload
    return _onnx._length_delimited(number, payload)


@pytest.mark.parametrize(("data_type", "field", "code", "values"), [(7, 7, "q", [0, 0, -1]), (6, 5, "i", [3, -2])])
@pytest.mark.parametrize("packed", [True, False])
def test_typed_integers_decode_signed(data_type, field, code, values, packed) -> None:
    wire = [value & ((1 << 64) - 1) for value in values]
    if packed:
        body = _field(field, 2, b"".join(_onnx._varint(value) for value in wire))
    else:
        body = b"".join(_field(field, 0, _onnx._varint(value)) for value in wire)
    message = (_field(1, 0, _onnx._varint(len(values))) + _field(2, 0, _onnx._varint(data_type))
               + _field(8, 2, b"shape") + body)
    tensor = _onnx._decode_tensor(memoryview(message))
    assert tensor.raw_data == struct.pack(f"<{len(values)}{code}", *values)


def test_triplet_rewrite_shapes_parse() -> None:
    graph, _, _ = _loaded("triplet_path")
    shapes = {name: struct.unpack(f"<{len(t.raw_data) // 8}q", t.raw_data)
              for name, t in graph.initializers.items() if name.startswith("rw_shape_") and t.data_type == 7}
    assert shapes["rw_shape_f3_2"] == (0, 0, -1)


@pytest.mark.parametrize(("label", "cap", "contraction"),
                         [("gcap", True, None), ("triplet_path", False, "path"), ("triplet_ag", False, "ag")])
def test_edge_stream_structure(label, cap, contraction) -> None:
    graph, network, plans = _loaded(label)
    egt = network.egt
    assert egt is not None
    assert network.architecture.blocks == len(egt.blocks) == 15  # noqa: PLR2004
    assert network.architecture.edge_channels == 34 and egt.state_channels == 16 and egt.site_hidden == 64  # noqa: PLR2004
    assert [site.after_block for site in egt.sites] == [3, 7, 11]
    assert egt.cap is cap and all(block.cap is cap for block in egt.blocks)
    assert [site.triplet.contraction if site.triplet else None for site in egt.sites] == [contraction] * 3
    expected_state = egt.blocks[0].state
    for index, block in enumerate(egt.blocks):
        assert block.state == expected_state, index
        site = next((site for site in egt.sites if site.after_block == index), None)
        if site is not None:
            expected_state = site.state
            assert graph.produced_by(site.state).index == site.last_node
    assert not [name for name in plans if name.endswith("_codes") and name != "/prologue/mix_codes"]
    for name, plan in plans.items():
        if "/mha/egt/" in name or "/edge_site/" in name or name.startswith("/prologue/"):
            assert plan.data_type == FLOAT32, name
        if name.endswith(("/mha/qkv/w", "/mha/out/w", "/ffn/dense1/w", "/ffn/dense2/w")) and "/edge_site/" not in name:
            assert plan.data_type == FLOAT16, name


def test_gcap_names_are_the_maps() -> None:
    _, network, _ = _loaded("gcap")
    block, edge, site, prologue = network.blocks[0], network.egt.blocks[0], network.egt.sites[0], network.egt.prologue
    assert (block.edge_coefficients, block.pair_key, block.pair_query, block.mix_coefficients) == (
        "const_1219", "const_1220", "const_1221", "const_1222")
    assert (edge.node_temperature, edge.edge_temperature, edge.edge_read, edge.door, edge.gate_weight,
            edge.gate_bias) == ("const_1225", "const_1227", "const_1223", "const_1228", "const_1231", "const_1233")
    assert (edge.softmax_node, edge.logits_node, edge.weights_node) == (10421, 10420, 10433)
    assert (site.readback_weight, site.ffn_in_weight, site.ffn_in_bias, site.ffn_out_weight) == (
        "const_1329", "const_1333", "const_1335", "const_1336")
    assert (site.readback_node, site.state_node, site.last_node) == (10897, 10898, 10923)
    assert (prologue.edge_mix, prologue.edge_offset_table, prologue.relative_index) == ("const_1213", "const_25", "const_23")
    assert (prologue.edges_node, prologue.pair_node, prologue.state_node) == (10279, 10305, 10331)


def test_triplet_path_names_are_the_maps() -> None:
    _, network, _ = _loaded("triplet_path")
    triplet = network.egt.sites[0].triplet
    assert (triplet.value_weight, triplet.gate_weight, triplet.gate_bias, triplet.output_weight) == (
        "const_1337", "const_1338", "const_1340", "const_1341")
    assert (triplet.split_node, triplet.add_node, network.egt.sites[0].last_node) == (10999, 11038, 11069)


def test_prologue_plans_are_k1s() -> None:
    _, _, plans = _loaded("gcap")
    wired = [plan for name, plan in plans.items() if name.startswith("/prologue/")]
    assert wired == plan_prologue_egt(**GCAP_TABLES)


def _block_folds(graph, network, index: int) -> dict[str, torch.Tensor]:
    """The map's section 5.1 folds for one block, from the export's initializers in float64."""
    shape, block, edge, pair = network.architecture, network.blocks[index], network.egt.blocks[index], network.pair
    heads = shape.heads
    t_node = _raw(graph, edge.node_temperature).reshape(heads)
    t_edge = _raw(graph, edge.edge_temperature).reshape(heads)
    head_mix = _raw(graph, block.mix_coefficients)[0]
    channel_mix = _raw(graph, pair.channel_mix)[0]
    table = _raw(graph, pair.offset_table)
    relidx = _as_int32(graph, pair.relative_index).long().reshape(shape.tokens, shape.tokens)
    return {
        "qk_scale": t_node / math.sqrt(shape.head_dim),
        "attack": t_node[:, None] * _raw(graph, block.edge_coefficients)[0],
        "key": t_node[:, None, None] * _raw(graph, block.pair_key)[0],
        "query": t_node[:, None, None] * _raw(graph, block.pair_query)[0],
        "scaled_coefficients": t_node[:, None] * (head_mix @ channel_mix.T),
        "constant_bias": t_node[:, None, None] * torch.einsum("ht,trc->hrc", head_mix, table.T[:, relidx]),
        "edge_read/w": t_edge[:, None] * _raw(graph, edge.edge_read)[0],
        "door/w": _raw(graph, edge.door)[0],
        "gate/w": _raw(graph, edge.gate_weight)[0],
        "gate/b": _raw(graph, edge.gate_bias).reshape(heads),
    }


@pytest.mark.parametrize("label", ["gcap", "triplet_path"])
@pytest.mark.parametrize("index", [0, 7, 14])
def test_block_folds_match_float64(label, index) -> None:
    graph, network, plans = _loaded(label)
    for name, expected in _block_folds(graph, network, index).items():
        got = _decode(graph, plans[f"/encoder{index}/mha/egt/{name}"])
        assert _error(f"{label} block {index} {name}", got, expected) <= _STORED_RELATIVE, name


def _einsum_map(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("hc,cij->hij", weights, values)


@pytest.mark.parametrize(("label", "index"), [("gcap", 0), ("gcap", 14), ("triplet_path", 3)])
def test_folded_logits_and_gate_equal_the_export_formula(label, index) -> None:
    graph, network, plans = _loaded(label)
    shape, block, edge, pair = network.architecture, network.blocks[index], network.egt.blocks[index], network.pair
    heads, depth, tokens = shape.heads, shape.head_dim, shape.tokens
    generator = torch.Generator().manual_seed(0x0E + index)
    q = torch.randn(heads, tokens, depth, generator=generator, dtype=torch.float64)
    k = torch.randn(heads, tokens, depth, generator=generator, dtype=torch.float64)
    edges = (torch.rand(shape.edge_channels, tokens, tokens, generator=generator) < 0.05).double()  # noqa: PLR2004
    state = torch.randn(network.egt.state_channels, tokens, tokens, generator=generator, dtype=torch.float64)
    relidx = _as_int32(graph, pair.relative_index).long().reshape(tokens, tokens)
    mixed = torch.einsum("cij,cd->dij", edges, _raw(graph, pair.channel_mix)[0]) + _raw(graph, pair.offset_table).T[:, relidx]
    norm = 1.0 / torch.sqrt(mixed.pow(2).mean(0, keepdim=True) + pair.epsilon)
    pair_stream = mixed * norm

    # The export's formula (item E map section 3), from raw initializers.
    qk = q @ k.transpose(1, 2)
    logits = (qk / math.sqrt(depth)
              + _einsum_map(_raw(graph, block.edge_coefficients)[0], edges)
              + torch.einsum("hjc,cij->hij", torch.einsum("hjd,hcd->hjc", k, _raw(graph, block.pair_key)[0]), edges)
              + torch.einsum("hic,cij->hij", torch.einsum("hid,hcd->hic", q, _raw(graph, block.pair_query)[0]), edges)
              + _einsum_map(_raw(graph, block.mix_coefficients)[0], pair_stream))
    t_node = _raw(graph, edge.node_temperature).reshape(heads, 1, 1)
    t_edge = _raw(graph, edge.edge_temperature).reshape(heads, 1, 1)
    direct = ((t_node * logits + t_edge * _einsum_map(_raw(graph, edge.edge_read)[0], state))
              * (1.0 + _einsum_map(_raw(graph, edge.door)[0], state)))
    gate = 2.0 * torch.sigmoid(_einsum_map(_raw(graph, edge.gate_weight)[0], state)
                               + _raw(graph, edge.gate_bias).reshape(heads, 1, 1))
    if edge.cap:
        gate = gate - torch.relu(gate - 1.0)

    # The served form, from the carrier's FP32 tables only.
    def served(name: str) -> torch.Tensor:
        return _decode(graph, plans[f"/encoder{index}/mha/egt/{name}"])

    coefficient_terms = (_einsum_map(served("attack"), edges)
                         + torch.einsum("hjc,cij->hij", torch.einsum("hjd,hcd->hjc", k, served("key")), edges)
                         + torch.einsum("hic,cij->hij", torch.einsum("hid,hcd->hic", q, served("query")), edges))
    folded_pair = norm * (_einsum_map(served("scaled_coefficients"), edges) + served("constant_bias"))
    folded = ((served("qk_scale")[:, None, None] * qk + coefficient_terms + folded_pair
               + _einsum_map(served("edge_read/w"), state))
              * (1.0 + _einsum_map(served("door/w"), state)))
    folded_gate = 2.0 * torch.sigmoid(_einsum_map(served("gate/w"), state) + served("gate/b")[:, None, None])
    if network.egt.cap:
        folded_gate = torch.clamp(folded_gate, max=1.0)
    assert _error(f"{label} block {index} assembled H", folded, direct) <= _ASSEMBLED_RELATIVE
    assert _error(f"{label} block {index} assembled gate", folded_gate, gate) <= _ASSEMBLED_RELATIVE
    if edge.cap:
        assert bool((gate == 1.0).any()), "the cap never clipped, so the capped gate is untested"


@pytest.mark.parametrize(("label", "position"), [("gcap", 0), ("gcap", 2), ("triplet_path", 1), ("triplet_ag", 0)])
def test_site_maps_follow_the_export_orientation(label, position) -> None:
    graph, network, plans = _loaded(label)
    site = network.egt.sites[position]
    prefix = f"/encoder{site.after_block}/edge_site"
    maps = {"readback/w": site.readback_weight, "ffn/dense1/w": site.ffn_in_weight, "ffn/dense2/w": site.ffn_out_weight}
    biases = {"ffn/dense1/b": site.ffn_in_bias}
    if site.triplet is not None:
        maps |= {"triplet/value/w": site.triplet.value_weight, "triplet/gate/w": site.triplet.gate_weight,
                 "triplet/out/w": site.triplet.output_weight}
        biases |= {"triplet/gate/b": site.triplet.gate_bias}
    generator = torch.Generator().manual_seed(0x5173 + position)
    for name, constant in maps.items():
        exported = _raw(graph, constant)[0]  # [1, a, b], applied as the export's `zab,zacd->zbcd`
        values = torch.randn(exported.shape[0], 8, 8, generator=generator, dtype=torch.float64)
        expected = torch.einsum("ab,acd->bcd", exported, values)
        got = _einsum_map(_decode(graph, plans[f"{prefix}/{name}"]), values)
        assert _error(f"{label} site {site.after_block} {name}", got, expected) <= _STORED_RELATIVE, name
    for name, constant in biases.items():
        expected = _raw(graph, constant).reshape(-1)
        assert _error(f"{label} site {site.after_block} {name}", _decode(graph, plans[f"{prefix}/{name}"]), expected) <= _STORED_RELATIVE


def test_policy_pair_bias_reads_the_34_channel_mix() -> None:
    graph, network, plans = _loaded("gcap")
    head_mix = _raw(graph, network.heads.policy_mix_coefficients).reshape(1, -1)
    expected = head_mix @ _raw(graph, network.pair.channel_mix)[0].T / network.policy_divisor
    plan = plans["/policy/pair/scaled_coefficients"]
    assert plan.shape == (1, 34) and plan.data_type == FLOAT16
    assert _error("gcap policy scaled_coefficients (FP16)", _decode(graph, plan), expected) <= 1e-3  # noqa: PLR2004


def test_every_gcap_plan_builds_its_declared_size() -> None:
    graph, _, plans = _loaded("gcap")
    for plan in plans.values():
        payload, _ = _build_payload(graph, plan)
        assert len(payload) == plan.element_count * (2 if plan.data_type == FLOAT16 else 4), plan.name


@pytest.mark.parametrize("label", ["gcap", "triplet_path", "triplet_ag"])
def test_builder_accepts_the_served_egt2_family(label: str) -> None:
    """W wired gcap and K5 the two triplet exports: the builder's contract check raises for none of them."""
    from lczero_triton.lab.network import _check_egt_supported  # noqa: PLC0415

    _, network, _ = _loaded(label)
    _check_egt_supported(network)


@pytest.mark.parametrize("label", ["gcap", "triplet_path", "triplet_ag"])
def test_triplet_sites_plan_k4s_tables(label: str) -> None:
    """A triplet export plans exactly K4's four tables at each of the three sites; gcap plans none."""
    from lczero_triton.bt4.kernels.triplet_site import triplet_table_names  # noqa: PLC0415

    _, network, plans = _loaded(label)
    for site in network.egt.sites:
        names = triplet_table_names(site.after_block)
        planned = [name for name in names if name in plans]
        if label == "gcap":
            assert planned == [], f"gcap plans triplet tables at site {site.after_block}: {planned}"
        else:
            assert planned == list(names), f"{label} site {site.after_block} plans {planned}"
            assert plans[names[0]].shape == (2 * network.egt.state_channels, network.egt.state_channels)


def test_static_family_has_no_edge_stream() -> None:
    _, network, plans = _loaded("static")
    assert network.egt is None
    assert not [name for name in plans if "/egt/" in name or "/edge_site/" in name]
