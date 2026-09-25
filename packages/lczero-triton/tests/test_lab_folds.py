"""Build-time folds of the static net's pair stream, on the real export (CPU).

The attention kernel receives `S (G E + K)` in place of the export's
`sum_t C2[h,t] M[t]`, `M = (einsum(E, D) + T) S`. These tests check, on the
actual step-50,000 tensors, that the recovery finds the right constants, that
`T` is rebuilt exactly as the export's runtime gather, and that the folded
payloads reproduce the direct 16-channel term within FP16 rounding.
"""

import os
from pathlib import Path

import pytest
import torch
from lczero_triton.lab import _onnx
from lczero_triton.lab._mapping import read_network
from lczero_triton.lab._names import plan_network
from lczero_triton.lab._onnx import FLOAT16, FLOAT32
from lczero_triton.lab.carrier import _as_float32, _as_int32, _build_payload

_NET = Path(os.environ.get("LC0EX_STATIC_NET", str(Path.home() / "Kovax/nets/static_bs4g_512x15_50000_vw.pb.gz")))

pytestmark = pytest.mark.skipif(not _NET.exists(), reason=f"{_NET} not present")


@pytest.fixture(scope="module")
def loaded():
    _, graph = _onnx.load_carrier(_NET)
    network = read_network(graph)
    plans = {plan.name: plan for plan in plan_network(network)}
    return graph, network, plans


def _decode(graph, plan) -> torch.Tensor:
    payload, _ = _build_payload(graph, plan)
    dtype = torch.float32 if plan.data_type == FLOAT32 else torch.float16
    return torch.frombuffer(bytearray(payload), dtype=dtype).double().reshape(plan.shape)


def test_pair_tables_are_recovered(loaded) -> None:
    _, network, _ = loaded
    assert network.pair.channel_mix == "const_727"
    assert network.pair.offset_table == "const_21"
    assert network.pair.relative_index == "const_22"
    assert abs(network.pair.epsilon - 1e-6) < 1e-12


def test_offsets_are_the_exports_runtime_gather(loaded) -> None:
    graph, network, plans = loaded
    offsets = _decode(graph, plans["/prologue/offsets"])
    squares = torch.arange(64)
    rank, file = squares // 8, squares % 8
    relidx = 15 * (rank[:, None] - rank[None, :] + 7) + (file[:, None] - file[None, :] + 7)
    assert bool((_as_int32(graph, network.pair.relative_index).reshape(64, 64) == relidx).all())
    table = _as_float32(graph, network.pair.offset_table).reshape(225, 16).double()
    assert bool((offsets == table.T[:, relidx]).all())


@pytest.mark.parametrize("prefix", ["/encoder0", "/encoder14", "/policy"])
def test_folds_reproduce_the_sixteen_channel_term(loaded, prefix) -> None:
    graph, network, plans = loaded
    torch.manual_seed(0x7015)
    edges = (torch.rand(4, 64, 64) < 0.05).double()
    channel_mix = _decode(graph, plans["/prologue/channel_mix"])
    offsets = _decode(graph, plans["/prologue/offsets"])
    mixed = torch.einsum("cij,cd->dij", edges, channel_mix) + offsets
    norm = 1.0 / torch.sqrt(mixed.pow(2).mean(0, keepdim=True) + network.pair.epsilon)
    if prefix == "/policy":
        # The export divides Q K^T + bias TOGETHER by sqrt(policy width), so the policy folds carry that divisor
        # (`_plan_heads`: `divisor=network.policy_divisor`); the blocks' bias is added after their own scaling.
        head_mix = _as_float32(graph, network.heads.policy_mix_coefficients).double().reshape(1, 16)
        head_mix = head_mix / network.policy_divisor
    else:
        index = int(prefix.removeprefix("/encoder"))
        head_mix = _as_float32(graph, network.blocks[index].mix_coefficients).double().reshape(32, 16)
    direct = torch.einsum("ht,trc->hrc", head_mix, mixed * norm)
    scaled = _decode(graph, plans[f"{prefix}/pair/scaled_coefficients"])
    constant = _decode(graph, plans[f"{prefix}/pair/constant_bias"])
    folded = norm * (torch.einsum("he,erc->hrc", scaled, edges) + constant)
    error = (folded - direct).abs().max().item()
    scale = direct.abs().max().item()
    assert error <= 1e-3 * scale, f"{prefix}: max |folded - direct| = {error:.3e} against max |direct| = {scale:.3e}"


def test_every_plan_builds_a_payload_of_its_declared_size(loaded) -> None:
    graph, _, plans = loaded
    for plan in plans.values():
        payload, _ = _build_payload(graph, plan)
        width = 2 if plan.data_type == FLOAT16 else 4
        assert len(payload) == plan.element_count * width, plan.name


def test_input_gate_is_served_as_relu_of_the_export_gate(loaded) -> None:
    graph, network, plans = loaded
    raw = _as_float32(graph, network.embedding.input_gate).double()
    assert bool((raw < 0).any()), "the gate has no negative weights, so this test would prove nothing"
    served = _decode(graph, plans["/attn_body/mult_gate/w"]).reshape(-1)
    expected = torch.relu(raw)
    assert bool((served >= 0).all())
    assert (served - expected).abs().max().item() <= 1e-3 * expected.abs().max().item()
