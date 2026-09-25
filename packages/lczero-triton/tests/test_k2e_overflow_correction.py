"""CUDA tests for K2e's two-launch E path: main launch list-only (`overflow_exact=False`) + correction launch
(`overflow_only=True`). Reuses K2's reference set and helpers. ``K2E-GATE`` lines under ``pytest -s``."""

import pytest
import torch
from lczero_triton.bt4.kernels.attention_egt import (
    BLOCK_TABLES,
    AttentionEgtSpecialization,
    launch_attention_egt,
)

from test_attention_egt import (  # noqa: I001
    _CAPACITY,
    _DEPTH,
    _HEADS,
    _architecture,
    _block_inputs,
    _edge_list,
    _error,
    _run,
    _tables,
    reference,  # noqa: F401
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")]
_F32_RELATIVE = 1e-5


def _two_launches(reference, inputs, tables, capacity, scaled_stream=False):
    edge_list = _edge_list(reference, capacity)[:4]
    count = inputs["qkv"].shape[0]
    batch_count = count * _HEADS
    common = {"batch_count": batch_count, "heads": _HEADS, "head_dim": _DEPTH, "architecture": _architecture(),
              "cap": inputs["cap"], "export_h": True, "capacity": capacity, "norm_f32": inputs["S"].dtype == torch.float32,
              "export_weights": True, "scaled_stream": scaled_stream}
    output = torch.full((count, 64, 512), -7.0, dtype=torch.float16, device="cuda")
    logits = torch.full((batch_count, 64, 64), -7.0, dtype=torch.float32, device="cuda")
    weights = torch.full((batch_count, 64, 64), -7.0, dtype=torch.float32, device="cuda")
    for overflow_only in (False, True):
        spec = AttentionEgtSpecialization(**common, overflow_exact=False, overflow_only=overflow_only)
        launch_attention_egt(output, inputs["qkv"], edge_list, reference["edges"], inputs["S"], inputs["state"],
                             tables, spec, logits=logits, weights=weights)
    torch.cuda.synchronize()
    return {"H": logits.view(count, 32, 64, 64), "A": weights.view(count, 32, 64, 64),
            "attended": output.view(count, 64, 32, 16).permute(0, 2, 1, 3)}


@pytest.mark.parametrize("capacity", [64, 128, 256, 1024])
def test_two_launches_match_ort(reference, capacity):
    """Main launch list-only + correction launch == ORT, at capacities that force 62 / 55 / 45 / 0 dense positions."""
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    got = _two_launches(reference, inputs, _tables("gcap", 3), capacity)
    errors = {name: _error(got[name], inputs[name]) for name in ("H", "A")}
    print(f"K2E-GATE two launches capacity {capacity} gcap block 3: "
          + "  ".join(f"{k} rel {v['rel']:.2e}" for k, v in errors.items()), flush=True)
    assert all(e["rel"] <= _F32_RELATIVE for e in errors.values())


def test_two_launches_equal_the_branch_kernel_bit_for_bit(reference):
    """At capacity 256 (45 of 64 dense) the two-launch result must equal K2b's single-kernel branch result exactly."""
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    tables = _tables("gcap", 3)
    two = _two_launches(reference, inputs, tables, 256)
    one = _run(reference, inputs, tables, capacity=256)
    for name in ("H", "A", "attended"):
        assert torch.equal(two[name], one[name]), f"{name} differs between the two-launch and the branch form"


def test_scaled_stream_is_in_class(reference):
    inputs = _block_inputs(reference, "gcap", 3, served=False)
    got = _two_launches(reference, inputs, _tables("gcap", 3), _CAPACITY, scaled_stream=True)
    errors = {name: _error(got[name], inputs[name]) for name in ("H", "A")}
    print("K2E-GATE scaled stream gcap block 3: " + "  ".join(f"{k} rel {v['rel']:.2e}" for k, v in errors.items()), flush=True)
    assert all(e["rel"] <= _F32_RELATIVE for e in errors.values())
