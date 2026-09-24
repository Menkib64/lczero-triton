"""CUDA tests for `attention_egt`'s edge-state read modes (round 21, K2c), gated per block against onnxruntime.

Reuses K2's reference set and helpers (`test_attention_egt`): for every reference block, the head program reading
(a) the FP16 copy of the state and (b) the block's FP16 read tiles in 1, 2 and 4 head rounds must match ORT's H, A
and A.v within the class of FP16 projections (the served rule), and match the FP32-state kernel within the class
of an FP16 rounding of the read terms. ``K2C-GATE`` lines under ``pytest -s``.
"""

import pytest
import torch
from lczero_triton.bt4.kernels.attention_egt import (
    BLOCK_TABLES,
    AttentionEgtSpecialization,
    launch_attention_egt,
)
from lczero_triton.bt4.kernels.egt_state_tiles import (
    CastStateSpecialization,
    StateTilesSpecialization,
    launch_cast_state,
    launch_state_tiles,
)

from test_attention_egt import (  # noqa: I001  (the test module next to this one)
    _BLOCKS,
    _CAPACITY,
    _DEPTH,
    _HEADS,
    _SERVED_RELATIVE,
    _architecture,
    _block_inputs,
    _edge_list,
    _error,
    _tables,
    reference,  # noqa: F401  (fixture)
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]
# An FP16 rounding of the read, door and gate terms, against the FP32-state kernel: a few ulps at peak.
_MODE_VS_F32_RELATIVE = 4e-3


def _run_mode(reference: dict, inputs: dict, tables: dict, mode: str, rounds: int) -> dict[str, torch.Tensor]:
    """H, A and A.v of one block with the head program reading the state as `mode` says."""
    edge_list = _edge_list(reference, _CAPACITY)[:4]
    count = inputs["qkv"].shape[0]
    batch_count = count * _HEADS
    architecture = _architecture()
    output = torch.full((count, 64, 512), -7.0, dtype=torch.float16, device="cuda")
    logits = torch.full((batch_count, 64, 64), -7.0, dtype=torch.float32, device="cuda")
    weights = torch.full((batch_count, 64, 64), -7.0, dtype=torch.float32, device="cuda")
    common = {"batch_count": batch_count, "heads": _HEADS, "head_dim": _DEPTH, "architecture": architecture,
              "cap": inputs["cap"], "export_h": True, "capacity": _CAPACITY,
              "norm_f32": inputs["S"].dtype == torch.float32, "export_weights": True}
    if mode == "f32":
        launch_attention_egt(output, inputs["qkv"], edge_list, reference["edges"], inputs["S"], inputs["state"],
                             tables, AttentionEgtSpecialization(**common), logits=logits, weights=weights)
    elif mode == "f16":
        copy = torch.empty((count, 16, 64, 64), dtype=torch.float16, device="cuda")
        launch_cast_state(copy, inputs["state"], CastStateSpecialization(count, architecture))
        launch_attention_egt(output, inputs["qkv"], edge_list, reference["edges"], inputs["S"], copy, tables,
                             AttentionEgtSpecialization(**common, state_f32=False), logits=logits, weights=weights)
    else:
        round_heads = _HEADS // rounds
        for round_index in range(rounds):
            head_base = round_index * round_heads
            tiles = torch.empty((count, round_heads, 3, 64, 64), dtype=torch.float16, device="cuda")
            launch_state_tiles(tiles, inputs["state"], tables["edge_read/w"], tables["door/w"], tables["gate/w"],
                               StateTilesSpecialization(count, architecture, round_heads=round_heads,
                                                        head_base=head_base))
            launch_attention_egt(output, inputs["qkv"], edge_list, reference["edges"], inputs["S"], tiles, tables,
                                 AttentionEgtSpecialization(**common, state_f32=False, state_tiles=True,
                                                            round_heads=round_heads, head_base=head_base),
                                 logits=logits, weights=weights)
    torch.cuda.synchronize()
    return {"H": logits.view(count, 32, 64, 64), "A": weights.view(count, 32, 64, 64),
            "attended": output.view(count, 64, 32, 16).permute(0, 2, 1, 3)}


_MODES = [("f16", 1), ("tiles", 1), ("tiles", 2), ("tiles", 4)]


@pytest.mark.parametrize(("mode", "rounds"), _MODES, ids=[f"{m}-r{r}" for m, r in _MODES])
@pytest.mark.parametrize(("label", "block"), _BLOCKS, ids=[f"{a}-b{b}" for a, b in _BLOCKS])
def test_read_mode_matches_ort_served_precision(reference: dict, label: str, block: int, mode: str,
                                                rounds: int) -> None:
    inputs = _block_inputs(reference, label, block, served=True)
    tables = _tables(label, block)
    got = _run_mode(reference, inputs, tables, mode, rounds)
    base = _run_mode(reference, inputs, tables, "f32", 1)
    errors = {name: _error(got[name], inputs[name]) for name in ("H", "A", "attended")}
    drift = {name: _error(got[name], base[name]) for name in ("H", "A", "attended")}
    print(f"K2C-GATE {label} block {block} {mode} r{rounds} served vs ORT: "
          + "  ".join(f"{k} rel {v['rel']:.2e}" for k, v in errors.items())
          + " | vs f32 kernel: " + "  ".join(f"{k} rel {v['rel']:.2e}" for k, v in drift.items()), flush=True)
    assert all(e["rel"] <= _SERVED_RELATIVE for e in errors.values())
    assert all(d["rel"] <= _MODE_VS_F32_RELATIVE for d in drift.values())


def test_rounds_write_every_head_once(reference: dict) -> None:
    """Two and four rounds must reproduce the one-round result exactly: same tiles, same programs, only split."""
    inputs = _block_inputs(reference, "gcap", 3, served=True)
    tables = _tables("gcap", 3)
    one = _run_mode(reference, inputs, tables, "tiles", 1)
    for rounds in (2, 4):
        split = _run_mode(reference, inputs, tables, "tiles", rounds)
        for name in ("H", "A", "attended"):
            assert torch.equal(split[name], one[name]), f"{name} differs with {rounds} rounds"
