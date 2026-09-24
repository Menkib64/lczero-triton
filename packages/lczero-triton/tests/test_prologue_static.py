"""CUDA tests for the static-net prologue.

``E`` is gated **exactly** against the lab's own ``attack_graph.edge_channels``
run in JAX (``ref/prologue_ref_*``: 4,096 seeded synthetic boards plus the 8
random-float boards of the step-50,000 reference dump, whose multi-occupied
squares make the gate stricter than legal positions would). ``S`` is gated
against a float64 evaluation of the export's own node chain with random ``D``
and ``T``, which tests the formula independently of the carrier's fold.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from lczero_triton.bt4.kernels.prologue_static import (
    PAIR_EPSILON,
    PrologueStaticSpecialization,
    _autotune_grid,
    _prologue_static_kernel,
    compile_prologue_static,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_REFERENCE = Path(os.environ.get("LC0EX_PROLOGUE_REF", str(Path.home() / "spsa/static_net_r20/ref")))
_CHUNK = 256


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _unpack(path: Path, count: int, bits: int) -> torch.Tensor:
    """numpy.packbits (big-endian per byte) -> ``[count, bits]`` bool, without numpy."""
    raw = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).view(count, -1)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    return ((raw[..., None] >> shifts) & 1).view(count, -1)[:, :bits].bool()


def _masks(piece_bits: torch.Tensor) -> torch.Tensor:
    """``[count, 12, 64]`` bool -> ``[count, 112]`` uint64 plane masks."""
    count = piece_bits.shape[0]
    weights = torch.tensor([1 << s if s < 63 else -(1 << 63) for s in range(64)], dtype=torch.int64)
    packed = (piece_bits.to(torch.int64) * weights).sum(-1)
    masks = torch.zeros((count, 112), dtype=torch.int64)
    masks[:, :12] = packed
    return masks.view(torch.uint64)


def _load_reference():
    header_path = _REFERENCE / "prologue_ref.json"
    if not header_path.exists():
        pytest.skip(f"reference not found at {_REFERENCE}")
    header = json.loads(header_path.read_text())
    count = header["count"]
    planes = _unpack(_REFERENCE / header["planes12"]["file"], count, 12 * 64).view(count, 12, 64)
    edges = _unpack(_REFERENCE / header["E"]["file"], count, 4 * 64 * 64).view(count, 4, 64, 64)
    return count, planes, edges


def _run(masks: torch.Tensor, channel_mix: torch.Tensor, offsets: torch.Tensor):
    batch = masks.shape[0]
    codes = torch.full((batch, 64, 64), 255, dtype=torch.uint8, device="cuda")
    edge_norm = torch.full((batch, 64, 64), -1.0, dtype=torch.float16, device="cuda")
    values = torch.zeros((batch, 112), dtype=torch.float32, device="cuda")
    values[:, :12] = 1.0
    _prologue_static_kernel[_autotune_grid](
        codes, edge_norm, masks.cuda(), values, channel_mix, offsets, batch, PAIR_EPSILON,
    )
    torch.cuda.synchronize()
    assert bool((codes < 16).all()), "a packed E code uses only its four low bits"
    edges = torch.stack([((codes >> channel) & 1) for channel in range(4)], dim=1).to(torch.float16)
    return edges, edge_norm


def test_edges_match_the_lab_attack_graph_exactly() -> None:
    count, planes, reference = _load_reference()
    channel_mix = torch.zeros((4, 16), dtype=torch.float32, device="cuda")
    offsets = torch.ones((16, 64, 64), dtype=torch.float32, device="cuda")
    mismatched_boards = []
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        batch_masks = _masks(planes[start:stop])
        if stop - start < _CHUNK:
            pad = torch.zeros((_CHUNK - (stop - start), 112), dtype=torch.uint64)
            batch_masks = torch.cat([batch_masks, pad])
        edges, _ = _run(batch_masks, channel_mix, offsets)
        got = edges[: stop - start].float().cpu()
        assert bool(((got == 0) | (got == 1)).all()), "E must be exactly 0 or 1"
        wrong = (got.bool() != reference[start:stop]).flatten(1).any(1)
        mismatched_boards += [start + int(i) for i in wrong.nonzero().flatten()]
    assert not mismatched_boards, f"{len(mismatched_boards)} of {count} boards differ, first {mismatched_boards[:10]}"


def test_normalization_matches_the_export_node_chain() -> None:
    count, planes, reference = _load_reference()
    torch.manual_seed(0x5171)
    channel_mix = torch.randn((4, 16), dtype=torch.float32, device="cuda") * 0.7
    offsets = torch.randn((16, 64, 64), dtype=torch.float32, device="cuda") * 0.5
    batch_masks = _masks(planes[:_CHUNK])
    edges, edge_norm = _run(batch_masks, channel_mix, offsets)
    e = reference[:_CHUNK].double()
    mixed = torch.einsum("bcij,cd->bdij", e, channel_mix.double().cpu()) + offsets.double().cpu()
    expected = 1.0 / torch.sqrt(mixed.pow(2).mean(dim=1) + PAIR_EPSILON)
    got = edge_norm.double().cpu()
    relative = ((got - expected).abs() / expected).max().item()
    assert relative < 1e-3, f"max relative error {relative:.3e}"


def test_artifact_compiles() -> None:
    artifact = compile_prologue_static(PrologueStaticSpecialization(batch_count=8, architecture=_architecture()))
    assert artifact is not None
