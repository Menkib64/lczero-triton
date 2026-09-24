"""CUDA tests for the EGT2 prologue (item E, K1).

``E`` is gated **bitwise** against onnxruntime evaluating the gcap export's own graph (node 10279) on every
position of ``ref/egt`` and all 34 channels. The set holds 1,024 positions from real games, with pins, x-rays,
clipped king rings, promotions, en passant, castling, partial histories and both side-to-move colours; see
``egt_ref.json`` and ``positions.json``. ``S`` (node 10304), ``Z0`` (node 10305) and ``e0`` (node 10331) are gated
against the same run in both storage precisions. K1b (round 22) replaced the packed-``E`` kernel by a row kernel
and a packing kernel; the gates are unchanged, only the launch helper and the artifact shapes are. ``Z0`` is rebuilt as ``S * (D.E + T)`` from the kernels' own
``E`` and ``S``, which is exactly what its folds consume. The reference manifest is verified once per module.
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
from lczero_triton.bt4.kernels.prologue_egt import (
    CODE_GROUP_BASES,
    CODE_GROUPS,
    CODE_ROWS,
    CODE_WIDTH,
    EDGE_CHANNELS,
    EDGE_SPLIT,
    EPSILON,
    SEED_TABLES,
    PrologueEgtSpecialization,
    _autotune_grid,
    _prologue_egt_seed_kernel,
    buffer_bytes,
    compile_prologue_egt_edges,
    compile_prologue_egt_rows,
    compile_prologue_egt_seed,
    fold_mix_codes,
    launch_prologue_egt_edges,
    prologue_egt,
    unpack_edges,
)
from lczero_triton.lab import _names
from lczero_triton.lab._names import plan_prologue_egt

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_REFERENCE = Path(os.environ.get("LC0EX_EGT_REF", str(Path(__file__).resolve().parents[3] / "ref/egt")))
_NETWORK = Path(os.environ.get(
    "LC0EX_EGT_NET", str(Path.home() / "spsa/lc0ex_5080/work/nets_r20/egt2_gcap_512x15_50000_vw.pb.gz")))
# The gcap export's prologue initializers (item E step 0, `check_gcap.json` roles). R0 will recover them by structure.
GCAP_TABLES = {
    "channel_mix": "const_1205",
    "offset_table": "const_22",
    "edge_mix": "const_1213",
    "edge_offset_table": "const_25",
    "relative_index": "const_23",
}
_CHUNK = 256
# FP32 class: the map's pass rule for recomputations. FP16 storage: half an FP16 unit in the last place is
# 2**-11 relative, plus the FP32 floor.
_F32_RELATIVE = 1e-5
_F16_RELATIVE = 6e-4


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _read(name: str, dtype: torch.dtype, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray((_REFERENCE / name).read_bytes()), dtype=dtype).view(*shape)


@pytest.fixture(scope="module")
def reference() -> dict:
    header_path = _REFERENCE / "egt_ref.json"
    if not header_path.exists():
        pytest.skip(f"reference not found at {_REFERENCE}")
    for line in (_REFERENCE / "SHA256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((_REFERENCE / name).read_bytes()).hexdigest() == digest, f"{name} fails its sha256"
    header = json.loads(header_path.read_text())
    count = header["count"]
    planes = _read("planes.bin", torch.float32, count, 112, 64)
    present = planes != 0
    values = planes.amax(-1)
    assert bool((~present | (planes == values[..., None])).all()), "a plane is not uniform over its set squares"
    weights = torch.tensor([1 << s if s < 63 else -(1 << 63) for s in range(64)], dtype=torch.int64)
    masks = (present.to(torch.int64) * weights).sum(-1).view(torch.uint64)
    raw = _read("E_bits.bin", torch.uint8, count, -1)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    edges = ((raw[..., None] >> shifts) & 1).view(count, -1)[:, : EDGE_CHANNELS * 4096].bool()
    tables = {
        "channel_mix": _read("channel_mix.bin", torch.float32, 34, 16),
        "offsets": _read("offsets.bin", torch.float32, 16, 64, 64),
        "edge_mix": _read("edge_mix.bin", torch.float32, 34, 16),
        "edge_offsets": _read("edge_offsets.bin", torch.float32, 16, 64, 64),
    }
    tables["mix_codes"] = fold_mix_codes(tables["channel_mix"], tables["edge_mix"])
    return {
        "count": count,
        "masks": masks,
        "values": values,
        "edges": edges.view(count, EDGE_CHANNELS, 64, 64),
        "S": _read("S.bin", torch.float32, count, 64, 64),
        "Z0": _read("Z0.bin", torch.float32, count, 16, 64, 64),
        "e0": _read("e0.bin", torch.float32, count, 16, 64, 64),
        "tables": tables,
        "gpu_tables": {name: table.cuda() for name, table in tables.items()},
    }


def _run(reference: dict, start: int, *, norm_f32: bool, state_f32: bool, seed: bool = True):
    batch = _CHUNK
    stop = min(start + batch, reference["count"])
    masks = torch.zeros((batch, 112), dtype=torch.int64).view(torch.uint64)
    values = torch.zeros((batch, 112), dtype=torch.float32)
    masks[: stop - start] = reference["masks"][start:stop]
    values[: stop - start] = reference["values"][start:stop]
    codes = torch.full((batch, 64, 64), -1, dtype=torch.int64, device="cuda").view(torch.uint64)
    launch_prologue_egt_edges(codes, masks.cuda(), values.cuda())  # K1b: rows kernel, then the packing kernel
    norm = torch.full((batch, 64, 64), -1.0, dtype=torch.float32 if norm_f32 else torch.float16, device="cuda")
    state = torch.full((batch, 16, 64, 64), -7.0, dtype=torch.float32 if state_f32 else torch.float16, device="cuda")
    if seed:
        t = reference["gpu_tables"]
        _prologue_egt_seed_kernel[_autotune_grid](
            norm, state, codes, t["offsets"], t["edge_offsets"], t["mix_codes"], batch, EPSILON, norm_f32, state_f32,
        )
    torch.cuda.synchronize()
    n = stop - start
    return stop, codes[:n], norm[:n], state[:n]


def test_edges_match_ort_bitwise(reference: dict) -> None:
    count = reference["count"]
    mismatched = torch.zeros(EDGE_CHANNELS, dtype=torch.int64)
    boards: list[int] = []
    for start in range(0, count, _CHUNK):
        stop, codes, _, _ = _run(reference, start, norm_f32=False, state_f32=True, seed=False)
        assert bool(((codes.view(torch.int64) >> EDGE_CHANNELS) == 0).all()), "bits 34-63 of a code must be zero"
        difference = unpack_edges(codes).cpu() != reference["edges"][start:stop]
        mismatched += difference.sum((0, 2, 3))
        boards += [start + int(i) for i in difference.flatten(1).any(1).nonzero().flatten()]
    print(f"K1-GATE E: {int(mismatched.sum())} mismatched cells over {count} positions x 34 channels; "
          f"per channel {mismatched.tolist()}", flush=True)
    assert not boards, f"{len(boards)} of {count} positions differ, first {boards[:10]}"


@pytest.mark.parametrize("storage", ["f32", "f16"])
def test_norm_and_seed_match_ort(reference: dict, storage: str) -> None:
    wide = storage == "f32"
    relative = _F32_RELATIVE if wide else _F16_RELATIVE
    channel_mix = reference["gpu_tables"]["channel_mix"].double()
    offsets = reference["gpu_tables"]["offsets"].double()
    worst = dict.fromkeys(("S_abs", "S_rel", "Z0_abs", "Z0_peak", "e0_abs", "e0_peak", "e0_excess", "Z0_excess"), 0.0)
    for start in range(0, reference["count"], _CHUNK):
        stop, codes, norm, state = _run(reference, start, norm_f32=wide, state_f32=wide)
        assert bool(torch.isfinite(norm).all()) and bool(torch.isfinite(state).all())
        s_ref = reference["S"][start:stop].cuda().double()
        s = norm.double()
        worst["S_abs"] = max(worst["S_abs"], float((s - s_ref).abs().max()))
        worst["S_rel"] = max(worst["S_rel"], float(((s - s_ref).abs() / s_ref).max()))
        z0 = s[:, None] * (torch.einsum("zcij,cd->zdij", unpack_edges(codes).double(), channel_mix) + offsets)
        z_ref = reference["Z0"][start:stop].cuda().double()
        dz = (z0 - z_ref).abs()
        worst["Z0_abs"] = max(worst["Z0_abs"], float(dz.max()))
        worst["Z0_peak"] = max(worst["Z0_peak"], float(z_ref.abs().max()))
        worst["Z0_excess"] = max(worst["Z0_excess"], float((dz - relative * z_ref.abs() - 1e-6).max()))
        e_ref = reference["e0"][start:stop].cuda().double()
        de = (state.double() - e_ref).abs()
        worst["e0_abs"] = max(worst["e0_abs"], float(de.max()))
        worst["e0_peak"] = max(worst["e0_peak"], float(e_ref.abs().max()))
        worst["e0_excess"] = max(worst["e0_excess"], float((de - relative * e_ref.abs() - 1e-6).max()))
    worst["Z0_rel"] = worst["Z0_abs"] / worst["Z0_peak"]
    worst["e0_rel"] = worst["e0_abs"] / worst["e0_peak"]
    print(f"K1-GATE seeds storage={storage}: " + json.dumps({k: f"{v:.3e}" for k, v in worst.items()}), flush=True)
    assert worst["S_rel"] <= relative, f"S max relative error {worst['S_rel']:.3e}"
    assert worst["Z0_excess"] <= 0.0, f"Z0 exceeds {relative:g} relative + 1e-6 by {worst['Z0_excess']:.3e}"
    assert worst["e0_excess"] <= 0.0, f"e0 exceeds {relative:g} relative + 1e-6 by {worst['e0_excess']:.3e}"
    assert worst["Z0_rel"] <= relative and worst["e0_rel"] <= relative


def test_code_rows_sum_the_channel_rows() -> None:
    torch.manual_seed(0x4B31)
    channel_mix, edge_mix = torch.randn(34, 16), torch.randn(34, 16)
    rows = fold_mix_codes(channel_mix, edge_mix).double()
    assert rows.shape == (CODE_ROWS, CODE_WIDTH)
    assert _names.EGT_CODE_GROUPS == CODE_GROUPS
    codes = torch.randint(0, 1 << EDGE_CHANNELS, (4096,), dtype=torch.int64)
    bits = ((codes[:, None] >> torch.arange(EDGE_CHANNELS)[None, :]) & 1).double()
    expected = bits @ torch.cat((channel_mix, edge_mix), dim=1).double()
    got = sum(rows[CODE_GROUP_BASES[g] + ((codes >> first) & ((1 << width) - 1))]
              for g, (first, width) in enumerate(CODE_GROUPS))
    assert float((got - expected).abs().max()) < 1e-4  # noqa: PLR2004


def test_plans_match_the_kernel() -> None:
    plans = {plan.name: plan for plan in plan_prologue_egt(**GCAP_TABLES)}
    assert set(SEED_TABLES) <= set(plans)
    assert plans["/prologue/channel_mix"].shape == (EDGE_CHANNELS, 16)
    assert plans["/prologue/edge_mix"].shape == (EDGE_CHANNELS, 16)
    assert plans["/prologue/offsets"].shape == plans["/prologue/edge_offsets"].shape == (16, 64, 64)
    assert plans["/prologue/mix_codes"].shape == (CODE_ROWS, CODE_WIDTH)
    assert all(plan.data_type == _names.FLOAT32 for plan in plans.values())


def test_carrier_tables_match_the_reference(reference: dict) -> None:
    if not _NETWORK.exists():
        pytest.skip(f"network not found at {_NETWORK}")
    from lczero_triton.lab._onnx import load_carrier  # noqa: PLC0415
    from lczero_triton.lab.carrier import prologue_egt_tables  # noqa: PLC0415

    _, graph = load_carrier(_NETWORK)
    tables = prologue_egt_tables(graph, plan_prologue_egt(**GCAP_TABLES))
    for name in ("channel_mix", "offsets", "edge_mix", "edge_offsets", "mix_codes"):
        assert torch.equal(tables[f"/prologue/{name}"], reference["tables"][name]), name


def test_compiles_to_an_lc0ex_artifact() -> None:
    specialization = PrologueEgtSpecialization(batch_count=8, architecture=_architecture())
    null = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2
    rows_artifact = compile_prologue_egt_rows(specialization.edges)
    assert rows_artifact.grid == (8, 1, 1)
    assert rows_artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 3 + null
    edges_artifact = compile_prologue_egt_edges(specialization.edges)
    assert edges_artifact.grid == (8 * EDGE_SPLIT, 1, 1)
    assert edges_artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 2 + null
    seed_artifact = compile_prologue_egt_seed(specialization)
    assert seed_artifact.grid == (8, 1, 1)
    assert seed_artifact.parameters == (lc0ex_pb2.PARAMETER_TYPE_POINTER,) * 6 + null

    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)
    sizes = buffer_bytes(specialization)
    tables = {
        plan.name: builder.persistent_buffer(
            name=plan.name, shape=plan.shape, dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, alignment_bytes=256,
        )
        for plan in plan_prologue_egt(**GCAP_TABLES)
    }
    prologue_egt(
        builder,
        kernels,
        builder.temporary_buffer(size_bytes=sizes["edges"], alignment_bytes=256),
        builder.temporary_buffer(size_bytes=sizes["edge_norm"], alignment_bytes=256),
        builder.temporary_buffer(size_bytes=sizes["edge_state"], alignment_bytes=256),
        builder.buffer(name="/input/plane_masks", shape=(8, 112), dtype=lc0ex_pb2.Buffer.DATA_TYPE_U64),
        builder.buffer(name="/input/plane_values", shape=(8, 112), dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32),
        tables,
        specialization,
    )
