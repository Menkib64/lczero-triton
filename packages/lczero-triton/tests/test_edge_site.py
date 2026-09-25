"""CUDA tests for the EGT2 edge update site (item E, K3).

``e'`` from the kernel is gated against onnxruntime evaluating the gcap export's own graph (nodes 10923, 11513 and
12103). The reference set ``ref/egt_k3`` holds 128 of K1's positions (every 8th row of ``ref/egt``). Each site is given
ORT's own state ``e`` (nodes 10331, 10923, 11513) and its block's ``H`` (10840, 11430, 12020) in K2's layout.

Gates:
* ``e'``: FP32 class, ``max|d| / max|ref| <= 1e-5`` (the map's pass rule), at batch 64 and at batch 8;
* the three sites chained from the kernel's own output: the map's end-to-end rule, 1e-4;
* the K4 split (stage "readback", then stage "ffn"): against a float64 recompute of ``e_hat``, against the one-call
  site, and against ORT;
* the carrier's site tables: bitwise equal to the reference tables;
* the artifact compile and builder call, for every stage;
* ``reverse`` (the lab's ``rev_edge``, BT6-test): stage "ffn" and the one-call site against a float64 recompute with
  a random reverse table -- no reference set carries one, and none is needed: the term is two lines of algebra.

FP16-stored ``H`` is an info row, with no gate. The reference manifest is verified once per module.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.edge_site import (
    _STAGES,
    CELLS,
    EPSILON,
    HEADS,
    HIDDEN,
    SITE_BLOCKS,
    STATES,
    EdgeSiteSpecialization,
    _autotune_grid,
    _edge_site_kernel,
    buffer_bytes,
    compile_edge_site,
    edge_site,
    site_table_names,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_REFERENCE = Path(os.environ.get("LC0EX_EGT_K3_REF", str(Path(__file__).resolve().parents[3] / "ref/egt_k3")))
_CARRIER = Path(os.environ.get(
    "LC0EX_EGT_R0_CARRIER", str(Path.home() / "spsa/lc0ex_5080/work/r20c_itemE_R0/egt2_gcap_512x15_lc0ex_r0.pb.gz")))
_F32_RELATIVE = 1e-5
_CHAINED_RELATIVE = 1e-4
_TABLE_SHAPES = ((STATES, HEADS), (HIDDEN, STATES), (HIDDEN,), (STATES, HIDDEN))


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _read(name: str, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray((_REFERENCE / name).read_bytes()), dtype=torch.float32).view(*shape)


@pytest.fixture(scope="module")
def reference() -> dict:
    header_path = _REFERENCE / "k3_ref.json"
    if not header_path.exists():
        pytest.skip(f"reference not found at {_REFERENCE}")
    for line in (_REFERENCE / "SHA256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((_REFERENCE / name).read_bytes()).hexdigest() == digest, f"{name} fails its sha256"
    header = json.loads(header_path.read_text())
    count = header["count"]
    sites = {}
    for after_block in SITE_BLOCKS:
        entry = header["sites"][f"site{after_block}"]
        assert tuple(entry["tables"]) == site_table_names(after_block), "table order is not the kernel's argument order"
        sites[after_block] = {
            "e": _read(entry["e_in"]["file"], count, STATES, 64, 64),
            "H": _read(entry["H"]["file"], count * HEADS, 64, 64),
            "e_out": _read(entry["e_out"]["file"], count, STATES, 64, 64),
            "tables": [_read(table["file"], *table["shape"]).cuda() for table in entry["tables"].values()],
        }
    return {"count": count, "sites": sites}


def _run(site: dict, states: torch.Tensor, logits: torch.Tensor, batch: int, stage: str = "site") -> torch.Tensor:
    """Run one site over every position in chunks of `batch` (the last chunk padded); returns CPU FP32."""
    count = states.shape[0]
    results = torch.empty_like(states)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        filled = stop - start
        state = torch.zeros((batch, STATES, 64, 64), dtype=torch.float32, device="cuda")
        state[:filled] = states[start:stop]
        heads = torch.zeros((batch * HEADS, 64, 64), dtype=torch.float32, device="cuda")
        heads[: filled * HEADS] = logits[start * HEADS: stop * HEADS]
        output = torch.full_like(state, float("nan"))
        # The eighth pointer is the reverse-edge table; with `reverse` False it is never read.
        _edge_site_kernel[_autotune_grid](
            output, state, heads, *site["tables"], site["tables"][1], batch, HEADS, STATES, HIDDEN, _STAGES[stage],
            EPSILON, False,
        )
        torch.cuda.synchronize()
        results[start:stop] = output[:filled].cpu()
    assert bool(torch.isfinite(results).all()), "a cell was not written or is not finite"
    return results


def _error(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (got.double() - expected.double()).abs().max().item()
    return difference, difference / expected.double().abs().max().item()


@pytest.mark.parametrize("batch", [64, 8])
@pytest.mark.parametrize("after_block", SITE_BLOCKS)
def test_site_matches_ort(reference: dict, after_block: int, batch: int) -> None:
    site = reference["sites"][after_block]
    got = _run(site, site["e"], site["H"], batch)
    max_abs, relative = _error(got, site["e_out"])
    print(f"K3-GATE site{after_block} batch {batch} e' (FP32 H): max_abs {max_abs:.3e} rel {relative:.3e}", flush=True)
    if batch == 64:  # noqa: PLR2004
        rounded = _run(site, site["e"], site["H"].half().float(), batch)
        info_abs, info_relative = _error(rounded, site["e_out"])
        print(f"K3-INFO site{after_block} batch {batch} e' (FP16-stored H, no gate): max_abs {info_abs:.3e} "
              f"rel {info_relative:.3e}", flush=True)
    assert relative <= _F32_RELATIVE, f"site {after_block}: e' relative error {relative:.3e}"


def test_sites_chained_from_the_kernel(reference: dict) -> None:
    state = reference["sites"][SITE_BLOCKS[0]]["e"]
    for after_block in SITE_BLOCKS:
        site = reference["sites"][after_block]
        state = _run(site, state, site["H"], 64)
        max_abs, relative = _error(state, site["e_out"])
        print(f"K3-GATE chained through site{after_block}: max_abs {max_abs:.3e} rel {relative:.3e}", flush=True)
        assert relative <= _CHAINED_RELATIVE


@pytest.mark.parametrize("after_block", SITE_BLOCKS)
def test_k4_split_matches_float64_and_the_site(reference: dict, after_block: int) -> None:
    site = reference["sites"][after_block]
    e_hat = _run(site, site["e"], site["H"], 64, stage="readback")
    readback = site["tables"][0].double().cpu()
    expected = site["e"].double() + torch.einsum(
        "ch,zhij->zcij", readback, site["H"].double().view(-1, HEADS, 64, 64))
    max_abs, relative = _error(e_hat, expected)
    print(f"K3-GATE site{after_block} stage readback vs float64 e_hat: max_abs {max_abs:.3e} rel {relative:.3e}")
    assert relative <= _F32_RELATIVE
    split = _run(site, e_hat, site["H"], 64, stage="ffn")
    fused = _run(site, site["e"], site["H"], 64)
    split_abs, split_relative = _error(split, fused)
    ort_abs, ort_relative = _error(split, site["e_out"])
    print(f"K3-GATE site{after_block} readback+ffn vs site: max_abs {split_abs:.3e} rel {split_relative:.3e}; "
          f"vs ORT: max_abs {ort_abs:.3e} rel {ort_relative:.3e}", flush=True)
    assert split_relative <= _F32_RELATIVE and ort_relative <= _F32_RELATIVE


def test_carrier_tables_match_the_reference(reference: dict) -> None:
    if not _CARRIER.exists():
        pytest.skip(f"carrier not found at {_CARRIER}")
    from lczero_triton.lab._onnx import load_carrier  # noqa: PLC0415
    from lczero_triton.lab.carrier import _as_float32  # noqa: PLC0415

    _, graph = load_carrier(_CARRIER)
    for after_block in SITE_BLOCKS:
        for name, table in zip(site_table_names(after_block), reference["sites"][after_block]["tables"], strict=True):
            assert name in graph.initializers, f"{name} is not in the carrier"
            assert torch.equal(_as_float32(graph, name).reshape(-1), table.cpu().reshape(-1)), name


@pytest.mark.parametrize("stage", ["site", "readback", "ffn"])
def test_compiles_to_an_lc0ex_artifact(stage: str) -> None:
    specialization = EdgeSiteSpecialization(batch_count=8, architecture=_architecture(), stage=stage)
    started = time.perf_counter()
    artifact = compile_edge_site(specialization)
    seconds = time.perf_counter() - started
    pointer, null = lc0ex_pb2.PARAMETER_TYPE_POINTER, lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
    expected = {
        "site": (pointer,) * 7,
        "readback": (pointer,) * 4 + (null,) * 3,
        "ffn": (pointer, pointer, null, null, pointer, pointer, pointer),
    }[stage]
    # Slot eight is the reverse-edge table: NULL without `reverse`. Then Triton's two scratch slots.
    assert artifact.parameters == expected + (null,) + (null, null)
    assert artifact.grid[1:] == (1, 1) and (8 * CELLS) % artifact.grid[0] == 0
    print(f"K3-COMPILE stage {stage} batch 8: {seconds:.1f} s, grid {artifact.grid}, block {artifact.block}, "
          f"cubin {len(artifact.binary_data)} B", flush=True)

    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)
    sizes = buffer_bytes(specialization)
    assert sizes["output"] == 4 * STATES * 8 * CELLS and sizes["hidden"] == 0
    tables = {
        name: builder.persistent_buffer(name=name, shape=shape, dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
                                        alignment_bytes=256)
        for name, shape in zip(site_table_names(3), _TABLE_SHAPES, strict=True)
    }
    output = builder.temporary_buffer(size_bytes=sizes["output"], alignment_bytes=256)
    state = builder.temporary_buffer(size_bytes=sizes["state"], alignment_bytes=256)
    logits = None if stage == "ffn" else builder.temporary_buffer(size_bytes=sizes["logits"], alignment_bytes=256)
    edge_site(builder, kernels, output, state, logits, tables, specialization, after_block=3)
    with pytest.raises(ValueError, match="separate buffer"):
        edge_site(builder, kernels, state, state, logits, tables, specialization, after_block=3)


def _rms64(values: torch.Tensor) -> torch.Tensor:
    return values / torch.sqrt((values * values).mean(dim=1, keepdim=True) + EPSILON)


@pytest.mark.parametrize("stage", ["ffn", "site"])
@pytest.mark.parametrize("heads", [HEADS, 16])
def test_reverse_edge_matches_float64(stage: str, heads: int) -> None:
    """`rev_edge`: hidden = relu(W1 rms(e_hat)[i, j] + b1 + Wr rms(e_hat)[j, i]); e' = rms(e_hat + W2 hidden)."""
    batch = 8
    generator = torch.Generator(device="cpu").manual_seed(20260921)
    state = torch.randn((batch, STATES, 64, 64), generator=generator, dtype=torch.float32)
    logits = torch.randn((batch * heads, 64, 64), generator=generator, dtype=torch.float32)
    readback = 0.1 * torch.randn((STATES, heads), generator=generator, dtype=torch.float32)
    dense1 = 0.3 * torch.randn((HIDDEN, STATES), generator=generator, dtype=torch.float32)
    bias1 = 0.1 * torch.randn((HIDDEN,), generator=generator, dtype=torch.float32)
    dense2 = 0.3 * torch.randn((STATES, HIDDEN), generator=generator, dtype=torch.float32)
    reverse = 0.3 * torch.randn((HIDDEN, STATES), generator=generator, dtype=torch.float32)

    e_hat = state.double()
    if stage == "site":
        e_hat = e_hat + torch.einsum("ch,zhij->zcij", readback.double(), logits.double().view(batch, heads, 64, 64))
    normed = _rms64(e_hat)
    hidden = (torch.einsum("dc,zcij->zdij", dense1.double(), normed) + bias1.double()[None, :, None, None]
              + torch.einsum("dc,zcij->zdij", reverse.double(), normed.transpose(2, 3)))
    expected = _rms64(e_hat + torch.einsum("cd,zdij->zcij", dense2.double(), torch.relu(hidden)))
    without = _rms64(e_hat + torch.einsum("cd,zdij->zcij", dense2.double(), torch.relu(
        torch.einsum("dc,zcij->zdij", dense1.double(), normed) + bias1.double()[None, :, None, None])))

    output = torch.full((batch, STATES, 64, 64), float("nan"), dtype=torch.float32, device="cuda")
    _edge_site_kernel[_autotune_grid](
        output, state.cuda(), logits.cuda(), readback.cuda(), dense1.cuda(), bias1.cuda(), dense2.cuda(),
        reverse.cuda(), batch, heads, STATES, HIDDEN, _STAGES[stage], EPSILON, True,
    )
    torch.cuda.synchronize()
    got = output.cpu()
    assert bool(torch.isfinite(got).all()), "a cell was not written or is not finite"
    max_abs, relative = _error(got, expected)
    moved = (expected - without).abs().max().item()
    print(f"REVEDGE-GATE stage {stage} heads {heads}: max_abs {max_abs:.3e} rel {relative:.3e}; the reverse term "
          f"moves e' by up to {moved:.3e}", flush=True)
    assert moved > 1e-2, "the test's reverse term is too small to detect a kernel that ignores it"
    assert relative <= _F32_RELATIVE


def test_reverse_edge_artifact_takes_the_fifth_table() -> None:
    specialization = EdgeSiteSpecialization(batch_count=8, architecture=_architecture(), stage="ffn", reverse=True)
    artifact = compile_edge_site(specialization)
    pointer, null = lc0ex_pb2.PARAMETER_TYPE_POINTER, lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
    assert artifact.parameters == (pointer, pointer, null, null, pointer, pointer, pointer, pointer, null, null)
    assert site_table_names(3, reverse=True)[-1] == "/encoder3/edge_site/ffn/dense1_rev/w"
    with pytest.raises(ValueError, match="readback"):
        EdgeSiteSpecialization(batch_count=8, architecture=120, stage="readback", reverse=True)


def test_specialization_rejects_bad_widths() -> None:
    with pytest.raises(ValueError, match="power of two"):
        EdgeSiteSpecialization(batch_count=8, architecture=120, hidden=48)
    with pytest.raises(ValueError, match="unknown stage"):
        EdgeSiteSpecialization(batch_count=8, architecture=120, stage="triplet")  # type: ignore[arg-type]
