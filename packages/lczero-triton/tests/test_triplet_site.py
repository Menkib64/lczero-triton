"""CUDA tests for the EGT2 triplet operator (item E K4; round 22 K4b: the fused forms).

``e_hat2`` from the kernel is gated against onnxruntime evaluating each triplet export's own graph (nodes 11038,
11736 and 12434 on `triplet_path`, the same ids on `triplet_ag`), for **both** contraction orders at all three
sites. The reference set ``ref/egt_k4`` holds 64 of K1's positions (every 16th row of ``ref/egt``), with each
site's input state ``e``, its block's ``H`` in K2's layout, ``e_hat`` (K3's readback output, this kernel's input),
``a_in``/``a_out``, ``e_hat2`` and ``e'``.

Gates, per site and variant:
* ``e_hat2``: FP32 class, ``max|d| / max|ref| <= 1e-5`` (the map's single-site rule), at batch 64 and batch 8;
* ``a_in`` and ``a_out``, rebuilt in float64 from the kernel's own ``gates`` buffer: the same rule. This is what
  pins the rms, the gate projection and bias, the split order ``e_in, g_in, e_out, g_out`` and the two softmax
  axes (last axis inward, row axis outward);
* the K3 chain end to end at every site: ``edge_site`` stage "readback" -> this triplet -> ``edge_site`` stage
  "ffn", from ORT's ``e`` and ``H``, against ORT's ``e'``;
* the contraction flag: the wrong order must miss ORT by orders of magnitude (so the flag is exercised, not inert);
* the four triplet tables planned from the export by R0's reader: bitwise equal to the reference tables;
* the artifact compile and the builder call, for every stage and form.

K4b adds, per site and variant:
* step (a) ``readback_prep``: ``e_hat``, ``values``, ``gates`` and hence ``e_hat2`` **bit-identical** to the split
  chain (``edge_site`` stage "readback" then ``prep``) -- the dots are the same expressions -- and in class vs ORT;
* step (b) ``fused`` in K4's class (``dot="ieee"``): ``e_hat2`` in class vs ORT at
  batch 64 and 8, ``va`` in class vs the scratch chain's (not bit-identical: the rms scale is applied after the
  channel sum, ``W rms(x) = inv (W x)``), the flag exercised, every autotune candidate (group, warps) in class on
  the same reference, and the K3 chain around it;
* step (b) in the served class (``dot="fp16"``: A and V rounded once, FP32 accumulation on the tensor cores; and
  ``state_f16``: e_hat read from the FP16 copy `triplet_readback` writes): ``e_hat2`` and ``e'`` within the
  served rule (5e-3 vs ORT) for every candidate, and `triplet_readback`'s e_hat bit-identical to K3's readback
  with its copy exactly ``e_hat.half()``.

The reference manifest is verified once per module.
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
    HEADS,
    HIDDEN,
    EdgeSiteSpecialization,
    _edge_site_kernel,
    site_table_names,
)
from lczero_triton.bt4.kernels.edge_site import _autotune_grid as _site_grid
from lczero_triton.bt4.kernels.triplet_site import (
    _DOTS,
    _FUSED_CONFIGURATIONS,
    CELLS,
    EPSILON,
    SITE_BLOCKS,
    STATES,
    TRIPLET_DOTS,
    TRIPLET_HEADS,
    TripletSiteSpecialization,
    _contract_grid,
    _fused_programs,
    _planar_grid,
    _triplet_contract_kernel,
    _triplet_fused_body,
    _triplet_fused_kernel,
    _triplet_out_ffn_kernel,
    _triplet_out_kernel,
    _triplet_prep_kernel,
    _triplet_readback_kernel,
    _triplet_readback_prep_kernel,
    buffer_bytes,
    compile_triplet_contract,
    compile_triplet_fused,
    compile_triplet_out,
    compile_triplet_out_ffn,
    compile_triplet_prep,
    compile_triplet_readback,
    compile_triplet_readback_prep,
    launch_triplet_fused,
    ffn_table_names,
    readback_table_name,
    triplet_out_ffn,
    triplet_readback,
    triplet_site,
    triplet_site_with_readback,
    triplet_table_names,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_REFERENCE = Path(os.environ.get("LC0EX_EGT_K4_REF", str(Path(__file__).resolve().parents[3] / "ref/egt_k4")))
_NET_DIRECTORY = Path(os.environ.get("LC0EX_EGT_NETS", str(Path.home() / "spsa/lc0ex_5080/work/nets_r20")))
_VARIANTS = ("path", "ag")
_F32_RELATIVE = 1e-5
_ULP_CLASS = 1e-6  # the same arithmetic in another reduction order: an ulp or two, not a different program
_SERVED_RELATIVE = 5e-3  # the served rule (K2c section 3): the FP16-operand contraction and the FP16 e_hat read
_FLAG_MINIMUM = 1e-2  # the wrong contraction must miss ORT by at least this, or the flag is inert
_TRIPLET_SHAPES = {"triplet/value/w": (2 * STATES, STATES), "triplet/gate/w": (STATES, STATES),
                   "triplet/gate/b": (STATES,), "triplet/out/w": (STATES, 2 * STATES)}


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _read(name: str, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray((_REFERENCE / name).read_bytes()), dtype=torch.float32).view(*shape)


@pytest.fixture(scope="module")
def reference() -> dict:
    header_path = _REFERENCE / "k4_ref.json"
    if not header_path.exists():
        pytest.skip(f"reference not found at {_REFERENCE}")
    for line in (_REFERENCE / "SHA256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((_REFERENCE / name).read_bytes()).hexdigest() == digest, f"{name} fails its sha256"
    header = json.loads(header_path.read_text())
    count = header["count"]
    sites = {}
    for key, entry in header["sites"].items():
        assert entry["contraction"] == entry["variant"], f"{key}: the reference's own contraction flag disagrees"
        dumps = {name: _read(dump["file"], *dump["shape"]) for name, dump in entry["dumps"].items()}
        dumps["H"] = dumps["H"].reshape(count * HEADS, 64, 64)
        tables = {name: _read(table["file"], *table["shape"]).cuda() for name, table in entry["tables"].items()}
        assert set(triplet_table_names(entry["after_block"])) <= set(tables), f"{key}: a triplet table is missing"
        sites[(entry["variant"], entry["after_block"])] = {"dumps": dumps, "tables": tables}
    return {"count": count, "sites": sites, "header": header}


def _triplet_tables(site: dict, after_block: int) -> list[torch.Tensor]:
    return [site["tables"][name] for name in triplet_table_names(after_block)]


def _chunks(count: int, batch: int):
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        yield start, stop, stop - start


def _padded(source: torch.Tensor, start: int, stop: int, batch: int) -> torch.Tensor:
    """A zero-padded batch of `source[start:stop]` on the GPU (the reference has 64 positions)."""
    chunk = torch.zeros((batch, *source.shape[1:]), dtype=torch.float32, device="cuda")
    chunk[: stop - start] = source[start:stop]
    return chunk


def _run_triplet(site: dict, after_block: int, contraction: str, e_hat: torch.Tensor, batch: int) -> dict:
    """Run prep, contract and out over every position in chunks of `batch`; return e_hat2, gates and va."""
    count = e_hat.shape[0]
    flag = 1 if contraction == "ag" else 0
    value_weight, gate_weight, gate_bias, output_weight = _triplet_tables(site, after_block)
    results = torch.empty_like(e_hat)
    all_gates = torch.empty_like(e_hat)
    all_va = torch.empty((count, 2 * STATES, 64, 64), dtype=torch.float32)
    for start, stop, filled in _chunks(count, batch):
        state = _padded(e_hat, start, stop, batch)
        values = torch.full((batch, 2 * STATES, 64, 64), float("nan"), dtype=torch.float32, device="cuda")
        gates = torch.full_like(state, float("nan"))
        _triplet_prep_kernel[_planar_grid](values, gates, state, value_weight, gate_weight, gate_bias, batch,
                                           STATES, EPSILON)
        _triplet_contract_kernel[_contract_grid](values, gates, batch, STATES, TRIPLET_HEADS, TRIPLET_DOTS, flag)
        all_va[start:stop] = values[:filled].cpu()
        _triplet_out_kernel[_planar_grid](state, values, output_weight, batch, STATES)
        torch.cuda.synchronize()
        results[start:stop] = state[:filled].cpu()
        all_gates[start:stop] = gates[:filled].cpu()
    assert bool(torch.isfinite(results).all()), "a cell of e_hat2 was not written or is not finite"
    assert bool(torch.isfinite(all_gates).all()), "a cell of the gates buffer was not written or is not finite"
    return {"e_hat2": results, "gates": all_gates, "va": all_va}


def _run_fused(  # noqa: PLR0913
    site: dict, after_block: int, contraction: str, e_hat: torch.Tensor, batch: int, dot: str = "ieee",
    state_f16: bool = False, configuration: tuple[int, int] | None = None,
) -> dict:
    """K4b (b): run fused then out over every position in chunks of `batch`; return e_hat2 and va.

    `configuration` = (group, warps) launches the body directly with that candidate; None runs the autotuner.
    With `state_f16` the launches read ``e_hat.half()``, as `triplet_readback` would have written it.
    """
    count = e_hat.shape[0]
    flag = 1 if contraction == "ag" else 0
    specialization = TripletSiteSpecialization(batch_count=batch, architecture=_architecture(),
                                               contraction=contraction, form="fused", dot=dot, state_f16=state_f16)
    value_weight, gate_weight, gate_bias, output_weight = _triplet_tables(site, after_block)
    results = torch.empty_like(e_hat)
    all_va = torch.empty((count, 2 * STATES, 64, 64), dtype=torch.float32)
    for start, stop, filled in _chunks(count, batch):
        state = _padded(e_hat, start, stop, batch)
        source = state.half() if state_f16 else state
        va = torch.full((batch, 2 * STATES, 64, 64), float("nan"), dtype=torch.float32, device="cuda")
        if configuration is None:
            launch_triplet_fused(va, source, value_weight, gate_weight, gate_bias, specialization)
        else:
            group, warps = configuration
            grid = (_fused_programs(batch, TRIPLET_HEADS, TRIPLET_DOTS, group),)
            _triplet_fused_body[grid](va, source, value_weight, gate_weight, gate_bias, batch, STATES, TRIPLET_HEADS,
                                      TRIPLET_DOTS, flag, EPSILON, _DOTS[dot], group=group, num_warps=warps)
        torch.cuda.synchronize()
        all_va[start:stop] = va[:filled].cpu()
        _triplet_out_kernel[_planar_grid](state, va, output_weight, batch, STATES)
        torch.cuda.synchronize()
        results[start:stop] = state[:filled].cpu()
    assert bool(torch.isfinite(all_va).all()), "a cell of va was not written or is not finite"
    assert bool(torch.isfinite(results).all()), "a cell of e_hat2 was not written or is not finite"
    return {"e_hat2": results, "va": all_va}


def _run_readback(site: dict, after_block: int, e_in: torch.Tensor, logits: torch.Tensor, batch: int) -> dict:
    """`triplet_readback`: K3's readback plus the FP16 copy, over every position; e_hat (FP32) and copy (FP16)."""
    count = e_in.shape[0]
    readback = site["tables"][readback_table_name(after_block)]
    e_hat = torch.empty_like(e_in)
    copies = torch.empty(e_in.shape, dtype=torch.float16)
    for start, stop, filled in _chunks(count, batch):
        state = _padded(e_in, start, stop, batch)
        heads = torch.zeros((batch * HEADS, 64, 64), dtype=torch.float32, device="cuda")
        heads[: filled * HEADS] = logits[start * HEADS: stop * HEADS]
        output = torch.full_like(state, float("nan"))
        copy = torch.full(state.shape, float("nan"), dtype=torch.float16, device="cuda")
        _triplet_readback_kernel[_planar_grid](output, copy, state, heads, readback, batch, HEADS, STATES)
        torch.cuda.synchronize()
        e_hat[start:stop] = output[:filled].cpu()
        copies[start:stop] = copy[:filled].cpu()
    assert bool(torch.isfinite(e_hat).all()) and bool(torch.isfinite(copies).all()), "readback left a cell unwritten"
    return {"e_hat": e_hat, "copy": copies}


def _run_out_ffn(site: dict, after_block: int, e_hat: torch.Tensor, va: torch.Tensor, batch: int) -> torch.Tensor:
    """K4b (c): `out` folded into the FFN stage, over every position; return e'."""
    count = e_hat.shape[0]
    _, _, _, output_weight = _triplet_tables(site, after_block)
    dense1_weight, dense1_bias, dense2_weight = (site["tables"][name] for name in ffn_table_names(after_block))
    results = torch.empty_like(e_hat)
    for start, stop, filled in _chunks(count, batch):
        state = _padded(e_hat, start, stop, batch)
        chunk_va = _padded(va, start, stop, batch)
        output = torch.full_like(state, float("nan"))
        _triplet_out_ffn_kernel[_planar_grid](output, state, chunk_va, output_weight, dense1_weight, dense1_bias,
                                              dense2_weight, batch, STATES, HIDDEN, EPSILON)
        torch.cuda.synchronize()
        results[start:stop] = output[:filled].cpu()
    assert bool(torch.isfinite(results).all()), "out_ffn left a cell unwritten"
    return results


def _run_readback_prep(site: dict, after_block: int, e_in: torch.Tensor, logits: torch.Tensor,
                       batch: int) -> dict:
    """K4b (a): run readback_prep over every position; return e_hat, values and gates as written."""
    count = e_in.shape[0]
    value_weight, gate_weight, gate_bias, _ = _triplet_tables(site, after_block)
    readback = site["tables"][readback_table_name(after_block)]
    e_hat = torch.empty_like(e_in)
    all_values = torch.empty((count, 2 * STATES, 64, 64), dtype=torch.float32)
    all_gates = torch.empty_like(e_in)
    for start, stop, filled in _chunks(count, batch):
        state = _padded(e_in, start, stop, batch)
        heads = torch.zeros((batch * HEADS, 64, 64), dtype=torch.float32, device="cuda")
        heads[: filled * HEADS] = logits[start * HEADS: stop * HEADS]
        output = torch.full_like(state, float("nan"))
        values = torch.full((batch, 2 * STATES, 64, 64), float("nan"), dtype=torch.float32, device="cuda")
        gates = torch.full_like(state, float("nan"))
        _triplet_readback_prep_kernel[_planar_grid](values, gates, output, state, heads, readback, value_weight,
                                                    gate_weight, gate_bias, batch, HEADS, STATES, EPSILON)
        torch.cuda.synchronize()
        e_hat[start:stop] = output[:filled].cpu()
        all_values[start:stop] = values[:filled].cpu()
        all_gates[start:stop] = gates[:filled].cpu()
    for name, tensor in (("e_hat", e_hat), ("values", all_values), ("gates", all_gates)):
        assert bool(torch.isfinite(tensor).all()), f"readback_prep left a cell of {name} unwritten"
    return {"e_hat": e_hat, "values": all_values, "gates": all_gates}


def _run_prep(site: dict, after_block: int, e_hat: torch.Tensor, batch: int) -> dict:
    """K4's prep alone over every position; return values and gates as written."""
    count = e_hat.shape[0]
    value_weight, gate_weight, gate_bias, _ = _triplet_tables(site, after_block)
    all_values = torch.empty((count, 2 * STATES, 64, 64), dtype=torch.float32)
    all_gates = torch.empty_like(e_hat)
    for start, stop, filled in _chunks(count, batch):
        state = _padded(e_hat, start, stop, batch)
        values = torch.full((batch, 2 * STATES, 64, 64), float("nan"), dtype=torch.float32, device="cuda")
        gates = torch.full_like(state, float("nan"))
        _triplet_prep_kernel[_planar_grid](values, gates, state, value_weight, gate_weight, gate_bias, batch,
                                           STATES, EPSILON)
        torch.cuda.synchronize()
        all_values[start:stop] = values[:filled].cpu()
        all_gates[start:stop] = gates[:filled].cpu()
    return {"values": all_values, "gates": all_gates}


def _run_contract_out(site: dict, after_block: int, contraction: str, e_hat: torch.Tensor,
                      values: torch.Tensor, gates: torch.Tensor, batch: int) -> torch.Tensor:
    """K4's contract and out from given scratch buffers; return e_hat2."""
    count = e_hat.shape[0]
    flag = 1 if contraction == "ag" else 0
    _, _, _, output_weight = _triplet_tables(site, after_block)
    results = torch.empty_like(e_hat)
    for start, stop, filled in _chunks(count, batch):
        state = _padded(e_hat, start, stop, batch)
        chunk_values = _padded(values, start, stop, batch)
        chunk_gates = _padded(gates, start, stop, batch)
        _triplet_contract_kernel[_contract_grid](chunk_values, chunk_gates, batch, STATES, TRIPLET_HEADS,
                                                 TRIPLET_DOTS, flag)
        _triplet_out_kernel[_planar_grid](state, chunk_values, output_weight, batch, STATES)
        torch.cuda.synchronize()
        results[start:stop] = state[:filled].cpu()
    return results


def _run_site(site: dict, after_block: int, state: torch.Tensor, logits: torch.Tensor, batch: int,
              stage: str) -> torch.Tensor:
    """Run one `edge_site` stage ("readback" or "ffn") over every position, in K3's own harness shape."""
    count = state.shape[0]
    tables = [site["tables"][name] for name in site_table_names(after_block)]
    results = torch.empty_like(state)
    for start, stop, filled in _chunks(count, batch):
        chunk = _padded(state, start, stop, batch)
        heads = torch.zeros((batch * HEADS, 64, 64), dtype=torch.float32, device="cuda")
        if logits is not None:
            heads[: filled * HEADS] = logits[start * HEADS: stop * HEADS]
        output = torch.full_like(chunk, float("nan"))
        # The eighth pointer is the reverse-edge table (`rev_edge`); these exports have none, so it is never read.
        _edge_site_kernel[_site_grid](output, chunk, heads, *tables, tables[1], batch, HEADS, STATES, HIDDEN,
                                      _STAGES[stage], EPSILON, False)
        torch.cuda.synchronize()
        results[start:stop] = output[:filled].cpu()
    assert bool(torch.isfinite(results).all()), f"stage {stage} left a cell unwritten"
    return results


def _error(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (got.double() - expected.double()).abs().max().item()
    return difference, difference / expected.double().abs().max().item()


def _weights_from_gates(gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild a_in and a_out in float64 from the kernel's own `gates`, in the export's split order."""
    logit_in, door_in, logit_out, door_out = torch.split(gates.double(), TRIPLET_HEADS, dim=1)
    a_in = torch.softmax(logit_in, -1) * torch.sigmoid(door_in)
    a_out = torch.softmax(logit_out, -2) * torch.sigmoid(door_out)
    return a_in, a_out


# ----------------------------------------------------------------------------------------------- K4: the scratch chain
@pytest.mark.parametrize("batch", [64, 8])
@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_e_hat2_matches_ort(reference: dict, variant: str, after_block: int, batch: int) -> None:
    site = reference["sites"][(variant, after_block)]
    got = _run_triplet(site, after_block, variant, site["dumps"]["e_hat"], batch)
    max_abs, relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    print(f"K4-GATE {variant} site{after_block} batch {batch} e_hat2: max_abs {max_abs:.3e} rel {relative:.3e}",
          flush=True)
    assert relative <= _F32_RELATIVE, f"{variant} site {after_block}: e_hat2 relative error {relative:.3e}"


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_softmax_gate_weights_match_ort(reference: dict, variant: str, after_block: int) -> None:
    site = reference["sites"][(variant, after_block)]
    got = _run_triplet(site, after_block, variant, site["dumps"]["e_hat"], 64)
    a_in, a_out = _weights_from_gates(got["gates"])
    for name, value in (("a_in", a_in), ("a_out", a_out)):
        max_abs, relative = _error(value, site["dumps"][name])
        print(f"K4-GATE {variant} site{after_block} {name}: max_abs {max_abs:.3e} rel {relative:.3e}", flush=True)
        assert relative <= _F32_RELATIVE, f"{variant} site {after_block}: {name} relative error {relative:.3e}"


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_chain_readback_triplet_ffn(reference: dict, variant: str, after_block: int) -> None:
    """K3's split contract end to end: readback writes e_hat, the triplet adds its branch, the FFN closes the site."""
    site = reference["sites"][(variant, after_block)]
    e_hat = _run_site(site, after_block, site["dumps"]["e_in"], site["dumps"]["H"], 64, "readback")
    hat_abs, hat_relative = _error(e_hat, site["dumps"]["e_hat"])
    got = _run_triplet(site, after_block, variant, e_hat, 64)
    hat2_abs, hat2_relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    e_new = _run_site(site, after_block, got["e_hat2"], None, 64, "ffn")
    new_abs, new_relative = _error(e_new, site["dumps"]["e_new"])
    print(f"K4-GATE {variant} site{after_block} chain: e_hat {hat_abs:.3e}/{hat_relative:.3e}, "
          f"e_hat2 {hat2_abs:.3e}/{hat2_relative:.3e}, e' {new_abs:.3e}/{new_relative:.3e}", flush=True)
    assert hat_relative <= _F32_RELATIVE and hat2_relative <= _F32_RELATIVE and new_relative <= _F32_RELATIVE


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_the_contraction_flag_is_exercised(reference: dict, variant: str, after_block: int) -> None:
    """Running the other contraction order on the same net must miss ORT by orders of magnitude."""
    site = reference["sites"][(variant, after_block)]
    wrong = "ag" if variant == "path" else "path"
    got = _run_triplet(site, after_block, wrong, site["dumps"]["e_hat"], 64)
    _, relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    print(f"K4-GATE {variant} site{after_block} with contraction {wrong}: rel {relative:.3e} (must be large)",
          flush=True)
    assert relative >= _FLAG_MINIMUM, f"the {wrong} flag changes nothing at {variant} site {after_block}"


@pytest.mark.parametrize("variant", _VARIANTS)
def test_planned_tables_match_the_reference(reference: dict, variant: str) -> None:
    """R0's reader must plan the four triplet tables exactly as the reference stores them ([out, in], FP32)."""
    net = _NET_DIRECTORY / f"egt2_triplet_{variant}_512x15_50000_vw.pb.gz"
    if not net.exists():
        pytest.skip(f"export not found at {net}")
    from lczero_triton.lab._mapping import read_network  # noqa: PLC0415
    from lczero_triton.lab._names import plan_network  # noqa: PLC0415
    from lczero_triton.lab._onnx import FLOAT32, load_carrier  # noqa: PLC0415
    from lczero_triton.lab.carrier import _build_payload  # noqa: PLC0415

    _, graph = load_carrier(net)
    network = read_network(graph)
    assert network.egt is not None and [site.after_block for site in network.egt.sites] == list(SITE_BLOCKS)
    assert [site.triplet.contraction for site in network.egt.sites] == [variant] * len(SITE_BLOCKS)
    plans = {plan.name: plan for plan in plan_network(network)}
    for after_block in SITE_BLOCKS:
        site = reference["sites"][(variant, after_block)]
        for name in triplet_table_names(after_block):
            plan = plans[name]
            assert plan.data_type == FLOAT32, f"{name} is not planned FP32"
            leaf = name.split("/edge_site/")[1]
            assert tuple(plan.shape) == _TRIPLET_SHAPES[leaf], (name, plan.shape)
            payload, _ = _build_payload(graph, plan)
            planned = torch.frombuffer(bytearray(payload), dtype=torch.float32)
            assert torch.equal(planned, site["tables"][name].cpu().reshape(-1)), name
    print(f"K4-GATE {variant}: 12 planned triplet tables are bitwise equal to the reference", flush=True)


@pytest.mark.parametrize("contraction", list(_VARIANTS))
def test_compiles_to_an_lc0ex_artifact(contraction: str) -> None:
    specialization = TripletSiteSpecialization(batch_count=8, architecture=_architecture(), contraction=contraction)
    pointer, null = lc0ex_pb2.PARAMETER_TYPE_POINTER, lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
    for name, compile_stage, pointers in (("prep", compile_triplet_prep, 6),
                                          ("contract", compile_triplet_contract, 2),
                                          ("out", compile_triplet_out, 3)):
        started = time.perf_counter()
        artifact = compile_stage(specialization)
        seconds = time.perf_counter() - started
        assert artifact.parameters[:pointers] == (pointer,) * pointers
        assert set(artifact.parameters[pointers:]) <= {null}
        assert artifact.grid[1:] == (1, 1) and artifact.grid[0] > 0
        print(f"K4-COMPILE {contraction} stage {name} batch 8: {seconds:.1f} s, grid {artifact.grid}, "
              f"block {artifact.block}, cubin {len(artifact.binary_data)} B", flush=True)

    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)
    sizes = buffer_bytes(specialization)
    assert sizes["values"] == 4 * 2 * STATES * 8 * CELLS and sizes["va"] == 0
    tables = {
        name: builder.persistent_buffer(name=name, shape=_TRIPLET_SHAPES[name.split("/edge_site/")[1]],
                                        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, alignment_bytes=256)
        for name in triplet_table_names(3)
    }
    state = builder.temporary_buffer(size_bytes=sizes["state"], alignment_bytes=256)
    values = builder.temporary_buffer(size_bytes=sizes["values"], alignment_bytes=256)
    gates = builder.temporary_buffer(size_bytes=sizes["gates"], alignment_bytes=256)
    triplet_site(builder, kernels, state, values, gates, tables, specialization, after_block=3)
    with pytest.raises(ValueError, match="distinct buffers"):
        triplet_site(builder, kernels, state, state, gates, tables, specialization, after_block=3)
    with pytest.raises(ValueError, match="gates scratch"):
        triplet_site(builder, kernels, state, values, None, tables, specialization, after_block=3)


def test_specialization_rejects_bad_widths() -> None:
    with pytest.raises(ValueError, match="power of two"):
        TripletSiteSpecialization(batch_count=8, architecture=120, dots=3)
    with pytest.raises(ValueError, match="must equal"):
        TripletSiteSpecialization(batch_count=8, architecture=120, heads=2)
    with pytest.raises(ValueError, match="unknown contraction"):
        TripletSiteSpecialization(batch_count=8, architecture=120, contraction="triplet")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown form"):
        TripletSiteSpecialization(batch_count=8, architecture=120, form="split")  # type: ignore[arg-type]


# ------------------------------------------------------------------------- K4b (a): readback fused into prep
@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_readback_prep_is_bit_identical_to_the_split_chain(reference: dict, variant: str, after_block: int) -> None:
    """(a): e_hat, values, gates and e_hat2 must equal the split chain's to the bit, and e_hat2 must be in class."""
    site = reference["sites"][(variant, after_block)]
    e_in, logits = site["dumps"]["e_in"], site["dumps"]["H"]
    split_e_hat = _run_site(site, after_block, e_in, logits, 64, "readback")
    split_prep = _run_prep(site, after_block, split_e_hat, 64)
    fused = _run_readback_prep(site, after_block, e_in, logits, 64)
    # The readback dot is K3's expression and must reproduce e_hat to the bit. The prep dots then run on the dot's
    # output layout instead of a load layout, and the rms reduction (`tl.sum` over the 16 channels) is ordered by
    # the layout, so `values` and `gates` may differ from the standalone prep by an ulp or two (K4b report).
    for name, got, expected, exact in (("e_hat", fused["e_hat"], split_e_hat, True),
                                       ("values", fused["values"], split_prep["values"], False),
                                       ("gates", fused["gates"], split_prep["gates"], False)):
        equal = torch.equal(got, expected)
        max_abs, relative = _error(got, expected)
        print(f"K4B-GATE(a) {variant} site{after_block} {name}: bit-identical {equal} "
              f"(max_abs {max_abs:.3e} rel {relative:.3e})", flush=True)
        if exact:
            assert equal, f"(a) {variant} site {after_block}: {name} differs from the split chain"
        else:
            assert relative <= _ULP_CLASS, f"(a) {variant} site {after_block}: {name} rel {relative:.3e}"
    # Diagnostic for the reason: if only the per-cell rms scale differs, values_fused / values_split is one factor
    # per cell across all 32 channels (up to the dot's own rounding).
    ratio = fused["values"].double() / split_prep["values"].double()
    large = split_prep["values"].abs() > 1e-2
    per_cell = torch.where(large, ratio, torch.nan)
    spread = torch.nanmean((per_cell - torch.nanmean(per_cell, dim=1, keepdim=True)).abs(), dim=1)
    print(f"K4B-GATE(a) {variant} site{after_block} values ratio: |ratio - 1| max "
          f"{(per_cell[large] - 1).abs().max().item():.3e}, spread across channels (mean |dev|) max "
          f"{torch.nanmax(spread).item() if hasattr(torch, 'nanmax') else float(spread[~spread.isnan()].max()):.3e}",
          flush=True)
    e_hat2 = _run_contract_out(site, after_block, variant, fused["e_hat"], fused["values"], fused["gates"], 64)
    split_e_hat2 = _run_triplet(site, after_block, variant, split_e_hat, 64)["e_hat2"]
    chain_abs, chain_relative = _error(e_hat2, split_e_hat2)
    max_abs, relative = _error(e_hat2, site["dumps"]["e_hat2"])
    print(f"K4B-GATE(a) {variant} site{after_block} e_hat2: vs split chain max_abs {chain_abs:.3e} rel "
          f"{chain_relative:.3e} (bit-identical {torch.equal(e_hat2, split_e_hat2)}); vs ORT max_abs {max_abs:.3e} "
          f"rel {relative:.3e}", flush=True)
    assert chain_relative <= _ULP_CLASS and relative <= _F32_RELATIVE


@pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS[:1]])
def test_readback_prep_at_batch_8(reference: dict, variant: str, after_block: int, batch: int) -> None:
    """(a) at the small rung: the same bit-identity, through the batch-8 autotune."""
    site = reference["sites"][(variant, after_block)]
    e_in, logits = site["dumps"]["e_in"], site["dumps"]["H"]
    split_e_hat = _run_site(site, after_block, e_in, logits, batch, "readback")
    fused = _run_readback_prep(site, after_block, e_in, logits, batch)
    assert torch.equal(fused["e_hat"], split_e_hat)
    _, relative = _error(fused["e_hat"], site["dumps"]["e_hat"])
    assert relative <= _F32_RELATIVE


# ------------------------------------------------------------------------- K4b (b): fused prep + contract
@pytest.mark.parametrize("batch", [64, 8])
@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_fused_e_hat2_matches_ort(reference: dict, variant: str, after_block: int, batch: int) -> None:
    """(b) in K4's class: dot='ieee', FP32 state."""
    site = reference["sites"][(variant, after_block)]
    got = _run_fused(site, after_block, variant, site["dumps"]["e_hat"], batch)
    max_abs, relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    print(f"K4B-GATE(b) {variant} site{after_block} batch {batch} ieee e_hat2: max_abs {max_abs:.3e} rel "
          f"{relative:.3e}; best {_triplet_fused_kernel.best_config}", flush=True)
    assert relative <= _F32_RELATIVE, f"(b) {variant} site {after_block}: e_hat2 relative error {relative:.3e}"


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_fused_va_matches_the_scratch_chain(reference: dict, variant: str, after_block: int) -> None:
    """(b)'s va against K4's contract output: in class (the rms scale is applied after the channel sum)."""
    site = reference["sites"][(variant, after_block)]
    e_hat = site["dumps"]["e_hat"]
    scratch = _run_triplet(site, after_block, variant, e_hat, 64)
    fused = _run_fused(site, after_block, variant, e_hat, 64)
    max_abs, relative = _error(fused["va"], scratch["va"])
    equal = torch.equal(fused["va"], scratch["va"])
    print(f"K4B-GATE(b) {variant} site{after_block} va vs scratch chain: max_abs {max_abs:.3e} rel {relative:.3e} "
          f"bit-identical {equal}", flush=True)
    assert relative <= _F32_RELATIVE


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_fused_contraction_flag_is_exercised(reference: dict, variant: str, after_block: int) -> None:
    site = reference["sites"][(variant, after_block)]
    wrong = "ag" if variant == "path" else "path"
    got = _run_fused(site, after_block, wrong, site["dumps"]["e_hat"], 64)
    _, relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    print(f"K4B-GATE(b) {variant} site{after_block} with contraction {wrong}: rel {relative:.3e} (must be large)",
          flush=True)
    assert relative >= _FLAG_MINIMUM


@pytest.mark.parametrize("configuration", list(_FUSED_CONFIGURATIONS))
@pytest.mark.parametrize("variant", _VARIANTS)
def test_every_fused_candidate_is_in_class(reference: dict, variant: str, configuration: tuple[int, int]) -> None:
    """Every (group, warps) the autotuner may pick must gate on its own in K4's class (site 3, batch 64)."""
    site = reference["sites"][(variant, 3)]
    got = _run_fused(site, 3, variant, site["dumps"]["e_hat"], 64, configuration=configuration)
    max_abs, relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    print(f"K4B-GATE(b) {variant} site3 ieee candidate group/warps {configuration}: max_abs {max_abs:.3e} "
          f"rel {relative:.3e}", flush=True)
    assert relative <= _F32_RELATIVE, f"candidate {configuration}: {relative:.3e}"


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_chain_readback_fused_ffn(reference: dict, variant: str, after_block: int) -> None:
    """The served fused chain in K4's class: edge_site readback -> fused -> out -> edge_site ffn, vs ORT's e'."""
    site = reference["sites"][(variant, after_block)]
    e_hat = _run_site(site, after_block, site["dumps"]["e_in"], site["dumps"]["H"], 64, "readback")
    got = _run_fused(site, after_block, variant, e_hat, 64)
    e_new = _run_site(site, after_block, got["e_hat2"], None, 64, "ffn")
    hat2_abs, hat2_relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    new_abs, new_relative = _error(e_new, site["dumps"]["e_new"])
    print(f"K4B-GATE(b) {variant} site{after_block} chain: e_hat2 {hat2_abs:.3e}/{hat2_relative:.3e}, "
          f"e' {new_abs:.3e}/{new_relative:.3e}", flush=True)
    assert hat2_relative <= _F32_RELATIVE and new_relative <= _F32_RELATIVE


# ------------------------------------------------------------ K4b (b), the served class: FP16 A.V, FP16 e_hat read
@pytest.mark.parametrize("batch", [64, 8])
@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_fused_fp16_within_the_served_rule(reference: dict, variant: str, after_block: int, batch: int) -> None:
    """dot='fp16' (A and V rounded once, FP32 accumulation): e_hat2 within the served rule, not K4's 1e-5."""
    site = reference["sites"][(variant, after_block)]
    got = _run_fused(site, after_block, variant, site["dumps"]["e_hat"], batch, dot="fp16")
    max_abs, relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    print(f"K4B-GATE(b16) {variant} site{after_block} batch {batch} fp16 e_hat2: max_abs {max_abs:.3e} rel "
          f"{relative:.3e}; best {_triplet_fused_kernel.best_config}", flush=True)
    assert relative <= _SERVED_RELATIVE, f"(b16) {variant} site {after_block}: e_hat2 rel {relative:.3e}"


@pytest.mark.parametrize("configuration", list(_FUSED_CONFIGURATIONS))
@pytest.mark.parametrize("variant", _VARIANTS)
def test_every_fp16_candidate_is_within_the_served_rule(reference: dict, variant: str,
                                                        configuration: tuple[int, int]) -> None:
    site = reference["sites"][(variant, 3)]
    got = _run_fused(site, 3, variant, site["dumps"]["e_hat"], 64, dot="fp16", configuration=configuration)
    max_abs, relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    print(f"K4B-GATE(b16) {variant} site3 fp16 candidate group/warps {configuration}: max_abs {max_abs:.3e} "
          f"rel {relative:.3e}", flush=True)
    assert relative <= _SERVED_RELATIVE


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_readback_copy_is_bit_identical(reference: dict, variant: str, after_block: int) -> None:
    """`triplet_readback`: e_hat equals K3's readback to the bit, and the copy is exactly e_hat.half()."""
    site = reference["sites"][(variant, after_block)]
    e_in, logits = site["dumps"]["e_in"], site["dumps"]["H"]
    split_e_hat = _run_site(site, after_block, e_in, logits, 64, "readback")
    got = _run_readback(site, after_block, e_in, logits, 64)
    assert torch.equal(got["e_hat"], split_e_hat), "triplet_readback: e_hat differs from K3's readback"
    assert torch.equal(got["copy"], split_e_hat.half()), "triplet_readback: the copy is not e_hat.half()"
    _, relative = _error(got["copy"].float(), site["dumps"]["e_hat"])
    print(f"K4B-GATE(b16) {variant} site{after_block} readback copy: e_hat bit-identical, copy = half(e_hat), "
          f"copy vs ORT e_hat rel {relative:.3e}", flush=True)


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_chain_readback_copy_fused16_ffn(reference: dict, variant: str, after_block: int) -> None:
    """The served FP16 chain: triplet_readback -> fused(fp16, state_f16) -> out -> edge_site ffn, vs ORT."""
    site = reference["sites"][(variant, after_block)]
    got_readback = _run_readback(site, after_block, site["dumps"]["e_in"], site["dumps"]["H"], 64)
    got = _run_fused(site, after_block, variant, got_readback["e_hat"], 64, dot="fp16", state_f16=True)
    e_new = _run_site(site, after_block, got["e_hat2"], None, 64, "ffn")
    hat2_abs, hat2_relative = _error(got["e_hat2"], site["dumps"]["e_hat2"])
    new_abs, new_relative = _error(e_new, site["dumps"]["e_new"])
    print(f"K4B-GATE(b16) {variant} site{after_block} chain fp16+state16: e_hat2 {hat2_abs:.3e}/{hat2_relative:.3e}, "
          f"e' {new_abs:.3e}/{new_relative:.3e}", flush=True)
    assert hat2_relative <= _SERVED_RELATIVE and new_relative <= _SERVED_RELATIVE


# ------------------------------------------------------------------------- K4b (c): out folded into the FFN
@pytest.mark.parametrize("batch", [64, 8])
@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_out_ffn_matches_ort_and_k3(reference: dict, variant: str, after_block: int, batch: int) -> None:
    """(c): e' from out_ffn vs ORT's e' (class) and vs K3's out -> ffn on the same va (ulp class)."""
    site = reference["sites"][(variant, after_block)]
    e_hat = site["dumps"]["e_hat"]
    scratch = _run_triplet(site, after_block, variant, e_hat, 64)  # K4's va and e_hat2 (out applied)
    fused = _run_out_ffn(site, after_block, e_hat, scratch["va"], batch)
    split = _run_site(site, after_block, scratch["e_hat2"], None, batch, "ffn")
    split_abs, split_relative = _error(fused, split)
    max_abs, relative = _error(fused, site["dumps"]["e_new"])
    print(f"K4B-GATE(c) {variant} site{after_block} batch {batch} e': vs K3 out->ffn max_abs {split_abs:.3e} rel "
          f"{split_relative:.3e} (bit-identical {torch.equal(fused, split)}); vs ORT max_abs {max_abs:.3e} rel "
          f"{relative:.3e}", flush=True)
    assert split_relative <= _ULP_CLASS and relative <= _F32_RELATIVE


@pytest.mark.parametrize(("variant", "after_block"), [(v, b) for v in _VARIANTS for b in SITE_BLOCKS])
def test_chain_readback_fused16_out_ffn(reference: dict, variant: str, after_block: int) -> None:
    """The served chain with (c): edge_site readback -> fused(fp16) -> out_ffn, vs ORT's e' at the served rule."""
    site = reference["sites"][(variant, after_block)]
    e_hat = _run_site(site, after_block, site["dumps"]["e_in"], site["dumps"]["H"], 64, "readback")
    got = _run_fused(site, after_block, variant, e_hat, 64, dot="fp16")
    e_new = _run_out_ffn(site, after_block, e_hat, got["va"], 64)
    max_abs, relative = _error(e_new, site["dumps"]["e_new"])
    print(f"K4B-GATE(c) {variant} site{after_block} chain fp16 + out_ffn: e' max_abs {max_abs:.3e} rel "
          f"{relative:.3e}", flush=True)
    assert relative <= _SERVED_RELATIVE


@pytest.mark.parametrize("contraction", list(_VARIANTS))
def test_out_ffn_compiles_and_builds(contraction: str) -> None:
    pointer = lc0ex_pb2.PARAMETER_TYPE_POINTER
    fused = TripletSiteSpecialization(batch_count=8, architecture=_architecture(), contraction=contraction,
                                      form="fused", dot="fp16")
    started = time.perf_counter()
    artifact = compile_triplet_out_ffn(fused)
    print(f"K4B-COMPILE {contraction} stage out_ffn batch 8: {time.perf_counter() - started:.1f} s, grid "
          f"{artifact.grid}, block {artifact.block}, cubin {len(artifact.binary_data)} B, best "
          f"{_triplet_out_ffn_kernel.best_config}", flush=True)
    assert artifact.parameters[:7] == (pointer,) * 7 and artifact.grid[0] > 0
    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)
    tables = {
        name: builder.persistent_buffer(name=name, shape=_TRIPLET_SHAPES[name.split("/edge_site/")[1]],
                                        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, alignment_bytes=256)
        for name in triplet_table_names(3)
    }
    shapes = {"ffn/dense1/w": (HIDDEN, STATES), "ffn/dense1/b": (HIDDEN,), "ffn/dense2/w": (STATES, HIDDEN)}
    ffn_tables = {
        name: builder.persistent_buffer(name=name, shape=shapes[name.split("/edge_site/")[1]],
                                        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, alignment_bytes=256)
        for name in ffn_table_names(3)
    }
    sizes = buffer_bytes(fused)
    state = builder.temporary_buffer(size_bytes=sizes["state"], alignment_bytes=256)
    output = builder.temporary_buffer(size_bytes=sizes["state"], alignment_bytes=256)
    va = builder.temporary_buffer(size_bytes=sizes["va"], alignment_bytes=256)
    triplet_site(builder, kernels, state, va, None, tables, fused, after_block=3, out=False)
    triplet_out_ffn(builder, kernels, output, state, va, tables, ffn_tables, fused, after_block=3)
    with pytest.raises(ValueError, match="distinct buffers"):
        triplet_out_ffn(builder, kernels, state, state, va, tables, ffn_tables, fused, after_block=3)


@pytest.mark.parametrize("contraction", list(_VARIANTS))
def test_k4b_stages_compile_and_build(contraction: str) -> None:
    """The new stages compile to artifacts, and every K4b builder form appends its calls."""
    pointer, null = lc0ex_pb2.PARAMETER_TYPE_POINTER, lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
    scratch = TripletSiteSpecialization(batch_count=8, architecture=_architecture(), contraction=contraction)
    fused = TripletSiteSpecialization(batch_count=8, architecture=_architecture(), contraction=contraction,
                                      form="fused")
    fused16 = TripletSiteSpecialization(batch_count=8, architecture=_architecture(), contraction=contraction,
                                        form="fused", dot="fp16", state_f16=True)
    for name, compile_stage, key, pointers in (
        ("readback_prep", compile_triplet_readback_prep, scratch, 9),
        ("readback", compile_triplet_readback, fused16, 5),
        ("fused ieee", compile_triplet_fused, fused, 5),
        ("fused fp16 state16", compile_triplet_fused, fused16, 5),
    ):
        started = time.perf_counter()
        artifact = compile_stage(key)
        seconds = time.perf_counter() - started
        assert artifact.parameters[:pointers] == (pointer,) * pointers
        assert set(artifact.parameters[pointers:]) <= {null}
        assert artifact.grid[1:] == (1, 1) and artifact.grid[0] > 0
        print(f"K4B-COMPILE {contraction} stage {name} batch 8: {seconds:.1f} s, grid {artifact.grid}, "
              f"block {artifact.block}, cubin {len(artifact.binary_data)} B, "
              f"shared {artifact.dynamic_shared_memory_bytes} B", flush=True)
    print(f"K4B-COMPILE {contraction} fused best {_triplet_fused_kernel.best_config}", flush=True)

    executable = ExecutableBuilder()
    builder = executable.program(name="main")
    kernels = KernelCache(executable)
    tables = {
        name: builder.persistent_buffer(name=name, shape=_TRIPLET_SHAPES[name.split("/edge_site/")[1]],
                                        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, alignment_bytes=256)
        for name in triplet_table_names(3)
    }
    readback = builder.persistent_buffer(name=readback_table_name(3), shape=(STATES, HEADS),
                                         dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, alignment_bytes=256)
    sizes = buffer_bytes(fused)
    assert sizes["values"] == 0 and sizes["gates"] == 0 and sizes["copy"] == 0
    assert sizes["va"] == 4 * 2 * STATES * 8 * CELLS
    state = builder.temporary_buffer(size_bytes=sizes["state"], alignment_bytes=256)
    va = builder.temporary_buffer(size_bytes=sizes["va"], alignment_bytes=256)
    triplet_site(builder, kernels, state, va, None, tables, fused, after_block=3)
    sizes16 = buffer_bytes(fused16)
    assert sizes16["copy"] == 2 * STATES * 8 * CELLS
    copy = builder.temporary_buffer(size_bytes=sizes16["copy"], alignment_bytes=256)
    e_in = builder.temporary_buffer(size_bytes=sizes["state"], alignment_bytes=256)
    logits = builder.temporary_buffer(size_bytes=4 * HEADS * 8 * CELLS, alignment_bytes=256)
    triplet_readback(builder, kernels, state, copy, e_in, logits, readback, fused16)
    triplet_site(builder, kernels, state, va, None, tables, fused16, after_block=3, copy=copy)
    with pytest.raises(ValueError, match="exactly when state_f16"):
        triplet_site(builder, kernels, state, va, None, tables, fused16, after_block=3)
    with pytest.raises(ValueError, match="exactly when state_f16"):
        triplet_site(builder, kernels, state, va, None, tables, fused, after_block=3, copy=copy)
    sizes = buffer_bytes(scratch)
    values = builder.temporary_buffer(size_bytes=sizes["values"], alignment_bytes=256)
    gates = builder.temporary_buffer(size_bytes=sizes["gates"], alignment_bytes=256)
    triplet_site_with_readback(builder, kernels, state, e_in, logits, values, gates, tables, readback, scratch,
                               after_block=3)
    with pytest.raises(ValueError, match="form 'scratch'"):
        triplet_site_with_readback(builder, kernels, state, e_in, logits, values, gates, tables, readback, fused,
                                   after_block=3)
    with pytest.raises(ValueError, match="distinct buffers"):
        triplet_site_with_readback(builder, kernels, state, state, logits, values, gates, tables, readback, scratch,
                                   after_block=3)
    with pytest.raises(ValueError, match="distinct buffers"):
        triplet_readback(builder, kernels, state, copy, state, logits, readback, fused16)


def test_specialization_rejects_bad_forms() -> None:
    with pytest.raises(ValueError, match="unknown dot"):
        TripletSiteSpecialization(batch_count=8, architecture=120, form="fused", dot="tf32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="belong to form 'fused'"):
        TripletSiteSpecialization(batch_count=8, architecture=120, dot="fp16")
    with pytest.raises(ValueError, match="belong to form 'fused'"):
        TripletSiteSpecialization(batch_count=8, architecture=120, state_f16=True)
