"""Q1: the quantiser-vector file the analyser hands over, read by the builder (CPU).

The reader's whole job is to refuse. A wrong-width, mis-scaled or mis-keyed vector produces a net that
loads, runs and plays plausible-looking moves, so every refusal below is one silent failure removed.
"""

import struct
import zipfile
from pathlib import Path

import pytest
from lczero_triton.lab._names import quant_names
from lczero_triton.lab._quant import (
    QuantFormatError,
    buffer_names,
    activation_step,
    fold_into_affine,
    identity_prescale,
    load_prescale,
    prescale,
    producing_norm,
    smoothing_vector,
)

_WIDTHS = {"ffn_in": 8, "attn_in": 8, "ffn_mid": 16, "attn_out": 8}


def _npy(values, descr: str = "<f4") -> bytes:
    """One `.npy` v1 member, the format `numpy.savez` writes (pinned by the test below)."""
    code = {"<f4": "f", "<f8": "d", "<f2": "e"}[descr]
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({len(values)},), }}"
    padding = -(len(header) + 11) % 64
    header = header + " " * padding + "\n"
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode()
            + struct.pack(f"<{len(values)}{code}", *values))


def _write(path: Path, members: dict[str, object], descr: str = "<f4") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for key, values in members.items():
            archive.writestr(f"{key}.npy", _npy(values, descr))
    return path


def _minimal(path: Path) -> Path:
    return _write(path, {
        "encoder0/ffn_in/r": [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0],
        "encoder0/ffn_in/m": [0.0, 0.5, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0],
        "encoder1/attn_in/r": [3.0] * 8,
    })


def test_reads_the_sites_it_is_given(tmp_path: Path) -> None:
    loaded = load_prescale(_minimal(tmp_path / "q.npz"), blocks=2, widths=_WIDTHS)

    assert len(loaded) == 2  # noqa: PLR2004
    site = loaded.get("encoder0", "ffn_in")
    assert site is not None
    assert site.prescale == (1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0)
    assert site.offset is not None and site.offset[1] == 0.5  # noqa: PLR2004
    assert loaded.get("encoder1", "attn_in").offset is None
    assert loaded.get("encoder1", "ffn_in") is None
    assert len(loaded.digest) == 8  # noqa: PLR2004


def test_a_vector_of_the_wrong_width_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "wide.npz", {"encoder0/ffn_in/r": [1.0] * 7})
    with pytest.raises(QuantFormatError, match="7 channels"):
        load_prescale(path, blocks=2, widths=_WIDTHS)


def test_a_block_past_the_network_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "far.npz", {"encoder9/ffn_in/r": [1.0] * 8})
    with pytest.raises(QuantFormatError, match="past this network"):
        load_prescale(path, blocks=2, widths=_WIDTHS)


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path / "odd.npz", {"encoder0/ffn_out/r": [1.0] * 8})
    with pytest.raises(QuantFormatError, match="site 'ffn_out'"):
        load_prescale(path, blocks=2, widths=_WIDTHS)


def test_a_non_positive_scale_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "zero.npz", {"encoder0/ffn_in/r": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
    with pytest.raises(QuantFormatError, match="positive and finite"):
        load_prescale(path, blocks=2, widths=_WIDTHS)


def test_r_and_s_must_be_built_against_one_step(tmp_path: Path) -> None:
    """`r * s` is `1 / D`: constant over channels, or the two came from different calibrations."""
    good = _write(tmp_path / "good.npz", {
        "encoder0/ffn_in/r": [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0],
        "encoder0/ffn_in/s": [8.0, 4.0, 2.0, 1.0, 8.0, 4.0, 2.0, 1.0],
    })
    assert load_prescale(good, blocks=1, widths=_WIDTHS).get("encoder0", "ffn_in") is not None

    bad = _write(tmp_path / "bad.npz", {
        "encoder0/ffn_in/r": [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0],
        "encoder0/ffn_in/s": [8.0, 4.0, 2.0, 1.0, 8.0, 4.0, 2.0, 2.0],
    })
    with pytest.raises(QuantFormatError, match="not built against the same activation step"):
        load_prescale(bad, blocks=1, widths=_WIDTHS)


def test_a_two_dimensional_member_is_refused(tmp_path: Path) -> None:
    payload = (b"\x93NUMPY\x01\x00" + struct.pack("<H", 64)
               + b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 4), }".ljust(63) + b"\n"
               + struct.pack("<8f", *([1.0] * 8)))
    path = tmp_path / "matrix.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("encoder0/ffn_in/r.npy", payload)
    with pytest.raises(QuantFormatError, match="one-dimensional"):
        load_prescale(path, blocks=1, widths=_WIDTHS)


def test_the_fold_is_the_two_vectors_the_norm_reads(tmp_path: Path) -> None:
    loaded = load_prescale(_minimal(tmp_path / "q.npz"), blocks=2, widths=_WIDTHS)
    site = loaded.get("encoder0", "ffn_in")
    gammas = [0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5]
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    quant_gammas, quant_betas = fold_into_affine(gammas, betas, site)

    assert quant_gammas[1] == pytest.approx(1.5 * 2.0)
    assert quant_betas[1] == pytest.approx((0.2 - 0.5) * 2.0)
    # And the codes the folded pair produces are the codes the two-vector form produces.
    for index, (gamma, beta) in enumerate(zip(gammas, betas, strict=True)):
        normalized = 0.37
        folded = normalized * quant_gammas[index] + quant_betas[index]
        direct = ((normalized * gamma + beta) - site.offset[index]) * site.prescale[index]
        assert folded == pytest.approx(direct, rel=1e-6)


def test_the_migration_formulas_are_the_analysers(tmp_path: Path) -> None:
    """`s`, `D` and `r` as the SmoothQuant task defines them, so one lane's numbers read in the other."""
    amax = [1.0, 16.0, 4.0, 2.0]
    weights = [1.0, 0.25, 2.0, 1.0]
    smoothing = smoothing_vector(amax, weights, 0.5)

    assert smoothing[1] == pytest.approx((16.0 ** 0.5) / (0.25 ** 0.5))
    step = activation_step(amax, smoothing)
    assert step == pytest.approx(max(a / s for a, s in zip(amax, smoothing, strict=True)) / 127.0)
    vector = prescale(smoothing, step)
    assert vector[2] == pytest.approx(1.0 / (smoothing[2] * step))
    # The unsmoothed case is a flat vector, not a different code path.
    assert identity_prescale(4, step) == (1.0 / step,) * 4


def test_the_hand_written_members_are_what_numpy_writes(tmp_path: Path) -> None:
    """Pin the test's own writer to the real format; the venv that serves has no numpy, the analyser has."""
    numpy = pytest.importorskip("numpy")
    values = [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0]
    reference = tmp_path / "numpy.npz"
    numpy.savez(reference, **{"encoder0/ffn_in/r": numpy.array(values, dtype=numpy.float32)})

    mine = load_prescale(_write(tmp_path / "mine.npz", {"encoder0/ffn_in/r": values}), blocks=1)
    theirs = load_prescale(reference, blocks=1)

    assert mine.get("encoder0", "ffn_in").prescale == theirs.get("encoder0", "ffn_in").prescale


def test_a_site_is_resolved_to_the_norm_that_emits_it() -> None:
    """The off-by-one: a block's attention input comes from the PREVIOUS block's second norm."""
    assert producing_norm("encoder7", "ffn_in") == ("/encoder7", "ln1")
    assert producing_norm("encoder7", "attn_in") == ("/encoder6", "ln2")
    assert producing_norm("encoder0", "attn_in") == ("/attn_body", "ln1")
    assert producing_norm("embedding", "ffn_in") == ("/attn_body", "ln1")
    # The two GEMM-epilogue sites have no norm to fold into; they wait for Q1's epilogue.
    assert producing_norm("encoder7", "ffn_mid") is None
    assert producing_norm("encoder7", "attn_out") is None
    with pytest.raises(QuantFormatError, match="no attention to feed"):
        producing_norm("embedding", "attn_in")
    assert quant_names("/encoder7", "ln1") == ("/encoder7/ln1/quant/r", "/encoder7/ln1/quant/m")
    assert buffer_names("encoder7", "attn_in") == ("/encoder6/ln2/quant/r", "/encoder6/ln2/quant/m")
    assert buffer_names("encoder7", "ffn_mid") is None
