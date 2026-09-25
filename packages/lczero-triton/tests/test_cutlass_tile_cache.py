"""One process must resolve one CUTLASS specialization to one tile.

`_select_tile` is consulted three times for a single kernel -- when the dynamic shared
memory is probed (`shared_storage_bytes`), when the body is rendered (`_render`) and
again in `compile_cutlass_matmul` for the launch grid -- while the sweep cache is a JSON
file that several builders share. Round 22 (09-12) built two ladders beside a third build
against one shared file and shipped a rung-8 QKV GEMM compiled for one tile and launched
with another's dynamic shared memory: every rung gated clean and every `backendbench`
batch died on an invalid `__shared__` write of 16 bytes. These tests drive that race
deterministically -- the file is rewritten between calls, exactly as a concurrent builder
rewrites it -- so they need no GPU and no race luck.
"""

import json
from pathlib import Path

import pytest
from lczero_triton.bt4.kernels import cutlass_matmul
from lczero_triton.bt4.kernels.cutlass_matmul import (
    CutlassMatmulSpecialization,
    _select_tile,
    _store_sweep_result,
    render_source,
)

# The round-22 casualty: the rung-8 QKV projection of the 512-wide EGT2 nets.
_SPECIALIZATION = CutlassMatmulSpecialization(
    m=512,
    n=1536,
    k=512,
    architecture=120,
    has_bias=True,
)
_DEVICE = (120, 84)
_KEY = "120/84/512/1536/512/fused"
# Two candidates the sweep really chooses between at a 512-row GEMM, where the
# winner is within noise and so decided by what else the box is doing. They
# disagree in both threadblock dimensions, so the body and the grid differ.
_TILE_A: list[object] = [[64, 64, 32], [32, 32, 32], 4]
_TILE_B: list[object] = [[128, 128, 32], [64, 64, 32], 3]
_TILE_OURS: list[object] = [[128, 64, 32], [64, 32, 32], 6]


def _write_cache(path: Path, entries: dict[str, list[object]]) -> None:
    """Rewrite the cache the way a builder does: whole file, one shot, no lock."""
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")


@pytest.fixture
def cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a private cache file, off the device and off nvcc."""
    path = tmp_path / "cutlass_tiles.json"
    monkeypatch.setattr(cutlass_matmul, "_SWEEP_CACHE_PATH", path)
    monkeypatch.setattr(cutlass_matmul, "device_key", lambda: _DEVICE)
    # Per-process state, reset so these tests are order-independent.
    monkeypatch.setattr(cutlass_matmul, "_SWEEP_CACHE_MEMO", {}, raising=False)
    monkeypatch.setattr(cutlass_matmul, "_SWEEP_CACHE_OWN", {}, raising=False)
    monkeypatch.delenv("LC0EX_CUTLASS_TILE", raising=False)
    monkeypatch.delenv("LC0EX_CUTLASS_TILE_REUSE", raising=False)
    # A cache miss must never reach nvcc from a unit test.
    monkeypatch.setenv("LC0EX_CUTLASS_TILE_SWEEP", "0")
    return path


def test_a_concurrent_rewrite_cannot_change_a_tile_inside_one_process(
    cache_path: Path,
) -> None:
    """The defect itself: one specialization, two lookups, the file rewritten between."""
    _write_cache(cache_path, {_KEY: _TILE_A})

    first = _select_tile(_SPECIALIZATION)
    _write_cache(cache_path, {_KEY: _TILE_B})  # another builder records its winner
    second = _select_tile(_SPECIALIZATION)

    assert first == ((64, 64, 32), (32, 32, 32), 4)
    assert second == first


def test_the_rendered_body_and_the_launch_grid_come_from_one_tile(
    cache_path: Path,
) -> None:
    """Round 22 reconstructed: the tile compiled in is the tile launched with."""
    _write_cache(cache_path, {_KEY: _TILE_A})

    source = render_source(_SPECIALIZATION)  # the compiled body's tile
    _write_cache(cache_path, {_KEY: _TILE_B})
    threadblock, _warp, _stages = _select_tile(_SPECIALIZATION)  # the launch geometry

    shape = (
        f"cutlass::gemm::GemmShape<"
        f"{threadblock[0]}, {threadblock[1]}, {threadblock[2]}>"
    )
    assert f"using ThreadblockShape = {shape};" in source
    grid = (
        (_SPECIALIZATION.m + threadblock[0] - 1) // threadblock[0],
        (_SPECIALIZATION.n + threadblock[1] - 1) // threadblock[1],
        1,
    )
    assert grid == (8, 24, 1)


def test_a_swept_tile_outranks_another_builders_answer(cache_path: Path) -> None:
    """What this process measured is what it keeps, whatever lands in the file."""
    _write_cache(cache_path, {_KEY: _TILE_A})
    _store_sweep_result(_KEY, _TILE_OURS)

    _write_cache(cache_path, {_KEY: _TILE_B})

    assert _select_tile(_SPECIALIZATION) == ((128, 64, 32), (64, 32, 32), 6)


def test_storing_keeps_another_builders_entries_and_the_file_format(
    cache_path: Path,
) -> None:
    """The store merges onto the file as it stands now, in the format it had."""
    theirs = "120/84/512/688/512/residual"
    _write_cache(cache_path, {_KEY: _TILE_A})
    _select_tile(_SPECIALIZATION)  # this process has read the file
    _write_cache(cache_path, {_KEY: _TILE_A, theirs: _TILE_B})  # they add a shape

    _store_sweep_result(_KEY, _TILE_OURS)

    text = cache_path.read_text(encoding="utf-8")
    written = json.loads(text)
    assert written[theirs] == _TILE_B  # their entry survived
    assert written[_KEY] == _TILE_OURS  # ours won
    assert text == json.dumps(written, indent=1)  # the format is unchanged
    assert [entry.name for entry in cache_path.parent.iterdir()] == [cache_path.name]


def test_the_cache_file_is_replaced_not_rewritten_in_place(cache_path: Path) -> None:
    """A concurrent reader sees the whole old file or the whole new one, never half."""
    _write_cache(cache_path, {_KEY: _TILE_A})
    before = cache_path.stat().st_ino

    _store_sweep_result("120/84/8/1536/512/fused", _TILE_B)

    assert cache_path.stat().st_ino != before


def test_a_build_still_starts_from_the_cache_file_on_disk(
    cache_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading once must not stop the next build from using what earlier ones swept."""
    _write_cache(cache_path, {_KEY: _TILE_A})
    assert _select_tile(_SPECIALIZATION)[0] == (64, 64, 32)

    other = tmp_path / "another_builders_copy.json"
    _write_cache(other, {_KEY: _TILE_B})
    monkeypatch.setattr(cutlass_matmul, "_SWEEP_CACHE_PATH", other)

    assert _select_tile(_SPECIALIZATION)[0] == (128, 128, 32)


def test_a_swept_rung_is_still_there_for_the_next_rung_to_reuse(
    cache_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dense `auto` ladder borrows the rung it just swept -- reading once must not lose it."""
    monkeypatch.setenv("LC0EX_CUTLASS_TILE_REUSE", "1")
    _write_cache(cache_path, {})
    _store_sweep_result(_KEY, _TILE_OURS)  # rung 8, swept by this process

    rung16 = CutlassMatmulSpecialization(
        m=1024,
        n=1536,
        k=512,
        architecture=120,
        has_bias=True,
    )

    assert _select_tile(rung16) == ((128, 64, 32), (64, 32, 32), 6)


def test_the_replacement_keeps_the_caches_permissions(cache_path: Path) -> None:
    """A cache several builds read must not come back owner-only after a store."""
    _write_cache(cache_path, {_KEY: _TILE_A})
    cache_path.chmod(0o664)

    _store_sweep_result(_KEY, _TILE_OURS)

    assert cache_path.stat().st_mode & 0o777 == 0o664
