"""Seed a new rung's Triton autotune from the rungs already measured.

R38 priced a new ladder rung at **537 s of Triton autotuning** against 196 s of
CUTLASS tile sweeping, so the autotuner -- not the sweep -- is what makes the
dense 64-128 band expensive.  The cost is structural: `_MATMUL_CONFIGS` has 160
entries, `_prune_matmul_configs` typically leaves 27-112 of them, and each one is
timed by `cold_do_bench`, which re-uploads every operand from pinned host memory
over PCIe before each repetition so that L2 is genuinely cold.  That is the right
benchmark and it is why one config costs milliseconds rather than microseconds.

The observation this module rests on: `@triton.autotune(cache_results=True)`
already writes every result to `~/.triton/cache/<hash>/<kernel>.autotune.json`,
carrying the autotune key and every candidate with its timings.  For the encoder
GEMMs the winning TILE is stable across neighbouring M -- in the measured
`n=1024, k=1024` family a single 128x64x64 tile wins from M=512 to M=1920 -- so a
new rung does not need to rediscover it, only to re-decide the scheduling
parameters around it.

Seeding rule, deliberately conservative:
  * donors must match `(n, k)` and the rest of the autotune key exactly, so a
    bias/activation variant never seeds a different one;
  * the two nearest donors in **log M** are taken (log, because the ladder is
    geometric in effect and a 2x jump matters more at M=512 than at M=8192);
  * the kept set is every surviving config whose tile is a donor's tile, or one
    step away in `block_m` only -- block_m is the dimension M actually moves --
    which leaves the cheap scheduling knobs (`group_size_m`, warps, stages) fully
    searched at those tiles.

Off by default.  `LC0EX_TRITON_CONFIG_REUSE=1` enables it;
`LC0EX_TRITON_REUSE_INDEX=<dir>` points the donor index at a cache directory
other than `~/.triton/cache`, which is what makes an honest A/B possible: the
donors can be read from the box's accumulated cache while the results are
written to a fresh, cold `TRITON_CACHE_DIR`.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import triton

_ENABLED = os.environ.get("LC0EX_TRITON_CONFIG_REUSE") == "1"
_NEAREST = int(os.environ.get("LC0EX_TRITON_REUSE_NEAREST", "2"))
_index: dict[str, dict[tuple[Any, ...], dict[int, dict[str, Any]]]] | None = None
_stats = {"hit": 0, "miss": 0, "kept": 0, "offered": 0}


def enabled() -> bool:
    """Return whether config reuse is switched on."""
    return _ENABLED


def _index_root() -> pathlib.Path:
    override = os.environ.get("LC0EX_TRITON_REUSE_INDEX")
    if override:
        return pathlib.Path(override).expanduser()
    return pathlib.Path.home() / ".triton" / "cache"


def _best_config(entry: dict[str, Any]) -> tuple[dict[str, Any], float] | None:
    best: tuple[dict[str, Any], float] | None = None
    for conf, timings in entry.get("configs_timings", ()):
        if not timings:
            continue
        median = sorted(timings)[len(timings) // 2]
        if best is None or median < best[1]:
            best = (conf, median)
    return best


def _build_index() -> dict[str, dict[tuple[Any, ...], dict[int, dict[str, Any]]]]:
    root = _index_root()
    index: dict[str, dict[tuple[Any, ...], dict[int, dict[str, Any]]]] = {}
    best_time: dict[tuple[str, tuple[Any, ...], int], float] = {}
    if not root.is_dir():
        return index
    for path in root.glob("*/*.autotune.json"):
        kernel = path.name[: -len(".autotune.json")]
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        key = entry.get("key")
        if not key or len(key) < 3:
            continue
        try:
            m, n, k = int(key[0]), int(key[1]), int(key[2])
        except (TypeError, ValueError):
            continue
        best = _best_config(entry)
        if best is None or m <= 0:
            continue
        family = (n, k, *_normalise(key[3:]))
        slot = (kernel, family, m)
        # Several cache directories can hold the same key under different
        # specialization hashes; keep the fastest measurement of each.
        if slot in best_time and best_time[slot] <= best[1]:
            continue
        best_time[slot] = best[1]
        index.setdefault(kernel, {}).setdefault(family, {})[m] = best[0]
    return index


def _get_index() -> dict[str, dict[tuple[Any, ...], dict[int, dict[str, Any]]]]:
    global _index  # noqa: PLW0603
    if _index is None:
        _index = _build_index()
    return _index


def _normalise(rest: Sequence[Any]) -> tuple[int, ...]:
    """Reduce an autotune key's tail to the part that is comparable over time.

    The cached tails are not uniformly typed: `has_bias` appears as `True` in
    files written by one build and as `1` in another, and the number of trailing
    dtype strings has changed between versions.  Everything in this stack is
    fp16, so the dtypes carry no information; drop them and coerce the flags to
    int, or a donor and its consumer would never match.
    """
    return tuple(
        int(value) for value in rest if isinstance(value, (bool, int, float))
    )


def _tile(conf: Any) -> tuple[int, int, int]:  # noqa: ANN401
    kwargs = conf["kwargs"] if isinstance(conf, dict) else conf.kwargs
    return (
        int(kwargs["block_m"]),
        int(kwargs["block_n"]),
        int(kwargs["block_k"]),
    )


def seed(
    kernel: str,
    configs: Sequence[triton.Config],
    m: int,
    n: int,
    k: int,
    rest: tuple[Any, ...],
) -> list[triton.Config] | None:
    """Return the seeded candidate subset, or None to leave `configs` alone.

    None -- not a refusal but an absence of evidence -- is returned whenever the
    donors cannot justify a cut: reuse is off, no family matches, this exact M is
    already measured (so the autotuner will hit its own cache anyway), or the
    seeds match nothing that survived pruning.
    """
    if not _ENABLED:
        return None
    family = (n, k, *_normalise(rest))
    donors = _get_index().get(kernel, {}).get(family)
    if not donors:
        _stats["miss"] += 1
        return None
    if m in donors or m <= 0:
        return None
    nearest = sorted(donors, key=lambda mm: abs(math.log(mm) - math.log(m)))
    seeds = [donors[mm] for mm in nearest[:_NEAREST]]
    tiles: set[tuple[int, int, int]] = set()
    for conf in seeds:
        block_m, block_n, block_k = _tile(conf)
        tiles.add((block_m, block_n, block_k))
        # M is the dimension that moves between rungs, so allow block_m to move
        # one step with it; block_n and block_k are pinned by N and K, which do
        # not change within a shape family.
        tiles.add((max(16, block_m // 2), block_n, block_k))
        tiles.add((min(256, block_m * 2), block_n, block_k))
    kept = [conf for conf in configs if _tile(conf) in tiles]
    _stats["offered"] += len(configs)
    if not kept:
        _stats["miss"] += 1
        return None
    _stats["hit"] += 1
    _stats["kept"] += len(kept)
    return kept


def stats() -> dict[str, int]:
    """Return the hit/miss counters, for the build log."""
    return dict(_stats)
