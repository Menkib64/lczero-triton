#!/usr/bin/env python3
"""Is one artifact's CUTLASS launch geometry consistent with its dynamic shared memory?

`compile_cutlass_matmul` consults `_select_tile` three times for one kernel -- the launch
grid and block, the rendered body, and `shared_storage_bytes` for the dynamic shared
memory -- with an nvcc compile of the body sitting between the second and the third. When
another builder rewrites the shared tile cache inside that window, the node ends up with
the grid/block of one tile and the shared memory of another, and the kernel writes past
the end of its allocation at launch (round 22: `Invalid __shared__ write of size 16 bytes
... at 0x9c00 is out of bounds`).

This reads a built artifact and asks, node by node, whether ANY candidate tile explains
the grid, the block and the declared shared memory at once. It needs nvcc (one probe
compile per tile and epilogue family, all cached) and no GPU. A mismatch is proof that the
node was assembled from two different tiles; agreement is not proof of the opposite, since
`stages` is invisible in the launch geometry.

usage: LC0EX_CHECK_ARCH=89 check_artifact_tile_consistency.py ARTIFACT.lc0ex [ARTIFACT.lc0ex ...]

The architecture comes from LC0EX_CHECK_ARCH (default 120 = sm_120; set 89 for an RTX 4090 artifact). A candidate tile
whose shared memory is over the device ceiling (shared_storage_bytes raises ValueError) is skipped -- the builder never
selects one, so it cannot explain a node.
"""

import os
import re
import sys
from pathlib import Path

from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels.cutlass_matmul import (
    _SWEEP_CANDIDATES,
    _epilogue_family,
    _thread_count,
    CutlassMatmulSpecialization,
    shared_storage_bytes,
)

_NAME = re.compile(r"^lc0ex_cutlass_gemm_f16_m(\d+)_n(\d+)_k(\d+)(.*)$")
_WARP_SIZE = 32


def _specialization(function: str) -> CutlassMatmulSpecialization | None:
    """Rebuild the specialization an entry-point name encodes."""
    matched = _NAME.match(function)
    if matched is None:
        return None
    m, n, k, suffix = int(matched[1]), int(matched[2]), int(matched[3]), matched[4]
    activation = "none"
    for name in ("mish", "relu", "swish", "sigmoid", "selu"):
        if f"_{name}" in suffix:
            activation = name
    return CutlassMatmulSpecialization(
        m=m,
        n=n,
        k=k,
        architecture=int(os.environ.get("LC0EX_CHECK_ARCH", "120")),
        has_bias="_bias" in suffix,
        activation=activation,
        has_skip=suffix.endswith("_skip"),
        glu="_glu" in suffix,
    )


def _smem(specialization: CutlassMatmulSpecialization, tile: tuple) -> int:
    """`sizeof(SharedStorage)` for one forced tile, straight from nvcc."""
    threadblock, warp, stages = tile
    os.environ["LC0EX_CUTLASS_TILE"] = ",".join(
        str(value) for value in (*threadblock, *warp, stages)
    )
    return shared_storage_bytes(specialization)


def _check(path: Path) -> int:
    """Report every node whose geometry and shared memory cannot share one tile."""
    executable = lc0ex_pb2.NeuralExecutable()
    executable.ParseFromString(path.read_bytes())
    problems = 0
    checked = 0
    for program in executable.programs:
        for node in program.nodes:
            function = executable.kernels[node.kernel_idx].function
            specialization = _specialization(function)
            if specialization is None:
                continue
            checked += 1
            grid, block = tuple(node.grid), tuple(node.block)
            fitting = []
            for tile in _SWEEP_CANDIDATES:
                threadblock, warp, _stages = tile
                expected_grid = (
                    (specialization.m + threadblock[0] - 1) // threadblock[0],
                    (specialization.n + threadblock[1] - 1) // threadblock[1],
                    1,
                )
                if expected_grid != grid:
                    continue
                if (_thread_count(threadblock, warp), 1, 1) != block:
                    continue
                fitting.append(tile)
            if not fitting:
                print(f"  ?? {program.name} {function}: no candidate matches {grid}/{block}")
                continue
            sizes = {}
            for tile in fitting:
                try:
                    sizes[tile] = _smem(specialization, tile)
                except ValueError:
                    continue  # over the shared-memory ceiling: never selected by the builder
            if not sizes:
                print(f"  ?? {program.name} {function}: every candidate matching {grid}/{block} is over the ceiling")
                continue
            if node.dynamic_shared_memory_bytes not in set(sizes.values()):
                problems += 1
                print(
                    f"  MISMATCH {program.name} {function}: grid={grid} block={block} "
                    f"declares {node.dynamic_shared_memory_bytes} B of shared memory; the "
                    f"tile(s) that explain the geometry need "
                    f"{sorted(set(sizes.values()))} B "
                    f"({_epilogue_family(specialization)}, {sorted(sizes)})"
                )
    verdict = "INCONSISTENT" if problems else "consistent"
    print(f"{verdict}: {path.name}: {checked} CUTLASS nodes, {problems} mismatched")
    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    total = 0
    for argument in sys.argv[1:]:
        total += _check(Path(argument))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
