#!/usr/bin/env python3
"""G4 (round 26): which tensor-core instructions each kernel of an artifact actually issues.

For every distinct kernel of a `.lc0ex` artifact: write its CUBIN to a scratch directory, disassemble it with
`cuobjdump -sass`, and count the MMA-class opcodes (HMMA = FP16/BF16, IMMA = integer, QMMA / OMMA = FP8 / FP4 on
Ada and Blackwell). The node count per program says how many launches use it.

    sass_census.py ARTIFACT.lc0ex [--cuobjdump /usr/local/cuda-12.9/bin/cuobjdump] [--scratch DIR]

The Q1 gate reads: IMMA in the four int8 sites' kernels, no HMMA there, and no QMMA anywhere on sm_89.
"""

import argparse
import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "lc0ex" / "src"))

from lc0ex.proto import lc0ex_pb2  # noqa: E402

_OPCODES = re.compile(r"\b(HMMA|IMMA|QMMA|OMMA|HGMMA|IGMMA|QGMMA)(\.[A-Z0-9_.]+)?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--cuobjdump", default="/usr/local/cuda-12.9/bin/cuobjdump")
    parser.add_argument("--scratch", type=Path, help="where the CUBINs are written (default: a temporary directory)")
    arguments = parser.parse_args()
    executable = lc0ex_pb2.NeuralExecutable()
    executable.ParseFromString(arguments.artifact.read_bytes())
    uses: collections.Counter[int] = collections.Counter()
    for program in executable.programs:
        for node in program.nodes:
            uses[node.kernel_idx] += 1
    programs = max(1, len(executable.programs))
    scratch_context = tempfile.TemporaryDirectory(prefix="sass-") if arguments.scratch is None else None
    scratch = Path(scratch_context.name) if scratch_context else arguments.scratch
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"# {arguments.artifact.name}: {len(executable.kernels)} kernels, {len(executable.programs)} programs")
    print(f"# {'launches/prog':>13}  {'MMA opcodes (count)':<48}  kernel")
    for index, kernel in enumerate(executable.kernels):
        cubin = scratch / f"k{index:03d}.cubin"
        cubin.write_bytes(executable.binaries[kernel.binary_idx].data)
        listing = subprocess.run([arguments.cuobjdump, "-sass", str(cubin)], check=False, capture_output=True,
                                 text=True).stdout
        counts = collections.Counter(m.group(1) + (m.group(2) or "") for m in _OPCODES.finditer(listing))
        summary = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items())) or "-"
        print(f"  {uses[index] / programs:13.1f}  {summary[:48]:<48}  {kernel.function[:90]}")
    if scratch_context:
        scratch_context.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
