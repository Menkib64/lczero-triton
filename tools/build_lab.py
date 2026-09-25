"""Build an lc0ex artifact for a lab static net and report what it contains.

usage: build_lab.py --network EXPORT.pb.gz --output ART.lc0ex --batch-size 8[,16,...]
"""

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2

from lczero_triton.lab._format import load_lab_export
from lczero_triton.lab.network import build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # ValidateInitializer compares lc0ex data types to ONNX data types as integers.
    assert lc0ex_pb2.Buffer.DATA_TYPE_F16 == 10 and lc0ex_pb2.Buffer.DATA_TYPE_F32 == 1
    sizes = [int(size) for size in arguments.batch_size.split(",")]
    if arguments.output.exists():
        raise SystemExit(f"{arguments.output} exists; not overwriting")
    network, lab = load_lab_export(arguments.network)
    builder = ExecutableBuilder()
    started = time.time()
    build(builder, network, lab, batch_sizes=sizes)
    builder.build_and_write(arguments.output)
    executable = lc0ex_pb2.NeuralExecutable()
    executable.ParseFromString(arguments.output.read_bytes())
    for program in executable.programs:
        functions = Counter(executable.kernels[node.kernel_idx].function for node in program.nodes)
        cutlass = sum(count for function, count in functions.items() if "cutlass" in function)
        print(f"program {program.name}: nodes={len(program.nodes)} kernels={len(functions)} cutlass_nodes={cutlass}")
    print(f"BUILD-OK {arguments.output} {arguments.output.stat().st_size} B in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
