#!/usr/bin/env python3
"""Write a lab carrier (the ONNX initializers planned as lc0ex buffers) for one export: `lab.carrier.convert`.

usage: make_carrier.py EXPORT.pb.gz CARRIER.pb.gz
"""
import sys
from pathlib import Path

from lczero_triton.lab import carrier

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
if destination.exists():
    raise SystemExit(f"{destination} exists; not overwriting")
report = carrier.convert(source, destination)
print(report)
print(f"wrote {destination} {destination.stat().st_size} B")
