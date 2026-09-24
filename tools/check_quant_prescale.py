#!/usr/bin/env python3
"""Check a quantiser-vector `.npz` before it is handed to the builder (Q1, ask 3 of the 09-22 ruling).

The analyser runs this on its own file; the builder runs the same code at build time, so a file that passes here
cannot be refused later for a reason this did not print. It reads the architecture off the network when one is
given, which is what makes a wrong width detectable at all.

    tools/check_quant_prescale.py VECTORS.npz --network NET.pb.gz
    tools/check_quant_prescale.py VECTORS.npz --blocks 28 --d-model 1024 --ffn-hidden 1024

Exit status is 0 only if every vector is servable. `--require-all` additionally demands the analyser's recipe:
all four sites in every block (112 vectors on the flagship). ⛔ `m` is reported but not wanted — the 09-21
SmoothQuant read refuses the offset fold on this family; a file that carries `m` gets a warning, not an error.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "lczero-triton" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "lc0ex" / "src"))

from lczero_triton.lab._quant import (  # noqa: E402
    SITES,
    QuantFormatError,
    buffer_names,
    load_prescale,
)


def _widths(arguments: argparse.Namespace) -> tuple[dict[str, int], int]:
    """The channel count of each site, and the block count, from a network or from the flags."""
    if arguments.network is not None:
        from lczero_triton.lab import _onnx  # noqa: PLC0415
        from lczero_triton.lab._mapping import read_network  # noqa: PLC0415

        _, graph = _onnx.load_carrier(arguments.network)
        shape = read_network(graph).architecture
        print(f"# architecture read off {arguments.network.name}: {shape.blocks} blocks x {shape.d_model}, "
              f"{shape.heads} heads x {shape.head_dim}, ffn hidden {shape.ffn_hidden}, style {shape.block_style}")
        if shape.block_style != "postnorm":
            print("# NOTE: a pre-norm net folds the vector into the norm's own scale; r is then flat (1 / D).")
        return ({"ffn_in": shape.d_model, "attn_in": shape.d_model, "ffn_mid": shape.ffn_hidden,
                 "attn_out": shape.heads * shape.head_dim}, shape.blocks)
    if arguments.blocks is None or arguments.d_model is None:
        message = "give --network, or --blocks and --d-model (and --ffn-hidden if it differs from d_model)"
        raise SystemExit(message)
    hidden = arguments.ffn_hidden or arguments.d_model
    return ({"ffn_in": arguments.d_model, "attn_in": arguments.d_model, "ffn_mid": hidden,
             "attn_out": arguments.d_model}, arguments.blocks)


def main() -> int:
    """Validate one file and print what the builder would see."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vectors", type=Path, help="the analyser's .npz")
    parser.add_argument("--network", type=Path, help="the lab export the vectors were measured on")
    parser.add_argument("--blocks", type=int, help="block count, if no network is given")
    parser.add_argument("--d-model", type=int, help="trunk width, if no network is given")
    parser.add_argument("--ffn-hidden", type=int, help="FFN hidden width (defaults to d_model)")
    parser.add_argument("--require-all", action="store_true",
                        help="demand all four sites in every block (the analyser's recipe)")
    arguments = parser.parse_args()

    widths, blocks = _widths(arguments)
    try:
        loaded = load_prescale(arguments.vectors, blocks=blocks, widths=widths)
    except QuantFormatError as error:
        print(f"REFUSED: {error}")
        return 1

    print(f"# {arguments.vectors.name} sha {loaded.digest}, {len(loaded)} sites, "
          f"{arguments.vectors.stat().st_size / 1024:.0f} kB on disk")
    print(f"# {'scope':>12} {'site':>9} {'chan':>5} | {'r min':>10} {'r max':>10} {'spread':>8} | "
          f"{'m':>4} | buffers")
    offsets = 0
    for (scope, site), vectors in sorted(loaded.sites.items()):
        low, high = min(vectors.prescale), max(vectors.prescale)
        names = buffer_names(scope, site)
        where = names[0].rsplit("/quant/", 1)[0] if names else "GEMM epilogue (Q1)"
        offsets += vectors.offset is not None
        print(f"  {scope:>12} {site:>9} {vectors.channels:>5} | {low:>10.4g} {high:>10.4g} "
              f"{high / low:>8.1f} | {'yes' if vectors.offset else '-':>4} | {where}")

    expected = {(f"encoder{index}", site) for index in range(blocks) for site in SITES}
    missing = sorted(expected - set(loaded.sites))
    print(f"# {len(loaded)} of {len(expected)} sites present"
          + (f"; missing {len(missing)}: {', '.join(f'{a}/{b}' for a, b in missing[:6])}"
             + (" …" if len(missing) > 6 else "") if missing else ""))  # noqa: PLR2004
    if offsets:
        print(f"⚠ {offsets} site(s) carry an offset `m`. The 09-21 SmoothQuant read refuses the offset fold on "
              "this family (worse on all six nets); the builder will not serve it unless it is asked to.")
    if missing and arguments.require_all:
        print("REFUSED: --require-all and the recipe is incomplete. Dropping a site costs 0.1-0.6 pp of top-1.")
        return 1
    print("OK: every vector in this file is servable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
