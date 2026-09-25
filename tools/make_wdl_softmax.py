#!/usr/bin/env python3
"""Build the WDL-softmax reference copy of a lab export, the way round 20's static copy was built.

`export512.py` writes `/output/wdl` as LOGITS (a bare Gemm), while lc0's ONNX backend reads that tensor as
PROBABILITIES.  The round-20 gates therefore compare lc0ex against `onnx-cuda` running a *copy* of the export whose
value head ends in a Softmax.  Reading the existing static copy
(`work/static_net/static_bs4g_512x15_50000_vw_wdlsoftmax.pb.gz`, 4,305 nodes against the export's 4,304) shows exactly
what was done:

* the value Gemm's output is renamed `/output/wdl` -> `/output/wdl_logits`;
* one node is appended at the END of the node list: `Softmax` named `/output/wdl_softmax`,
  input `/output/wdl_logits`, output `/output/wdl`;
* nothing else changes -- initializers, the graph's declared outputs and the `Net.onnx_model` fields
  (`output_wdl` stays `/output/wdl`) are untouched.

This script reproduces that transformation at the protobuf-wire level with the tree's own `lab/_onnx.py` helpers (the
shared venv has no `onnx` and no `numpy`), and `--verify-against` proves the method by rebuilding a known copy and
comparing the DECOMPRESSED `Net` bytes (gzip headers carry an mtime, so the compressed bytes never match).

usage:
  make_wdl_softmax.py --source EXPORT.pb.gz --output COPY.pb.gz [--verify-against KNOWN_COPY.pb.gz]
  make_wdl_softmax.py --source STATIC.pb.gz --verify-against STATIC_wdlsoftmax.pb.gz      (check only, no output)
"""

import argparse
import gzip
import hashlib
from pathlib import Path

import lczero_triton.lab._onnx as ox

_WIRE_VARINT = 0
_WIRE_LENGTH = 2
_MODEL_GRAPH = 7
_GRAPH_NODE = 1
# NodeProto: input 1, output 2, name 3, op_type 4, attribute 5.
_NODE_INPUT, _NODE_OUTPUT, _NODE_NAME, _NODE_OP_TYPE, _NODE_ATTRIBUTE = 1, 2, 3, 4, 5
# AttributeProto: name 1, i 3, type 20 (AttributeType INT = 2).
_ATTRIBUTE_NAME, _ATTRIBUTE_INT, _ATTRIBUTE_TYPE, _ATTRIBUTE_TYPE_INT = 1, 3, 20, 2
_WDL = "/output/wdl"
_WDL_LOGITS = "/output/wdl_logits"
_SOFTMAX_NAME = "/output/wdl_softmax"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:8]


def _encode_axis_attribute() -> bytes:
    """AttributeProto `axis = 1` (INT), the attribute the known static copy's Softmax carries.

    On the `[batch, 3]` value output axis 1 is the last axis, so this is the ONNX default spelled out.
    """
    return b"".join((
        ox._length_delimited(_ATTRIBUTE_NAME, b"axis"),
        ox._tag(_ATTRIBUTE_INT, _WIRE_VARINT) + ox._varint(1),
        ox._tag(_ATTRIBUTE_TYPE, _WIRE_VARINT) + ox._varint(_ATTRIBUTE_TYPE_INT),
    ))


def _encode_softmax_node() -> bytes:
    """One NodeProto: Softmax(/output/wdl_logits) -> /output/wdl, named /output/wdl_softmax, axis 1."""
    return b"".join((
        ox._length_delimited(_NODE_INPUT, _WDL_LOGITS.encode("utf8")),
        ox._length_delimited(_NODE_OUTPUT, _WDL.encode("utf8")),
        ox._length_delimited(_NODE_NAME, _SOFTMAX_NAME.encode("utf8")),
        ox._length_delimited(_NODE_OP_TYPE, b"Softmax"),
        ox._length_delimited(_NODE_ATTRIBUTE, _encode_axis_attribute()),
    ))


def _rename_output(node_payload: memoryview) -> bytes:
    """Re-encode one NodeProto with its `/output/wdl` output renamed, every other field byte for byte."""
    parts = []
    renamed = 0
    for number, wire_type, value in ox._iter_fields(node_payload):
        if number == _NODE_OUTPUT and wire_type == _WIRE_LENGTH and bytes(value).decode("utf8") == _WDL:
            parts.append(ox._length_delimited(_NODE_OUTPUT, _WDL_LOGITS.encode("utf8")))
            renamed += 1
        else:
            parts.append(ox._reencode(number, wire_type, value))
    if renamed != 1:
        message = f"expected exactly one {_WDL} output on the value node, found {renamed}"
        raise SystemExit(message)
    return b"".join(parts)


def _rewrite_graph(graph_payload: memoryview) -> bytes:
    """Rename the value node's output and append the Softmax node after the last node field."""
    fields = list(ox._iter_fields(graph_payload))
    node_positions = [index for index, (number, wire, _) in enumerate(fields)
                      if number == _GRAPH_NODE and wire == _WIRE_LENGTH]
    if not node_positions:
        raise SystemExit("the graph has no nodes")
    producers = [index for index in node_positions
                 if _WDL in ox._decode_node(fields[index][2], 0).outputs]
    if len(producers) != 1:
        message = f"expected exactly one node writing {_WDL}, found {len(producers)}"
        raise SystemExit(message)
    producer_index, last_node_index = producers[0], node_positions[-1]
    node = ox._decode_node(fields[producer_index][2], 0)
    print(f"value node: {node.op_type} {node.name} inputs {list(node.inputs)} -> {list(node.outputs)}")
    parts = []
    for index, (number, wire_type, value) in enumerate(fields):
        if index == producer_index:
            parts.append(ox._length_delimited(_GRAPH_NODE, _rename_output(value)))
        else:
            parts.append(ox._reencode(number, wire_type, value))
        if index == last_node_index:
            parts.append(ox._length_delimited(_GRAPH_NODE, _encode_softmax_node()))
    return b"".join(parts)


def _rewrite_model(model: bytes) -> bytes:
    """Return the ModelProto with the rewritten graph, every other field byte for byte."""
    parts = []
    rewritten = 0
    for number, wire_type, value in ox._iter_fields(memoryview(model)):
        if number == _MODEL_GRAPH and wire_type == _WIRE_LENGTH:
            parts.append(ox._length_delimited(_MODEL_GRAPH, _rewrite_graph(value)))
            rewritten += 1
        else:
            parts.append(ox._reencode(number, wire_type, value))
    if rewritten != 1:
        raise SystemExit(f"expected one graph in the model, found {rewritten}")
    return b"".join(parts)


def softmax_copy(source: Path) -> bytes:
    """Return the uncompressed `Net` bytes of the WDL-softmax copy of `source`."""
    net = gzip.open(source, "rb").read()
    model = ox.extract_onnx_model(net)
    before = ox.parse_model(model)
    rewritten = _rewrite_model(model)
    after = ox.parse_model(rewritten)
    added = [n for n in after.nodes if n.op_type == "Softmax" and n.name == _SOFTMAX_NAME]
    print(f"{source.name}: nodes {len(before.nodes)} -> {len(after.nodes)}, "
          f"initializers {len(before.initializers)} -> {len(after.initializers)}, "
          f"appended {added[0].op_type} {added[0].name} {list(added[0].inputs)} -> {list(added[0].outputs)} "
          f"at index {added[0].index}")
    assert len(after.nodes) == len(before.nodes) + 1, "exactly one node must be added"
    assert after.initializers.keys() == before.initializers.keys(), "initializers must not change"
    assert after.producer[_WDL].op_type == "Softmax", "/output/wdl must now come from the Softmax"
    assert _WDL_LOGITS in after.producer, "/output/wdl_logits must be produced"
    return ox.replace_onnx_model(net, rewritten)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-against", type=Path)
    arguments = parser.parse_args()
    produced = softmax_copy(arguments.source)
    print(f"rebuilt Net: {len(produced)} B, sha256[0:8] of the decompressed bytes {_digest(produced)}")
    if arguments.verify_against is not None:
        known = gzip.open(arguments.verify_against, "rb").read()
        print(f"known copy {arguments.verify_against.name}: {len(known)} B, decompressed sha {_digest(known)}")
        print("VERIFY", "IDENTICAL" if known == produced else "DIFFERENT")
        if known != produced:
            return 1
    if arguments.output is not None:
        if arguments.output.exists():
            raise SystemExit(f"{arguments.output} exists; not overwriting")
        with gzip.open(arguments.output, "wb", compresslevel=6) as target:
            target.write(produced)
        print(f"wrote {arguments.output} {arguments.output.stat().st_size} B "
              f"sha256[0:8] {_digest(arguments.output.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
