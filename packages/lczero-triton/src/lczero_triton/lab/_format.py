"""Load and validate a lab export: an ONNX-carrier Leela network.

The lab exporter writes `network_format.network = 7` with `onnx_model` set and
no `weights` stanza (`export512.py`), which `bt4._format.load_network` rejects
by design. This module accepts exactly that shape and nothing else.
"""

import gzip
from pathlib import Path

from lc0ex.proto import net_pb2

from lczero_triton.lab import _onnx
from lczero_triton.lab._mapping import LabNetwork, read_network


class LabFormatError(ValueError):
    """The file is not a lab ONNX-carrier network this package can build."""


def validate_lab_format(network: net_pb2.Net) -> None:
    """Require an ONNX-carrier attention-body network with no encoder weights."""
    if not network.HasField("onnx_model"):
        message = "a lab export carries its weights as ONNX initializers; this net has no onnx_model"
        raise LabFormatError(message)
    network_format = network.format.network_format
    if network_format.network != net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
        message = f"network_format.network is {network_format.network}, expected an attention body"
        raise LabFormatError(message)
    if network_format.input != net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE:
        message = f"network_format.input is {network_format.input}, expected 112 classical planes"
        raise LabFormatError(message)
    if network.HasField("weights") and len(network.weights.encoder) > 0:
        message = "this net has an encoder weights stanza; build it with the BT4 builder"
        raise LabFormatError(message)


def load_lab_export(path: Path) -> tuple[net_pb2.Net, LabNetwork]:
    """Return the parsed network and the architecture recovered from its graph."""
    network = net_pb2.Net()
    network.ParseFromString(gzip.open(path, "rb").read())
    validate_lab_format(network)
    _, graph = _onnx.load_carrier(path)
    return network, read_network(graph)
