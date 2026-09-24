"""A dependency-free reader and writer for the ONNX subset the lab nets need.

The lab's 512x15 arms are exported as **ONNX-carrier** Lc0 networks: the `Net`
message carries an empty `weights` stub and the whole graph in `onnx_model`.
Nothing in this repository's environment can parse that -- `onnx` is not
installed and adding it would modify a shared virtual environment -- so the
handful of fields that matter are read straight off the protobuf wire.

Only three things are needed, and the runtime contract is what makes that
enough. `UploadWeights` in `network_lc0ex_cuda.cc` parses the carrier, indexes
`graph().initializer()` **by name**, and requires every buffer the artifact
declares to have an initializer of the same name, data type, shape and byte
length. It never looks at a node. So this module reads nodes only to *recover
the architecture* at build time, and the model it writes back carries
initializers alone.
"""

import gzip
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ONNX TensorProto.DataType, which lc0ex's Buffer.DataType mirrors numerically
# (`lc0ex.proto`: DATA_TYPE_F32 = 1, DATA_TYPE_F16 = 10) -- that shared numbering
# is why `ValidateInitializer` can compare the two enums with a plain cast.
FLOAT32 = 1
# Q1: the int8 GEMM's weights ship as raw bytes (`lc0ex.proto` DATA_TYPE_U8 = 2 = ONNX UINT8); the kernel reads
# them as signed codes. No runtime change: `DataTypeSize` already knows U8.
UINT8 = 2
INT32 = 6
INT64 = 7
FLOAT16 = 10

_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LENGTH = 2
_WIRE_32BIT = 5

# ModelProto.graph, and the GraphProto/TensorProto/NodeProto fields below.
_MODEL_IR_VERSION = 1
_MODEL_OPSET = 8
_MODEL_GRAPH = 7
_OPSET_DOMAIN = 1
_OPSET_VERSION = 2
_GRAPH_NODE = 1
_GRAPH_NAME = 2
_GRAPH_INITIALIZER = 5
_GRAPH_VALUE_INFO = (11, 12, 13)
_TENSOR_DIMS = 1
_TENSOR_DATA_TYPE = 2
_TENSOR_FLOAT_DATA = 4
_TENSOR_INT32_DATA = 5
_TENSOR_INT64_DATA = 7
_TENSOR_NAME = 8
_TENSOR_RAW_DATA = 9
_NODE_INPUT = 1
_NODE_OUTPUT = 2
_NODE_NAME = 3
_NODE_OP_TYPE = 4
_NODE_ATTRIBUTE = 5
_ATTRIBUTE_NAME = 1
_ATTRIBUTE_INT = 3
_ATTRIBUTE_STRING = 4
_ATTRIBUTE_INTS = 8

_IR_VERSION = 8
_OPSET_VERSION_VALUE = 17


class OnnxFormatError(ValueError):
    """The carrier cannot be traversed as an ONNX model."""


def _read_varint(buffer: memoryview, offset: int) -> tuple[int, int]:
    """Return (value, next offset) for one base-128 varint."""
    result = 0
    shift = 0
    while True:
        byte = buffer[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def _iter_fields(buffer: memoryview) -> Iterator[tuple[int, int, object]]:
    """Yield (field number, wire type, payload) for one protobuf message."""
    offset = 0
    limit = len(buffer)
    while offset < limit:
        key, offset = _read_varint(buffer, offset)
        number, wire_type = key >> 3, key & 7
        if wire_type == _WIRE_VARINT:
            value, offset = _read_varint(buffer, offset)
            yield number, wire_type, value
        elif wire_type == _WIRE_LENGTH:
            length, offset = _read_varint(buffer, offset)
            yield number, wire_type, buffer[offset : offset + length]
            offset += length
        elif wire_type == _WIRE_64BIT:
            yield number, wire_type, buffer[offset : offset + 8]
            offset += 8
        elif wire_type == _WIRE_32BIT:
            yield number, wire_type, buffer[offset : offset + 4]
            offset += 4
        else:
            message = f"unsupported protobuf wire type {wire_type}"
            raise OnnxFormatError(message)


def _packed_varints(payload: memoryview) -> list[int]:
    """Decode one packed repeated varint field."""
    values: list[int] = []
    offset = 0
    while offset < len(payload):
        value, offset = _read_varint(payload, offset)
        values.append(value)
    return values


_SIGN_BIT_64 = 1 << 63
_RANGE_64 = 1 << 64


def _signed(values: list[int]) -> list[int]:
    """Reinterpret varints of a protobuf `int32`/`int64` field as signed (R0).

    Both types put a negative value on the wire as its 64-bit two's complement, so -1 reads back as 2**64 - 1,
    which `struct.pack` rejects. The EGT2 triplet exports store their Einsum-rewrite shapes (`[0, 0, -1]`) this
    way in `int64_data`. Non-negative values are returned unchanged, so every other export decodes as before.
    """
    return [value - _RANGE_64 if value >= _SIGN_BIT_64 else value for value in values]


@dataclass(slots=True)
class Tensor:
    """One ONNX initializer, kept as raw bytes rather than decoded values."""

    name: str
    data_type: int
    dims: tuple[int, ...]
    raw_data: bytes

    @property
    def element_count(self) -> int:
        """Return the number of elements the declared shape holds."""
        count = 1
        for dimension in self.dims:
            count *= dimension
        return count


@dataclass(slots=True)
class Node:
    """One ONNX node, reduced to what architecture recovery reads."""

    index: int
    op_type: str
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    integers: dict[str, list[int]] = field(default_factory=dict)
    strings: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Graph:
    """The parts of an ONNX graph this package traverses."""

    nodes: list[Node]
    initializers: dict[str, Tensor]
    producer: dict[str, Node]

    def produced_by(self, tensor: str) -> Node | None:
        """Return the node that produces `tensor`, or None for an input."""
        return self.producer.get(tensor)


def _decode_tensor(payload: memoryview) -> Tensor | None:  # noqa: C901, PLR0912
    """Decode one TensorProto, returning None when it carries no name."""
    dims: list[int] = []
    data_type = 0
    name: str | None = None
    raw_data = b""
    floats: list[float] = []
    integers: list[int] = []
    wide: list[int] = []
    for number, wire_type, value in _iter_fields(payload):
        if number == _TENSOR_DIMS and wire_type == _WIRE_VARINT:
            dims.append(value)
        elif number == _TENSOR_DIMS and wire_type == _WIRE_LENGTH:
            dims.extend(_packed_varints(value))
        elif number == _TENSOR_DATA_TYPE and wire_type == _WIRE_VARINT:
            data_type = value
        elif number == _TENSOR_NAME and wire_type == _WIRE_LENGTH:
            name = bytes(value).decode("utf8")
        elif number == _TENSOR_RAW_DATA and wire_type == _WIRE_LENGTH:
            raw_data = bytes(value)
        elif number == _TENSOR_FLOAT_DATA and wire_type == _WIRE_LENGTH:
            payload_bytes = bytes(value)
            floats.extend(
                struct.unpack(f"<{len(payload_bytes) // 4}f", payload_bytes)
            )
        elif number == _TENSOR_FLOAT_DATA and wire_type == _WIRE_32BIT:
            floats.append(struct.unpack("<f", bytes(value))[0])
        elif number == _TENSOR_INT32_DATA and wire_type == _WIRE_VARINT:
            integers.append(value)
        elif number == _TENSOR_INT32_DATA and wire_type == _WIRE_LENGTH:
            integers.extend(_packed_varints(value))
        elif number == _TENSOR_INT64_DATA and wire_type == _WIRE_VARINT:
            wide.append(value)
        elif number == _TENSOR_INT64_DATA and wire_type == _WIRE_LENGTH:
            wide.extend(_packed_varints(value))
    if name is None:
        return None
    if not raw_data:
        # An exporter may use a typed data field instead of `raw_data`;
        # normalize to raw bytes so every consumer sees exactly one form -- and
        # so a byte-length check against an lc0ex buffer means the same thing
        # whichever field the writer chose.
        if floats:
            raw_data = struct.pack(f"<{len(floats)}f", *floats)
        elif integers:
            raw_data = struct.pack(f"<{len(integers)}i", *_signed(integers))
        elif wide:
            raw_data = struct.pack(f"<{len(wide)}q", *_signed(wide))
    return Tensor(
        name=name, data_type=data_type, dims=tuple(dims), raw_data=raw_data
    )


def _decode_node(payload: memoryview, index: int) -> Node:
    """Decode one NodeProto."""
    inputs: list[str] = []
    outputs: list[str] = []
    name = ""
    op_type = ""
    integers: dict[str, list[int]] = {}
    strings: dict[str, str] = {}
    for number, wire_type, value in _iter_fields(payload):
        if wire_type != _WIRE_LENGTH:
            continue
        if number == _NODE_INPUT:
            inputs.append(bytes(value).decode("utf8"))
        elif number == _NODE_OUTPUT:
            outputs.append(bytes(value).decode("utf8"))
        elif number == _NODE_NAME:
            name = bytes(value).decode("utf8")
        elif number == _NODE_OP_TYPE:
            op_type = bytes(value).decode("utf8")
        elif number == _NODE_ATTRIBUTE:
            _decode_attribute(value, integers, strings)
    return Node(
        index=index,
        op_type=op_type,
        name=name,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        integers=integers,
        strings=strings,
    )


def _decode_attribute(
    payload: memoryview,
    integers: dict[str, list[int]],
    strings: dict[str, str],
) -> None:
    """Decode one AttributeProto into the node's integer and string maps."""
    name: str | None = None
    collected: list[int] = []
    text: str | None = None
    for number, wire_type, value in _iter_fields(payload):
        if number == _ATTRIBUTE_NAME and wire_type == _WIRE_LENGTH:
            name = bytes(value).decode("utf8")
        elif wire_type == _WIRE_VARINT and number in (
            _ATTRIBUTE_INT,
            _ATTRIBUTE_INTS,
        ):
            collected.append(value)
        elif number == _ATTRIBUTE_INTS and wire_type == _WIRE_LENGTH:
            collected.extend(_packed_varints(value))
        elif number == _ATTRIBUTE_STRING and wire_type == _WIRE_LENGTH:
            text = bytes(value).decode("utf8", "replace")
    if name is None:
        return
    if collected:
        integers[name] = collected
    if text is not None:
        strings[name] = text


def parse_model(model: bytes) -> Graph:  # noqa: C901
    """Parse an ONNX ModelProto into nodes, initializers and a producer map."""
    buffer = memoryview(model)
    graph_payload: memoryview | None = None
    for number, wire_type, value in _iter_fields(buffer):
        if number == _MODEL_GRAPH and wire_type == _WIRE_LENGTH:
            graph_payload = value
            break
    if graph_payload is None:
        message = "the ONNX model carries no graph"
        raise OnnxFormatError(message)

    nodes: list[Node] = []
    initializers: dict[str, Tensor] = {}
    for number, wire_type, value in _iter_fields(graph_payload):
        if wire_type != _WIRE_LENGTH:
            continue
        if number == _GRAPH_NODE:
            nodes.append(_decode_node(value, len(nodes)))
        elif number == _GRAPH_INITIALIZER:
            tensor = _decode_tensor(value)
            if tensor is not None:
                initializers[tensor.name] = tensor

    producer: dict[str, Node] = {}
    for node in nodes:
        for output in node.outputs:
            producer[output] = node
    return Graph(nodes=nodes, initializers=initializers, producer=producer)


def _tag(number: int, wire_type: int) -> bytes:
    """Encode one protobuf field tag."""
    return _varint(number << 3 | wire_type)


def _varint(value: int) -> bytes:
    """Encode one base-128 varint."""
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            encoded.append(byte | 0x80)
        else:
            encoded.append(byte)
            return bytes(encoded)


def _length_delimited(number: int, payload: bytes) -> bytes:
    """Encode one length-delimited field."""
    return _tag(number, _WIRE_LENGTH) + _varint(len(payload)) + payload


def encode_tensor(tensor: Tensor) -> bytes:
    """Encode one TensorProto with its payload in `raw_data`."""
    parts = [
        _tag(_TENSOR_DIMS, _WIRE_VARINT) + _varint(dimension)
        for dimension in tensor.dims
    ]
    parts.append(_tag(_TENSOR_DATA_TYPE, _WIRE_VARINT) + _varint(tensor.data_type))
    parts.append(_length_delimited(_TENSOR_NAME, tensor.name.encode("utf8")))
    parts.append(_length_delimited(_TENSOR_RAW_DATA, tensor.raw_data))
    return b"".join(parts)


def encode_initializer_model(tensors: list[Tensor], *, graph_name: str) -> bytes:
    """Serialize a ModelProto holding `tensors` as initializers and no nodes.

    `UploadWeights` reads initializers only, so a carrier for an lc0ex artifact
    needs nothing else. Leaving the nodes out keeps the generated file honest:
    it is a weight container, not a graph that anything could be tempted to run.
    """
    graph = b"".join(
        [_length_delimited(_GRAPH_NAME, graph_name.encode("utf8"))]
        + [
            _length_delimited(_GRAPH_INITIALIZER, encode_tensor(tensor))
            for tensor in tensors
        ]
    )
    opset = _length_delimited(_OPSET_DOMAIN, b"") + _tag(
        _OPSET_VERSION, _WIRE_VARINT
    ) + _varint(_OPSET_VERSION_VALUE)
    return b"".join(
        [
            _tag(_MODEL_IR_VERSION, _WIRE_VARINT) + _varint(_IR_VERSION),
            _length_delimited(_MODEL_OPSET, opset),
            _length_delimited(_MODEL_GRAPH, graph),
        ]
    )


def load_carrier(path: Path) -> tuple[bytes, Graph]:
    """Read a gzip-compressed carrier `Net` and return (net bytes, its graph).

    The `Net` is returned undecoded so a caller can rewrite one field of it
    without a protobuf dependency on the exact `net.proto` revision that wrote
    it -- the lab's exporter and this tree do not have to agree on anything but
    the field number of `onnx_model`.
    """
    with gzip.open(path, "rb") as source:
        encoded = source.read()
    return encoded, parse_model(extract_onnx_model(encoded))


# net.proto: Net.onnx_model = 11, OnnxModel.model = 1.
_NET_ONNX_MODEL = 11
_ONNX_MODEL_MODEL = 1


def extract_onnx_model(net: bytes) -> bytes:
    """Return the ONNX ModelProto bytes carried by one serialized `Net`."""
    for number, wire_type, value in _iter_fields(memoryview(net)):
        if number != _NET_ONNX_MODEL or wire_type != _WIRE_LENGTH:
            continue
        for inner, inner_wire, payload in _iter_fields(value):
            if inner == _ONNX_MODEL_MODEL and inner_wire == _WIRE_LENGTH:
                return bytes(payload)
    message = "the network carries no onnx_model.model"
    raise OnnxFormatError(message)


def replace_onnx_model(net: bytes, model: bytes) -> bytes:
    """Return `net` with its `onnx_model.model` replaced by `model`.

    Every other field is copied through byte for byte, which is what keeps the
    generated carrier a faithful descendant of the export: magic, format enums
    and the `weights` stub survive untouched. This is the wire-level equivalent
    of round 16's `SerializePartialToString` -- and stronger, because it cannot
    drop a field this tree's `net.proto` happens not to know about.
    """
    parts: list[bytes] = []
    replaced = False
    for number, wire_type, value in _iter_fields(memoryview(net)):
        if number == _NET_ONNX_MODEL and wire_type == _WIRE_LENGTH:
            parts.append(
                _length_delimited(_NET_ONNX_MODEL, _rewrite_onnx_model(value, model))
            )
            replaced = True
        else:
            parts.append(_reencode(number, wire_type, value))
    if not replaced:
        message = "the network carries no onnx_model to replace"
        raise OnnxFormatError(message)
    return b"".join(parts)


def _rewrite_onnx_model(payload: memoryview, model: bytes) -> bytes:
    """Return one OnnxModel message with its `model` field replaced."""
    parts: list[bytes] = []
    for number, wire_type, value in _iter_fields(payload):
        if number == _ONNX_MODEL_MODEL and wire_type == _WIRE_LENGTH:
            parts.append(_length_delimited(_ONNX_MODEL_MODEL, model))
        else:
            parts.append(_reencode(number, wire_type, value))
    return b"".join(parts)


def _reencode(number: int, wire_type: int, value: object) -> bytes:
    """Re-encode one field exactly as it was read."""
    if wire_type == _WIRE_VARINT:
        return _tag(number, wire_type) + _varint(int(value))
    if wire_type == _WIRE_LENGTH:
        return _length_delimited(number, bytes(value))  # type: ignore[arg-type]
    return _tag(number, wire_type) + bytes(value)  # type: ignore[arg-type]
