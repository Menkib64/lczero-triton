"""Tests for generic kernel registration and opaque buffer dependencies."""

# ruff: noqa: PLR2004

from lc0ex import (
    Buffer,
    ExecutableBuilder,
    KernelArtifact,
    KernelHandle,
    SymbolArtifact,
    SymbolHandle,
)
from lc0ex.proto import lc0ex_pb2

F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16
POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
NULL_POINTER = lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
TARGET_ARCHITECTURE = "sm_80"


def _builder() -> ExecutableBuilder:
    """Create a target-configured executable builder."""
    return ExecutableBuilder(io_data_type=F16).set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        TARGET_ARCHITECTURE,
    )


def _artifact(
    *,
    function: str = "kernel",
    parameters: tuple[lc0ex_pb2.ParameterType, ...] = (POINTER,),
    runtime_ns: int | None = None,
) -> KernelArtifact:
    """Create a small generic pointer-ABI kernel artifact."""
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=b"fake cubin",
        function=function,
        parameters=parameters,
        grid=(2, 3, 1),
        block=(128, 1, 1),
        dynamic_shared_memory_bytes=256,
        runtime_ns=runtime_ns,
    )


def _external(
    builder: ExecutableBuilder,
    name: str,
    *,
    writable: bool = False,
) -> Buffer:
    """Create a small named external buffer for one graph test."""
    return builder.persistent_buffer(
        name=name,
        shape=(1,),
        dtype=F16,
        writable=writable,
    )


def test_add_kernel_and_call_serializes_generic_metadata() -> None:
    """The generic builder serializes registered pointer kernels and calls."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact(parameters=(POINTER, POINTER, POINTER)))
    first = _external(builder, "first")
    second = _external(builder, "second")
    third = _external(builder, "third")
    program.call(kernel, first, second, third)

    executable = builder.build()

    assert executable.kernels[0].function == "kernel"
    assert list(executable.programs[0].nodes[0].grid) == [2, 3, 1]
    assert [
        argument.allocation.offset
        for argument in executable.programs[0].nodes[0].arguments
    ] == [0, 2, 4]


def test_null_pointer_parameters_are_omitted_from_node_arguments() -> None:
    """Null pointer ABI slots are supplied by the runtime, not the graph."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact(parameters=(POINTER, NULL_POINTER, POINTER)))
    first = _external(builder, "first")
    second = _external(builder, "second")
    program.call(kernel, first, second)

    executable = builder.build()
    node = executable.programs[0].nodes[0]

    assert list(executable.kernels[0].parameters) == [POINTER, NULL_POINTER, POINTER]
    assert [argument.allocation.offset for argument in node.arguments] == [0, 2]


def test_readonly_external_buffers_have_no_dependency() -> None:
    """Named persistent ranges default to readonly accesses."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact())
    buffer = _external(builder, "weights")
    program.call(kernel, buffer)
    program.call(kernel, buffer)

    nodes = builder.build().programs[0].nodes

    assert not nodes[0].dependencies
    assert not nodes[1].dependencies


def test_writer_and_reader_dependencies_follow_opaque_identity() -> None:
    """Writers order readers and later writers of the same storage handle."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact())
    buffer = _external(builder, "output", writable=True)
    program.call(kernel, buffer)
    program.call(kernel, buffer, readonly=(buffer,))
    program.call(kernel, buffer)

    nodes = builder.build().programs[0].nodes

    assert list(nodes[1].dependencies) == [0]
    assert list(nodes[2].dependencies) == [1]


def test_ordered_raw_temporaries_reuse_one_range() -> None:
    """Temporaries may alias only after dependencies order their accesses."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    first = program.temporary_buffer(size_bytes=64, alignment_bytes=32)
    second = program.temporary_buffer(size_bytes=64, alignment_bytes=32)
    bridge = _external(builder, "bridge", writable=True)
    program.call(kernel, first, bridge)
    program.call(kernel, bridge, second, readonly=(bridge,))

    executable = builder.build()
    nodes = executable.programs[0].nodes

    assert executable.programs[0].execution_allocation.size_bytes == 64
    assert (
        nodes[0].arguments[0].allocation.offset
        == nodes[1].arguments[1].allocation.offset
    )
    assert list(nodes[1].dependencies) == [0]


def test_independent_raw_temporaries_do_not_reuse_storage() -> None:
    """Potentially concurrent temporary accesses require distinct ranges."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact())
    first = program.temporary_buffer(size_bytes=64, alignment_bytes=32)
    second = program.temporary_buffer(size_bytes=64, alignment_bytes=32)
    program.call(kernel, first)
    program.call(kernel, second)

    executable = builder.build()

    assert executable.programs[0].execution_allocation.size_bytes == 128


def test_add_kernel_deduplicates_identical_artifacts() -> None:
    """An artifact identity maps to one opaque kernel handle."""
    builder = _builder()
    first = builder.add_kernel(_artifact())
    second = builder.add_kernel(_artifact())

    assert isinstance(first, KernelHandle)
    assert second is first


def test_symbol_argument_serializes_as_immutable_module_pointer() -> None:
    """A module symbol is a pointer argument, not a callable graph node."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact(parameters=(POINTER, POINTER)))
    output = _external(builder, "output", writable=True)
    symbol = builder.add_symbol(
        SymbolArtifact(
            binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
            binary_data=b"fake cubin",
            symbol_name="mapping_table",
        )
    )
    program.call(kernel, output, symbol)

    executable = builder.build()
    argument = executable.programs[0].nodes[0].arguments[1]

    assert isinstance(symbol, SymbolHandle)
    assert len(executable.binaries) == 1
    assert argument.symbol.binary_idx == 0
    assert argument.symbol.symbol_name == "mapping_table"
    assert not argument.HasField("allocation")


def test_kernel_runtime_ns_is_serialized() -> None:
    """Kernel runtime_ns is serialized into the NeuralExecutable protobuf."""
    builder = _builder()
    program = builder.program(name="main")
    kernel = builder.add_kernel(_artifact(runtime_ns=123_456))
    first = _external(builder, "first")
    program.call(kernel, first)

    executable = builder.build()

    assert executable.kernels[0].runtime_ns == 123_456
