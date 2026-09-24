"""Builder for serialized Lc0 neural executables."""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Self

from lc0ex.buffer_builder import (
    AllocationPlan,
    Buffer,
    BufferBuilder,
    BufferLocation,
    data_type_size_bytes,
)
from lc0ex.kernel_builder import (
    KernelArtifact,
    KernelHandle,
    SymbolArtifact,
    SymbolHandle,
)
from lc0ex.proto import lc0ex_pb2

_MAGIC = 0x1C0E
_FORMAT = 1
# Round 26 C1 (the bytes census): when set, every `ProgramBuilder.call` appends one JSON line to this file -- the
# kernel, its grid, and per argument the byte span it can touch and whether the call may write it. The artifact
# itself records only allocation offsets, so this is the only place a node's IDEAL bytes (each input read once,
# each output written once) can be read from. Off by default; the artifact is identical either way.
_TRACE_CALLS = os.environ.get("LC0EX_TRACE_CALLS", "")


@dataclass(frozen=True, slots=True)
class _KernelInvocation:
    """One invocation of a registered kernel."""

    kernel: KernelHandle
    arguments: tuple[Buffer | SymbolHandle, ...]
    readonly: frozenset[Buffer]


class ProgramBuilder:
    """Build one program and its private execution allocation."""

    def __init__(
        self,
        owner: "ExecutableBuilder",
        name: str,
        metadata: bytes | None,
    ) -> None:
        """Initialize a program owned by *owner*."""
        self._owner = owner
        self._name = name
        self.metadata = metadata
        self._allocation = owner._buffers.execution_allocation()  # noqa: SLF001
        self._invocations: list[_KernelInvocation] = []

    @property
    def name(self) -> str:
        """Return the immutable program name."""
        return self._name

    def buffer(
        self,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create a named execution buffer in this program's allocation."""
        return self._owner._buffers.external_buffer(  # noqa: SLF001
            self._allocation,
            name=name,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def temporary_buffer(self, *, size_bytes: int, alignment_bytes: int) -> Buffer:
        """Create an unnamed execution buffer in this program's allocation."""
        return self._owner._buffers.temporary_buffer(  # noqa: SLF001
            self._allocation,
            size_bytes=size_bytes,
            alignment_bytes=alignment_bytes,
        )

    def tensor(
        self,
        *,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create an unnamed execution tensor in this program's allocation."""
        return self._owner._buffers.tensor(  # noqa: SLF001
            self._allocation,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def temporary_tensor(
        self,
        *,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        alignment_bytes: int = 256,
    ) -> Buffer:
        """Create an unnamed temporary tensor in this program's allocation."""
        return self.tensor(
            shape=shape,
            dtype=dtype,
            writable=True,
            alignment_bytes=alignment_bytes,
        )

    def persistent_tensor(
        self,
        *,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create an unnamed persistent tensor in the executable allocation."""
        return self._owner.persistent_tensor(
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def persistent_buffer(
        self,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create a named persistent buffer in the executable allocation."""
        return self._owner.persistent_buffer(
            name=name,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def set_target(
        self,
        vendor: lc0ex_pb2.Target.Vendor,
        architecture: str,
    ) -> Self:
        """Set the executable target and return this program builder."""
        self._owner.set_target(vendor, architecture)
        return self

    def add_kernel(self, kernel: KernelArtifact) -> KernelHandle:
        """Register a kernel in the owning executable."""
        return self._owner.add_kernel(kernel)

    def add_symbol(self, symbol: SymbolArtifact) -> SymbolHandle:
        """Register a symbol in the owning executable."""
        return self._owner.add_symbol(symbol)

    def call(
        self,
        kernel: KernelHandle,
        *arguments: Buffer | SymbolHandle,
        readonly: Sequence[Buffer] = (),
    ) -> None:
        """Append a call to this program."""
        buffers = self._owner._buffers  # noqa: SLF001
        artifact = self._owner._kernels[kernel]  # noqa: SLF001
        expected_argument_count = sum(
            parameter != lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER
            for parameter in artifact.parameters
        )
        if len(arguments) != expected_argument_count:
            message = (
                f"Kernel call has {len(arguments)} arguments; expected "
                f"{expected_argument_count}."
            )
            raise ValueError(message)
        self._invocations.append(
            _KernelInvocation(
                kernel=kernel,
                arguments=arguments,
                readonly=frozenset(
                    set(readonly)
                    | {
                        argument
                        for argument in arguments
                        if isinstance(argument, Buffer)
                        if not buffers.is_reusable(argument)
                        and not buffers.is_writable(argument)
                    },
                ),
            ),
        )
        if _TRACE_CALLS:
            self._owner._trace_call(self._name, len(self._invocations) - 1, artifact,  # noqa: SLF001
                                    self._invocations[-1])


class ExecutableBuilder:
    """Build an Lc0 neural executable with shared persistent storage."""

    def __init__(self) -> None:
        """Initialize an empty executable builder."""
        self._target: tuple[lc0ex_pb2.Target.Vendor, str] | None = None
        self._metadata: bytes | None = None
        self._buffers = BufferBuilder()
        self._kernels: dict[KernelHandle, KernelArtifact] = {}
        self._kernel_handles: dict[KernelArtifact, KernelHandle] = {}
        self._symbols: dict[SymbolHandle, SymbolArtifact] = {}
        self._symbol_handles: dict[SymbolArtifact, SymbolHandle] = {}
        self._programs: list[ProgramBuilder] = []

    def persistent_tensor(
        self,
        *,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create an unnamed persistent tensor in the executable allocation."""
        return self._buffers.persistent_tensor(
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def persistent_buffer(
        self,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create a named persistent buffer in the executable allocation."""
        return self._buffers.external_buffer(
            self._buffers.persistent_allocation(),
            name=name,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def program(
        self,
        *,
        name: str,
        metadata: bytes | None = None,
    ) -> ProgramBuilder:
        """Create a named program with a private execution allocation."""
        normalized_metadata = None if metadata is None else bytes(metadata)
        result = ProgramBuilder(self, name, normalized_metadata)
        self._programs.append(result)
        return result

    def set_target(
        self,
        vendor: lc0ex_pb2.Target.Vendor,
        architecture: str,
    ) -> Self:
        """Set the executable's compilation target and return this builder."""
        self._target = (vendor, architecture)
        return self

    def set_metadata(self, metadata: bytes) -> Self:
        """Set opaque executable metadata and return this builder."""
        self._metadata = bytes(metadata)
        return self

    def add_kernel(self, kernel: KernelArtifact) -> KernelHandle:
        """Register a compiled kernel and return its opaque handle."""
        existing = self._kernel_handles.get(kernel)
        if existing is not None:
            return existing

        handle = KernelHandle()
        self._kernels[handle] = kernel
        self._kernel_handles[kernel] = handle
        return handle

    def add_symbol(self, symbol: SymbolArtifact) -> SymbolHandle:
        """Register an immutable module symbol and return its opaque handle."""
        existing = self._symbol_handles.get(symbol)
        if existing is not None:
            return existing

        handle = SymbolHandle()
        self._symbols[handle] = symbol
        self._symbol_handles[symbol] = handle
        return handle

    def build(self) -> lc0ex_pb2.NeuralExecutable:
        """Create and return a new neural executable."""
        executable = lc0ex_pb2.NeuralExecutable(magic=_MAGIC, format=_FORMAT)
        if self._metadata is not None:
            executable.metadata = self._metadata
        if self._target is not None:
            vendor, architecture = self._target
            executable.target.vendor = vendor
            executable.target.architecture = architecture

        persistent_conflicts: dict[Buffer, set[Buffer]] = {}
        persistent_plan = self._buffers.plan(
            self._buffers.persistent_allocation(),
            persistent_conflicts,
        )
        if persistent_plan is not None:
            executable.persistent_allocation.size_bytes = persistent_plan.size_bytes
            executable.persistent_allocation.alignment_bytes = (
                persistent_plan.alignment_bytes
            )
            self._build_buffers(executable, persistent_plan)

        kernel_indices, symbol_locations = self._build_exports(executable)
        for program in self._programs:
            dependencies, reusable_conflicts = self._analyze_invocations(program)
            plan = self._buffers.plan(
                program._allocation,  # noqa: SLF001
                reusable_conflicts,
            )
            destination = executable.programs.add(name=program.name)
            if program.metadata is not None:
                destination.metadata = program.metadata
            if plan is not None:
                destination.execution_allocation.size_bytes = plan.size_bytes
                destination.execution_allocation.alignment_bytes = plan.alignment_bytes
                self._build_buffers(destination, plan)
            self._build_invocations(
                destination=destination,
                program=program,
                dependencies=dependencies,
                plans=(persistent_plan, plan),
                exports=(kernel_indices, symbol_locations),
            )
        return executable

    def build_and_write(
        self,
        path: str | PathLike[str],
    ) -> lc0ex_pb2.NeuralExecutable:
        """Build, serialize, and return the executable."""
        executable = self.build()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(executable.SerializeToString())
        return executable

    def _build_exports(
        self,
        executable: lc0ex_pb2.NeuralExecutable,
    ) -> tuple[dict[KernelHandle, int], dict[SymbolHandle, tuple[int, str]]]:
        """Append registered binaries and kernels and return export locations."""
        binary_indices: dict[tuple[lc0ex_pb2.Binary.Format, bytes], int] = {}
        kernel_indices: dict[KernelHandle, int] = {}

        def binary_index(
            binary_format: lc0ex_pb2.Binary.Format,
            binary_data: bytes,
        ) -> int:
            binary = (binary_format, binary_data)
            binary_idx = binary_indices.get(binary)
            if binary_idx is None:
                executable.binaries.add(format=binary_format, data=binary_data)
                binary_idx = len(executable.binaries) - 1
                binary_indices[binary] = binary_idx
            return binary_idx

        for handle, artifact in self._kernels.items():
            binary_idx = binary_index(artifact.binary_format, artifact.binary_data)
            kernel_proto = executable.kernels.add(
                binary_idx=binary_idx,
                function=artifact.function,
                parameters=artifact.parameters,
            )
            if artifact.runtime_ns is not None:
                kernel_proto.runtime_ns = artifact.runtime_ns
            kernel_indices[handle] = len(executable.kernels) - 1
        symbol_locations = {
            handle: (
                binary_index(artifact.binary_format, artifact.binary_data),
                artifact.symbol_name,
            )
            for handle, artifact in self._symbols.items()
        }
        return kernel_indices, symbol_locations

    def _analyze_invocations(
        self,
        program: ProgramBuilder,
    ) -> tuple[list[list[int]], dict[Buffer, set[Buffer]]]:
        """Return reduced dependencies and reusable-buffer conflicts."""
        writes: list[tuple[Buffer, int, int, int]] = []
        reads: list[tuple[Buffer, int, int, int]] = []
        ancestors: list[int] = []
        accesses: dict[Buffer, set[int]] = {}

        dependencies: list[list[int]] = []
        for index, invocation in enumerate(program._invocations):  # noqa: SLF001
            for buffer in {
                argument
                for argument in invocation.arguments
                if isinstance(argument, Buffer)
            }:
                # Views share their root's storage, so the reuse planner must see
                # every access to any view as an access to the root.
                if self._buffers.is_reusable(buffer):
                    accesses.setdefault(buffer.root_storage, set()).add(index)
            reduced_dependencies, ancestor_mask = self._invocation_dependencies(
                invocation,
                index,
                writes,
                reads,
                ancestors,
            )
            ancestors.append(ancestor_mask)
            dependencies.append(reduced_dependencies)

        return dependencies, self._reusable_conflicts(accesses, ancestors)

    def _build_buffers(
        self,
        destination: lc0ex_pb2.NeuralExecutable | lc0ex_pb2.Program,
        plan: AllocationPlan,
    ) -> None:
        """Serialize named external buffers into an executable or program."""
        buffers = destination.buffers
        for buffer, external in plan.external_buffers:
            location = plan.locations[buffer]
            entry = buffers.add(
                name=external.name,
                offset=location.offset,
                data_type=external.dtype,
                shape=external.shape,
            )
            if external.strides is not None and not buffer.is_contiguous():
                entry.layout.strides.extend(external.strides)

    def _build_invocations(
        self,
        *,
        destination: lc0ex_pb2.Program,
        program: ProgramBuilder,
        dependencies: list[list[int]],
        plans: tuple[AllocationPlan | None, AllocationPlan | None],
        exports: tuple[
            dict[KernelHandle, int],
            dict[SymbolHandle, tuple[int, str]],
        ],
    ) -> None:
        """Serialize one program's invocation graph."""
        persistent_plan, plan = plans
        kernel_indices, symbol_locations = exports
        locations = {}
        if persistent_plan is not None:
            locations.update(persistent_plan.locations)
        if plan is not None:
            locations.update(plan.locations)
        for invocation, invocation_dependencies in zip(
            program._invocations,  # noqa: SLF001
            dependencies,
            strict=True,
        ):
            artifact = self._kernels[invocation.kernel]
            node = destination.nodes.add(
                kernel_idx=kernel_indices[invocation.kernel],
                dependencies=invocation_dependencies,
                grid=artifact.grid,
                block=artifact.block,
                dynamic_shared_memory_bytes=artifact.dynamic_shared_memory_bytes,
            )
            for argument in invocation.arguments:
                if isinstance(argument, Buffer):
                    # A view is not in the plan: its address is its root's
                    # planned offset plus its own byte offset.
                    location = locations.get(argument)
                    if location is None:
                        root_location = locations[argument.root_storage]
                        location = BufferLocation(
                            root_location.allocation,
                            root_location.offset + argument.offset,
                        )
                    allocation_kind = (
                        lc0ex_pb2.Node.Argument.AllocationLocation.AllocationKind
                    )
                    allocation = (
                        allocation_kind.ALLOCATION_PERSISTENT
                        if location.allocation.is_persistent()
                        else allocation_kind.ALLOCATION_EXECUTION
                    )
                    node.arguments.add(
                        allocation=lc0ex_pb2.Node.Argument.AllocationLocation(
                            kind=allocation,
                            offset=location.offset,
                        )
                    )
                else:
                    binary_idx, symbol_name = symbol_locations[argument]
                    node.arguments.add(
                        symbol=lc0ex_pb2.Node.Argument.Symbol(
                            binary_idx=binary_idx,
                            symbol_name=symbol_name,
                        )
                    )

    def _trace_call(self, program: str, index: int, artifact: KernelArtifact, invocation: _KernelInvocation) -> None:
        """Append one call's argument spans to `LC0EX_TRACE_CALLS` (round 26 C1)."""
        entries = []
        for argument in invocation.arguments:
            if not isinstance(argument, Buffer):
                entries.append({"kind": "symbol"})
                continue
            root, low, high = self._hazard_range(argument)
            record = self._buffers._records[root]  # noqa: SLF001
            external = record.external
            entries.append({
                "root": id(root), "low": low, "high": high, "bytes": high - low,
                "persistent": record.allocation.is_persistent(),
                "readonly": argument in invocation.readonly,
                "name": getattr(external, "name", None) if external is not None else None,
            })
        line = {"program": program, "index": index, "function": artifact.function, "grid": list(artifact.grid),
                "arguments": entries}
        with Path(_TRACE_CALLS).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line) + "\n")

    def _hazard_range(self, buffer: Buffer) -> tuple[Buffer, int, int]:
        """Return the (root storage, first byte, last byte + 1) a buffer touches.

        A root buffer covers its whole allocation record.  A view covers the
        bounding box of its strided extent, which over-approximates a strided
        view and is exact for a contiguous one; over-approximation only ever adds
        an edge, never drops one.
        """
        root = buffer.root_storage
        size = self._buffers.storage_size_bytes(buffer)
        if buffer is root:
            return (root, 0, size)
        shape, strides = buffer.shape, buffer.strides
        if not shape:
            return (root, 0, size)
        element = data_type_size_bytes(buffer.dtype) if buffer.dtype else 1
        span = element
        for extent, stride in zip(shape, strides, strict=True):
            if extent > 0:
                span += (extent - 1) * abs(stride) * element
        low = buffer.offset
        return (root, low, min(low + span, size) if size else low + span)

    def _invocation_dependencies(
        self,
        invocation: _KernelInvocation,
        index: int,
        writes: list[tuple[Buffer, int, int, int]],
        reads: list[tuple[Buffer, int, int, int]],
        ancestors: list[int],
    ) -> tuple[list[int], int]:
        """Update access hazards and return reduced dependencies for one node."""
        dependencies: set[int] = set()
        for buffer in {
            argument
            for argument in invocation.arguments
            if isinstance(argument, Buffer)
        }:
            root, low, high = self._hazard_range(buffer)
            # read-after-write, for a reader and a writer alike
            dependencies.update(
                entry[3]
                for entry in writes
                if entry[0] is root and entry[1] < high and low < entry[2]
            )
            if buffer in invocation.readonly:
                reads.append((root, low, high, index))
                continue
            # write-after-read
            dependencies.update(
                entry[3]
                for entry in reads
                if entry[0] is root and entry[1] < high and low < entry[2]
            )
            writes.append((root, low, high, index))

        reduced_dependencies: list[int] = []
        covered_ancestors = 0
        for dependency in sorted(dependencies, reverse=True):
            if covered_ancestors & (1 << dependency):
                continue
            reduced_dependencies.append(dependency)
            covered_ancestors |= ancestors[dependency] | (1 << dependency)
        reduced_dependencies.reverse()
        ancestor_mask = 0
        for dependency in reduced_dependencies:
            ancestor_mask |= ancestors[dependency] | (1 << dependency)
        return reduced_dependencies, ancestor_mask

    def _reusable_conflicts(
        self,
        accesses: dict[Buffer, set[int]],
        ancestors: list[int],
    ) -> dict[Buffer, set[Buffer]]:
        """Return reusable-buffer pairs whose accesses can overlap."""
        buffers = list(accesses)
        conflicts: dict[Buffer, set[Buffer]] = {buffer: set() for buffer in accesses}
        for first_index, first in enumerate(buffers):
            for second in buffers[first_index + 1 :]:
                if not self._buffers.share_allocation(first, second):
                    continue
                first_before_second = all(
                    ancestors[second_access] & (1 << first_access)
                    for first_access in accesses[first]
                    for second_access in accesses[second]
                )
                second_before_first = all(
                    ancestors[first_access] & (1 << second_access)
                    for first_access in accesses[first]
                    for second_access in accesses[second]
                )
                if first_before_second or second_before_first:
                    continue
                conflicts[first].add(second)
                conflicts[second].add(first)
        return conflicts
