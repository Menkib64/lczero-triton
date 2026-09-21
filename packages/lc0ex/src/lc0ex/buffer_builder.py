"""Opaque logical buffers, tensor views, and contiguous allocation planning."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Self

from lc0ex.proto import lc0ex_pb2


def default_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute contiguous row-major strides in element counts for *shape*."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 1, 0, -1):
        strides[i - 1] = strides[i] * shape[i]
    return tuple(strides)


def data_type_size_bytes(dtype: lc0ex_pb2.Buffer.DataType) -> int:
    """Return the size in bytes of one value of *dtype*.

    Raises:
        KeyError: If *dtype* is not a supported concrete data type.

    """
    sizes = {
        lc0ex_pb2.Buffer.DATA_TYPE_F32: 4,
        lc0ex_pb2.Buffer.DATA_TYPE_U32: 4,
        lc0ex_pb2.Buffer.DATA_TYPE_U8: 1,
        lc0ex_pb2.Buffer.DATA_TYPE_F16: 2,
        lc0ex_pb2.Buffer.DATA_TYPE_U64: 8,
        lc0ex_pb2.Buffer.DATA_TYPE_BF16: 2,
    }
    return sizes[dtype]

class BufferType(Enum):
    """The type of a logical buffer."""

    PERSISTENT = 1
    EXECUTION = 2
    HOST = 3


@dataclass(frozen=True, slots=True, eq=False)
class Allocation:
    """A logical device-memory allocation owned by a buffer builder."""

    _type: BufferType

    def is_persistent(self) -> bool:
        """Return whether this allocation belongs to the executable."""
        return self._type == BufferType.PERSISTENT

    def is_execution(self) -> bool:
        """Return whether this allocation belongs to one program execution."""
        return self._type == BufferType.EXECUTION

    def is_host(self) -> bool:
        """Return whether this allocation belongs to host memory."""
        return self._type == BufferType.HOST

class Buffer:
    """An opaque identity or view for one logical device-memory range."""

    __slots__ = (
        "_allocation",
        "_builder",
        "_dtype",
        "_offset",
        "_shape",
        "_storage",
        "_strides",
        "_writable",
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        storage: "Buffer | None" = None,
        offset: int = 0,
        shape: Sequence[int] = (),
        strides: Sequence[int] | None = None,
        dtype: lc0ex_pb2.Buffer.DataType = lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN,
        allocation: Allocation | None = None,
        builder: "BufferBuilder | None" = None,
        writable: bool = True,
    ) -> None:
        """Initialize an opaque buffer or sliced view."""
        self._storage = storage
        self._offset = offset
        self._shape = tuple(shape)
        self._strides = default_strides(shape) if strides is None else tuple(strides)
        self._dtype = dtype
        self._allocation = allocation
        self._builder = builder
        self._writable = writable

    @property
    def root_storage(self) -> "Buffer":
        """Return the underlying root physical storage buffer."""
        return self if self._storage is None else self._storage.root_storage

    @property
    def offset(self) -> int:
        """Return byte offset relative to root physical storage."""
        return self._offset

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the tensor shape."""
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Return the tensor strides in element count."""
        return self._strides

    @property
    def dtype(self) -> lc0ex_pb2.Buffer.DataType:
        """Return the tensor data type."""
        return self._dtype

    @property
    def writable(self) -> bool:
        """Return whether this buffer is writable."""
        return self._writable

    @property
    def size_bytes(self) -> int:
        """Return total element byte count (dense size)."""
        if not self._shape or self._dtype == lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN:
            return 0
        return prod(self._shape) * data_type_size_bytes(self._dtype)

    def is_contiguous(self) -> bool:
        """Return whether this buffer is contiguous in row-major order."""
        return self._strides == default_strides(self._shape)

    def is_host(self) -> bool:
        """Return whether this buffer is in host memory."""
        return self._allocation is not None and self._allocation.is_host()

    def external(self, name: str) -> Self:
        """Register this buffer view as a named external buffer."""
        if self._builder is None or self._allocation is None:
            message = "Buffer is not attached to a BufferBuilder."
            raise RuntimeError(message)
        self._builder.register_external_view(self, name=name)
        return self

    def as_tensor(
        self,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType | None = None,
    ) -> "Buffer":
        """Return a shaped, contiguous view of this buffer's storage.

        `temporary_buffer` hands out an opaque range with no shape, which cannot
        be indexed.  This attaches a row-major interpretation to it so it can be
        sliced by sample; the storage, offset and writability are unchanged.
        """
        return Buffer(
            storage=self.root_storage,
            offset=self._offset,
            shape=shape,
            strides=None,
            dtype=self._dtype if dtype is None else dtype,
            allocation=self._allocation,
            builder=self._builder,
            writable=self._writable,
        )

    def transpose(self, dim0: int, dim1: int) -> "Buffer":
        """Return a transposed view swapping *dim0* and *dim1*."""
        ndim = len(self._shape)
        norm_dim0 = dim0 + ndim if dim0 < 0 else dim0
        norm_dim1 = dim1 + ndim if dim1 < 0 else dim1
        if not (0 <= norm_dim0 < ndim and 0 <= norm_dim1 < ndim):
            message = f"Dimension out of range: {dim0}, {dim1} for ndim {ndim}"
            raise IndexError(message)
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        new_shape[norm_dim0], new_shape[norm_dim1] = (
            new_shape[norm_dim1],
            new_shape[norm_dim0],
        )
        new_strides[norm_dim0], new_strides[norm_dim1] = (
            new_strides[norm_dim1],
            new_strides[norm_dim0],
        )
        return Buffer(
            storage=self.root_storage,
            offset=self._offset,
            shape=new_shape,
            strides=new_strides,
            dtype=self._dtype,
            allocation=self._allocation,
            builder=self._builder,
            writable=self._writable,
        )

    def __getitem__(self, index: object) -> "Buffer":  # noqa: C901
        """Return a sliced view into this tensor."""
        ndim = len(self._shape)
        if ndim == 0:
            message = "Cannot index a 0-dimensional tensor"
            raise IndexError(message)

        indices: list[object] = list(index) if isinstance(index, tuple) else [index]

        ellipsis_count = indices.count(Ellipsis)
        if ellipsis_count > 1:
            message = "An index can only have a single ellipsis ('...')"
            raise IndexError(message)
        if ellipsis_count == 1:
            ellipsis_idx = indices.index(Ellipsis)
            num_missing = ndim - (len(indices) - 1)
            indices[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * num_missing

        if len(indices) < ndim:
            indices.extend([slice(None)] * (ndim - len(indices)))
        elif len(indices) > ndim:
            message = (
                f"Too many indices for tensor of dimension {ndim}: got {len(indices)}"
            )
            raise IndexError(message)

        element_size = (
            data_type_size_bytes(self._dtype)
            if self._dtype != lc0ex_pb2.Buffer.DATA_TYPE_UNKNOWN
            else 1
        )
        new_offset = self._offset
        new_shape: list[int] = []
        new_strides: list[int] = []

        for i, idx in enumerate(indices):
            dim_size = self._shape[i]
            stride = self._strides[i]
            if isinstance(idx, int):
                norm_idx = idx + dim_size if idx < 0 else idx
                if not (0 <= norm_idx < dim_size):
                    message = (
                        f"Index {idx} out of bounds for dimension {i} of size "
                        f"{dim_size}"
                    )
                    raise IndexError(message)
                new_offset += norm_idx * stride * element_size
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(dim_size)
                if step == 0:
                    message = "Slice step cannot be zero"
                    raise ValueError(message)
                dim_len = max(
                    0,
                    (stop - start + (step - 1 if step > 0 else step + 1)) // step,
                )
                new_offset += start * stride * element_size
                new_shape.append(dim_len)
                new_strides.append(stride * step)
            else:
                message = f"Invalid index type: {type(idx)}"
                raise TypeError(message)

        return Buffer(
            storage=self.root_storage,
            offset=new_offset,
            shape=new_shape,
            strides=new_strides,
            dtype=self._dtype,
            allocation=self._allocation,
            builder=self._builder,
            writable=self._writable,
        )


@dataclass(frozen=True, slots=True)
class _ExternalBuffer:
    """Canonical metadata for one named external range."""

    name: str
    shape: tuple[int, ...]
    dtype: lc0ex_pb2.Buffer.DataType
    strides: tuple[int, ...] | None = None
    offset_in_storage: int = 0

@dataclass(frozen=True, slots=True)
class _HostBuffer:
    """Canonical metadata for one named host buffer range."""

    name: str
    shape: tuple[int, ...]
    dtype: lc0ex_pb2.Buffer.DataType
    strides: tuple[int, ...] | None = None

@dataclass(frozen=True, slots=True)
class _BufferRecord:
    """Private physical metadata for one opaque buffer handle."""

    allocation: Allocation
    size_bytes: int
    alignment_bytes: int
    external: _ExternalBuffer | None
    writable: bool


@dataclass(slots=True)
class _AllocationSlot:
    """One reusable raw-storage range in an allocation."""

    buffers: list[Buffer]
    size_bytes: int
    alignment_bytes: int


@dataclass(frozen=True, slots=True)
class BufferLocation:
    """The planned location of one logical buffer."""

    allocation: Allocation
    offset: int


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    """The packed layout of one persistent or execution allocation."""

    size_bytes: int
    alignment_bytes: int
    locations: dict[Buffer, BufferLocation]
    external_buffers: tuple[tuple[Buffer, _ExternalBuffer], ...]


class BufferBuilder:
    """Collect opaque buffers and pack each allocation."""

    def __init__(self) -> None:
        """Initialize an empty allocation collection."""
        self._persistent = Allocation(_type=BufferType.PERSISTENT)
        self._buffers_by_allocation: dict[Allocation, list[Buffer]] = {
            self._persistent: [],
        }
        self._records: dict[Buffer, _BufferRecord] = {}
        self._external_by_allocation_and_name: dict[tuple[Allocation, str], Buffer] = {}
        self._external_views: list[tuple[Buffer, _ExternalBuffer]] = []
        self._external_roots: set[Buffer] = set()

    def persistent_allocation(self) -> Allocation:
        """Return the executable-wide persistent allocation."""
        return self._persistent

    def execution_allocation(self) -> Allocation:
        """Create and return a new program execution allocation."""
        allocation = Allocation(_type=BufferType.EXECUTION)
        self._buffers_by_allocation[allocation] = []
        return allocation

    def host_allocation(self) -> Allocation:
        """Create and return a new host memory allocation."""
        allocation = Allocation(_type=BufferType.HOST)
        self._buffers_by_allocation[allocation] = []
        return allocation

    def register_external_view(self, buffer: Buffer, *, name: str) -> None:
        """Register a buffer view as a named external buffer."""
        root = buffer.root_storage
        record = self._records[root]
        key = (record.allocation, name)
        if key in self._external_by_allocation_and_name:
            message = f"Duplicate external buffer name '{name}' in allocation."
            raise ValueError(message)
        self._external_by_allocation_and_name[key] = buffer
        self._external_roots.add(root)
        external = _ExternalBuffer(
            name=name,
            shape=buffer.shape,
            dtype=buffer.dtype,
            strides=buffer.strides,
            offset_in_storage=buffer.offset,
        )
        self._external_views.append((buffer, external))

    def persistent_tensor(
        self,
        *,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create an unnamed persistent tensor in the persistent allocation."""
        return self.tensor(
            self._persistent,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )

    def tensor(
        self,
        allocation: Allocation,
        *,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool = False,
        alignment_bytes: int | None = None,
    ) -> Buffer:
        """Create an unnamed tensor in *allocation*."""
        normalized_shape = tuple(shape)
        element_size = data_type_size_bytes(dtype)
        resolved_alignment = (
            element_size if alignment_bytes is None else alignment_bytes
        )
        size_bytes = prod(normalized_shape) * element_size

        buffer = Buffer(
            shape=normalized_shape,
            dtype=dtype,
            allocation=allocation,
            builder=self,
            writable=writable,
        )
        self._add_buffer(
            buffer,
            _BufferRecord(
                allocation,
                size_bytes,
                resolved_alignment,
                None,
                writable,
            ),
        )
        return buffer

    def external_buffer(  # noqa: PLR0913
        self,
        allocation: Allocation,
        *,
        name: str,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool,
        alignment_bytes: int | None,
    ) -> Buffer:
        """Create or retrieve a named external buffer in *allocation*."""
        key = (allocation, name)
        existing = self._external_by_allocation_and_name.get(key)
        if existing is not None:
            return existing

        buf = self.tensor(
            allocation,
            shape=shape,
            dtype=dtype,
            writable=writable,
            alignment_bytes=alignment_bytes,
        )
        buf.external(name)
        return buf

    def host_buffer(
        self,
        *,
        shape: Sequence[int],
        dtype: lc0ex_pb2.Buffer.DataType,
        writable: bool,
    ) -> Buffer:
        """Create or retrieve a named host buffer in the persistent allocation."""
        return self.tensor(
            self.host_allocation(),
            shape=shape,
            dtype=dtype,
            writable=writable,
        )

    def temporary_buffer(
        self,
        allocation: Allocation,
        *,
        size_bytes: int,
        alignment_bytes: int,
    ) -> Buffer:
        """Create an unnamed raw-storage buffer in an execution allocation."""
        buffer = Buffer(
            allocation=allocation,
            builder=self,
            writable=True,
        )
        self._add_buffer(
            buffer,
            _BufferRecord(
                allocation,
                size_bytes,
                alignment_bytes,
                None,
                writable=True,
            ),
        )
        return buffer

    def storage_size_bytes(self, buffer: Buffer) -> int:
        """Return the byte extent of *buffer*'s root storage.

        `Buffer.size_bytes` is the dense size of a SHAPED buffer and is 0 for the
        opaque raw temporaries `temporary_buffer` hands out, so the physical
        extent lives only in the record.
        """
        return self._records[buffer.root_storage].size_bytes

    def is_reusable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* is an internal reusable range."""
        root = buffer.root_storage
        return self._records[root].external is None and root not in self._external_roots

    def is_writable(self, buffer: Buffer) -> bool:
        """Return whether *buffer* may be modified by graph nodes."""
        return buffer.writable and self._records[buffer.root_storage].writable

    def share_allocation(self, first: Buffer, second: Buffer) -> bool:
        """Return whether two opaque ranges belong to one allocation."""
        return (
            self._records[first.root_storage].allocation
            is self._records[second.root_storage].allocation
        )

    def plan(
        self,
        allocation: Allocation,
        reusable_conflicts: dict[Buffer, set[Buffer]],
    ) -> AllocationPlan | None:
        """Pack one allocation and return its serialized plan."""
        buffers = self._buffers_by_allocation[allocation]
        fixed_buffers = [
            buffer
            for buffer in buffers
            if self._records[buffer].external is not None
            or buffer in self._external_roots
            or allocation.is_persistent()
        ]
        reusable_buffers = [
            buffer
            for buffer in buffers
            if buffer not in fixed_buffers and buffer in reusable_conflicts
        ]
        if not fixed_buffers and not reusable_buffers:
            return None

        locations: dict[Buffer, BufferLocation] = {}
        allocation_size = 0
        allocation_alignment = 1
        for buffer in fixed_buffers:
            record = self._records[buffer]
            allocation_alignment = max(allocation_alignment, record.alignment_bytes)
            allocation_size = _align_up(allocation_size, record.alignment_bytes)
            locations[buffer] = BufferLocation(allocation, allocation_size)
            allocation_size += record.size_bytes

        allocation_size, reusable_alignment, reusable_offsets = (
            self._build_reusable_ranges(
                reusable_buffers,
                reusable_conflicts,
                allocation_size,
            )
        )
        allocation_alignment = max(allocation_alignment, reusable_alignment)
        for buffer, offset in reusable_offsets.items():
            locations[buffer] = BufferLocation(allocation, offset)

        external_buffers_list: list[tuple[Buffer, _ExternalBuffer]] = []
        for buffer, external in self._external_views:
            if buffer._allocation is allocation:  # noqa: SLF001
                root_loc = locations[buffer.root_storage]
                locations[buffer] = BufferLocation(
                    allocation, root_loc.offset + buffer.offset
                )
                external_buffers_list.append((buffer, external))

        for buffer in fixed_buffers:
            record = self._records[buffer]
            if record.external is not None and not any(
                b is buffer for b, _ in external_buffers_list
            ):
                external_buffers_list.append((buffer, record.external))

        return AllocationPlan(
            allocation_size,
            allocation_alignment,
            locations,
            tuple(external_buffers_list),
        )

    def _add_buffer(self, buffer: Buffer, record: _BufferRecord) -> None:
        """Register one newly created opaque buffer handle."""
        self._buffers_by_allocation[record.allocation].append(buffer)
        self._records[buffer] = record

    def _build_reusable_ranges(
        self,
        buffers: list[Buffer],
        conflicts: dict[Buffer, set[Buffer]],
        allocation_size: int,
    ) -> tuple[int, int, dict[Buffer, int]]:
        """Pack internal raw ranges into reusable slots."""
        slots: list[_AllocationSlot] = []
        for buffer in buffers:
            record = self._records[buffer]
            compatible_slots = [
                slot
                for slot in slots
                if all(member not in conflicts[buffer] for member in slot.buffers)
            ]
            if compatible_slots:
                slot = min(
                    compatible_slots,
                    key=lambda candidate: (
                        candidate.size_bytes != record.size_bytes,
                        max(candidate.size_bytes, record.size_bytes),
                        max(candidate.alignment_bytes, record.alignment_bytes),
                    ),
                )
                slot.size_bytes = max(slot.size_bytes, record.size_bytes)
                slot.alignment_bytes = max(slot.alignment_bytes, record.alignment_bytes)
                slot.buffers.append(buffer)
            else:
                slots.append(
                    _AllocationSlot([buffer], record.size_bytes, record.alignment_bytes)
                )

        allocation_alignment = 1
        offsets: dict[Buffer, int] = {}
        for slot in slots:
            allocation_alignment = max(allocation_alignment, slot.alignment_bytes)
            allocation_size = _align_up(allocation_size, slot.alignment_bytes)
            for buffer in slot.buffers:
                offsets[buffer] = allocation_size
            allocation_size += slot.size_bytes
        return allocation_size, allocation_alignment, offsets


def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to a multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment
