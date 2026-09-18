"""Tests for on-demand BT4 kernel specialization caching."""

import logging
from dataclasses import dataclass

import pytest
from lc0ex import ExecutableBuilder, KernelArtifact
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache

POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER


@dataclass(frozen=True, slots=True)
class _Specialization:
    """A minimal immutable compile-time kernel contract."""

    width: int


def _compiler(spec: _Specialization) -> KernelArtifact:
    """Build a unique test artifact for one specialization."""
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=spec.width.to_bytes(),
        function=f"kernel_{spec.width}",
        parameters=(POINTER,),
        grid=(1, 1, 1),
        block=(32, 1, 1),
        dynamic_shared_memory_bytes=0,
    )


def test_kernel_cache_reuses_one_compiler_specialization_pair() -> None:
    """Equivalent immutable requests compile once and reuse one handle."""
    cache = KernelCache(ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16))

    first = cache.get(_compiler, _Specialization(32))
    second = cache.get(_compiler, _Specialization(32))

    assert second is first


def test_kernel_cache_logs_compile_and_reuse(caplog: pytest.LogCaptureFixture) -> None:
    """Compilation progress is visible while cache hits remain debug-level."""
    cache = KernelCache(ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16))
    caplog.set_level(logging.DEBUG)

    cache.get(_compiler, _Specialization(32))
    cache.get(_compiler, _Specialization(32))

    messages = [record.getMessage() for record in caplog.records]
    assert sum("compiling kernel" in message for message in messages) == 1
    assert (
        sum("kernel _compiler compilation completed" in message for message in messages)
        == 1
    )
    assert sum("reusing compiled kernel" in message for message in messages) == 1


def test_kernel_cache_keeps_distinct_specializations_separate() -> None:
    """A changed compile-time parameter receives a distinct artifact handle."""
    cache = KernelCache(ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16))

    first = cache.get(_compiler, _Specialization(32))
    second = cache.get(_compiler, _Specialization(64))

    assert second is not first
