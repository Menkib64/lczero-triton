"""Tests for real BT4 Triton compilation and graph registration."""

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.add_bias_batched import (
    AddBiasBatchedSpecialization,
    _add_bias_batched_kernel,
    compile_add_bias_batched,
)
from lczero_triton.bt4.kernels.add_bias_batched import (
    _artifact_grid as add_bias_artifact_grid,
)
from lczero_triton.bt4.kernels.add_vectors import (
    AddVectorsSpecialization,
    _add_vectors_kernel,
    compile_add_vectors,
)
from lczero_triton.bt4.kernels.add_vectors import (
    _artifact_grid as add_vectors_artifact_grid,
)
from lczero_triton.bt4.kernels.copy_type_converted import (
    CopyTypeConvertedSpecialization,
    _copy_type_converted_kernel,
    compile_copy_type_converted,
    copy_type_converted,
)
from lczero_triton.bt4.kernels.copy_type_converted import (
    _artifact_grid as copy_artifact_grid,
)
from lczero_triton.bt4.kernels.expand_planes import (
    ExpandPlanesSpecialization,
    _expand_planes_kernel,
    compile_expand_planes,
)
from lczero_triton.bt4.kernels.expand_planes import (
    _artifact_grid as expand_artifact_grid,
)
from lczero_triton.bt4.kernels.input_gating import (
    InputGatingSpecialization,
    _input_gating_kernel,
    compile_input_gating,
)
from lczero_triton.bt4.kernels.input_gating import (
    _artifact_grid as gating_artifact_grid,
)
from lczero_triton.bt4.kernels.mapping_table import compile_symbol
from lczero_triton.bt4.kernels.nchw_to_nhwc import (
    NchwToNhwcSpecialization,
    _nchw_to_nhwc_kernel,
    compile_nchw_to_nhwc,
)
from lczero_triton.bt4.kernels.nchw_to_nhwc import (
    _artifact_grid as nchw_artifact_grid,
)
from lczero_triton.bt4.kernels.policy_map import (
    PolicyMapSpecialization,
    _policy_map_kernel,
    compile_policy_map,
    policy_map,
)
from lczero_triton.bt4.kernels.policy_map import (
    _artifact_grid as policy_artifact_grid,
)
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    PreprocessAttentionBodySpecialization,
    _preprocess_attention_body_kernel,
    compile_preprocess_attention_body,
)
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    _artifact_grid as preprocess_artifact_grid,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_THREE_POINTERS = 3
_NULL_POINTERS = (lc0ex_pb2.PARAMETER_TYPE_NULL_POINTER,) * 2
_WARP_SIZE = 32


def _architecture() -> int:
    """Return the active CUDA device's `sm_*` integer suffix."""
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def test_real_compilation_produces_pointer_abi_and_static_launch() -> None:
    """A real Triton result becomes an in-memory linker artifact."""
    artifact = compile_copy_type_converted(
        CopyTypeConvertedSpecialization(257, _architecture())
    )
    selected = _copy_type_converted_kernel.best_config

    assert artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert artifact.binary_data
    assert artifact.function
    assert artifact.parameters == (
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        lc0ex_pb2.PARAMETER_TYPE_POINTER,
        *_NULL_POINTERS,
    )
    assert artifact.grid == copy_artifact_grid(selected.kwargs, 257)
    assert artifact.block == (selected.num_warps * _WARP_SIZE, 1, 1)


def test_copy_graph_call_preserves_output_input_argument_order() -> None:
    """The family API serializes destination before its readonly source."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    output = program.buffer(
        name="output",
        shape=(257,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32,
        writable=True,
    )
    input_ = program.buffer(
        name="input",
        shape=(257,),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    copy_type_converted(
        program,
        KernelCache(builder),
        output,
        input_,
        CopyTypeConvertedSpecialization(257, _architecture()),
    )

    executable = builder.build()
    node = executable.programs[0].nodes[0]
    locations = {
        buffer.name: (buffer.offset,)
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }

    assert executable.target.architecture == f"sm_{_architecture()}"
    assert (node.arguments[0].allocation.offset,) == locations["output"]
    assert (node.arguments[1].allocation.offset,) == locations["input"]


def test_remaining_step_five_families_capture_autotuned_artifacts() -> None:
    """Every remaining family captures its selected launch and pointer ABI."""
    architecture = _architecture()
    expand_artifact = compile_expand_planes(ExpandPlanesSpecialization(5, architecture))
    nchw_artifact = compile_nchw_to_nhwc(
        NchwToNhwcSpecialization(2, 7, 5, 3, 5, architecture)
    )
    preprocess_artifact = compile_preprocess_attention_body(
        PreprocessAttentionBodySpecialization(2, 5, 11, architecture, 7)
    )
    gating_artifact = compile_input_gating(
        InputGatingSpecialization(2, 3, 43, architecture)
    )
    add_vectors_artifact = compile_add_vectors(
        AddVectorsSpecialization(259, 7, "mish", architecture)
    )
    add_bias_artifact = compile_add_bias_batched(
        AddBiasBatchedSpecialization(3, 5, 37, "mish", architecture)
    )
    policy_artifact = compile_policy_map(PolicyMapSpecialization(2, architecture))

    artifacts_and_kernels = (
        (expand_artifact, _expand_planes_kernel, _THREE_POINTERS),
        (nchw_artifact, _nchw_to_nhwc_kernel, 2),
        (preprocess_artifact, _preprocess_attention_body_kernel, _THREE_POINTERS),
        (gating_artifact, _input_gating_kernel, 4),
        (add_vectors_artifact, _add_vectors_kernel, _THREE_POINTERS),
        (add_bias_artifact, _add_bias_batched_kernel, _THREE_POINTERS),
        (policy_artifact, _policy_map_kernel, _THREE_POINTERS),
    )
    for artifact, kernel, pointer_count in artifacts_and_kernels:
        assert len(artifact.parameters) == pointer_count + len(_NULL_POINTERS)
        assert tuple(artifact.parameters[-len(_NULL_POINTERS) :]) == _NULL_POINTERS
        assert artifact.binary_data
        assert artifact.block == (kernel.best_config.num_warps * _WARP_SIZE, 1, 1)

    assert expand_artifact.grid == expand_artifact_grid(
        _expand_planes_kernel.best_config.kwargs, 5, 64
    )
    assert nchw_artifact.grid == nchw_artifact_grid(
        _nchw_to_nhwc_kernel.best_config.kwargs, 2 * 3 * 5 * 5
    )
    assert preprocess_artifact.grid == preprocess_artifact_grid(
        _preprocess_attention_body_kernel.best_config.kwargs, 2, 7, 16
    )
    assert gating_artifact.grid == gating_artifact_grid(
        _input_gating_kernel.best_config.kwargs, 2 * 3 * 43
    )
    assert add_vectors_artifact.grid == add_vectors_artifact_grid(
        _add_vectors_kernel.best_config.kwargs, 259
    )
    assert add_bias_artifact.grid == add_bias_artifact_grid(
        _add_bias_batched_kernel.best_config.kwargs, 3 * 5 * 37
    )
    assert policy_artifact.grid == policy_artifact_grid(
        _policy_map_kernel.best_config.kwargs, 2 * 1858
    )


def test_policy_map_serializes_embedded_symbol_argument() -> None:
    """Policy gathering uses a module symbol rather than an external buffer."""
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    program = builder.program(name="main")
    output = program.buffer(
        name="output",
        shape=(2, 1858),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
        writable=True,
    )
    input_ = program.buffer(
        name="input",
        shape=(2, 4288),
        dtype=lc0ex_pb2.Buffer.DATA_TYPE_F16,
    )
    architecture = _architecture()
    mapping = program.add_symbol(compile_symbol(architecture=f"sm_{architecture}"))
    policy_map(
        program,
        KernelCache(builder),
        output,
        input_,
        mapping,
        PolicyMapSpecialization(2, architecture),
    )

    executable = builder.build()
    mapping_argument = executable.programs[0].nodes[0].arguments[2]

    assert mapping_argument.symbol.symbol_name == "lczero_bt4_mapping_table"
    assert not mapping_argument.HasField("allocation")
    assert "/const/mapping_table" not in {buffer.name for buffer in executable.buffers}
