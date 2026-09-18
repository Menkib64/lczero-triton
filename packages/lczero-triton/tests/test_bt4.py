"""Tests for BT4 graph construction and its input-embedding segment."""

from collections.abc import Callable
from typing import cast

import lczero_triton.bt4.kernels.add_bias_batched as add_bias_batched_module
import lczero_triton.bt4.kernels.add_vectors as add_vectors_module
import lczero_triton.bt4.kernels.batched_matmul as batched_matmul_module
import lczero_triton.bt4.kernels.copy_type_converted as copy_type_converted_module
import lczero_triton.bt4.kernels.expand_planes as expand_planes_module
import lczero_triton.bt4.kernels.fused_attention as fused_attention_module
import lczero_triton.bt4.kernels.fused_qkv as fused_qkv_module
import lczero_triton.bt4.kernels.input_gating as input_gating_module
import lczero_triton.bt4.kernels.layer_norm as layer_norm_module
import lczero_triton.bt4.kernels.mapping_table as mapping_table_module
import lczero_triton.bt4.kernels.matmul as matmul_module
import lczero_triton.bt4.kernels.nchw_to_nhwc as nchw_to_nhwc_module
import lczero_triton.bt4.kernels.policy_map as policy_map_module
import lczero_triton.bt4.kernels.preprocess_attention_body as preprocess_module
import lczero_triton.bt4.kernels.promotion_logits as promotion_logits_module
import lczero_triton.bt4.kernels.softmax_64 as softmax_64_module
import lczero_triton.bt4.network as network_module
import pytest
from lc0ex import Buffer, ExecutableBuilder, KernelArtifact, SymbolArtifact
from lc0ex.proto import lc0ex_metadata_pb2, lc0ex_pb2, net_pb2
from lczero_triton.bt4.kernels.add_vectors import AddVectorsSpecialization
from lczero_triton.bt4.kernels.batched_matmul import BatchedMatmulSpecialization
from lczero_triton.bt4.kernels.expand_planes import ExpandPlanesSpecialization
from lczero_triton.bt4.kernels.fused_attention import (
    FusedAttentionSpecialization,
)
from lczero_triton.bt4.kernels.input_gating import InputGatingSpecialization
from lczero_triton.bt4.kernels.layer_norm import LayerNormSpecialization
from lczero_triton.bt4.kernels.matmul import MatmulSpecialization
from lczero_triton.bt4.kernels.nchw_to_nhwc import NchwToNhwcSpecialization
from lczero_triton.bt4.kernels.policy_map import PolicyMapSpecialization
from lczero_triton.bt4.kernels.preprocess_attention_body import (
    PreprocessAttentionBodySpecialization,
)
from lczero_triton.bt4.kernels.promotion_logits import (
    PromotionLogitsSpecialization,
)
from lczero_triton.bt4.network import (
    _BuildContext,
    _default_activation,
    _resolve_activation,
    build,
)

_POINTER = lc0ex_pb2.PARAMETER_TYPE_POINTER
_ARCHITECTURE = 80
_REUSED_TEMPORARY_COUNT = 3
_ENCODER_LOCAL_BUFFER_COUNT = 28
_EMBEDDING_FINGERPRINT_LAYER_COUNT = 14
_FULL_FINGERPRINT_LAYER_COUNT = 59


def _active_architecture() -> int:
    """Return a stable target for compiler-free structural tests."""
    return _ARCHITECTURE


def _target_skeleton() -> net_pb2.Net:
    """Create a minimal supported multihead network without large payloads."""
    network = net_pb2.Net()
    network.format.weights_encoding = net_pb2.Format.LINEAR16
    network.format.network_format.input = (
        net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    )
    network.format.network_format.network = (
        net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    )
    network.format.network_format.policy = net_pb2.NetworkFormat.POLICY_ATTENTION
    network.format.network_format.value = net_pb2.NetworkFormat.VALUE_WDL
    network.format.network_format.moves_left = net_pb2.NetworkFormat.MOVES_LEFT_V1
    network.format.network_format.default_activation = (
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH
    )
    network.weights.headcount = 1
    network.weights.encoder.add()
    for field in (
        "ip2_pol_w",
        "ip2_pol_b",
        "ip3_pol_w",
        "ip3_pol_b",
        "ip4_pol_w",
    ):
        getattr(network.weights.policy_heads.vanilla, field).params = b"\0\0"
    for field in (
        "ip_val_w",
        "ip_val_b",
        "ip1_val_w",
        "ip1_val_b",
        "ip2_val_w",
        "ip2_val_b",
    ):
        getattr(network.weights.value_heads.winner, field).params = b"\0\0"
    for field in (
        "ip_mov_w",
        "ip_mov_b",
        "ip1_mov_w",
        "ip1_mov_b",
        "ip2_mov_w",
        "ip2_mov_b",
    ):
        getattr(network.weights, field).params = b"\0\0"
    return network


def _set_elements(layer: net_pb2.Weights.Layer, element_count: int) -> None:
    """Populate one synthetic LINEAR16 layer with the requested element count."""
    layer.params = b"\0\0" * element_count


def _embedding_network(
    *,
    encoding_width: int,
    body_width: int,
    hidden_width: int,
) -> net_pb2.Net:
    """Create a compact format-valid network with inferred embedding widths."""
    network = _target_skeleton()
    weights = network.weights
    position_output_width = 64 * encoding_width
    embedding_input_width = 112 + encoding_width

    _set_elements(weights.ip_emb_preproc_w, 64 * 12 * position_output_width)
    _set_elements(weights.ip_emb_preproc_b, position_output_width)
    _set_elements(weights.ip_emb_w, embedding_input_width * body_width)
    _set_elements(weights.ip_emb_b, body_width)
    _set_elements(weights.ip_emb_ln_gammas, body_width)
    _set_elements(weights.ip_emb_ln_betas, body_width)
    _set_elements(weights.ip_mult_gate, 64 * body_width)
    _set_elements(weights.ip_add_gate, 64 * body_width)
    _set_elements(weights.ip_emb_ffn.dense1_w, body_width * hidden_width)
    _set_elements(weights.ip_emb_ffn.dense1_b, hidden_width)
    _set_elements(weights.ip_emb_ffn.dense2_w, hidden_width * body_width)
    _set_elements(weights.ip_emb_ffn.dense2_b, body_width)
    _set_elements(weights.ip_emb_ffn_ln_gammas, body_width)
    _set_elements(weights.ip_emb_ffn_ln_betas, body_width)
    return network


def _add_encoder(  # noqa: PLR0913
    network: net_pb2.Net,
    *,
    body_width: int,
    compression_width: int,
    hidden_width: int,
    generated_width: int,
    model_width: int,
    ffn_width: int,
) -> None:
    """Populate the skeleton encoder with locally inferred dimensions."""
    encoder = network.weights.encoder[0]
    heads = network.weights.headcount
    smolgen = encoder.mha.smolgen
    _set_elements(smolgen.compress, body_width * compression_width)
    _set_elements(smolgen.dense1_w, 64 * compression_width * hidden_width)
    _set_elements(smolgen.dense1_b, hidden_width)
    _set_elements(smolgen.ln1_gammas, hidden_width)
    _set_elements(smolgen.ln1_betas, hidden_width)
    _set_elements(smolgen.dense2_w, hidden_width * heads * generated_width)
    _set_elements(smolgen.dense2_b, heads * generated_width)
    _set_elements(smolgen.ln2_gammas, heads * generated_width)
    _set_elements(smolgen.ln2_betas, heads * generated_width)
    for weight, bias in (
        (encoder.mha.q_w, encoder.mha.q_b),
        (encoder.mha.k_w, encoder.mha.k_b),
        (encoder.mha.v_w, encoder.mha.v_b),
    ):
        _set_elements(weight, body_width * model_width)
        _set_elements(bias, model_width)
    _set_elements(encoder.mha.dense_w, model_width * body_width)
    _set_elements(encoder.mha.dense_b, body_width)
    _set_elements(encoder.ln1_gammas, body_width)
    _set_elements(encoder.ln1_betas, body_width)
    _set_elements(encoder.ffn.dense1_w, body_width * ffn_width)
    _set_elements(encoder.ffn.dense1_b, ffn_width)
    _set_elements(encoder.ffn.dense2_w, ffn_width * body_width)
    _set_elements(encoder.ffn.dense2_b, body_width)
    _set_elements(encoder.ln2_gammas, body_width)
    _set_elements(encoder.ln2_betas, body_width)
    _set_elements(network.weights.smolgen_w, generated_width * 64 * 64)
    network.format.network_format.smolgen_activation = (
        net_pb2.NetworkFormat.ACTIVATION_SWISH
    )


def _artifact(function: str, parameter_count: int) -> KernelArtifact:
    """Return a CUDA-shaped artifact without compiling or launching a kernel."""
    return KernelArtifact(
        binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
        binary_data=b"stub",
        function=function,
        parameters=(_POINTER,) * parameter_count,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        dynamic_shared_memory_bytes=0,
    )


def _compiler(
    records: list[tuple[str, object]],
    function: str,
    parameter_count: int,
) -> Callable[[object], KernelArtifact]:
    """Create a compiler stub that records its immutable specialization."""

    def compile_stub(specialization: object) -> KernelArtifact:
        records.append((function, specialization))
        return _artifact(function, parameter_count)

    return compile_stub


def _stub_compilers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, object]]:
    """Replace item-11 compilers while retaining the real graph wrappers."""
    records: list[tuple[str, object]] = []
    monkeypatch.setattr(
        expand_planes_module,
        "compile_expand_planes",
        _compiler(records, "expand_planes", 3),
    )
    monkeypatch.setattr(
        nchw_to_nhwc_module,
        "compile_nchw_to_nhwc",
        _compiler(records, "nchw_to_nhwc", 2),
    )

    def compile_matmul(specialization: MatmulSpecialization) -> KernelArtifact:
        records.append(("matmul", specialization))
        return _artifact("matmul", 6 if specialization.has_skip else 4)

    monkeypatch.setattr(
        matmul_module,
        "compile_matmul",
        compile_matmul,
    )
    monkeypatch.setattr(
        add_vectors_module,
        "compile_add_vectors",
        _compiler(records, "add_vectors", 3),
    )
    monkeypatch.setattr(
        preprocess_module,
        "compile_preprocess_attention_body",
        _compiler(records, "preprocess_attention_body", 3),
    )
    monkeypatch.setattr(
        input_gating_module,
        "compile_input_gating",
        _compiler(records, "input_gating", 4),
    )
    monkeypatch.setattr(
        add_bias_batched_module,
        "compile_add_bias_batched",
        _compiler(records, "add_bias_batched", 3),
    )
    monkeypatch.setattr(
        fused_qkv_module,
        "compile_fused_qkv_projection",
        _compiler(records, "fused_qkv_projection", 10),
    )
    monkeypatch.setattr(
        fused_qkv_module,
        "compile_fused_qkv_bias",
        _compiler(records, "fused_qkv_bias", 9),
    )
    monkeypatch.setattr(
        fused_attention_module,
        "compile_fused_attention",
        _compiler(records, "fused_attention", 4),
    )

    def compile_batched_matmul(
        specialization: BatchedMatmulSpecialization,
    ) -> KernelArtifact:
        records.append(("batched_matmul", specialization))
        parameter_count = 3 if specialization.operation == "body_attention_v" else 4
        return _artifact(specialization.operation, parameter_count)

    monkeypatch.setattr(
        batched_matmul_module,
        "compile_batched_matmul",
        compile_batched_matmul,
    )
    monkeypatch.setattr(
        softmax_64_module,
        "compile_softmax_64",
        _compiler(records, "softmax_64", 3),
    )

    def compile_layer_norm(specialization: LayerNormSpecialization) -> KernelArtifact:
        function = "layer_norm_skip" if specialization.has_skip else "layer_norm"
        records.append((function, specialization))
        return _artifact(function, 7 if specialization.has_skip else 5)

    monkeypatch.setattr(
        layer_norm_module,
        "compile_layer_norm",
        compile_layer_norm,
    )
    monkeypatch.setattr(network_module, "active_architecture", _active_architecture)
    return records


def _stub_head_compilers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, object]]:
    """Replace all graph compilers, including output-head-only families."""
    records = _stub_compilers(monkeypatch)
    monkeypatch.setattr(
        promotion_logits_module,
        "compile_promotion_logits",
        _compiler(records, "promotion_logits", 3),
    )
    monkeypatch.setattr(
        policy_map_module,
        "compile_policy_map",
        _compiler(records, "policy_map", 3),
    )
    monkeypatch.setattr(
        copy_type_converted_module,
        "compile_copy_type_converted",
        _compiler(records, "copy_type_converted", 2),
    )
    monkeypatch.setattr(
        mapping_table_module,
        "compile_symbol",
        lambda *, architecture: SymbolArtifact(  # noqa: ARG005
            binary_format=lc0ex_pb2.Binary.FORMAT_CUBIN,
            binary_data=b"mapping",
            symbol_name="lczero_bt4_mapping_table",
        ),
    )
    monkeypatch.setattr(
        network_module, "compile_symbol", mapping_table_module.compile_symbol
    )
    return records


def _add_output_heads(  # noqa: PLR0913
    network: net_pb2.Net,
    *,
    body_width: int,
    policy_width: int,
    policy_model_width: int,
    value_width: int,
    value_hidden_width: int,
    moves_width: int,
    moves_hidden_width: int,
) -> None:
    """Populate compact, dimensionally valid policy, WDL, and MLH heads."""
    weights = network.weights
    policy = weights.policy_heads.vanilla
    _set_elements(weights.policy_heads.ip_pol_w, body_width * policy_width)
    _set_elements(weights.policy_heads.ip_pol_b, policy_width)
    _set_elements(policy.ip2_pol_w, policy_width * policy_model_width)
    _set_elements(policy.ip2_pol_b, policy_model_width)
    _set_elements(policy.ip3_pol_w, policy_width * policy_model_width)
    _set_elements(policy.ip3_pol_b, policy_model_width)
    _set_elements(policy.ip4_pol_w, policy_model_width * 4)

    winner = weights.value_heads.winner
    _set_elements(winner.ip_val_w, body_width * value_width)
    _set_elements(winner.ip_val_b, value_width)
    _set_elements(winner.ip1_val_w, 64 * value_width * value_hidden_width)
    _set_elements(winner.ip1_val_b, value_hidden_width)
    _set_elements(winner.ip2_val_w, value_hidden_width * 3)
    _set_elements(winner.ip2_val_b, 3)

    _set_elements(weights.ip_mov_w, body_width * moves_width)
    _set_elements(weights.ip_mov_b, moves_width)
    _set_elements(weights.ip1_mov_w, 64 * moves_width * moves_hidden_width)
    _set_elements(weights.ip1_mov_b, moves_hidden_width)
    _set_elements(weights.ip2_mov_w, moves_hidden_width)
    _set_elements(weights.ip2_mov_b, 1)


def _keep_body(
    context: _BuildContext,
    body: Buffer,
    body_width: int,
    weights: net_pb2.Weights,
) -> Buffer:
    """Skip encoder construction while retaining the embedding body."""
    del context, body_width, weights
    return body


def _stop_after_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace later grammar productions with one read of the returned body."""

    def consume_body(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> Buffer:
        del body_width, weights
        kernel = context.builder.add_kernel(_artifact("consume_body", 1))
        context.builder.call(kernel, body, readonly=(body,))
        return body

    def ignore_policy(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del context, body, body_width, weights

    def ignore_value(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        winner: net_pb2.Weights.ValueHead,
    ) -> None:
        del context, body, body_width, winner

    def ignore_moves_left(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del context, body, body_width, weights

    monkeypatch.setattr(network_module, "_encoder_tower", consume_body)
    monkeypatch.setattr(network_module, "_policy_head", ignore_policy)
    monkeypatch.setattr(network_module, "_value_head", ignore_value)
    monkeypatch.setattr(network_module, "_moves_left_head", ignore_moves_left)


def _stop_after_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace output heads with one read of the final encoder body."""

    def consume_policy(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del body_width, weights
        kernel = context.builder.add_kernel(_artifact("consume_body", 1))
        context.builder.call(kernel, body, readonly=(body,))

    def ignore_value(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        winner: net_pb2.Weights.ValueHead,
    ) -> None:
        del context, body, body_width, winner

    def ignore_moves_left(
        context: _BuildContext,
        body: Buffer,
        body_width: int,
        weights: net_pb2.Weights,
    ) -> None:
        del context, body, body_width, weights

    monkeypatch.setattr(network_module, "_policy_head", consume_policy)
    monkeypatch.setattr(network_module, "_value_head", ignore_value)
    monkeypatch.setattr(network_module, "_moves_left_head", ignore_moves_left)


def _location(argument: lc0ex_pb2.Node.Argument) -> tuple[int, int]:
    """Return one serialized allocation argument location."""
    return argument.allocation.kind, argument.allocation.offset


def _parse_fingerprint(builder: ExecutableBuilder) -> net_pb2.Net:
    """Parse the sparse network fingerprint stored in executable metadata."""
    fingerprint = net_pb2.Net()
    fingerprint.ParseFromString(builder.build().metadata)
    return fingerprint


def _present_fingerprint_layers(
    fingerprint: net_pb2.Net,
) -> list[net_pb2.Weights.Layer]:
    """Return fingerprint layers whose proto2 parent fields are present."""
    weights = fingerprint.weights
    present: list[tuple[bool, net_pb2.Weights.Layer]] = [
        (weights.HasField(field), getattr(weights, field))
        for field in (
            "ip_emb_preproc_w",
            "ip_emb_preproc_b",
            "ip_emb_w",
            "ip_emb_b",
            "ip_emb_ln_gammas",
            "ip_emb_ln_betas",
            "ip_mult_gate",
            "ip_add_gate",
            "ip_emb_ffn_ln_gammas",
            "ip_emb_ffn_ln_betas",
            "smolgen_w",
            "ip_mov_w",
            "ip_mov_b",
            "ip1_mov_w",
            "ip1_mov_b",
            "ip2_mov_w",
            "ip2_mov_b",
        )
    ]
    if weights.HasField("ip_emb_ffn"):
        present.extend(
            (weights.ip_emb_ffn.HasField(field), getattr(weights.ip_emb_ffn, field))
            for field in ("dense1_w", "dense1_b", "dense2_w", "dense2_b")
        )
    for encoder in weights.encoder:
        present.extend(
            (encoder.mha.smolgen.HasField(field), getattr(encoder.mha.smolgen, field))
            for field in (
                "compress",
                "dense1_w",
                "dense1_b",
                "ln1_gammas",
                "ln1_betas",
                "dense2_w",
                "dense2_b",
                "ln2_gammas",
                "ln2_betas",
            )
        )
        present.extend(
            (encoder.mha.HasField(field), getattr(encoder.mha, field))
            for field in (
                "q_w",
                "q_b",
                "k_w",
                "k_b",
                "v_w",
                "v_b",
                "dense_w",
                "dense_b",
            )
        )
        present.extend(
            (encoder.HasField(field), getattr(encoder, field))
            for field in ("ln1_gammas", "ln1_betas", "ln2_gammas", "ln2_betas")
        )
        present.extend(
            (encoder.ffn.HasField(field), getattr(encoder.ffn, field))
            for field in ("dense1_w", "dense1_b", "dense2_w", "dense2_b")
        )
    if weights.HasField("policy_heads"):
        present.extend(
            (
                weights.policy_heads.vanilla.HasField(field),
                getattr(weights.policy_heads.vanilla, field),
            )
            for field in (
                "ip_pol_w",
                "ip_pol_b",
                "ip2_pol_w",
                "ip2_pol_b",
                "ip3_pol_w",
                "ip3_pol_b",
                "ip4_pol_w",
            )
        )
    if weights.HasField("value_heads"):
        present.extend(
            (
                weights.value_heads.winner.HasField(field),
                getattr(weights.value_heads.winner, field),
            )
            for field in (
                "ip_val_w",
                "ip_val_b",
                "ip1_val_w",
                "ip1_val_b",
                "ip2_val_w",
                "ip2_val_b",
            )
        )
    return [layer for is_present, layer in present if is_present]


def test_build_entry_point_is_importable() -> None:
    """BT4 graph construction is exposed through the protobuf-driven API."""
    assert callable(build)


def test_build_normalizes_and_builds_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normalized network builds through its encoder production."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    _add_encoder(
        network,
        body_width=16,
        compression_width=2,
        hidden_width=16,
        generated_width=16,
        model_width=16,
        ffn_width=32,
    )
    _stub_compilers(monkeypatch)
    _stop_after_encoders(monkeypatch)

    build(ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16), network, batch_sizes=(1,))

    assert (
        network.format.network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    assert (
        network.format.network_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    )


def test_build_stores_sparse_network_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executable metadata contains effective format and active layer slots."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(2,))

    fingerprint = _parse_fingerprint(builder)
    network_format = fingerprint.format.network_format
    assert network_format.input == net_pb2.NetworkFormat.INPUT_CLASSICAL_112_PLANE
    assert (
        network_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    assert network_format.output == net_pb2.NetworkFormat.OUTPUT_UNKNOWN
    assert (
        network_format.default_activation
        == net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH
    )
    assert network_format.ffn_activation == net_pb2.NetworkFormat.ACTIVATION_MISH
    assert network_format.smolgen_activation == net_pb2.NetworkFormat.ACTIVATION_MISH
    assert fingerprint.weights.headcount == 1
    assert len(fingerprint.weights.encoder) == 1
    assert fingerprint.weights.HasField("policy_heads") is False

    active_layers = _present_fingerprint_layers(fingerprint)
    assert len(active_layers) == _EMBEDDING_FINGERPRINT_LAYER_COUNT
    assert all(not layer.ListFields() for layer in active_layers)
    assert all(not layer.params for layer in active_layers)
    assert all(not layer.HasField("min_val") for layer in active_layers)
    assert all(not layer.HasField("max_val") for layer in active_layers)
    assert all(not layer.HasField("encoding") for layer in active_layers)
    assert all(not layer.dims for layer in active_layers)


def test_fingerprint_ignores_weight_values_encodings_and_batch_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport details and executable batch variants do not affect identity."""
    first = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    second = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    second.weights.ip_emb_preproc_w.params = b"\x01\x02" * (64 * 12 * 64)
    second.weights.ip_emb_preproc_w.min_val = -3.0
    second.weights.ip_emb_preproc_w.max_val = 7.0
    second.weights.ip_emb_preproc_w.encoding = net_pb2.Weights.Layer.LINEAR16

    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    first_builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    second_builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    build(first_builder, first, batch_sizes=(1,))
    build(second_builder, second, batch_sizes=(3,))

    assert first_builder.build().metadata == second_builder.build().metadata


def test_build_can_emit_multiple_program_specific_batch_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each fixed batch variant receives private input/output storage."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(1, 3))
    executable = builder.build()

    assert [program.name for program in executable.programs] == ["batch-1", "batch-3"]
    metadata: list[int] = []
    for program in executable.programs:
        program_metadata = lc0ex_metadata_pb2.ProgramMetadata()
        program_metadata.ParseFromString(program.metadata)
        metadata.append(program_metadata.batch_size)
    assert metadata == [1, 3]
    assert [tuple(program.buffers[0].shape) for program in executable.programs] == [
        (1, 112),
        (3, 112),
    ]
    assert executable.programs[0].execution_allocation.size_bytes != (
        executable.programs[1].execution_allocation.size_bytes
    )


def test_full_graph_fingerprint_contains_all_active_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete BT4 graph marks embedding, body, and selected heads."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=24)
    _add_encoder(
        network,
        body_width=16,
        compression_width=2,
        hidden_width=16,
        generated_width=16,
        model_width=16,
        ffn_width=24,
    )
    _add_output_heads(
        network,
        body_width=16,
        policy_width=12,
        policy_model_width=8,
        value_width=4,
        value_hidden_width=6,
        moves_width=3,
        moves_hidden_width=5,
    )
    _stub_head_compilers(monkeypatch)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(1,))

    active_layers = _present_fingerprint_layers(_parse_fingerprint(builder))
    assert len(active_layers) == _FULL_FINGERPRINT_LAYER_COUNT
    assert all(not layer.ListFields() for layer in active_layers)


def test_fingerprint_changes_when_semantic_enum_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Format enums are retained even when they do not change buffer shapes."""
    first = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    second = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    first.format.network_format.output = net_pb2.NetworkFormat.OUTPUT_CLASSICAL
    second.format.network_format.output = net_pb2.NetworkFormat.OUTPUT_WDL

    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    first_builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    second_builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)
    build(first_builder, first, batch_sizes=(1,))
    build(second_builder, second, batch_sizes=(1,))

    assert first_builder.build().metadata != second_builder.build().metadata


def test_encoder_builds_names_order_and_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One block preserves ONNX names and the required evaluation order."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=32)
    network.weights.headcount = 2
    _add_encoder(
        network,
        body_width=16,
        compression_width=2,
        hidden_width=16,
        generated_width=16,
        model_width=16,
        ffn_width=32,
    )
    records = _stub_compilers(monkeypatch)
    _stop_after_encoders(monkeypatch)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(2,))
    executable = builder.build()
    nodes = executable.programs[0].nodes
    functions = [executable.kernels[node.kernel_idx].function for node in nodes]
    encoder_functions = functions[11:-1]

    assert encoder_functions == [
        "matmul",
        "matmul",
        "layer_norm",
        "matmul",
        "layer_norm",
        "matmul",
        "matmul",
        "fused_attention",
        "matmul",
        "layer_norm",
        "matmul",
        "matmul",
        "layer_norm",
    ]
    buffers = {
        buffer.name: tuple(buffer.shape)
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }
    encoder_names = {name for name in buffers if name.startswith("/encoder0/")}
    assert len(encoder_names) == _ENCODER_LOCAL_BUFFER_COUNT
    assert buffers["/encoder0/mha/Q/w/w"] == (16, 16)
    assert buffers["/encoder0/smolgen/dense2/w/w"] == (16, 32)
    assert buffers["/const/smolgen_w"] == (16, 4096)
    assert buffers["/encoder0/alpha*input/w"] == (1,)
    assert buffers["/encoder0/ffn/alpha/w"] == (1,)

    specializations = [specialization for _name, specialization in records]
    assert (
        LayerNormSpecialization(
            row_count=2,
            width=32,
            activation="swish",
            has_skip=False,
            has_bias=False,
            architecture=_ARCHITECTURE,
        )
        in specializations
    )
    assert FusedAttentionSpecialization(4, 16, 8, 2, _ARCHITECTURE) in specializations

    arguments = [[_location(argument) for argument in node.arguments] for node in nodes]
    encoder_start = 11
    dense_out = arguments[encoder_start + 8]
    ln1 = arguments[encoder_start + 9]
    dense2 = arguments[encoder_start + 11]
    ln2 = arguments[encoder_start + 12]
    assert dense_out[4] == arguments[10][0]
    assert dense2[4] == ln1[0]
    assert arguments[-1][0] == ln2[0]


@pytest.mark.parametrize(
    ("encoding_width", "body_width", "hidden_width"),
    [(1, 16, 32), (3, 32, 48)],
)
def test_embedding_infers_external_shapes_from_layer_counts(
    monkeypatch: pytest.MonkeyPatch,
    encoding_width: int,
    body_width: int,
    hidden_width: int,
) -> None:
    """Vector counts and known matrix inputs determine every learned shape."""
    network = _embedding_network(
        encoding_width=encoding_width,
        body_width=body_width,
        hidden_width=hidden_width,
    )
    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(2,))
    executable = builder.build()
    buffers = {
        buffer.name: buffer
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }

    expected_shapes = {
        "/input/plane_masks": (2, 112),
        "/input/plane_values": (2, 112),
        "/attn_body/embedding/preprocess/matmul/w": (
            64 * 12,
            64 * encoding_width,
        ),
        "/attn_body/embedding/preprocess/add/w": (64 * encoding_width,),
        "/attn_body/matmul/w": (112 + encoding_width, body_width),
        "/attn_body/add/w": (body_width,),
        "/attn_body/ln/w/scale": (body_width,),
        "/attn_body/ln/w/bias": (body_width,),
        "/ip_mul_gate/w": (64, body_width),
        "/ip_add_gate/w": (64, body_width),
        "/attn_body/ffn/dense1/w/w": (body_width, hidden_width),
        "/attn_body/ffn/dense1/b/w": (hidden_width,),
        "/attn_body/ffn/dense2/w/w": (hidden_width, body_width),
        "/attn_body/ffn/dense2/b/w": (body_width,),
        "/attn_body/ffn/alpha/w": (1,),
        "/attn_body/ln2/w/scale": (body_width,),
        "/attn_body/ln2/w/bias": (body_width,),
    }

    assert {name: tuple(buffer.shape) for name, buffer in buffers.items()} == (
        expected_shapes
    )
    assert buffers["/input/plane_masks"].data_type == lc0ex_pb2.Buffer.DATA_TYPE_U64
    assert buffers["/input/plane_values"].data_type == lc0ex_pb2.Buffer.DATA_TYPE_F32
    assert all(
        buffer.data_type == lc0ex_pb2.Buffer.DATA_TYPE_F16
        for name, buffer in buffers.items()
        if not name.startswith("/input/")
    )
    assert all(
        name in {buffer.name for buffer in executable.programs[0].buffers}
        for name in ("/input/plane_masks", "/input/plane_values")
    )
    assert all(
        name in {buffer.name for buffer in executable.buffers}
        for name in buffers
        if not name.startswith("/input/")
    )


def test_embedding_builds_expected_operations_and_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The segment composes tuned families with inferred immutable dimensions."""
    network = _embedding_network(encoding_width=3, body_width=32, hidden_width=48)
    records = _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(2,))
    executable = builder.build()
    nodes = executable.programs[0].nodes
    functions = [executable.kernels[node.kernel_idx].function for node in nodes]

    assert functions == [
        "expand_planes",
        "nchw_to_nhwc",
        "matmul",
        "add_vectors",
        "preprocess_attention_body",
        "matmul",
        "layer_norm",
        "input_gating",
        "matmul",
        "matmul",
        "layer_norm",
        "consume_body",
    ]
    assert [specialization for _name, specialization in records] == [
        ExpandPlanesSpecialization(2 * 112, _ARCHITECTURE),
        NchwToNhwcSpecialization(2, 112, 12, 8, 8, _ARCHITECTURE),
        MatmulSpecialization(2, 64 * 3, 64 * 12, _ARCHITECTURE),
        AddVectorsSpecialization(2 * 64 * 3, 64 * 3, "none", _ARCHITECTURE),
        PreprocessAttentionBodySpecialization(2, 112, 3, _ARCHITECTURE),
        MatmulSpecialization(2 * 64, 32, 112 + 3, _ARCHITECTURE, has_bias=True),
        LayerNormSpecialization(
            row_count=2 * 64,
            width=32,
            activation="mish",
            has_skip=False,
            has_bias=False,
            architecture=_ARCHITECTURE,
        ),
        InputGatingSpecialization(2, 64, 32, _ARCHITECTURE),
        MatmulSpecialization(
            2 * 64, 48, 32, _ARCHITECTURE, has_bias=True, activation="mish"
        ),
        MatmulSpecialization(
            2 * 64, 32, 48, _ARCHITECTURE, has_bias=True, has_skip=True
        ),
        LayerNormSpecialization(
            row_count=2 * 64,
            width=32,
            activation="none",
            has_skip=False,
            has_bias=False,
            architecture=_ARCHITECTURE,
        ),
    ]


def test_embedding_preserves_pointer_reinterpretation_skip_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialized locations prove in-place calls, view-free reshaping, and reuse."""
    network = _embedding_network(encoding_width=3, body_width=32, hidden_width=48)
    _stub_compilers(monkeypatch)
    _stop_after_embedding(monkeypatch)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(2,))
    executable = builder.build()
    nodes = executable.programs[0].nodes
    arguments = [[_location(argument) for argument in node.arguments] for node in nodes]

    assert [list(node.dependencies) for node in nodes] == [
        [],
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
    ]
    assert arguments[1][0] == arguments[2][1]
    assert arguments[3][0] == arguments[3][1] == arguments[4][2]
    assert arguments[4][0] == arguments[5][1]
    assert arguments[5][0] == arguments[6][1]
    assert arguments[6][0] == arguments[7][0] == arguments[7][1]
    assert arguments[7][0] == arguments[8][1] == arguments[9][4]
    assert arguments[8][0] == arguments[9][1]
    assert arguments[9][0] == arguments[10][1]
    assert arguments[11][0] == arguments[10][0]

    temporary_outputs = {arguments[index][0] for index in (0, 1, 2, 4, 5, 6, 8, 9, 10)}
    assert len(temporary_outputs) == _REUSED_TEMPORARY_COUNT
    assert (
        len({arguments[4][0], arguments[4][1], arguments[4][2]})
        == _REUSED_TEMPORARY_COUNT
    )
    assert (
        len({arguments[9][0], arguments[9][1], arguments[9][4]})
        == _REUSED_TEMPORARY_COUNT
    )


def test_output_heads_build_contracts_and_independent_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All selected heads preserve names, operations, outputs, and body fan-out."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=24)
    _add_output_heads(
        network,
        body_width=16,
        policy_width=12,
        policy_model_width=8,
        value_width=4,
        value_hidden_width=6,
        moves_width=3,
        moves_hidden_width=5,
    )
    records = _stub_head_compilers(monkeypatch)
    monkeypatch.setattr(network_module, "_encoder_tower", _keep_body)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(2,))
    executable = builder.build()
    nodes = executable.programs[0].nodes
    functions = [executable.kernels[node.kernel_idx].function for node in nodes]
    head_functions = functions[11:]

    assert head_functions == [
        "matmul",
        "matmul",
        "matmul",
        "policy_qk",
        "promotion_logits",
        "policy_map",
        "matmul",
        "matmul",
        "matmul",
        "matmul",
        "matmul",
        "matmul",
    ]
    assert "softmax_64" not in head_functions

    buffers = {
        buffer.name: buffer
        for buffer in (
            *executable.buffers,
            *executable.programs[0].buffers,
        )
    }
    expected_shapes = {
        "/policy/dense1/matmul/w": (16, 12),
        "/policy/dense1/add/w": (12,),
        "/policy/Q/matmul/w": (12, 8),
        "/policy/Q/add/w": (8,),
        "/policy/K/matmul/w": (12, 8),
        "/policy/K/add/w": (8,),
        "/policy/promotion/matmul/w": (8, 4),
        "/policy/scale/w": (1,),
        "/value/embed/matmul/w": (16, 4),
        "/value/embed/add/w": (4,),
        "/value/dense1/matmul/w": (64 * 4, 6),
        "/value/dense1/add/w": (6,),
        "/value/dense2/matmul/w": (6, 3),
        "/value/dense2/add/w": (3,),
        "/mlh/embed/matmul/w": (16, 3),
        "/mlh/embed/add/w": (3,),
        "/mlh/dense1/matmul/w": (64 * 3, 5),
        "/mlh/dense1/add/w": (5,),
        "/mlh/dense2/matmul/w": (5, 1),
        "/mlh/dense2/add/w": (1,),
        "/output/policy": (2, 1858),
        "/output/wdl": (2, 3),
        "/output/mlh": (2, 1),
    }
    assert {
        name: tuple(buffers[name].shape) for name in expected_shapes
    } == expected_shapes
    for name in ("/output/policy", "/output/wdl", "/output/mlh"):
        buffer = buffers[name]
        assert buffer.data_type == lc0ex_pb2.Buffer.DATA_TYPE_F32
        assert buffer.name in {item.name for item in executable.programs[0].buffers}
    assert "/const/mapping_table" not in buffers

    head_specializations = [specialization for _name, specialization in records][11:]
    assert (
        BatchedMatmulSpecialization("policy_qk", 2, 64, 64, 8, 1, _ARCHITECTURE)
        in head_specializations
    )
    assert PromotionLogitsSpecialization(2, 8, _ARCHITECTURE) in head_specializations
    assert PolicyMapSpecialization(2, _ARCHITECTURE) in head_specializations
    assert (
        MatmulSpecialization(
            2, 1, 5, _ARCHITECTURE, has_bias=True, activation="relu", output_f32=True
        )
        in head_specializations
    )

    arguments = [[_location(argument) for argument in node.arguments] for node in nodes]
    head_start = 11
    final_body = arguments[10][0]
    assert arguments[head_start][1] == final_body
    assert arguments[head_start + 6][1] == final_body
    assert arguments[head_start + 9][1] == final_body
    assert arguments[head_start + 3][0] == arguments[head_start + 4][0]
    assert arguments[head_start + 4][0] == arguments[head_start + 5][1]
    assert nodes[head_start + 5].arguments[2].HasField("symbol")
    assert (
        nodes[head_start + 5].arguments[2].symbol.symbol_name
        == "lczero_bt4_mapping_table"
    )
    assert head_start + 5 not in nodes[head_start + 6].dependencies
    assert head_start + 8 not in nodes[head_start + 9].dependencies


def test_policy_embedding_prefers_head_local_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonempty vanilla embedding overrides the shared multihead operands."""
    network = _embedding_network(encoding_width=1, body_width=16, hidden_width=24)
    _add_output_heads(
        network,
        body_width=16,
        policy_width=12,
        policy_model_width=8,
        value_width=4,
        value_hidden_width=6,
        moves_width=3,
        moves_hidden_width=5,
    )
    _set_elements(network.weights.policy_heads.vanilla.ip_pol_w, 16 * 10)
    _set_elements(network.weights.policy_heads.vanilla.ip_pol_b, 10)
    _set_elements(network.weights.policy_heads.vanilla.ip2_pol_w, 10 * 8)
    _set_elements(network.weights.policy_heads.vanilla.ip3_pol_w, 10 * 8)
    _stub_head_compilers(monkeypatch)
    monkeypatch.setattr(network_module, "_encoder_tower", _keep_body)
    builder = ExecutableBuilder(io_data_type=lc0ex_pb2.Buffer.DATA_TYPE_F16)

    build(builder, network, batch_sizes=(1,))
    buffers = {buffer.name: tuple(buffer.shape) for buffer in builder.build().buffers}

    assert buffers["/policy/dense1/matmul/w"] == (16, 10)


@pytest.mark.parametrize(
    ("default", "expected"),
    [
        (
            net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU,
            net_pb2.NetworkFormat.ACTIVATION_RELU,
        ),
        (
            net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH,
            net_pb2.NetworkFormat.ACTIVATION_MISH,
        ),
    ],
)
def test_default_activation(default: int, expected: int) -> None:
    """LC0 default activation enums resolve to concrete operations."""
    resolved_default = cast("net_pb2.NetworkFormat.DefaultActivation", default)
    assert _default_activation(resolved_default) == expected


def test_explicit_activation_overrides_default() -> None:
    """An explicit operation activation is not replaced by the default."""
    assert (
        _resolve_activation(
            net_pb2.NetworkFormat.ACTIVATION_SWISH,
            net_pb2.NetworkFormat.ACTIVATION_MISH,
        )
        == net_pb2.NetworkFormat.ACTIVATION_SWISH
    )
