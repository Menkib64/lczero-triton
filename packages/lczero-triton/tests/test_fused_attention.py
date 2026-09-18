"""CUDA numerical and execution tests for fused attention kernel."""

import pytest
import torch
import triton
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._autotune import active_architecture
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.batched_matmul import _attention_v_kernel, _qk_kernel
from lczero_triton.bt4.kernels.fused_attention import (
    FusedAttentionSpecialization,
    _fused_attention_kernel,
    fused_attention,
)
from lczero_triton.bt4.kernels.softmax_64 import _softmax_64_kernel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_F16 = lc0ex_pb2.Buffer.DATA_TYPE_F16
_TOLERANCE = 0.01


def test_fused_attention_matches_reference() -> None:
    """Verify that fused attention matches separate QK+softmax+V execution."""
    batch = 2
    heads = 32
    head_depth = 32
    model_width = heads * head_depth
    square_count = 64
    batch_count = batch * heads

    queries = torch.randn(
        batch * square_count, model_width, dtype=torch.float16, device="cuda"
    )
    keys = torch.randn(
        batch * square_count, model_width, dtype=torch.float16, device="cuda"
    )
    values = torch.randn(
        batch * square_count, model_width, dtype=torch.float16, device="cuda"
    )
    smolgen = torch.randn(
        batch_count, square_count, square_count, dtype=torch.float16, device="cuda"
    )
    scale = torch.tensor([0.17677], dtype=torch.float16, device="cuda")

    # Reference using separate kernels
    qk_sep = torch.empty(
        batch_count * square_count * square_count,
        dtype=torch.float16,
        device="cuda",
    )
    out_sep = torch.empty(
        batch * square_count * model_width, dtype=torch.float16, device="cuda"
    )
    grid_qk = (triton.cdiv(64, 64), triton.cdiv(64, 64), batch_count)
    _qk_kernel.fn[grid_qk](
        qk_sep,
        queries,
        keys,
        scale,
        0,
        batch_count,
        heads,
        64,
        64,
        head_depth,
        64 * 64,
        64,
        64,
        32,
        num_warps=8,
        num_stages=3,
    )
    _softmax_64_kernel.fn[(batch_count * 64 // 8,)](
        qk_sep, qk_sep, smolgen, batch_count * 64, 8, num_warps=4
    )
    grid_att_v = (triton.cdiv(64, 64), triton.cdiv(32, 32), batch_count)
    _attention_v_kernel.fn[grid_att_v](
        out_sep,
        qk_sep,
        values,
        batch_count,
        heads,
        64,
        head_depth,
        64,
        64,
        32,
        32,
        num_warps=4,
        num_stages=4,
    )

    qkv = torch.cat([queries, keys, values], dim=-1).contiguous()
    out_fused = torch.empty(
        batch * square_count * model_width, dtype=torch.float16, device="cuda"
    )
    _fused_attention_kernel.fn[(batch_count,)](
        out_fused,
        qkv,
        smolgen,
        scale,
        batch_count,
        model_width,
        head_depth,
        heads,
        head_depth,
        num_warps=4,
    )

    diff = (out_fused - out_sep).abs().max().item()
    assert diff < _TOLERANCE

    builder = ExecutableBuilder(io_data_type=_F16)
    program = builder.program(name="main")
    cache = KernelCache(builder)

    out_buf = program.buffer(
        name="out",
        shape=(batch * square_count, model_width),
        dtype=_F16,
        writable=True,
    )
    qkv_buf = program.buffer(
        name="qkv", shape=(batch * square_count, 3 * model_width), dtype=_F16
    )
    smolgen_buf = program.buffer(
        name="smolgen", shape=(batch_count, 64, 64), dtype=_F16
    )
    scale_buf = program.persistent_buffer(name="scale", shape=(1,), dtype=_F16)

    fused_attention(
        program,
        cache,
        out_buf,
        qkv_buf,
        smolgen_buf,
        scale_buf,
        FusedAttentionSpecialization(
            batch_count=batch_count,
            model_width=model_width,
            head_depth=head_depth,
            heads_per_sample=heads,
            architecture=active_architecture(),
        ),
    )
    # Check that it builds
    builder.build()
