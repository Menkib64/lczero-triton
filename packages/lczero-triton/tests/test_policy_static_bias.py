"""CUDA tests for the static-net policy pair bias."""

import pytest
import torch
from lczero_triton.bt4.kernels.policy_static_bias import (
    PolicyStaticBiasSpecialization,
    _autotune_grid,
    _policy_static_bias_kernel,
    compile_policy_static_bias,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.mark.parametrize("batch", [1, 7, 64])
def test_bias_matches_the_folded_reference_and_leaves_promotions_alone(batch: int) -> None:
    torch.manual_seed(0x9011 + batch)
    records = (torch.randn(batch, 4288, device="cuda") * 2.0).half()
    edges = (torch.rand(batch, 4, 64, 64, device="cuda") < 0.05).half()
    edge_norm = (torch.rand(batch, 64, 64, device="cuda") * 1.5 + 0.25).half()
    scaled = (torch.randn(1, 4, device="cuda") * 0.5).half()
    constant = (torch.randn(1, 64, 64, device="cuda") * 0.5).half()
    before = records.clone()
    codes = sum((edges[:, channel].to(torch.uint8) << channel) for channel in range(4)).contiguous()
    _policy_static_bias_kernel[_autotune_grid](records, codes, edge_norm, scaled, constant, batch)
    torch.cuda.synchronize()
    bias = edge_norm.float() * (torch.einsum("e,zerc->zrc", scaled[0].float(), edges.float()) + constant[0].float())
    expected = before[:, :4096].float() + bias.reshape(batch, 4096)
    assert torch.allclose(records[:, :4096].float(), expected, atol=2e-3, rtol=1e-3)
    assert torch.equal(records[:, 4096:], before[:, 4096:]), "promotion entries must be untouched"


def test_artifact_compiles() -> None:
    assert compile_policy_static_bias(PolicyStaticBiasSpecialization(batch_count=8, architecture=_architecture())) is not None
