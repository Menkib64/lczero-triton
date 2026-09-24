"""CUDA tests for the EGT2 policy pair bias at 34 channels (item E, K2), gated against onnxruntime.

ORT's term is gcap node 12601, ``Einsum(Z0, c)``; the logits divide it by 22.6274 (node 12665). The kernel reads K1's
packed uint64 E and R0's FP16 plans ``/policy/pair/scaled_coefficients [1, 34]`` and ``/policy/pair/constant_bias
[1, 64, 64]`` (the divisor folded in). A random-table test reaches every channel bit, 33 included.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from lczero_triton.bt4.kernels.policy_egt_bias import (
    EDGE_CHANNELS,
    PolicyEgtBiasSpecialization,
    _autotune_grid,
    _policy_egt_bias_kernel,
    compile_policy_egt_bias,
)
from lczero_triton.bt4.kernels.prologue_egt import unpack_edges

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_TREE = Path(__file__).resolve().parents[3]
_REFERENCE = Path(os.environ.get("LC0EX_EGT_K2_REF", str(_TREE / "ref/egt_k2")))
_CARRIER = Path(os.environ.get(
    "LC0EX_EGT_CARRIER", str(Path.home() / "spsa/lc0ex_5080/work/r20c_itemE_R0/egt2_gcap_512x15_lc0ex_r0.pb.gz")))
_COUNT = 64
# FP16 tables (R0: 2.7e-4 fold error) and FP16 S (K1: 4.9e-4) and FP16 storage of the logit.
_F16_RELATIVE = 2e-3


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _read(relative: str, dtype: torch.dtype, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray((_REFERENCE / relative).read_bytes()), dtype=dtype).view(*shape)


def _run(records: torch.Tensor, codes: torch.Tensor, norm: torch.Tensor, scaled: torch.Tensor,
         constant: torch.Tensor) -> None:
    _policy_egt_bias_kernel[_autotune_grid](records, codes, norm, scaled, constant, records.shape[0])
    torch.cuda.synchronize()


@pytest.mark.parametrize("storage", ["S16", "S32"])
def test_bias_matches_ort(storage: str) -> None:
    header_path = _REFERENCE / "k2_ref.json"
    if not header_path.exists() or not _CARRIER.exists():
        pytest.skip("reference or carrier not present")
    for line in (_REFERENCE / "SHA256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in ("edges.bin", "gcap/S.bin", "gcap/policy_term.bin", "k2_ref.json"):
            assert hashlib.sha256((_REFERENCE / name).read_bytes()).hexdigest() == digest, name
    header = json.loads(header_path.read_text())
    divisor = header["arms"]["gcap"]["policy"]["divisor"]
    from lczero_triton.lab._onnx import load_carrier  # noqa: PLC0415

    _, graph = load_carrier(_CARRIER)

    def plan16(name: str) -> torch.Tensor:
        tensor = graph.initializers[name]
        assert tensor.data_type == 10, name  # noqa: PLR2004
        return torch.frombuffer(bytearray(tensor.raw_data), dtype=torch.float16).view(*tensor.dims).cuda()

    scaled, constant = plan16("/policy/pair/scaled_coefficients"), plan16("/policy/pair/constant_bias")
    assert scaled.shape == (1, EDGE_CHANNELS)
    codes = _read("edges.bin", torch.int64, _COUNT, 64, 64).view(torch.uint64).cuda()
    norm = _read("gcap/S.bin", torch.float32, _COUNT, 64, 64).cuda()
    norm = norm.half() if storage == "S16" else norm
    expected = _read("gcap/policy_term.bin", torch.float32, _COUNT, 64, 64).cuda().double() / divisor
    torch.manual_seed(0x2B34)
    # The 64x64 logits start at zero: on a random FP16 base the FP16 rounding of the stored sum (spacing 2e-3 near
    # |x| = 4) would swamp a term whose peak is 9e-3. The promotion tail is random, to catch writes past 4,096.
    records = torch.zeros(_COUNT, 4288, dtype=torch.float16, device="cuda")
    records[:, 4096:] = (torch.randn(_COUNT, 192, device="cuda") * 2.0).half()
    before = records.clone()
    _run(records, codes, norm, scaled, constant)
    added = records[:, :4096].double()
    difference = (added - expected.view(_COUNT, 4096)).abs()
    relative = float(difference.max() / expected.abs().max())
    print(f"K2-GATE policy bias ({storage}) vs ORT node 12601 / {divisor:.4f}: abs {float(difference.max()):.2e} "
          f"rel {relative:.2e} (term peak {float(expected.abs().max()):.3e})", flush=True)
    assert relative <= _F16_RELATIVE
    assert torch.equal(records[:, 4096:], before[:, 4096:]), "promotion entries must be untouched"


@pytest.mark.parametrize("batch", [1, 7, 64])
def test_every_channel_bit_against_the_folded_reference(batch: int) -> None:
    torch.manual_seed(0x9034 + batch)
    edges = torch.rand(batch, EDGE_CHANNELS, 64, 64, device="cuda") < 0.05  # noqa: PLR2004
    weights = (1 << torch.arange(EDGE_CHANNELS, device="cuda", dtype=torch.int64))
    codes = (edges.long() * weights[None, :, None, None]).sum(1).view(torch.uint64).contiguous()
    assert torch.equal(unpack_edges(codes), edges)
    records = torch.zeros(batch, 4288, dtype=torch.float16, device="cuda")
    norm = (torch.rand(batch, 64, 64, device="cuda") * 1.5 + 0.25).half()
    scaled = (torch.randn(1, EDGE_CHANNELS, device="cuda") * 0.5).half()
    constant = (torch.randn(1, 64, 64, device="cuda") * 0.5).half()
    _run(records, codes, norm, scaled, constant)
    bias = norm.float() * (torch.einsum("c,zcij->zij", scaled[0].float(), edges.float()) + constant[0].float())
    assert torch.allclose(records[:, :4096].float(), bias.reshape(batch, 4096), atol=2e-3, rtol=1e-3)
    assert torch.equal(records[:, 4096:], torch.zeros_like(records[:, 4096:]))


def test_artifact_compiles() -> None:
    assert compile_policy_egt_bias(PolicyEgtBiasSpecialization(batch_count=8, architecture=_architecture())) is not None
