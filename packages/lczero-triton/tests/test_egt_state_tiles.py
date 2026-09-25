"""CUDA tests for the EGT2 edge-state read tiles and the FP16 state copy (round 21, K2c).

Gates (``K2C-GATE`` lines, ``pytest -s``):
* the tiles against a float64 einsum of the same state and plans, in FP32 storage (FP32 class) and FP16 storage
  (half a unit in the last place of the tile), for every round split and head base;
* `reverse` writes the same tiles;
* the FP16 copy is the exact rounding of the state;
* every (sample, head, term) cell is written (a NaN-filled buffer has no NaN left);
* the artifact compile and the builder calls, both kernels.
"""

import math

import pytest
import torch
from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2
from lczero_triton.bt4.kernels._cache import KernelCache
from lczero_triton.bt4.kernels.egt_state_tiles import (
    TILE_TABLES,
    CastStateSpecialization,
    StateTilesSpecialization,
    cast_state,
    compile_cast_state,
    compile_state_tiles,
    egt_state_tiles,
    launch_cast_state,
    launch_state_tiles,
    tiles_bytes,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]
_HEADS, _STATES = 32, 16
_F32_RELATIVE = 1e-5
_F16_RELATIVE = 2 ** -11  # half a unit in the last place, peak-relative


def _architecture() -> int:
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


def _data(batch: int, seed: int = 0x2C) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn((batch, _STATES, 64, 64), device="cuda", generator=generator)
    state = state / state.pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt()  # rms-normalised, as the sites leave it
    weights = [torch.randn((_HEADS, _STATES), device="cuda", generator=generator) * 0.3 for _ in TILE_TABLES]
    return state, weights


def _expected(state: torch.Tensor, weights: list[torch.Tensor], round_heads: int, head_base: int) -> torch.Tensor:
    """float64 ``[batch, round_heads, 3, 64, 64]``."""
    rows = slice(head_base, head_base + round_heads)
    terms = [torch.einsum("hc,zcij->zhij", w[rows].double(), state.double()) for w in weights]
    return torch.stack(terms, dim=2)


def _relative(got: torch.Tensor, expected: torch.Tensor) -> float:
    return float((got.double() - expected).abs().max() / expected.abs().max())


@pytest.mark.parametrize("batch", [8, 64])
@pytest.mark.parametrize(("rounds", "round_index"), [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("tiles_f32", [True, False])
def test_tiles_match_float64(batch: int, rounds: int, round_index: int, tiles_f32: bool) -> None:
    state, weights = _data(batch)
    round_heads = _HEADS // rounds
    head_base = round_index * round_heads
    specialization = StateTilesSpecialization(batch, _architecture(), round_heads=round_heads, head_base=head_base,
                                              tiles_f32=tiles_f32)
    tiles = torch.full((batch, round_heads, 3, 64, 64), math.nan,
                       dtype=torch.float32 if tiles_f32 else torch.float16, device="cuda")
    launch_state_tiles(tiles, state, *weights, specialization)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(tiles).all()), "a tile cell was not written"
    relative = _relative(tiles, _expected(state, weights, round_heads, head_base))
    print(f"K2C-GATE tiles b{batch} rounds {rounds} round {round_index} {'f32' if tiles_f32 else 'f16'}: "
          f"rel {relative:.3e}", flush=True)
    assert relative <= (_F32_RELATIVE if tiles_f32 else _F16_RELATIVE)
    assert tiles_bytes(specialization) == tiles.numel() * tiles.element_size()


def test_reverse_writes_the_same_tiles() -> None:
    state, weights = _data(16)
    outputs = []
    for reverse in (False, True):
        tiles = torch.full((16, _HEADS, 3, 64, 64), math.nan, dtype=torch.float16, device="cuda")
        launch_state_tiles(tiles, state, *weights, StateTilesSpecialization(16, _architecture(), reverse=reverse))
        torch.cuda.synchronize()
        outputs.append(tiles.clone())
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("batch", [8, 64])
def test_cast_is_the_exact_rounding(batch: int) -> None:
    state, _ = _data(batch)
    copy = torch.full((batch, _STATES, 64, 64), math.nan, dtype=torch.float16, device="cuda")
    launch_cast_state(copy, state, CastStateSpecialization(batch, _architecture()))
    torch.cuda.synchronize()
    assert torch.equal(copy, state.half())


def test_specialization_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="power of two"):
        StateTilesSpecialization(8, _architecture(), round_heads=24)
    with pytest.raises(ValueError, match="power of two"):
        StateTilesSpecialization(8, _architecture(), states=8)
    with pytest.raises(ValueError, match="multiple of round_heads"):
        StateTilesSpecialization(8, _architecture(), round_heads=16, head_base=8)


def test_compiles_to_lc0ex_artifacts() -> None:
    architecture = _architecture()
    tiles_artifact = compile_state_tiles(StateTilesSpecialization(8, architecture, round_heads=16, head_base=16))
    cast_artifact = compile_cast_state(CastStateSpecialization(8, architecture))
    assert tiles_artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    assert cast_artifact.binary_format == lc0ex_pb2.Binary.FORMAT_CUBIN
    executable = ExecutableBuilder()
    program = executable.program(name="k2c")
    kernels = KernelCache(executable)
    cells = 8 * 64 * 64
    state = program.temporary_buffer(size_bytes=4 * _STATES * cells, alignment_bytes=256)
    copy = program.temporary_buffer(size_bytes=2 * _STATES * cells, alignment_bytes=256)
    specialization = StateTilesSpecialization(8, architecture, round_heads=16, head_base=16)
    tiles = program.temporary_buffer(size_bytes=tiles_bytes(specialization), alignment_bytes=256)
    tables = {name: program.persistent_buffer(name=f"/encoder0/mha/egt/{name}", shape=(_HEADS, _STATES),
                                              dtype=lc0ex_pb2.Buffer.DATA_TYPE_F32, alignment_bytes=256)
              for name in TILE_TABLES}
    cast_state(program, kernels, copy, state, CastStateSpecialization(8, architecture))
    egt_state_tiles(program, kernels, tiles, state, tables, specialization)
