# lc0ex speedups — work in progress

A branch on top of **`c4a2f79`** of `mooskagh/lczero-triton`, carrying the changes that
make the artifact faster. It is the *builder* half: everything here runs at
artifact-build time and is baked into the `.lc0ex` file. The runtime half (execution
slots, CUDA-graph mode, the `multi` backend, `backendcompare`) is a matching branch on
`mooskagh/lc0`.

All numbers below were measured on an **RTX 4090 (sm_89, 128 SM, 72 MB L2)**.

## What is in the branch

| file | what it does |
|---|---|
| `bt4/kernels/cutlass_matmul.py` | CUTLASS route for the encoder GEMMs — QKV, out-proj, FFN1/2. Fused bias + per-column output scale via `LinearCombinationBiasElementwise` with `BinaryOp = cutlass::multiplies`. Carries a table of measured tile winners. |
| `bt4/kernels/_config_reuse.py` | Seeds a new ladder rung's Triton autotune from rungs already measured. A new rung cost **537 s** of autotuning vs 196 s of CUTLASS sweeping; the winning tile is stable across neighbouring M, so it is re-used and only the scheduling parameters are re-decided. |
| `bt4/network.py` | Segment chunking — split a batch into L2-resident chunks and chain them. This is the single largest lever. |
| `bt4/kernels/fused_attention.py` | Chunk-aware attention. |
| `bt4/kernels/matmul.py` | Tile-reuse hook. |
| `lc0ex/builder.py`, `buffer_builder.py`, `cubin_module_compiler.py` | Plumbing to emit CUTLASS cubins alongside the Triton ones and to reuse compiled modules. |

## Build

```bash
uv sync          # the *_pb2.py proto stubs are GENERATED at package-build time and are
                 # not in git; a raw checkout cannot import until they exist.
```

## Build an artifact

```bash
env LC0EX_CUTLASS_FFN1=1 LC0EX_CUTLASS_FFN2=1 LC0EX_CUTLASS_OUTPROJ=1 LC0EX_CUTLASS_QKV=1 \
    LC0EX_CUTLASS_TILE_REUSE=1 LC0EX_TRITON_CONFIG_REUSE=1 \
    LC0EX_SEGMENT_CHUNK=64 LC0EX_CHUNK_MODE=chain LC0EX_CHUNK_LN=whole \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m lczero_triton.cli graph \
    --network <net.pb.gz> --batch-size 8,16,24,32,40,48,56,64,96,128 --output bt4.lc0ex
```

## The one number to retune per GPU: `LC0EX_SEGMENT_CHUNK`

The win is **L2 residency**. Live bytes are `slots x batch x bytes_per_position`; when that
crosses L2 the attention kernels fall off a cliff. Pick the largest chunk that still fits:

```
c = floor( (0.92 * L2_bytes - W) / (slots * bytes_per_position) )   then round DOWN to a divisor of the rung
```

`bytes_per_position` (BT4, fp16) = **786,432**; `W` (weights + workspace) ~ **15.8 MB** for BT4.

| GPU | L2 | `c` at 1 slot |
|---|---:|---:|
| RTX 4090 | 72 MB | **64** |
| RTX 5080 | 64 MB | **48** |

* ⚠ **Do not go below 32.** `c = 16` measures **-4.2 %** at one slot and **-4.2/-4.3 %** on four
  GPUs — below the useful floor at any slot count. The optimum is a **step, not a slope**: take
  the largest chunk that fits, then check one step down.
* ⚠ **`LC0EX_CHUNK_MODE=chain` (serialise), never `free`.** Letting the graph overlap chunks is
  *worse than not chunking at all* (attention 85.3 -> 129.4 us/layer): two chunks in flight put
  the same bytes in L2 as one whole batch. **Overlap and residency are the same resource.**

## ⭐ Before benchmarking on a different SM count: the grid may not divide

The tile space here is entirely powers of two. At the four encoder shapes, with the tile the
sweep picks (256x128), the tile counts are **768 / 384 / 256 / 256** — every one a multiple of
128. So **128 SMs** run 6.00 / 3.00 / 2.00 / 2.00 **whole** waves with **0.0 % tail-wave loss**.
On **84 SMs** the last wave leaves **8.6-23.8 %** of the machine idle; on 170 SMs, **9.6-24.7 %**.

**No block size can fix it.** 84 = 2^2 x 3 x 7, and the tile count is `(8192/BM) x (N/BN)` with
8192 = 2^13 and N contributing at most a factor 3. A block size can supply the 3 and **never the
7**, because division cannot introduce a prime factor. The lever is a **persistent / stream-K
schedule** (`GemmUniversalStreamk`, already in CUTLASS). Expect the tail wave in your numbers and
read a per-shape shortfall of that size as the *schedule*, not the kernel.

The `_MEASURED_TILES` table in `cutlass_matmul.py` is sm_89-specific for the same reason and
should be re-swept on another architecture.

## Compiler flags: what does and does not apply to a Triton kernel

Nsight Compute's "FP32/64 Instructions" rule fires on `values /= 1.0 + tl.exp(-values)` (SiLU, in
`bt4/kernels/layer_norm.py` and `matmul.py`) and suggests `--use_fast_math` / `--fmad=true`.

**It is a false positive on a Triton kernel, and there is nothing to collect.** Those are **nvcc**
flags; Triton goes Python -> Triton IR -> LLVM NVPTX -> `ptxas`, with no nvcc in the path. Every
fast-math transform the rule is asking for **is already applied**. Compiled with Triton 3.7.1 and
counted in the emitted PTX, for both `sm_89` and `sm_120`:

| source spelling | PTX float ops | detail |
|---|---:|---|
| `v / (1.0 + tl.exp(-v))` — **current** | **10** | `ex2.approx x2`, `div.full x2`, `mul x2`, `add x2`, `sub x2` |
| `v * tl.sigmoid(v)` | 12 | identical transcendentals, 2 extra `mul` |
| `0.5*v*(1.0 + tanh(0.5*v))` | 14 | libdevice `tanh` does **not** lower to `tanh.approx.f32`; it expands |
| `v / (1.0 + tl.exp2(-v*log2e))` | 10 | byte-identical — Triton already does this rewrite |

* **The exp is already fast-math.** `tl.exp` lowers to a multiply by `log2(e)` plus
  `ex2.approx.f32` — it is already CUDA's `__expf`. `--use_fast_math` cannot improve it.
* **The divide is already fast-math.** Triton emits **`div.full.f32`**, not the IEEE `div.rn.f32`.
  `div.full` is the `-prec-div=false` path that `--use_fast_math` would select. Already there.
* **`--fmad=true` is already the default** — both in `ptxas` and in Triton's own switch for it,
  `enable_fp_fusion`, which defaults to `True`.
* **Hand-rewriting the line makes it worse**, in all three directions tried above.

What the rule is really seeing is the SASS expansion of `div.full.f32` (a `MUFU.RCP` plus FFMA
refinement steps). So the line costs **two SFU ops per element** — one `EX2`, one `RCP` — and on
sm_89 the SFU is 16/SM/clk against 128 for FMA. If such a kernel is genuinely compute-bound it is
bound on **MUFU throughput**, which no compiler flag addresses and which is irreducible for a
sigmoid.

Two things worth doing instead:

1. **Check the bound before optimising the mix.** These are layer-norm and elementwise kernels;
   if they are DRAM-bound the FP32 instruction count is not what limits them, and the NCU rule
   fires on instruction mix without regard to that.
2. **Fuse the activation into the producer's epilogue** so it is not a standalone pass at all —
   the same move `cutlass_matmul.py` already makes for bias and the per-column output scale.

Reproduce with `triton.compile(..., target=GPUTarget("cuda", 89, 32)).asm["ptx"]` — no GPU needed,
and you can target `sm_120` from any machine.

## Gating — what to check before believing a number

* `backendcompare --backend=lc0ex-cuda ... --ref-backend=lc0ex-cuda ... --ref-weights=<net>` against
  the **unchunked** artifact. At `c = 64` on a 4090 this is **bit-identical** (policy KL 0.000000,
  top-1 100 %). Below that, expect in-class, not exact. ⚠ `--fens=<file>` is **required**.
* Self-determinism: the same artifact against itself, x3, must be 0.000000.
* ⚠ **Do not gate by comparing artifact sha256s.** Two builds of an identical tree differ in ~12 of
  93 cubins while the emitted graph is byte-identical — the *compilation* is not reproducible, the
  *graph* is.
* ⚠ `cuda-fp16` is exact at 16/32/48/64/96 and first deviates at **112**, so it cannot arbitrate
  above 96; gate against lc0ex itself there.
