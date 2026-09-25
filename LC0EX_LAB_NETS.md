# lc0ex for the lab nets — static, EGT2 (gcap / triplet) and the BT6-test sponsor shape

The *builder* half of serving a lab research net with lc0ex. It sits on top of `LC0EX_SPEEDUPS.md` (CUTLASS encoder
GEMMs, autotune reuse) and adds a second front end, `lczero_triton.lab`, next to the BT4 one. The *runtime* half needs
**no change**: `Kovax007/lc0@lc0ex-speedups-20260908` (`fe9345fa`) runs every artifact described here.

## Why a second front end: a lab net is not a `.pb.gz` of weights

`net.proto` has no fields for a gated FFN (`dense_gate_w`), for the pair / edge stream, the update sites or the triplet
operator, so `lc0-jax2leela` cannot write these nets and adding one field would not change that. The lab exports
**ONNX inside the `.pb.gz`** instead (`Net.onnx_model`, `network_format.network = 7`, no `weights` stanza), and this
package reads the architecture *off that graph*:

```
orbax checkpoint ──dump_ckpt_for_export.py──▶ weights.npz + ref.npz          (CPU, lab tooling, not in this repo)
                 ──export512.py──────────────▶ NET.pb.gz  (ONNX carrier; parity-gated against the JAX forward)
NET.pb.gz ──tools/make_carrier.py──▶ NET_lc0ex_carrier.pb.gz   ← lc0's --weights (initializers planned as lc0ex buffers)
NET.pb.gz ──tools/build_lab.py─────▶ NET_sm89.lc0ex            ← backend-opts lc0ex=...
```

`lab/_mapping.py` matches on **structure** (the exporter names every constant `const_<n>`, so names carry nothing):
blocks around their `Softmax` anchors, the GLU by dataflow, doors / sites / triplets by their channel maps. It refuses
a graph it does not fully understand rather than guess — a dropped projection is a silently broken net.

⚠ **The artifact emits WDL *logits*.** lc0ex's runtime softmaxes them itself (`DecodeWdl` in
`network_lc0ex_cuda.cc`), so the reader skips an export's terminal `Softmax`. lc0's *ONNX* backends read
probabilities, which is why the gate below compares against the export's WDL-softmax copy.

## What is served

| family | what it adds | kernels |
|---|---|---|
| static | pair-stream logit terms folded to per-head tables, sigmoid-GLU FFN | `attention_static`, `prologue_static`, `policy_static_bias`, `glu_matmul` / CUTLASS `glu` |
| smolgen twin | BT4's smolgen on the lab trunk | `fused_attention` |
| pre-norm + output gate | `x += a·attn(ln(x))`, `2·sigmoid` gate on the merged heads, final norm | as static |
| EGT2 `gcap` | 16-channel edge state: doors (E, M, G) in every block, update sites (readback + edge FFN) | `prologue_egt`, `attention_egt`, `egt_state_tiles`, `edge_site`, `policy_egt_bias` |
| EGT2 `triplet_path` / `triplet_ag` | the triplet operator between readback and edge FFN at every site | `triplet_site` |
| **BT6-test** (`bt6_test_1024x28x16_triplet_cv4`) | the four items below | — |

### BT6-test: what the sponsor shape needed (2026-09-21)

1024 × 28, 16 heads × 64, sigmoid-GLU dff 1024, `triplet_path` at 6 sites (after blocks 3/7/11/15/19/23), d_e 16, doors
in all 28 blocks — plus four things no earlier lab net had:

| model config | in the graph | served as |
|---|---|---|
| `encoder.dff: 1024` = d_model | all seven block projections are square | the GLU is found by **dataflow** (`_glu_by_dataflow`), not by weight shape |
| `defaults.ffn_softcap: 12` | `Mul(12, Tanh(Div(x, 12)))` on BOTH GLU branches, in every block and in the embedding FFN | CUTLASS `glu_softcap` (one more constant in the dual-GEMM epilogue) or the Triton gate `glu_capped`; a block with a `Tanh` the reader cannot attribute is refused, so a cap can never be dropped silently |
| `edge_stream.rev_edge: true` | the edge FFN's hidden layer also reads `rms(e_hat)` with the board axes swapped, through a second in-projection | `EdgeSiteSpecialization.reverse`: the site program gathers the mirrored `[16, pixels]` tile; table `/encoder{b}/edge_site/ffn/dense1_rev/w` |
| `embedding.dense_size: 512`, `shared_policy_embedding_size: 512` on a 1024 trunk | preprocess dense 768 → 64·512, embedding 112+512 → 1024, policy 1024 → 512 → 512 | `Architecture.embedding_dense`, `Architecture.policy_width`, read off the constants; `/policy/qk_scale` = 1/√512 |

What they cost on an RTX 4090 at batch 64 (FP16 edge state, three interleaved `backendbench` passes, one card):

| artifact | nps | ms / batch |
|---|---:|---:|
| the same trunk WITHOUT the cap and the reverse edge, d_model-wide preprocess and policy (the 09-20 speed twin) | 6,268 | 10.21 |
| BT6-test as trained, step 47,500 | **6,100** | **10.49** |

In nsys kernel time per batch (graph=off): the capped GLU **+0.24 ms** (28 blocks + the embedding; the gate's cap is
a three-term series — a sigmoid lies in (0, 1), so `12·tanh(g/12) = g(1 − u/3 + 2u²/15)`, u = (g/12)², to 1.8e-8 — and
only the up branch pays the exp and the reciprocal), the reverse edge **+0.20 ms** (six sites; the site's programs may
run over 8 × 8 or 4 × 8 tiles there, because a tile's mirror is a tile while a run's mirror is a column), the narrower
preprocess and policy **−0.18 ms**. Served (graph=dag) the difference is the table's +0.28 ms: a site runs beside the
next block's QKV GEMM, so kernel time does not add up to batch time. Neither new term is touched by int8 GEMMs.

Heads: `export512.py --policy-head … --value-head …` picks ONE policy and ONE value head per export, and an artifact
serves that pair. The run trains five policy heads (`vanilla`, `optimistic_st`, `soft`, `grill`, and `order`, which
never plays) and three value heads (`winner`, `q`, `st`); which pair plays is a decision for the head matches, not for
the backend — every pair is the same graph shape and the same speed.

## Build

```bash
# once: uv sync  (generates the *_pb2.py stubs; see LC0EX_SPEEDUPS.md)
tools/build_lab_artifact.sh LABEL GPU NET.pb.gz 8,16,32,64 cold none          # sweeps CUTLASS tiles on this card
tools/build_lab_artifact.sh LABEL GPU NET.pb.gz 64 cold TILECACHE.json        # reuses a measured tile cache
```
The script writes the carrier first, gives the build a **private** Triton cache and tile cache, refuses to overwrite an
artifact or reuse a label (a failed build leaves its cache behind — pick a new label), and refuses a GPU that is
running anything else: a disturbed autotune ships foreign tile choices invisibly. It sets the served switches:

| switch | served value | meaning |
|---|---|---|
| `LC0EX_CUTLASS_QKV/OUTPROJ/FFN2/GLU` | `1` | encoder GEMMs on CUTLASS (FFN widths must be multiples of 8; **use multiples of 64** — 683/1366 have no int8 kernel later) |
| `LC0EX_EGT_STATE` | `auto` | blocks read FP16 tiles up to `LC0EX_EGT_TILES_MAX_BATCH=16`, an FP16 copy of the state above |
| `LC0EX_EGT_STATE_AUTO_UPPER` | `f16` | `i8` / `f8` = one-byte edge state, +2–3 % at b64; ⚠ per-channel scales default to the 512×15 family's amax — re-measure (`LC0EX_EGT_STATE_AMAX`) and gate before serving a new net with it |
| `LC0EX_EGT_OVERFLOW` | `correction` | positions past the edge list's capacity recomputed exactly by a second, early-exiting launch |
| `LC0EX_TRIPLET_FORM` / `_DOT` | `fused` / `fp16` | triplet prep + contraction in one program, FP16 operands with FP32 accumulation |
| `LC0EX_EGT_READ_BLOCKS` | `all` | pricing only: a net trained to read in every block must be served reading in every block |

Limits today: 16 edge-state channels and the 34-channel attack graph are compiled in; head count and edge FFN width
must be powers of two; `LC0EX_TRIPLET_OUT_FFN=1` does not carry `rev_edge` (it raises); EGT2 on pre-norm blocks is not
built; the largest rung is the largest batch lc0 may ask for (`--minibatch-size` ≤ it) and there is no L2 chunking on
this front end, so keep rungs ≤ 64 on a 4090.

### Quantisation (Q1) — the parts that exist, and the one that does not

⚠ **No lab artifact is quantised today**: the int8 GEMM is not built, so nothing consumes int8 codes and every
artifact above is FP16 (the one-byte *edge state* is a different thing and is served). What exists is the
conversion the GEMM will need, built with the per-channel pre-scale in it from the start:

* `bt4/kernels/quantise_int8.py` — `q_j = clamp(floor((x_j − m_j)·r_j + 0.5), −127, 127)`, `r = 1/(s·D)`, a
  standalone pass over any activation. The per-channel vector costs nothing over a scalar (measured: ±1.5 %).
* `LayerNormSpecialization.quantise` — the same conversion **inside the norm**, from the FP32 output the row
  already holds. Under post-norm the norm feeds the residual too, so the SmoothQuant vector cannot go into
  `gamma`/`beta`; it goes here instead. 4× cheaper than a separate pass (0.28 vs 1.12 ms per b64 batch over the
  56 norm sites of a 28-block net) and more accurate, since a separate pass only ever sees the FP16 copy.
* `lab/_quant.py` — the `.npz` of `r` and `m` the analyser delivers, and the map from its site names to the norm
  that emits each copy (⛔ a block's `attn_in` is the **previous** block's `ln2`).

`ffn_mid` and `attn_out` are GEMM outputs; their vector multiplies into the per-column output scale the epilogue
must apply anyway, and lands with the int8 epilogue. Report: `REPORT_backend_q1_per_channel_prescale_built_0921.md`.

## Run

```bash
lc0 --weights=NET_lc0ex_carrier.pb.gz --backend=lc0ex-cuda \
    --backend-opts="gpu=0,lc0ex=NET_sm89.lc0ex,concurrency=1,graph=dag" --minibatch-size=64
```
An artifact is tied to its carrier (fingerprint + buffer names) and to its GPU architecture.

## Gate — before believing a number or playing a game

```bash
tools/make_wdl_softmax.py --source NET.pb.gz --output NET_wdlsoftmax.pb.gz      # the ONNX backend's reference copy
LC0=/path/to/lc0 tools/gate_lab_artifact.sh GPU NET_sm89.lc0ex NET_lc0ex_carrier.pb.gz NET_wdlsoftmax.pb.gz FENS 16 64
```
lc0ex against lc0's own `onnx-cuda` running the same export, position by position, plus self-determinism. Needs an lc0
built with `-Dlc0ex-runtime=true -Donnx=true`. What a pass looks like, and what a miss looks like (BT6-test step
47,500, 256 positions, batch 64, FP16 artifact against the FP32 graph):

| artifact | policy KL mean / max | \|Δq\| mean / max | top-1 agreement |
|---|---|---|---:|
| as served | 0.000003 / 0.000019 | 0.0007 / 0.0051 | 100 % |
| GLU on the Triton route (`LC0EX_CUTLASS_GLU=0`) | 0.000003 / 0.000015 | 0.0006 / 0.0047 | 100 % |
| control: the softcap deliberately dropped | 0.0235 / 0.294 | 0.093 / 1.05 | 92.2 % |
| control: the reverse-edge read deliberately dropped | 0.0644 / 0.735 | 0.095 / 0.75 | 86.7 % |
| the artifact against itself | 0 / 0 | 0 / 0 | 100 % |

The two controls are why the gate can be believed: each new term moves the policy by four orders of magnitude when it
is missing, and a net served without either still loads, runs and returns sane-looking moves. ⚠ A random-init export has **zero** edge reads (E/M/G, `tri_o`, the
site FFN's out-projection and `ffn_rev` are zero-init): fill every all-zero leaf before exporting a *speed* net, or a
folding compiler prices a net that does not read. A trained checkpoint needs nothing of the sort.

`NET_wdlsoftmax.pb.gz` also plays on any stock lc0 with an ONNX backend (`--backend=onnx-cuda`), slowly — useful as a
second opinion and for anyone without an lc0ex build.
