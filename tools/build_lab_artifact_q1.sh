#!/bin/bash
# Round 26 Q1: build ONE .lc0ex artifact against a GIVEN carrier -- the int8 artifact or its FP16 twin.
#
#   usage: build_lab_artifact_q1.sh LABEL GPU NET.pb.gz CARRIER.pb.gz BATCHES WARM_TRITON_DIR|cold TILECACHE.json|none
#                                   int8|fp16 [VECTORS.npz D1.npz|none] [ENV=VALUE ...]
#
# `build_lab_artifact.sh` names the carrier after the net and writes it itself; a Q1 carrier (`make_q1_carrier.sh`)
# carries the FP16 buffers AND the int8 ones, so both artifacts of a checkpoint load the SAME --weights file and a
# twin comparison has one variable. The FP16 twin is built with every Q1 switch unset -- byte for byte today's build.
# Everything else is `build_lab_artifact.sh`: never overwrite an artifact, a private Triton cache and tile cache per
# label, never autotune on a busy card, the tile-consistency check after the build.
set -u
LABEL=${1:?label}; G=${2:?gpu}; NET=${3:?net}; CAR=${4:?carrier}; B=${5:?batches}; WARM=${6:?warm or cold}
WTC=${7:?tilecache or none}; MODE=${8:?int8 or fp16}; shift 8
Q1=()
if [ "$MODE" = int8 ]; then
  VEC=${1:?vectors npz}; D1=${2:?d1 npz or none}; shift 2
  Q1=(LC0EX_QUANT_PRESCALE="$VEC" LC0EX_QUANT_GEMM=int8 LC0EX_QUANT_OPERAND=int8)
  [ "$D1" != none ] && Q1+=(LC0EX_QUANT_D1="$D1")
elif [ "$MODE" != fp16 ]; then
  echo "ABORT: mode $MODE is not int8 or fp16"; exit 1
fi
TREE=$(cd "$(dirname "$0")/.." && pwd)
PY=${LC0EX_PYTHON:-$TREE/.venv/bin/python}
WORK=${LC0EX_WORK:-$PWD/work}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
SRC=$TREE/packages/lczero-triton/src:$TREE/packages/lc0ex/src
ARCH=${LC0EX_ARCH:-$(nvidia-smi -i "$G" --query-gpu=compute_cap --format=csv,noheader | tr -d '.')}
mkdir -p "$WORK"/{art,logs,cache}
ART=$WORK/art/${LABEL}_sm${ARCH}.lc0ex; CACHE=$WORK/cache/triton_${LABEL}; TC=$WORK/cache/tilecache_${LABEL}.json
LOG=$WORK/logs/build_${LABEL}.log
CUT="LC0EX_CUTLASS_QKV=1 LC0EX_CUTLASS_OUTPROJ=1 LC0EX_CUTLASS_FFN2=1 LC0EX_CUTLASS_GLU=1"
SERVED="LC0EX_EGT_STATE=auto LC0EX_EGT_TILES_MAX_BATCH=16 LC0EX_EGT_OVERFLOW=correction LC0EX_TRIPLET_FORM=fused LC0EX_TRIPLET_DOT=fp16"
[ -e "$ART" ] && { echo "ABORT: $ART exists (never overwrite an artifact)"; exit 1; }
[ -s "$NET" ] && [ -s "$CAR" ] || { echo "ABORT: missing net or carrier"; exit 1; }
[ -e "$CACHE" ] && { echo "ABORT: $CACHE exists (private cache per build; pick a new LABEL)"; exit 1; }
busy=$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { echo "ABORT: GPU $G runs $busy compute app(s); never autotune on a shared card"; exit 1; }
free=$(df --output=avail -BG "$WORK" | tail -1 | tr -dc 0-9)
[ "$free" -ge 4 ] || { echo "ABORT: ${free} G free under $WORK; low space = stop and report"; exit 1; }
if [ "$WARM" = cold ]; then mkdir -p "$CACHE"; else cp -a "$WARM" "$CACHE" || exit 1; fi
if [ "$WTC" = none ]; then echo '{}' > "$TC"; else cp "$WTC" "$TC" || exit 1; fi
K=$TREE/packages/lczero-triton/src/lczero_triton
echo "# build $LABEL [$MODE] gpu=$G sm_$ARCH batches=$B net $(sha256sum "$NET" | cut -c1-8) carrier $(sha256sum "$CAR" | cut -c1-8) warm=$(basename "$WARM") tilecache=$(basename "$WTC") q1: ${Q1[*]:-none} extra: $* start $(date -u +%T)Z"
echo "# tree: network $(sha256sum $K/lab/network.py | cut -c1-8) _names $(sha256sum $K/lab/_names.py | cut -c1-8) cutlass_gemm_i8 $(sha256sum $K/bt4/kernels/cutlass_gemm_i8.py | cut -c1-8) cutlass_matmul $(sha256sum $K/bt4/kernels/cutlass_matmul.py | cut -c1-8) layer_norm $(sha256sum $K/bt4/kernels/layer_norm.py | cut -c1-8)"
t0=$(date +%s)
env $CUT $SERVED "${Q1[@]}" "$@" LC0EX_CUTLASS_TILE_CACHE="$TC" TRITON_CACHE_DIR="$CACHE" PYTHONPATH=$SRC CUDA_VISIBLE_DEVICES=$G \
  "$PY" "$TREE/tools/build_lab.py" --network "$NET" --output "$ART" --batch-size "$B" > "$LOG" 2>&1
rc=$?
echo "# built in $(( $(date +%s) - t0 ))s rc=$rc sha $(sha256sum "$ART" 2>/dev/null | cut -c1-8) $(grep -m1 -oE 'cutlass_nodes=[0-9]+' "$LOG") $(grep -m1 -oE 'kernels=[0-9]+' "$LOG")"
grep -E "Traceback|Error|error:" "$LOG" | head -3
[ -s "$ART" ] || exit 1
echo "# int8 GEMM nodes per program: $(grep -c 'lc0ex_cutlass_gemm_i8' "$LOG" 2>/dev/null) log lines naming them; Q1 line: $(grep -m1 -oE 'Q1: int8 GEMM sites.*' "$LOG")"
LC0EX_CHECK_ARCH=$ARCH PYTHONPATH=$SRC "$PY" "$TREE/tools/check_artifact_tile_consistency.py" "$ART" 2>&1 | tail -1
echo "# finished $(date -u +%T)Z"
