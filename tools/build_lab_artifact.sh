#!/bin/bash
# Build ONE .lc0ex artifact (and its carrier) for a lab ONNX-carrier export -- static, smolgen, gcap, triplet, and the
# BT6-test sponsor shape (rev_edge, ffn_softcap, 512-wide preprocess dense, policy width 512).
#
#   usage: build_lab_artifact.sh LABEL GPU NET.pb.gz BATCHES WARM_TRITON_DIR|cold TILECACHE.json|none [ENV=VALUE ...]
#
#   LABEL      names the artifact (<WORK>/art/<LABEL>_sm<ARCH>.lc0ex), its PRIVATE Triton cache and tile cache.
#   BATCHES    comma list of batch sizes, one program each (e.g. 64, or 8,16,32,64).
#   WARM...    a Triton cache directory to start from (copied, never shared), or `cold`.
#   TILECACHE  a CUTLASS tile cache to start from (copied), or `none` to sweep every shape on this card.
#   ENV=VALUE  extra builder switches, e.g. LC0EX_EGT_STATE_AUTO_UPPER=i8 (one-byte edge state).
#
# Environment: LC0EX_PYTHON (the venv's python, default .venv/bin/python of this tree), LC0EX_WORK (default $PWD/work),
# CUDA_HOME (default /usr/local/cuda-12.9), LC0EX_ARCH (default: the GPU's compute capability).
#
# Rules this script enforces, each of which cost a run once:
#   * it never overwrites an artifact, and a label whose private cache exists is refused (a failed build leaves one);
#   * the carrier is written FIRST, from the same reader the builder uses -- lc0 loads the CARRIER as --weights;
#   * nothing else may use the GPU while it autotunes: a disturbed build ships foreign tile choices invisibly.
set -u
LABEL=${1:?label}; G=${2:?gpu}; NET=${3:?net}; B=${4:?batches}; WARM=${5:?warm triton dir or cold}; WTC=${6:?tilecache or none}; shift 6
TREE=$(cd "$(dirname "$0")/.." && pwd)
PY=${LC0EX_PYTHON:-$TREE/.venv/bin/python}
WORK=${LC0EX_WORK:-$PWD/work}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
SRC=$TREE/packages/lczero-triton/src:$TREE/packages/lc0ex/src
ARCH=${LC0EX_ARCH:-$(nvidia-smi -i "$G" --query-gpu=compute_cap --format=csv,noheader | tr -d '.')}
mkdir -p "$WORK"/{art,logs,cache,carriers}
ART=$WORK/art/${LABEL}_sm${ARCH}.lc0ex; CACHE=$WORK/cache/triton_${LABEL}; TC=$WORK/cache/tilecache_${LABEL}.json
LOG=$WORK/logs/build_${LABEL}.log
CUT="LC0EX_CUTLASS_QKV=1 LC0EX_CUTLASS_OUTPROJ=1 LC0EX_CUTLASS_FFN2=1 LC0EX_CUTLASS_GLU=1"
SERVED="LC0EX_EGT_STATE=auto LC0EX_EGT_TILES_MAX_BATCH=16 LC0EX_EGT_OVERFLOW=correction LC0EX_TRIPLET_FORM=fused LC0EX_TRIPLET_DOT=fp16"
[ -e "$ART" ] && { echo "ABORT: $ART exists (never overwrite an artifact)"; exit 1; }
[ -s "$NET" ] || { echo "ABORT: missing net $NET"; exit 1; }
[ -e "$CACHE" ] && { echo "ABORT: $CACHE exists (private cache per build; pick a new LABEL)"; exit 1; }
busy=$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { echo "ABORT: GPU $G runs $busy compute app(s); never autotune on a shared card"; exit 1; }
if [ "$WARM" = cold ]; then mkdir -p "$CACHE"; else cp -a "$WARM" "$CACHE" || exit 1; fi
if [ "$WTC" = none ]; then echo '{}' > "$TC"; else cp "$WTC" "$TC" || exit 1; fi
K=$TREE/packages/lczero-triton/src/lczero_triton
echo "# build $LABEL gpu=$G sm_$ARCH batches=$B net $(sha256sum "$NET" | cut -c1-8) warm=$(basename "$WARM") tilecache=$(basename "$WTC") extra: $* start $(date -u +%T)Z"
echo "# tree: network $(sha256sum $K/lab/network.py | cut -c1-8) _mapping $(sha256sum $K/lab/_mapping.py | cut -c1-8) cutlass_matmul $(sha256sum $K/bt4/kernels/cutlass_matmul.py | cut -c1-8) attention_egt $(sha256sum $K/bt4/kernels/attention_egt.py | cut -c1-8)"
CAR=$WORK/carriers/$(basename "$NET" .pb.gz)_lc0ex_carrier.pb.gz
if [ ! -s "$CAR" ]; then
  PYTHONPATH=$SRC "$PY" "$TREE/tools/make_carrier.py" "$NET" "$CAR" 2>&1 | grep -E "CarrierReport|wrote|Error|error" | sed 's/^/# carrier: /'
  [ -s "$CAR" ] || { echo "ABORT: carrier failed"; exit 1; }
fi
echo "# carrier $(basename "$CAR") $(sha256sum "$CAR" | cut -c1-8)   <- this is lc0's --weights"
t0=$(date +%s)
env $CUT $SERVED "$@" LC0EX_CUTLASS_TILE_CACHE="$TC" TRITON_CACHE_DIR="$CACHE" PYTHONPATH=$SRC CUDA_VISIBLE_DEVICES=$G \
  "$PY" "$TREE/tools/build_lab.py" --network "$NET" --output "$ART" --batch-size "$B" > "$LOG" 2>&1
rc=$?
echo "# built in $(( $(date +%s) - t0 ))s rc=$rc sha $(sha256sum "$ART" 2>/dev/null | cut -c1-8) $(grep -m1 -oE 'cutlass_nodes=[0-9]+' "$LOG") $(grep -m1 -oE 'kernels=[0-9]+' "$LOG")"
grep -E "Traceback|Error|error:" "$LOG" | head -3
[ -s "$ART" ] || exit 1
LC0EX_CHECK_ARCH=$ARCH PYTHONPATH=$SRC "$PY" "$TREE/tools/check_artifact_tile_consistency.py" "$ART" 2>&1 | tail -1
echo "# finished $(date -u +%T)Z"
