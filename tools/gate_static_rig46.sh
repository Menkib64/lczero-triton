#!/bin/bash
# Item A.8 whole-net gate, rig46 (sm_120): lc0ex on the lab static net vs onnx-cuda.
#   usage: gate_static_rig46.sh <artifact> <gpu> [batch]
# lc0ex loads the r20b CARRIER (initializers named by lab/_names.py); onnx-cuda
# loads the ORIGINAL export (graph + initializers) through --ref-weights.
# Binary: the published branch with ONNX, 383416a1. KL/top-1 lines are the
# result; exit codes from PIPESTATUS.
set -u
ART=$1 GPU=$2 BATCH=${3:-8}
cd ~/spsa/lc0ex_5080 || exit 1
V=$PWD/lczero-triton/.venv/lib/python3.14/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cudnn/lib:$V/cublas/lib:$V/cuda_nvrtc/lib:$PWD/ort/onnxruntime-linux-x64-gpu-1.23.2/lib:${LD_LIBRARY_PATH:-}
LC0=$PWD/lczero-triton/submodules/lc0/build/rel-onnx/lc0
EXPORT=$PWD/work/static_net/static_bs4g_512x15_50000_vw.pb.gz
CARRIER=$PWD/work/static_net/static_bs4g_512x15_lc0ex_r20c.pb.gz
FENS=${FENS:-$PWD/positions/rep128.fen}
MAXPOS=${MAXPOS:-512}
echo "# gate rig46: lc0 $(sha256sum $LC0 | cut -c1-8) artifact $(sha256sum $ART | cut -c1-8) carrier $(sha256sum $CARRIER | cut -c1-8) export $(sha256sum $EXPORT | cut -c1-8) gpu=$GPU batch=$BATCH fens $(sha256sum $FENS | cut -c1-8) maxpos=$MAXPOS"
LX="gpu=$GPU,lc0ex=$ART,concurrency=1,graph=dag"
echo "### 1. load + backendbench (lc0ex on the carrier)"
"$LC0" backendbench -w "$CARRIER" --backend=lc0ex-cuda --backend-opts="$LX" \
  --start-batch-size=$BATCH --max-batch-size=$BATCH --batch-step=16 --batches=50 2>&1 | grep -vE "^\s*$|initializer .* has no corresponding" | tail -12
echo "exit=${PIPESTATUS[0]}"
for i in 1 2 3; do
  echo "### 2.$i self (lc0ex vs lc0ex)"
  "$LC0" backendcompare -w "$CARRIER" --backend=lc0ex-cuda --backend-opts="$LX" \
    --ref-backend=lc0ex-cuda --ref-backend-opts="$LX" --batch-size=$BATCH --fens="$FENS" --max-positions=$MAXPOS 2>&1 | grep -vE "initializer .* has no corresponding" | tail -9
  echo "exit=${PIPESTATUS[0]}"
done
echo "### 3. reference self-determinism (onnx-cuda optimize=6, export)"
"$LC0" backendcompare -w "$EXPORT" --backend=onnx-cuda --backend-opts="gpu=$GPU,optimize=6" \
  --ref-backend=onnx-cuda --ref-backend-opts="gpu=$GPU,optimize=6" --batch-size=$BATCH --fens="$FENS" --max-positions=$MAXPOS 2>&1 | tail -9
echo "exit=${PIPESTATUS[0]}"
echo "### 4. lc0ex (carrier) vs onnx-cuda optimize=6 (export)"
"$LC0" backendcompare -w "$CARRIER" --backend=lc0ex-cuda --backend-opts="$LX" \
  --ref-weights="$EXPORT" --ref-backend=onnx-cuda --ref-backend-opts="gpu=$GPU,optimize=6" \
  --batch-size=$BATCH --fens="$FENS" --max-positions=$MAXPOS 2>&1 | grep -vE "initializer .* has no corresponding" | tail -9
echo "exit=${PIPESTATUS[0]}"
echo "# gate finished $(date +%T)"
