#!/bin/bash
# Static net speed, same card as the onnx-cuda baseline (GPU 3 by default).
#   1. lc0ex backendbench on the CUTLASS ladder at 8/16/32/48/64/96/128, 3 passes
#   2. two-thread search at minibatch 56 and 64, DEFAULT prefetch -- the ladder's
#      top rung (128) must absorb prefetch, which also confirms the single-rung
#      search crash was the rung and nothing else
#   3. onnx-cuda search at the same settings (the backendbench column exists)
set -u
cd ~/spsa/lc0ex_5080 || exit 1
V=$PWD/lczero-triton/.venv/lib/python3.14/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cudnn/lib:$V/cublas/lib:$V/cuda_nvrtc/lib:$PWD/ort/onnxruntime-linux-x64-gpu-1.23.2/lib:${LD_LIBRARY_PATH:-}
LC0=$PWD/lczero-triton/submodules/lc0/build/rel-onnx/lc0
CARRIER=$PWD/work/static_net/static_bs4g_512x15_lc0ex_r20e.pb.gz
REF=$PWD/work/static_net/static_bs4g_512x15_50000_vw_wdlsoftmax.pb.gz
ART=$PWD/work/static_net/static_r20e_ladder_cutlass_sm120.lc0ex
GPU=${1:-3}
[ -s "$ART" ] || { echo "missing $ART"; exit 1; }
echo "# static speed: lc0 $(sha256sum $LC0 | cut -c1-8) artifact $(sha256sum $ART | cut -c1-8) carrier $(sha256sum $CARRIER | cut -c1-8) ref $(sha256sum $REF | cut -c1-8) gpu=$GPU"
echo "## 1. lc0ex backendbench"
for pass in 1 2 3; do
  for b in 8 16 32 48 64 96 128; do
    printf "pass=%s B=%-4s " $pass $b
    $LC0 backendbench -w $CARRIER --backend=lc0ex-cuda --backend-opts=gpu=$GPU,lc0ex=$ART,concurrency=1,graph=dag \
      --start-batch-size=$b --max-batch-size=$b --batch-step=16 --batches=100 2>&1 | grep -E "Benchmark batch size|rror|free\(\)" | tail -1
  done
done
echo "## 2-3. two-thread search, 40,000 nodes x 8 positions, default prefetch"
for run in 1 2 3; do
  for mb in 56 64; do
    printf "run=%s mb=%-3s lc0ex     " $run $mb
    $LC0 benchmark -w $CARRIER --backend=lc0ex-cuda --backend-opts=gpu=$GPU,lc0ex=$ART,concurrency=8,graph=dag \
      --threads=2 --minibatch-size=$mb --nodes=40000 --num-positions=8 2>&1 | grep -E "Nodes/second|rror|free\(\)|xception" | tail -1
    printf "run=%s mb=%-3s onnx-cuda " $run $mb
    $LC0 benchmark -w $REF --backend=onnx-cuda --backend-opts=gpu=$GPU \
      --threads=2 --minibatch-size=$mb --nodes=40000 --num-positions=8 2>&1 | grep -E "Nodes/second|rror" | tail -1
  done
done
echo "# finished $(date +%T)"
