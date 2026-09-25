#!/bin/bash
# Static net, same-card speed on a quiet host (GPU 3 = the onnx-cuda baseline card).
#   A = Triton-only ladder, B = CUTLASS ladder with the alignment guard (FFN2 on Triton).
#   1. backendbench, A and B interleaved per batch, 8..128, 3 passes
#   2. two-thread search, 40,000 nodes x 8 positions, default prefetch, mb 56 and 64:
#      A, B and onnx-cuda interleaved, 3 runs
# Result lines are grepped for throughput AND for failure words (failed|Unhandled|
# rror|xception), so a fault prints instead of leaving an empty row.
set -u
cd ~/spsa/lc0ex_5080 || exit 1
V=$PWD/lczero-triton/.venv/lib/python3.14/site-packages/nvidia
export LD_LIBRARY_PATH=$V/cudnn/lib:$V/cublas/lib:$V/cuda_nvrtc/lib:$PWD/ort/onnxruntime-linux-x64-gpu-1.23.2/lib:${LD_LIBRARY_PATH:-}
LC0=$PWD/lczero-triton/submodules/lc0/build/rel-onnx/lc0
W=$PWD/work/static_net
CARRIER=$W/static_bs4g_512x15_lc0ex_r20e.pb.gz
REF=$W/static_bs4g_512x15_50000_vw_wdlsoftmax.pb.gz
A=$W/static_r20e_iso_tritonladder.lc0ex
B=$W/static_r20e_ladder_cutlass_sm120_v2.lc0ex
GPU=${1:-3}
FAIL="failed|Unhandled|rror|xception"
for f in $A $B; do [ -s $f ] || { echo "missing $f"; exit 1; }; done
echo "# static speed compare: lc0 $(sha256sum $LC0 | cut -c1-8) A(triton) $(sha256sum $A | cut -c1-8) B(cutlass guarded) $(sha256sum $B | cut -c1-8) carrier $(sha256sum $CARRIER | cut -c1-8) ref $(sha256sum $REF | cut -c1-8) gpu=$GPU"
echo "## 1. backendbench"
for pass in 1 2 3; do
  for b in 8 16 32 48 64 96 128; do
    for arm in "triton|$A" "cutlass|$B"; do
      label=${arm%%|*}; art=${arm#*|}
      printf "pass=%s B=%-4s %-8s " $pass $b $label
      $LC0 backendbench -w $CARRIER --backend=lc0ex-cuda --backend-opts=gpu=$GPU,lc0ex=$art,concurrency=1,graph=dag \
        --start-batch-size=$b --max-batch-size=$b --batch-step=16 --batches=100 2>&1 | grep -E "Benchmark batch size|$FAIL" | tail -1
    done
  done
done
echo "## 2. two-thread search"
for run in 1 2 3; do
  for mb in 56 64; do
    for arm in "triton|lc0ex-cuda|$CARRIER|gpu=$GPU,lc0ex=$A,concurrency=8,graph=dag" "cutlass|lc0ex-cuda|$CARRIER|gpu=$GPU,lc0ex=$B,concurrency=8,graph=dag" "onnx-cuda|onnx-cuda|$REF|gpu=$GPU"; do
      IFS="|" read -r label backend net opts <<< "$arm"
      printf "run=%s mb=%-3s %-9s " $run $mb $label
      $LC0 benchmark -w $net --backend=$backend --backend-opts=$opts \
        --threads=2 --minibatch-size=$mb --nodes=40000 --num-positions=8 2>&1 | grep -E "Nodes/second|$FAIL" | tail -1
    done
  done
done
echo "# finished $(date +%T)"
