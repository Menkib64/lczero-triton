#!/bin/bash
# Round 26 Q1 gates for an int8 artifact and its FP16 twin (both load the SAME Q1 carrier), `lc0 backendcompare`:
#   G1  int8 self-determinism (must print 0.000000 everywhere)
#   G3  int8 vs the FP16 twin, lc0ex against lc0ex: the served quantisation error itself (REPORT, not a gate)
#   ref int8 vs lc0's ONNX backend on the export's WDL-softmax copy, and the FP16 twin vs the same (the control: the
#       twin must read like every FP16 artifact, KL ~1e-6; the int8 must NOT)
#
#   usage: gate_lab_artifact_q1.sh GPU INT8.lc0ex FP16.lc0ex CARRIER.pb.gz EXPORT_wdlsoftmax.pb.gz FENS [BATCH ...]
# Needs LC0 (an lc0 with -Dlc0ex-runtime=true and -Donnx=true). Never on a card that is autotuning.
set -u
G=${1:?gpu}; I8=${2:?int8 artifact}; F16=${3:?fp16 artifact}; CAR=${4:?carrier}; REF=${5:?wdlsm export}; FENS=${6:?fens}
shift 6
BATCHES=${*:-64}
LC0=${LC0:-lc0}
cmp() {  # LABEL ART REFBACKEND REFWEIGHTS REFOPTS BATCH
  echo "### $1 batch $6   art $(basename "$2") $(sha256sum "$2" | cut -c1-8)"
  timeout 3600 stdbuf -oL -eL "$LC0" backendcompare -w "$CAR" --backend=lc0ex-cuda \
    --backend-opts="gpu=$G,lc0ex=$2,concurrency=1,graph=dag" --ref-weights="$4" --ref-backend="$3" \
    --ref-backend-opts="$5" --batch-size="$6" --fens="$FENS" --max-positions=256 2>&1 \
    | grep -E "policy KL|policy L1|q_test|top-1|top-3|rror|xception" | grep -v "has no corresponding"
}
echo "# Q1 gate on GPU $G start $(date -u +%T)Z; carrier $(basename "$CAR") $(sha256sum "$CAR" | cut -c1-8); compute apps on the card: $(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | wc -l)"
cmp "G1 int8 self (must be 0.000000)" "$I8" lc0ex-cuda "$CAR" "gpu=$G,lc0ex=$I8,concurrency=1,graph=dag" 64
for b in $BATCHES; do
  cmp "G3 int8 vs FP16 twin (lc0ex vs lc0ex)" "$I8" lc0ex-cuda "$CAR" "gpu=$G,lc0ex=$F16,concurrency=1,graph=dag" "$b"
  cmp "ref int8 vs onnx-cuda (WDL-softmax copy)" "$I8" onnx-cuda "$REF" "gpu=$G" "$b"
  cmp "control FP16 twin vs onnx-cuda (WDL-softmax copy)" "$F16" onnx-cuda "$REF" "gpu=$G" "$b"
done
echo "# finished $(date -u +%T)Z"
