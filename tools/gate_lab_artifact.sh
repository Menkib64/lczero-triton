#!/bin/bash
# Correctness gate for a lab artifact: lc0ex against lc0's own ONNX backend running the SAME export, position by
# position (`lc0 backendcompare`: policy KL, top-1 agreement, value error), plus self-determinism.
#
#   usage: gate_lab_artifact.sh GPU ARTIFACT.lc0ex CARRIER.pb.gz EXPORT_wdlsoftmax.pb.gz FENS [BATCH ...]
#
# * lc0ex serves the CARRIER (its initializers, planned as lc0ex buffers) and softmaxes the WDL logits itself
#   (`DecodeWdl` in network_lc0ex_cuda.cc), so the artifact emits LOGITS.
# * lc0's ONNX backend reads `/output/wdl` as PROBABILITIES, so the reference is the export's WDL-softmax copy
#   (`tools/make_wdl_softmax.py`), never the raw export.
# * Needs an lc0 built with -Dlc0ex-runtime=true AND -Donnx=true (LC0 env, default `lc0` on PATH); never run it on a
#   card that is autotuning.
set -u
G=${1:?gpu}; ART=${2:?artifact}; CAR=${3:?carrier}; REF=${4:?wdl-softmax export}; FENS=${5:?fens}; shift 5
BATCHES=${*:-16 64}
LC0=${LC0:-lc0}
cmp() {  # LABEL REFBACKEND REFWEIGHTS REFOPTS BATCH
  echo "### $1 batch $5   art $(basename "$ART") $(sha256sum "$ART" | cut -c1-8) carrier $(sha256sum "$CAR" | cut -c1-8)"
  timeout 3600 stdbuf -oL -eL "$LC0" backendcompare -w "$CAR" --backend=lc0ex-cuda \
    --backend-opts="gpu=$G,lc0ex=$ART,concurrency=1,graph=dag" --ref-weights="$3" --ref-backend="$2" \
    --ref-backend-opts="$4" --batch-size="$5" --fens="$FENS" --max-positions=256 2>&1 \
    | grep -E "policy KL|policy L1|q_test|top-1|top-3|rror|xception" | grep -v "has no corresponding"
}
echo "# gate on GPU $G start $(date -u +%T)Z; compute apps on the card: $(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | wc -l)"
cmp "self (must be 0.000000)" lc0ex-cuda "$CAR" "gpu=$G,lc0ex=$ART,concurrency=1,graph=dag" 64
for b in $BATCHES; do cmp "vs onnx-cuda (WDL-softmax copy)" onnx-cuda "$REF" "gpu=$G" "$b"; done
echo "# finished $(date -u +%T)Z"
