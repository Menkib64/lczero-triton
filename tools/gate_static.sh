#!/bin/bash
# Item A.8 -- whole-net gate for the lab static net served by lc0ex.
#   usage: gate_static.sh <lc0 binary> <artifact> <gpu> [batch]
# Weights: lc0ex loads the r20 CARRIER (initializers named by lab/_names.py);
# the onnx-cuda reference loads the ORIGINAL export (graph + initializers) via
# --ref-weights, because the carrier deliberately holds no graph.
# Reads the KL / top-1 lines; exit codes come from PIPESTATUS (R-trap 09-10).
set -u
LC0=$1 ART=$2 GPU=$3 BATCH=${4:-8}
cd ~/spsa/static_net_r20 || exit 1
EXPORT=$HOME/Kovax/nets/static_bs4g_512x15_50000_vw.pb.gz
CARRIER=$PWD/work/static_bs4g_512x15_lc0ex_r20b.pb.gz
FENS=${FENS:-$HOME/spsa/lc0ex_2026-09/positions/rep128.fen}
MAXPOS=${MAXPOS:-512}
echo "# gate: lc0 $(sha256sum $LC0 | cut -c1-8) artifact $(sha256sum $ART | cut -c1-8) carrier $(sha256sum $CARRIER | cut -c1-8) export $(sha256sum $EXPORT | cut -c1-8) gpu=$GPU batch=$BATCH fens $(sha256sum $FENS | cut -c1-8)"
LX="gpu=$GPU,lc0ex=$ART,concurrency=1,graph=dag"
echo "### 1. load + backendbench (lc0ex, carrier)"
"$LC0" backendbench -w "$CARRIER" --backend=lc0ex-cuda --backend-opts="$LX" \
  --start-batch-size=$BATCH --max-batch-size=$BATCH --batch-step=16 --batches=50 2>&1 | grep -vE "^\s*$" | tail -12
echo "exit=${PIPESTATUS[0]}"
for i in 1 2 3; do
  echo "### 2.$i self (lc0ex vs lc0ex)"
  "$LC0" backendcompare -w "$CARRIER" --backend=lc0ex-cuda --backend-opts="$LX" \
    --ref-backend=lc0ex-cuda --ref-backend-opts="$LX" --batch-size=$BATCH --fens="$FENS" --max-positions=$MAXPOS 2>&1 | tail -9
  echo "exit=${PIPESTATUS[0]}"
done
echo "### 3. reference self-determinism (onnx-cuda optimize=6 vs itself, export)"
"$LC0" backendcompare -w "$EXPORT" --backend=onnx-cuda --backend-opts="gpu=$GPU,optimize=6" \
  --ref-backend=onnx-cuda --ref-backend-opts="gpu=$GPU,optimize=6" --batch-size=$BATCH --fens="$FENS" --max-positions=$MAXPOS 2>&1 | tail -9
echo "exit=${PIPESTATUS[0]}"
echo "### 4. lc0ex (carrier) vs onnx-cuda optimize=6 (export)"
"$LC0" backendcompare -w "$CARRIER" --backend=lc0ex-cuda --backend-opts="$LX" \
  --ref-weights="$EXPORT" --ref-backend=onnx-cuda --ref-backend-opts="gpu=$GPU,optimize=6" \
  --batch-size=$BATCH --fens="$FENS" --max-positions=$MAXPOS 2>&1 | tail -9
echo "exit=${PIPESTATUS[0]}"
