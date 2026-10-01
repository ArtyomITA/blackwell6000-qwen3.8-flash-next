#!/bin/bash
# KV massima che regge lo stress (4 x 195-228k) per un candidato, poi velocita' completa (vspeed_f2 = vspeed.sh con
# PYTHONPATH fase2, KV decimale, CSV ms/step). Si ferma alla prima KV senza OOM e con server vivo.
# Da LEAD #122.4 la base ha il top-k deterministico: KVENV (default "-E VLLM_QSA_DET_TOPK_PY=1") va nella unit.
# Uso: run_kv.sh <tag> <vista> "<extra vllm>" <kv1> [kv2 ...]   (sotto flock /mnt/llmunity-models/gpu.lock)
B=/mnt/llmunity-models/fase2-tests/banco; TAG=$1; VIEW=$2; export EXTRA=$3; shift 3
echo "lock BANCO run_kv $TAG $(date -u +%T) env: ${KVENV--E VLLM_QSA_DET_TOPK_PY=1}"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
for g in "$@"; do
  t=$TAG-kv${g/./p}; csv=$B/steps_$t.csv; : > $csv
  STRESS=1 IMG=1 STENV="-E VLLM_STEP_TIMING_FILE=$csv ${KVENV--E VLLM_QSA_DET_TOPK_PY=1}" bash $B/${VSPEED:-vspeed_f2.sh} $t $VIEW $g 4096 3 '{"host_file_gather":true}' 12G | tee $B/vspeed_$t.log
  grep -q "righe OOM: 0" $B/vspeed_$t.log && grep -q "server vivo? 200" $B/vspeed_$t.log && { echo "KV_OK $g GB"; break; }
  echo "KV $g GB non regge"
done
echo "[$(date -u +%T)] RUN_KV_FINE $TAG"
