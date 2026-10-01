#!/bin/bash
# Smoke B.3 sulla combo base (MTP FP8, KV 13, chunk 4096, worktree fase2): patch spenta e accesa, greedy (temp 0),
# stessi prompt: tok/s spento contro acceso = costo della patch; con la patch accesa ms/step dal CSV.
# Lanciare: flock /mnt/llmunity-models/gpu.lock bash run_b3.sh
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-mtpfp8-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_b3 $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
for mode in off on; do
  csv=""; [ $mode = on ] && { csv=$B/steps_b3_on.csv; : > $csv; }
  bash $B/srv.sh start b3$mode $V 13 4096 3 $csv || exit 1
  $PY $B/sweep.py 30002 ${csv:-/dev/null} $V 32000,128000 1,4 256 0.0
  echo "acceptance: $(sudo journalctl -u llmunity-vllm-b3$mode --since '-30min' --no-pager -o cat | grep -oE 'Mean acceptance length: [0-9.]+' | awk '{s+=$4;n++} END {if(n) printf "%.2f su %d righe", s/n, n}')"
  bash $B/srv.sh stop b3$mode
done
echo "[$(date -u +%T)] RUN_B3_FINE"
