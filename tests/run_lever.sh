#!/bin/bash
# Una leva di velocita' su un candidato: server con CSV ms/step, sweep (temp 1.0, 512 token) e greedy top-20 a 1 utente
# per il gate "in distribuzione" (gate_cmp.py contro il riferimento della stessa vista con MTP 3 e leva spenta).
# NBT = max-num-batched-tokens (default 4096).
# Uso: run_lever.sh <tag> <vista> <kv_gb> <num_spec> "<extra vllm>" "<env -E ...>"  (sotto flock gpu.lock)
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; PY=$R/vllm-venv/bin/python
TAG=$1; V=$2; KV=$3; NS=$4; export EXTRA=$5; export ENVS=$6
echo "lock BANCO run_lever $TAG $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
csv=$B/steps_$TAG.csv; : > $csv
bash $B/srv.sh start $TAG $V $KV ${NBT:-4096} $NS $csv || exit 1
$PY $B/sweep.py 30002 $csv $V 32000,128000,180000 1,2,4 512 1.0
GATE_SALT=$TAG $PY $B/gate_gen.py 30002 $V $B/gate_$TAG.json 1 512  # salt: prefill a freddo, indipendente dalla cache lasciata dallo sweep
echo "acceptance: $(sudo journalctl -u llmunity-vllm-$TAG --since '-60min' --no-pager -o cat | grep -oE 'Mean acceptance length: [0-9.]+' | awk '{s+=$4;n++} END {if(n) printf "%.2f su %d righe", s/n, n}')"
$PY $B/steps_sum.py $csv
bash $B/srv.sh stop $TAG
echo "[$(date -u +%T)] RUN_LEVER_FINE $TAG"
