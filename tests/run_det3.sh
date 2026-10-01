#!/bin/bash
# Controllo diretto delle righe PLE (VLLM_PLE_CHECK_ROWS=1, patch FORGIA): base FP8 host_file_gather, det2.py; ogni
# riga caricata in _staging confrontata con un pread indipendente. Poi conteggio dei WARNING "PLE CHECK".
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-mtpfp8-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_det3 $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
export EXTRA=""
T0=$(date "+%Y-%m-%d %H:%M:%S")
ENVS="-E VLLM_PLE_CHECK_ROWS=1" bash $B/srv.sh start det3chk $V 13 4096 3 || exit 1
$PY $B/det2.py 30002 $V chk
sudo journalctl -u llmunity-vllm-det3chk --since "$T0" --no-pager -o cat | grep "PLE CHECK" | tail -20 | cut -c1-200
echo "righe con mismatch: $(sudo journalctl -u llmunity-vllm-det3chk --since "$T0" --no-pager -o cat | grep "PLE CHECK" | grep -vc " 0/")"
bash $B/srv.sh stop det3chk
echo "[$(date -u +%T)] RUN_DET3_FINE"
