#!/bin/bash
# Riferimento per il gate G: base FP8 + top-k deterministico + --block-size 3392 (stessa pagina e stesso chunk di G).
# Se gate_gref == gate_g1 bit a bit, il replay e' esatto e l'unica differenza di G dalla base e' il chunk 3392.
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-mtpfp8-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_gref $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
T0=$(date "+%Y-%m-%d %H:%M:%S")
EXTRA="--block-size 3392" ENVS="-E VLLM_QSA_DET_TOPK_PY=1" bash $B/srv.sh start gref $V 13 4096 3 || exit 1
sudo journalctl -u llmunity-vllm-gref --since "$T0" --no-pager -o cat | grep -E "block size|KV cache size" | cut -c1-200
GATE_SALT=gref $PY $B/gate_gen.py 30002 $V $B/gate_gref.json 1 128
GATE_SALT=gref $PY $B/gate_gen4.py 30002 $V $B/gate4_gref.json 128
bash $B/srv.sh stop gref
echo "== 1 utente: base+topk+pagina 3392 contro G"; $PY $B/gate_cmp.py $B/gate_gref.json $B/gate_g1.json
echo "== 4 utenti"; $PY $B/gate_cmp.py $B/gate4_gref.json $B/gate4_g1.json
echo "[$(date -u +%T)] RUN_GREF_FINE"
