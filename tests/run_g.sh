#!/bin/bash
# Gate server di G (replay stato GDN, worktree fase2g): base FP8 + top-k deterministico + --use-replayssm.
# Confronto greedy 1 utente a freddo contro gate_topk1.json (stessa base senza G, da run_det2). Log kernel e KV.
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-mtpfp8-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_g $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
T0=$(date "+%Y-%m-%d %H:%M:%S")
TREE=main EXTRA="--use-replayssm" ENVS="-E PYTHONPATH=$R/vllm-fase2g -E VLLM_GDN_REPLAY_LIB=$R/fase2-build/g/build_120/gdn_replay_C_120.so -E VLLM_QSA_DET_TOPK_PY=1" \
  bash $B/srv.sh start g1 $V 13 4096 3 || exit 1
sudo journalctl -u llmunity-vllm-g1 --since "$T0" --no-pager -o cat | grep -E "RecoverSSM|GDN decode kernel|Falling back|KV cache size|num_blocks|loading took" | cut -c1-200
GATE_SALT=g1 $PY $B/gate_gen.py 30002 $V $B/gate_g1.json 1 128
GATE_SALT=g1 $PY $B/gate_gen4.py 30002 $V $B/gate4_g1.json 128
bash $B/srv.sh stop g1
echo "== base+topk contro base+topk+G"; $PY $B/gate_cmp.py $B/gate_topk1.json $B/gate_g1.json
echo "== 4 utenti: base+topk contro base+topk+G"; $PY $B/gate_cmp.py $B/gate4_topk1.json $B/gate4_g1.json
echo "[$(date -u +%T)] RUN_G_FINE"
