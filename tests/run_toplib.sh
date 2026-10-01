#!/bin/bash
# Top-k deterministico in CUDA (backport #55122, VLLM_QSA_DET_TOPK_LIB): bit a bit contro l'overlay Python (gate_topk1,
# gate4_topk1 di run_det2) e costo del prefill a freddo con det2.py (2k/32k/200k).
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-mtpfp8-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_toplib $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
EXTRA="" ENVS="-E VLLM_QSA_DET_TOPK_LIB=$R/fase2-build/topk/build/det_topk_C.so" bash $B/srv.sh start toplib $V 13 4096 3 || exit 1
$PY $B/det2.py 30002 $V toplib
GATE_SALT=toplib $PY $B/gate_gen.py 30002 $V $B/gate_toplib.json 1 128
GATE_SALT=toplib $PY $B/gate_gen4.py 30002 $V $B/gate4_toplib.json 128
bash $B/srv.sh stop toplib
echo "== 1 utente: overlay contro kernel"; $PY $B/gate_cmp.py $B/gate_topk1.json $B/gate_toplib.json
echo "== 4 utenti: overlay contro kernel"; $PY $B/gate_cmp.py $B/gate4_topk1.json $B/gate4_toplib.json
echo "[$(date -u +%T)] RUN_TOPLIB_FINE"
