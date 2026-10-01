#!/bin/bash
# Determinismo: base FP8 (persistent_topk) contro overlay top-k deterministico (VLLM_QSA_DET_TOPK_PY=1, due avvii).
# det2.py: 2k/32k/200k x5 con cache_salt (prefill a freddo), stesso avvio; poi confronto tra i due avvii con overlay.
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-mtpfp8-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_det2 $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
export EXTRA=""
ENVS="" bash $B/srv.sh start det2base $V 13 4096 3 || exit 1
$PY $B/det2.py 30002 $V base; bash $B/srv.sh stop det2base
for k in 1 2; do
  ENVS="-E VLLM_QSA_DET_TOPK_PY=1" bash $B/srv.sh start det2topk$k $V 13 4096 3 || exit 1
  $PY $B/det2.py 30002 $V topk$k
  GATE_SALT=ref$k $PY $B/gate_gen.py 30002 $V $B/gate_topk$k.json 1 128
  GATE_SALT=ref$k $PY $B/gate_gen4.py 30002 $V $B/gate4_topk$k.json 128
  bash $B/srv.sh stop det2topk$k
done
$PY - <<'PY'
import json
a = json.load(open("/mnt/llmunity-models/fase2-tests/banco/det2_topk1.json"))
b = json.load(open("/mnt/llmunity-models/fase2-tests/banco/det2_topk2.json"))
for L in a:
    same = a[L][0]["top_logprobs"] == b[L][0]["top_logprobs"] and a[L][0]["tokens"] == b[L][0]["tokens"]
    print(f"topk avvio1 contro avvio2 L={L}: {'BIT A BIT' if same else 'DIVERSI'}")
PY
echo "== gate_gen topk avvio1 contro avvio2"; $PY $B/gate_cmp.py $B/gate_topk1.json $B/gate_topk2.json
echo "== gate_gen4 topk avvio1 contro avvio2"; $PY $B/gate_cmp.py $B/gate4_topk1.json $B/gate4_topk2.json
echo "[$(date -u +%T)] RUN_DET2_FINE"
