#!/bin/bash
# Gate C.2 (embed_tokens via UVA, ESATTO): MTP BF16, chunk 4096, KV 12 GB uguale nei due lati, worktree fase2.
# off = vision UVA come bf16uva-kv12; on = vision + embed_tokens UVA. Greedy top-20 logprob a 1 e 4 utenti, poi confronto.
# Lanciare: flock /mnt/llmunity-models/gpu.lock bash run_c2.sh
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-fp8ple-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_c2 $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
for mode in off on; do
  if [ $mode = off ]; then export EXTRA="--cpu-offload-gb 2 --cpu-offload-params visual"
  else export EXTRA="--cpu-offload-gb 3 --cpu-offload-params visual,embed_tokens"; fi
  bash $B/srv.sh start c2$mode $V 12 4096 3 || exit 1
  sudo journalctl -u llmunity-vllm-c2$mode --since '-10min' --no-pager -o cat | grep -iE "offload" | cut -c1-200 | tail -3
  $PY $B/gate_gen.py 30002 $V $B/c2_${mode}1.json 1 512
  $PY $B/gate_gen.py 30002 $V $B/c2_${mode}4.json 4 512
  echo "-- immagine ($mode)"; $PY $B/img.py 30002
  bash $B/srv.sh stop c2$mode
done
echo "== off1 contro on1"; $PY $B/gate_cmp.py $B/c2_off1.json $B/c2_on1.json
echo "== off4 contro on4"; $PY $B/gate_cmp.py $B/c2_off4.json $B/c2_on4.json
echo "== rumore: off1 contro off4"; $PY $B/gate_cmp.py $B/c2_off1.json $B/c2_off4.json
echo "[$(date -u +%T)] RUN_C2_FINE"
