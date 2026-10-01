#!/bin/bash
# Determinismo della base e gate C.2 rifatto (sintassi giusta: --cpu-offload-params visual embed_tokens).
# Ogni generazione con cache_salt nuovo (prefill a freddo), greedy top-20, 1 utente, 128 token, 8 prompt.
# A1/A2 = stesso avvio off; B1 = secondo avvio off; C1 = avvio on. Lanciare: flock /mnt/llmunity-models/gpu.lock bash run_det.sh
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-fp8ple-ramview; PY=$R/vllm-venv/bin/python
echo "lock BANCO run_det $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
run() {  # run <tag> <extra> <out1> [out2]
  export EXTRA=$2; bash $B/srv.sh start $1 $V 12 4096 3 || exit 1
  sudo journalctl -u llmunity-vllm-$1 --since '-10min' --no-pager -o cat | grep -E "offloaded|loading took|cpu_offload_params" | grep -oE "Total CPU offloaded parameters: [0-9.]+|loading took [0-9.]+ GiB|cpu_offload_params': \[[^]]*\]" | sort -u
  GATE_SALT=$1a $PY $B/gate_gen.py 30002 $V $B/$3 1 128
  [ -n "$4" ] && GATE_SALT=$1b $PY $B/gate_gen.py 30002 $V $B/$4 1 128
  bash $B/srv.sh stop $1
}
run detA "--cpu-offload-gb 2 --cpu-offload-params visual" det_A1.json det_A2.json
run detB "--cpu-offload-gb 2 --cpu-offload-params visual" det_B1.json
run detC "--cpu-offload-gb 3 --cpu-offload-params visual embed_tokens" det_C1.json
echo "== A1 contro A2 (stesso avvio)"; $PY $B/gate_cmp.py $B/det_A1.json $B/det_A2.json
echo "== A1 contro B1 (due avvii)"; $PY $B/gate_cmp.py $B/det_A1.json $B/det_B1.json
echo "== A1 contro C1 (embed UVA)"; $PY $B/gate_cmp.py $B/det_A1.json $B/det_C1.json
echo "[$(date -u +%T)] RUN_DET_FINE"
