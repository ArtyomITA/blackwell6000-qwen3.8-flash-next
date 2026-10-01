#!/bin/bash
# Avvia vLLM (stessa riga di vspeed.sh) dal worktree fase2 o dall'albero principale e aspetta /health.
# Uso: srv.sh start <tag> <vista> <kv_gb> <chunk> <num_spec> [csv_step]   |   srv.sh stop <tag>
# Variabili: PREFIX="comando davanti a vllm (es. nsys)", ENVS="-E VAR=val ...", EXTRA="argomenti vllm in piu'", TREE=main (default fase2), MEMMAX (default 12G).
# Da chiamare sotto flock /mnt/llmunity-models/gpu.lock. Esce 0 solo con server pronto.
R=/mnt/llmunity-models; O=$R/prod-tests; PORT=30002; UNIT=llmunity-vllm-$2
if [ "$1" = stop ]; then sudo systemctl stop $UNIT 2>/dev/null; sleep 10; echo "[$(date -u +%T)] fermato $UNIT"; exit 0; fi
TAG=$2; VIEW=$3; KVG=$4; NBT=$5; NSPEC=$6; CSV=$7
PP=""; [ "${TREE:-fase2}" = fase2 ] && PP="-E PYTHONPATH=$R/vllm-fase2"
ST=""; [ -n "$CSV" ] && ST="-E VLLM_STEP_TIMING_FILE=$CSV"
T0=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$(date -u +%T)] avvio $TAG: vista $VIEW KV $KVG GB chunk $NBT MTP $NSPEC albero ${TREE:-fase2} csv ${CSV:-no} extra: $EXTRA"
sudo systemd-run --unit=$UNIT --collect -p MemoryMax=${MEMMAX:-12G} -p MemorySwapMax=0 -p RuntimeMaxSec=14400 -p LimitNOFILE=65535 -p LimitMEMLOCK=infinity \
  --uid=ec2-user --working-directory=$R \
  -E HOME=/home/ec2-user -E CUDA_HOME=/usr/local/cuda -E PATH=/usr/local/cuda/bin:$R/vllm-venv/bin:/usr/bin:/bin -E HF_HUB_OFFLINE=1 -E TRANSFORMERS_OFFLINE=1 -E PYTHONUNBUFFERED=1 $PP $ST $ENVS \
  $PREFIX $R/vllm-venv/bin/vllm serve $VIEW --served-model-name qwen x --host 127.0.0.1 --port $PORT \
  --load-format safetensors --tensor-parallel-size 1 \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens $NBT --gpu-memory-utilization 0.90 \
  --kv-cache-memory-bytes $(awk "BEGIN{printf \"%d\", $KVG*1073741824}") --kv-cache-dtype fp8 \
  --engram-config '{"host_file_gather":true}' --reasoning-parser qwen3 \
  --chat-template $O/chat-template-qwen3.8-unsloth.jinja --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"full_decode_only"}' --no-enable-flashinfer-autotune \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}" $EXTRA > /dev/null 2>&1
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; systemctl is-active --quiet $UNIT || break; sleep 5; done
if ! curl -sf http://127.0.0.1:$PORT/health > /dev/null; then
  echo "vLLM $TAG non partito"; sudo journalctl -u $UNIT --since "$T0" --no-pager -o cat | grep -iE "error|memory" | grep -v "frame #" | tail -5 | cut -c1-300
  sudo systemctl stop $UNIT 2>/dev/null; sleep 8; exit 1
fi
sudo journalctl -u $UNIT --since "$T0" --no-pager -o cat | grep -E "KV cache size|Model loading took|offloaded|GDN|MoE backend|vllm-fase2|CUDA graph|Graph capturing" | cut -c1-220
echo "[$(date -u +%T)] PRONTO $TAG"
