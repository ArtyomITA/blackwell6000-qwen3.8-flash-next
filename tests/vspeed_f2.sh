#!/bin/bash
# Velocita' vLLM su una configurazione: 1 utente breve x3, 1 utente 245k, 4 utenti reali, 4 utenti pesanti + acceptance MTP.
# Uso: vspeed.sh <tag> <vista> <kv_gb> <max_num_batched_tokens> <num_spec> <engram_json> <memmax_cgroup>
TAG=$1; VIEW=$2; KVG=$3; NBT=$4; NSPEC=$5; ENGRAM=$6; MEMMAX=$7
R=/mnt/llmunity-models; O=$R/prod-tests; C=$O/convs; PORT=30002; UNIT=llmunity-vllm-$TAG
T0=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$(date -u +%T)] avvio $TAG: KV $KVG GB, chunk $NBT, MTP $NSPEC, engram $ENGRAM"
sudo systemd-run --unit=$UNIT --collect -p MemoryMax=$MEMMAX -p MemorySwapMax=0 -p RuntimeMaxSec=7200 -p LimitNOFILE=65535 -p LimitMEMLOCK=infinity \
  --uid=ec2-user --working-directory=$R \
  -E HOME=/home/ec2-user -E CUDA_HOME=/usr/local/cuda -E PATH=/usr/local/cuda/bin:$R/vllm-venv/bin:/usr/bin:/bin -E HF_HUB_OFFLINE=1 -E TRANSFORMERS_OFFLINE=1 -E PYTHONUNBUFFERED=1 -E PYTHONPATH=/mnt/llmunity-models/vllm-fase2 $STENV \
  $R/vllm-venv/bin/vllm serve $VIEW --served-model-name qwen x --host 127.0.0.1 --port $PORT \
  --load-format safetensors --tensor-parallel-size 1 \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens $NBT --gpu-memory-utilization 0.90 \
  --kv-cache-memory-bytes $(awk "BEGIN{printf \"%d\", $KVG*1073741824}") --kv-cache-dtype fp8 \
  --engram-config "$ENGRAM" --reasoning-parser qwen3 \
  --chat-template $O/chat-template-qwen3.8-unsloth.jinja --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"full_decode_only"}' --no-enable-flashinfer-autotune \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}" $EXTRA > /dev/null 2>&1
( while systemctl is-active --quiet $UNIT; do
    h=$(awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo)
    awk "BEGIN{exit !($h < 1.5)}" && { echo "GUARDIA RAM $h GiB: fermo $UNIT"; sudo systemctl stop $UNIT; }
    sleep 3; done ) &
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; systemctl is-active --quiet $UNIT || break; sleep 5; done
if ! curl -sf http://127.0.0.1:$PORT/health > /dev/null; then
  echo "vLLM $TAG non partito"; sudo journalctl -u $UNIT --since "$T0" --no-pager -o cat | grep -iE "error|memory" | grep -v "frame #" | tail -3 | cut -c1-250
  sudo systemctl stop $UNIT 2>/dev/null; sleep 8; exit 1
fi
sudo journalctl -u $UNIT --since "$T0" --no-pager -o cat | grep -E "KV cache size|Model loading took" | tail -2 | cut -c1-200
echo "-- 1 utente, richieste brevi (3 volte)"
python3 - $PORT <<'PY'
import json, sys, time, urllib.request
for i in range(3):
    body = {"model": "x", "messages": [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome."}],
            "max_tokens": 400, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "seed": i, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); j = json.loads(urllib.request.urlopen(req, timeout=600).read())
    print(f"   {j['usage']['completion_tokens'] / (time.time() - t0):.1f} tok/s (incluso primo token)")
PY
echo "-- 1 utente, 240k"; python3 $O/speed_replay_oai.py $PORT $C/c240k.txt
echo "-- 4 utenti, 32k + 61k + 87k + 120k"; python3 $O/speed_replay_oai.py $PORT $C/c30k.txt $C/c60k.txt $C/c90k.txt $C/c115k.txt
echo "-- 4 utenti pesanti, 61k + 87k + 120k + 245k"; python3 $O/speed_replay_oai.py $PORT $C/c60k.txt $C/c90k.txt $C/c115k.txt $C/c240k.txt
if [ -n "$STRESS" ]; then echo "-- stress 4 utenti 195-228k (832k richiesti)"; python3 $O/speed_replay_oai.py $PORT $C/c196k_a.txt $C/c196k_b.txt $C/c196k_c.txt $C/c196k_d.txt; fi
if [ -n "$IMG" ]; then echo "-- richiesta con immagine (vision)"; $R/vllm-venv/bin/python - $PORT <<'PY'
import base64, io, json, sys, time, urllib.request
from PIL import Image, ImageDraw
im = Image.new("RGB", (1280, 720), "white"); d = ImageDraw.Draw(im)
for i in range(12): d.text((40, 40 + 55 * i), f"line {i}: def f{i}(x): return x * {i} + {i * 3}", fill="black")
buf = io.BytesIO(); im.save(buf, "PNG"); url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
for k in range(2):
    body = {"model": "x", "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}, "stream": True,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": f"Trascrivi la riga {k + 3} del codice nell'immagine."}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; txt = ""
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if line.startswith("data:") and "[DONE]" not in line:
            j = json.loads(line[5:]); c = j["choices"][0]["delta"] if j.get("choices") else {}
            if c.get("content") or c.get("reasoning"): first = first or time.time(); txt += c.get("content") or ""
    print(f"   immagine {k + 1}: primo token {(first or time.time()) - t0:.2f} s, risposta: {txt.strip()[:80]!r}")
PY
fi
echo "-- acceptance MTP (media delle righe di questo avvio): $(sudo journalctl -u $UNIT --since "$T0" --no-pager -o cat | grep -oE 'Mean acceptance length: [0-9.]+' | awk '{s+=$4;n++} END {if(n) printf "%.2f su %d righe", s/n, n}')"
echo "-- server vivo? $(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:$PORT/health); righe OOM: $(sudo journalctl -u $UNIT --since "$T0" --no-pager -o cat | grep -ciE 'out of memory')"
sudo systemctl stop $UNIT; sleep 10
echo "[$(date -u +%T)] FINE $TAG"
