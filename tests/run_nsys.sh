#!/bin/bash
# B.5 nsys sulla combo base (FP8, KV 13, chunk 4096, MTP 3): cattura A = prefill freddo di 32k (solo step misti),
# cattura B = 1,5 s di decode puro a 4 utenti (graph a livello di nodo). Poi somme per kernel con nsys stats.
# Lanciare: flock /mnt/llmunity-models/gpu.lock bash run_nsys.sh
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V=$R/d0xin-mtpfp8-ramview; PY=$R/vllm-venv/bin/python
NS=/usr/local/bin/nsys; O=$B/nsys; mkdir -p $O
echo "lock BANCO run_nsys $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
export PREFIX="$NS profile -o $O/base -f true -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat:2"
export EXTRA="--profiler-config {\"profiler\":\"cuda\"}" MEMMAX=16G
bash $B/srv.sh start nsysbase $V 13 4096 3 || exit 1
$PY - <<'PY'
import json, threading, time, urllib.request
from transformers import AutoTokenizer
V = "/mnt/llmunity-models/d0xin-mtpfp8-ramview"; C = "/mnt/llmunity-models/prod-tests/convs"
tok = AutoTokenizer.from_pretrained(V)
ids = {x: tok.encode(open(f"{C}/c196k_{x}.txt", encoding="utf-8").read(), add_special_tokens=False) for x in "abcd"}
def post(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:30002{path}", data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600).read()
def gen(prompt, n):
    return post("/v1/completions", {"model": "qwen", "prompt": prompt, "max_tokens": n, "temperature": 0.0, "ignore_eos": True})
# A: prefill freddo di 32k (prompt mai visto: coda di c196k_a).
post("/start_profile"); t = time.time(); gen(ids["a"][-32000:], 1); print(f"A prefill 32k: {time.time() - t:.2f} s"); post("/stop_profile")
# B: 4 prompt da 16k gia' in cache, decode lungo; profilo 1,5 s a regime.
for x in "abcd": gen(ids[x][:16000], 1)
th = [threading.Thread(target=gen, args=(ids[x][:16000], 600)) for x in "abcd"]; [t.start() for t in th]
time.sleep(4); post("/start_profile"); time.sleep(1.5); post("/stop_profile"); [t.join() for t in th]
print("CATTURE_FATTE")
PY
bash $B/srv.sh stop nsysbase; sleep 20
ls -la $O
for f in $O/base*.nsys-rep; do
  echo "== $f"; $NS stats -q --report cuda_gpu_kern_sum --format csv -o ${f%.nsys-rep} $f > /dev/null 2>&1
  head -25 ${f%.nsys-rep}_cuda_gpu_kern_sum.csv | cut -c1-220
done
echo "[$(date -u +%T)] RUN_NSYS_FINE"
