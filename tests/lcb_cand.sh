#!/bin/bash
# Test finale di un candidato: i 6 problemi difficili x N a budget dato, come lcb_rerun.sh ma con il server di srv.sh
# (worktree fase2, EXTRA/ENVS del candidato, num_spec e chunk scelti) e la PLE tmpfs lasciata montata.
# Uso: lcb_cand.sh <tag> <vista> <kv_gb> <chunk> <num_spec> [ids] [reps=3] [max_tokens=100000]
# Variabili come srv.sh (EXTRA, ENVS); TOOLS=1 aggiunge toolcall_gate.py (50 chiamate) nello stesso avvio. Sotto flock /mnt/llmunity-models/gpu.lock.
B=/mnt/llmunity-models/fase2-tests/banco; D=/var/tmp/llmunity-lcb32
TAG=$1; V=$2; KV=$3; NBT=$4; NS=$5; ONLY=${6:-arc191_a,abc391_d,abc391_e,abc399_e,abc394_g,abc393_e}; REPS=${7:-3}; MAXTOK=${8:-100000}
echo "lock BANCO lcb_cand $TAG $(date -u +%T)"
mountpoint -q /mnt/llmunity-ple-ram || { echo "PLE tmpfs non montata"; exit 1; }
grep -q LCB_ONLY $D/lcb_eval.py || { echo "lcb_eval.py senza LCB_ONLY: lanciare prima lcb_rerun.sh una volta"; exit 1; }
bash $B/srv.sh start lcb-$TAG $V $KV $NBT $NS || exit 1
systemctl show llmunity-vllm-lcb-$TAG -p Environment -p ExecStart --no-pager | tr -s " " | cut -c1-3000   # comando completo per FINALE.md
[ -n "$TOOLS" ] && python3 $B/toolcall_gate.py 30002 $TAG 10   # gate tool call (>=50) nello stesso avvio
LCB_ONLY=$ONLY LCB_REPS=$REPS LCB_DATA=$D/test6.jsonl python3 $D/lcb_eval.py gen 30002 $D/lcb_vllm-$TAG.jsonl $MAXTOK 4 2>&1
echo "acceptance: $(sudo journalctl -u llmunity-vllm-lcb-$TAG --since '-12h' --no-pager -o cat | grep -oE 'Mean acceptance length: [0-9.]+' | awk '{s+=$4;n++} END {if(n) printf "%.2f su %d righe", s/n, n}'); OOM: $(sudo journalctl -u llmunity-vllm-lcb-$TAG --since '-12h' --no-pager -o cat | grep -ciE 'out of memory')"
bash $B/srv.sh stop lcb-$TAG
python3 - $D/lcb_vllm-$TAG.jsonl <<'PY'
import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
bad = sum(1 for r in rows if re.search(r"!{20,}", json.dumps(r)))
print(f"corruzione: {bad}/{len(rows)} generazioni con 20+ '!' di fila (token 0)")
PY
chmod 644 $D/lcb_vllm-$TAG.jsonl
sudo systemd-run --wait --pipe --collect --unit=llmunity-lcb32-eval-vllm-$TAG \
  -p DynamicUser=yes -p PrivateNetwork=yes -p ProtectSystem=strict -p ProtectHome=yes -p NoNewPrivileges=yes \
  -p RestrictAddressFamilies=AF_UNIX -p PrivateDevices=yes -p PrivateTmp=yes -p TemporaryFileSystem=/tmp:rw,size=512M \
  -p "BindReadOnlyPaths=$D:/opt/lcb32 /home/ec2-user/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu:/opt/llmunity-python" \
  -p InaccessiblePaths=/mnt/llmunity-models -p MemoryMax=4G -p MemorySwapMax=0 -p TasksMax=64 -p CPUQuota=200% -p RuntimeMaxSec=3600 \
  /usr/bin/env LCB_DATA=/opt/lcb32/test6.jsonl /opt/llmunity-python/bin/python3.12 -s /opt/lcb32/lcb_eval.py eval /opt/lcb32/lcb_vllm-$TAG.jsonl \
  > $D/eval_vllm-$TAG.log 2>&1
grep -v ESITI_JSON $D/eval_vllm-$TAG.log | grep -vE "^Running|^Finished|^Main|^Service|^CPU"
echo "[$(date -u +%T)] LCB_CAND_FINE $TAG"
