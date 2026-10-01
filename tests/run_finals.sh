#!/bin/bash
# Test finali dei 3 candidati: 50 tool call (qwen3_coder) + 6 difficili x3 a 100k (temp 1,0, top_p 0,95, top_k 20 come
# ieri) + conteggio corruzione, poi ple_cmp. K2 anche con --tool-call-parser qwen3_xml (solo tool call, 1 avvio in piu').
# FENV = env di F (vuota se F98 non entra). Lanciare: flock /mnt/llmunity-models/gpu.lock bash run_finals.sh
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; PY=$R/vllm-venv/bin/python
V1=$R/d0xin-fp8ple-ramview; V2=$R/d0xin-mtpfp8-ramview
X="--cpu-offload-gb 3 --cpu-offload-params visual embed_tokens"
T="-E VLLM_QSA_DET_TOPK_LIB=$R/fase2-build/topk/build/det_topk_C.so -E VLLM_QWEN4EXP_PLE_FILE_UVA=1 $FENV"
echo "lock BANCO run_finals $(date -u +%T) FENV=$FENV"
TOOLS=1 EXTRA="$X" ENVS="$T" bash $B/lcb_cand.sh K1 $V1 13.25 4096 3; bash $R/fase2-tests/ple_cmp.sh
TOOLS=1 EXTRA="$X" ENVS="$T" bash $B/lcb_cand.sh K2 $V2 15 4096 3; bash $R/fase2-tests/ple_cmp.sh
TOOLS=1 TREE=main EXTRA="$X --use-replayssm" ENVS="-E PYTHONPATH=$R/vllm-fase2g -E VLLM_GDN_REPLAY_LIB=$R/fase2-build/g/build_120/gdn_replay_C_120.so $T" \
  bash $B/lcb_cand.sh K3 $V1 13.5 4096 3; bash $R/fase2-tests/ple_cmp.sh
EXTRA="$X --tool-call-parser qwen3_xml" ENVS="$T" bash $B/srv.sh start xmlK2 $V2 15 4096 3 && { python3 $B/toolcall_gate.py 30002 K2-qwen3_xml 10; bash $B/srv.sh stop xmlK2; }
bash $R/fase2-tests/ple_cmp.sh
echo "[$(date -u +%T)] RUN_FINALS_FINE"
