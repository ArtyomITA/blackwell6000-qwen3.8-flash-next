#!/bin/bash
# Un avvio K3 (albero vllm-fase2gp, fix spento) per leggere il testo grezzo di fence_spiega (probe_fence.py).
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V1=$R/d0xin-fp8ple-ramview; PY=$R/vllm-venv/bin/python
X="--cpu-offload-gb 3 --cpu-offload-params visual embed_tokens --use-replayssm"
T="-E PYTHONPATH=$R/vllm-fase2gp -E VLLM_GDN_REPLAY_LIB=$R/fase2-build/g/build_120/gdn_replay_C_120.so -E VLLM_QSA_DET_TOPK_LIB=$R/fase2-build/topk/build/det_topk_C.so -E VLLM_QWEN4EXP_PLE_FILE_UVA=1 -E VLLM_QWEN4EXP_DRAFT_VOCAB=98304"
echo "lock BANCO run_probe_fence $(date -u +%T)"
TREE=main EXTRA="$X" ENVS="$T" bash $B/srv.sh start probefence $V1 13.5 4096 3 || exit 1
$PY $B/probe_fence.py 30002 $V1 $B/probe_fence.jsonl
bash $B/srv.sh stop probefence
bash $R/fase2-tests/ple_cmp.sh
echo "[$(date -u +%T)] RUN_PROBE_FENCE_FINE"
