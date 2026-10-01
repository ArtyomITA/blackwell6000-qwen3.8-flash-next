#!/bin/bash
# Gate del fix #57553 (VLLM_QWEN3_FENCE_FIX, worktree vllm-fase2p) su K2: toolcall_gate con il caso di rischio, fix spento
# e acceso (stesso albero), poi ple_cmp. Lanciare: flock /mnt/llmunity-models/gpu.lock bash run_fence.sh
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V2=$R/d0xin-mtpfp8-ramview
X="--cpu-offload-gb 3 --cpu-offload-params visual embed_tokens"
T="-E PYTHONPATH=$R/vllm-fase2p -E VLLM_QSA_DET_TOPK_LIB=$R/fase2-build/topk/build/det_topk_C.so -E VLLM_QWEN4EXP_PLE_FILE_UVA=1 -E VLLM_QWEN4EXP_DRAFT_VOCAB=98304"
echo "lock BANCO run_fence $(date -u +%T)"
for fix in 0 1; do
  TREE=main EXTRA="$X" ENVS="$T -E VLLM_QWEN3_FENCE_FIX=$fix" bash $B/srv.sh start fence$fix $V2 15 4096 3 || exit 1
  TOOLCALL_RISK=1 python3 $B/toolcall_gate.py 30002 K2-fence$fix 10
  bash $B/srv.sh stop fence$fix
done
bash $R/fase2-tests/ple_cmp.sh
echo "[$(date -u +%T)] RUN_FENCE_FINE"
