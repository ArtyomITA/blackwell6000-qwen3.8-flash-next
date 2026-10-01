#!/bin/bash
# Gate del fix #57553 sul vincitore proposto K3, albero di produzione vllm-fase2gp: per VLLM_QWEN3_FENCE_FIX=0 e =1
# toolcall_gate con il caso di rischio + gate_gen greedy salato (il fix tocca solo il parser: i token devono essere
# identici bit a bit), poi ple_cmp. Lanciare: flock /mnt/llmunity-models/gpu.lock bash run_fence_k3.sh
B=/mnt/llmunity-models/fase2-tests/banco; R=/mnt/llmunity-models; V1=$R/d0xin-fp8ple-ramview; PY=$R/vllm-venv/bin/python
X="--cpu-offload-gb 3 --cpu-offload-params visual embed_tokens --use-replayssm"
T="-E PYTHONPATH=$R/vllm-fase2gp -E VLLM_GDN_REPLAY_LIB=$R/fase2-build/g/build_120/gdn_replay_C_120.so -E VLLM_QSA_DET_TOPK_LIB=$R/fase2-build/topk/build/det_topk_C.so -E VLLM_QWEN4EXP_PLE_FILE_UVA=1 -E VLLM_QWEN4EXP_DRAFT_VOCAB=98304"
echo "lock BANCO run_fence_k3 $(date -u +%T)"
for fix in 0 1; do
  TREE=main EXTRA="$X" ENVS="$T -E VLLM_QWEN3_FENCE_FIX=$fix" bash $B/srv.sh start k3fence$fix $V1 13.5 4096 3 || exit 1
  TOOLCALL_RISK=1 python3 $B/toolcall_gate.py 30002 K3-fence$fix 10
  GATE_SALT=k3f$fix $PY $B/gate_gen.py 30002 $V1 $B/gate_k3fence$fix.json 1 256
  bash $B/srv.sh stop k3fence$fix
done
echo "== greedy K3 senza fix contro con fix"; $PY $B/gate_cmp.py $B/gate_k3fence0.json $B/gate_k3fence1.json
bash $R/fase2-tests/ple_cmp.sh
echo "[$(date -u +%T)] RUN_FENCE_K3_FINE"
