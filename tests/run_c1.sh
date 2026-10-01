#!/bin/bash
# Gate server C1 (PLE via UVA sugli shard tmpfs): run_lever con VLLM_QWEN4EXP_PLE_FILE_UVA=1 + top-k deterministico sulla
# base FP8, confronto bit a bit con baseref (stessa base senza C1), poi cmp dei 10 file tmpfs contro NVMe (LEAD #68).
B=/mnt/llmunity-models/fase2-tests/banco; PY=/mnt/llmunity-models/vllm-venv/bin/python
bash $B/run_lever.sh c1 /mnt/llmunity-models/d0xin-mtpfp8-ramview 13 3 "" "-E VLLM_QWEN4EXP_PLE_FILE_UVA=1 -E VLLM_QSA_DET_TOPK_PY=1"
echo "== baseref contro c1 (1 utente, a freddo? no: gate_gen senza salt, prefix cache vuota all'avvio)"; $PY $B/gate_cmp.py $B/gate_baseref.json $B/gate_c1.json
bash /mnt/llmunity-models/fase2-tests/ple_cmp.sh
