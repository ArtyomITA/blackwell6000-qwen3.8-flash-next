# Patch della fase 2 (FORGIA), tutte spente di default

Worktree sulla macchina (niente push):
- `/mnt/llmunity-models/vllm-fase2` (branch `fase2`): C.2, F, C1 + PLE check, top-k deterministico, finalize, B.3 di BANCO.
- `/mnt/llmunity-models/vllm-fase2g` (branch `fase2g`): tutto quello di `fase2` + G (#59366 Python con 3 modifiche).
  Si usa con `PYTHONPATH=<worktree>`; i `.so` sono link all'albero principale.
- Estensioni separate (nessun rebuild di vLLM): `/mnt/llmunity-models/fase2-build/g/build_120/gdn_replay_C_120.so`,
  `/mnt/llmunity-models/fase2-build/topk/build/det_topk_C.so`.

| patch | file diff | attivazione | classe | gate |
|---|---|---|---|---|
| C.2 embed_tokens in RAM (UVA) | C2_embed_uva.diff | `--cpu-offload-gb 3 --cpu-offload-params visual embed_tokens` (spazio, non virgola) | ESATTO | op bit a bit 10^6 id (emb_harness) + on/off dentro il rumore off/off (BANCO): PASSATO |
| F draft su ID 0..N-1 (F98) | D3_F_draft_vocab.diff | env `VLLM_QWEN4EXP_DRAFT_VOCAB=98304` (opzionale, non adottata: `VLLM_QWEN4EXP_DRAFT_TAIL=248044:248077`) | esatto in distribuzione | greedy bit a bit contro K1 senza F; 2 seed: 4 utenti +0,1%, 1 utente +3,6/+8%; acceptance LCB invariata (2,22 contro 2,21 di ieri). 65.536 scartato (``` = ID 71.093). ADOTTATO in K1/K2/K3 |
| C1 PLE via UVA dagli shard tmpfs | D2_C1_ple_file_uva_plus_PLECHECK.diff | `--engram-config {"host_file_gather":true}` + env `VLLM_QWEN4EXP_PLE_FILE_UVA=1` (fix: placeholder pinned con device="cpu") | ESATTO | op 10^6 righe == preadv (c1_harness) PASSATO; server bit a bit contro baseref (run_c1.log) PASSATO, gap step misti 30,6 -> 1,6 ms; `ple_cmp.sh` dopo ogni prova |
| PLE check (debug) | stesso diff di C1 | env `VLLM_PLE_CHECK_ROWS=1` | solo debug | det3: 0 mismatch |
| top-k deterministico indexer QSA (overlay) | Q_det_topk_overlay.diff | env `VLLM_QSA_DET_TOPK_PY=1` | correzione di qualita' (base non deterministica oggi) | det2: bit a bit tra avvii; prefill 200k +27% |
| top-k deterministico (backport #55122) | Q_det_topk_55122_csrc.diff + Q_det_topk_overlay.diff | env `VLLM_QSA_DET_TOPK_LIB=/mnt/llmunity-models/fase2-build/topk/build/det_topk_C.so` | come l'overlay | indici == overlay 16/16 casi, 13,6x piu' veloce (topk_chain.log) |
| finalize MoE non fuso | X_fi_unfused_finalize.diff | env `VLLM_FI_MOE_UNFUSED_FINALIZE=1` | non serve (finalize gia' separato, nsys) | - |
| G replay stato GDN | G_cuda_59366_gdn_kernel.diff (csrc) + worktree fase2g | `--use-replayssm` + env `VLLM_GDN_REPLAY_LIB=/mnt/llmunity-models/fase2-build/g/build_120/gdn_replay_C_120.so`, `PYTHONPATH=/mnt/llmunity-models/vllm-fase2g` | ESATTO nel replay; pagina 3392 (chunk) = M' | harness 1000 step bit a bit (g_run.log); server: KV +11,5%; bit a bit contro base+topk+`--block-size 3392` a 1 e 4 utenti (run_gref.log) PASSATO |
| fence nel parser tool Qwen3 (backport #57553 + chiusura su </think>) | T_qwen3_fence_fix_57553.diff | env `VLLM_QWEN3_FENCE_FIX=1`, worktree `/mnt/llmunity-models/vllm-fase2p` (= fase2 + fix; da portare in fase2g dopo i finali) | correzione di qualita' del parser (qwen3_coder e qwen3_xml sono lo stesso parser) | CPU: test PR 50/50, caso ``` aperta nel ragionamento + tool call vera OK, test qwen3 esistenti invariati con env 0/1; server: toolcall_gate su K2 FALLITO (run_fence.log: fence_spiega invariato 1/10, fantasma scritti FUORI dal fence; fence aperta nel testo + tool call vera 10/10 -> 1/10). NON ADOTTATO, env resta spenta |
| niente tool call implicita dal ragionamento | T_qwen3_parser_fase2p.diff (stesso file del fix fence) | env `VLLM_QWEN3_NO_REASONING_TOOL=1`, alberi fase2p/fase2gp | cambia il parser: DA DECIDERE (utente) | CPU (ghost_probe2.py): <tool_call> scritta nel ragionamento non diventa chiamata (oggi si'), chiamata vera dopo </think> invariata; PERDE la chiamata emessa senza chiudere </think> (7 test upstream di quel caso falliscono, atteso). Nessuna prova GPU. NON ADOTTATO |

Modifiche mie a #58863/#59366 per G: record = `decay = exp(g)` gia' calcolato dal kernel (non `g`), commit sequenziale
`h = fma(delta, k, h * decay)` al posto della forma chiusa, op replay presa dall'estensione separata.

Nota (1/10 sera): `G_cuda_59366_gdn_kernel.diff` e' il csrc di #59366 non modificato. La nostra unica modifica CUDA e' in
`kernels/gdn_replay/fused_gdn_decode_kernel.cu`: il record salva `shared_decay[t]` (= exp(g)) invece di `g`.
