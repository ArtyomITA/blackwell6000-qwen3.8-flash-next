# FINALE — 3 candidati per Swift 1.5 Qwen3.8 Flash-Next NVFP4 sulla Blackwell (RTX PRO 6000, SM120)

Stato: COMPLETO (scheletro RICERCA, numeri BANCO, patch FORGIA, 1/10/2026). Numeri dai log (dettaglio in CANDIDATI.md); i
comandi completi vengono dalla riga "non-default args" del journal di ogni corsa. Fonti web: RICERCA.md. Patch: patches/INDICE.md.

## 0. Obiettivo e base da battere
- Obiettivo (DOSSIER): 4 utenti con 160-210k di contesto ciascuno, velocita' >= oggi, zero perdita di qualita', MTP BF16
  preferito (FP8 se BF16 rallenta troppo); i 3 candidati rifanno i 6 problemi difficili x3 a budget 100k.
- Base (combo di oggi): Swift 1.5 NVFP4 (d0xin) + esperti MTP FP8, KV FP8 13 GB = 783.886 token (4 x 196k), chunk 4096
  (chunk reale 3.200, vedi §3), vision in GPU, PLE FP8 tmpfs host_file_gather, MTP 3, --no-enable-flashinfer-autotune.
  1 utente breve 156-162 tok/s; 245k TTFT 21,0 s, 169 tok/s; 4 utenti media ~93; 4 pesanti 84-94; 6 difficili x3 a
  100k: 16/18 (43,8k token medi). MTP BF16 di ieri: 17/18 (37,7k).

## 1. Correzione di qualita' trovata stanotte (entra in TUTTI i candidati)
- **Difetto della produzione di oggi:** il top-k dell'indexer QSA (`torch.ops._C.persistent_topk`, unico percorso su
  SM120, fork ops/qsa_indexer.py r.471-498, usato da decode e prefill) sceglie insiemi e ordini di blocchi diversi a
  input identici (posizioni via atomicAdd). Effetto misurato (det2, BANCO #139): stesso avvio, stesso prompt, prefill a
  freddo: logprob del token 0 diversi fino a 2,38 nat a 32k e 5,54 nat a 200k; a 2k (un chunk) stabile.
- Fonti: vLLM PR #55122 (aperta) https://github.com/vllm-project/vllm/pull/55122 ; caso documentato su Flash-Next
  (temp 0 instabile su 13/50 task, punteggio 97 -> 98 dopo la correzione)
  https://docai.hu/en/blog/qwen38-flash-next-nondeterministic-vllm-kernel
- Correzione adottata: backport #55122 come estensione separata, `VLLM_QSA_DET_TOPK_LIB=/mnt/llmunity-models/fase2-build/topk/build/det_topk_C.so`
  (contratto: valore decrescente, pari merito indice piu' basso, indici crescenti). Bit a bit con l'overlay Python
  (16/16 casi, gate_gen e gate_gen4 KLD 0); base riproducibile bit a bit tra avvii; costo prefill 200k +0,5% (16,47 s
  contro 16,39). Decode con LIB: K1 periodo 4 utenti 28,35 ms (k1ref) contro 27,86 senza LIB (+0,5 ms, dentro il rumore tra avvii); prefill invariato.
- Esclusa come causa la PLE host_file_gather (det3: 0 righe sbagliate su tutti gli step).

## 2. Candidati

Formato per ognuno (LEAD #6): configurazione + comando + log; KV ottenuta; velocita' 1/4 utenti, 4 pesanti, TTFT,
acceptance, ms/step; 6 difficili x3 a 100k (token medi/mediana/min-max/al tetto, risolti per difficolta'); tool call;
classe di accuratezza e gate; perche' e cosa resta incerto.

### K1 — MTP BF16 + vision UVA + embed UVA + F98 + top-k deterministico + C1
- Patch (tutte spente di default, attivate da flag/env): C.2 embed UVA (ESATTO, gate PASSATO: op bit a bit + dentro il
  rumore), vision UVA (offloader del fork), top-k LIB (correzione di qualita', bit a bit tra avvii), C1 PLE via UVA
  (ESATTO, bit a bit contro la base), F98 testa del draft ai primi 98.304 ID (esatto in distribuzione; greedy bit a bit
  contro K1 senza F, gate salato; velocita' su due seed), MTP 3.
- Comando completo (journal della corsa finale, run_finals.log):
  ```
  Environment: PYTHONPATH=/mnt/llmunity-models/vllm-fase2
    VLLM_QSA_DET_TOPK_LIB=/mnt/llmunity-models/fase2-build/topk/build/det_topk_C.so
    VLLM_QWEN4EXP_PLE_FILE_UVA=1 VLLM_QWEN4EXP_DRAFT_VOCAB=98304
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_HOME=/usr/local/cuda   (unit: LimitMEMLOCK=infinity, MemoryMax=12G)
  vllm serve /mnt/llmunity-models/d0xin-fp8ple-ramview --served-model-name qwen x --load-format safetensors
    --tensor-parallel-size 1 --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 4096
    --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 14227079168 --kv-cache-dtype fp8
    --engram-config {"host_file_gather":true} --reasoning-parser qwen3
    --chat-template /mnt/llmunity-models/prod-tests/chat-template-qwen3.8-unsloth.jinja --enable-auto-tool-choice
    --tool-call-parser qwen3_coder --compilation-config {"cudagraph_mode":"full_decode_only"}
    --no-enable-flashinfer-autotune --speculative-config {"method":"mtp","num_speculative_tokens":3}
    --cpu-offload-gb 3 --cpu-offload-params visual embed_tokens
  ```
  Controlli al boot: "Total CPU offloaded parameters: 2.02", loading 76,94 GiB, "GDN decode kernel: cuda",
  'FLASHINFER_CUTLASS' NvFp4 MoE backend.
- KV: 799.157 token = 4 x 199,8k (base 783.886 = 4 x 196k). Stress 4 x 195-228k senza OOM (run_kv_k1.log).
- Velocita' (sweep run_lever a temp 1,0, client 4 utenti medio a 32/128/180k; 1 utente; metodo diverso dal vspeed della
  base: il confronto "velocita' >= oggi" lo chiude il vspeed finale):
  - K1 + LIB + C1 + F98: 4 utenti 97,7 (seed 0) e 92,1 (seed 1); 1 utente 159,7 e 146,3 tok/s; decode 4 utenti
    26,14-26,23 ms/step (target 21,5, draft 3,3, gap 1,3); step misti 243 ms (gap 3,5 con C1); acceptance 2,68-2,69.
  - prima misura vspeed (senza C1/F98): 245k TTFT 21,5 s; 4 utenti t2 media 86,9; 4 pesanti 88,2. vspeed della
    combinazione finale (run_vs_K1fin.log, stesso vspeed della base, 08:52-09:01): brevi 171,0-186,2 tok/s; 245k TTFT
    19,5 s, 172,2 tok/s; 4 utenti t2 108,1/82,4/99,7/85,8 = **94,0** (base ~93); pesanti 78,0/75,6/92,7/101,4 = 86,9
    (base 84-94); acceptance 2,79; decode 1/2/3/4 utenti 17,01/20,07/23,49/25,68 ms; step misti 241-259 ms; stress senza
    OOM (832k > 799k: richieste in coda); immagine 3,96 s / 0,11 s.
- 6 difficili x3 a 100k (temp 1,0, top_p 0,95, top_k 20, 4 in parallelo): **15/18 (83,3%)**: medium 6/6, hard 9/12;
  sbagliati abc399_e 2/3, abc391_e 1/3; token medi 35.043, mediana 37.000, min 15.174, max 58.814, al tetto 0;
  34 min. Acceptance 2,22 (come ieri: fail100k-bf16 2,21, fail100k-fp8 2,19). Corruzione ("!" x20) 0/18. ple_cmp ok.
- Tool call (qwen3_coder, 50): 32/50 (fence_spiega 0/10 = #57541 del parser, vedi §4; edits_annidati 3/10 = il modello
  chiama prima run_command: nota, non difetto).
- Perche': BF16 tiene il draft migliore di ieri e la UVA libera 2,03 GiB che tornano KV; F98 e C1 recuperano il passo.
  Incerto: 15/18 contro 17/18 di ieri con lo stesso MTP BF16 = dentro il rumore di 18 tentativi a temperatura 1,0.
- Produzione: albero /mnt/llmunity-models/vllm-fase2p (= fase2 + fix fence #57553) con VLLM_QWEN3_FENCE_FIX=1 se il
  gate fence passa; con la env spenta e' identico all'albero del test (FORGIA #262).

### K2 — MTP FP8 + vision UVA + embed UVA + F98 + top-k deterministico + C1
- Patch: come K1 con la vista d0xin-mtpfp8-ramview (esperti MTP FP8 della base di oggi).
- Comando completo (journal della corsa finale):
  ```
  Environment: come K1 (PYTHONPATH=/mnt/llmunity-models/vllm-fase2, VLLM_QSA_DET_TOPK_LIB=..., VLLM_QWEN4EXP_PLE_FILE_UVA=1,
    VLLM_QWEN4EXP_DRAFT_VOCAB=98304; unit LimitMEMLOCK=infinity)
  vllm serve /mnt/llmunity-models/d0xin-mtpfp8-ramview ... (stessi argomenti di K1) --kv-cache-memory-bytes 16106127360
    --speculative-config {"method":"mtp","num_speculative_tokens":3} --cpu-offload-gb 3 --cpu-offload-params visual embed_tokens
  ```
  Controlli al boot: "Total CPU offloaded parameters: 2.02", loading 74,62 GiB.
- KV: 903.506 token = 4 x 225,9k a 15 GB (+119.620 sulla base). Stress 4 x 195-228k senza OOM (run_kv_k2.log). 15 GB era il
  primo valore provato: probabilmente si sale ancora.
- Velocita' (run_lever, temp 1,0, client medio a 32/128/180k, LIB + C1, senza F98): 4 utenti 97,9; 1 utente 143,8;
  decode 4 utenti 27,27 ms/step; step misti 242 ms; acceptance 2,90. Con F98 e vspeed della combinazione finale:
  vspeed della combinazione finale (run_vs_K2fin.log, 09:01-09:10): brevi 172,6-178,8 tok/s; 245k TTFT 19,4 s, 168,0
  tok/s (t2 205,1); 4 utenti t2 76,9/117,0/78,0/98,2 = **92,5**; pesanti 76,2/119,0/96,0/94,2 = 96,4; acceptance 2,63;
  decode 1/2/3/4 utenti 16,93/20,20/23,46/25,55 ms; step misti 243-259 ms; stress senza OOM (in coda); immagine 3,91 s / 0,11 s.
- 6 difficili x3 a 100k: **16/18 (88,9%)**: medium 5/6, hard 11/12; falliti arc191_a 1/3 al tetto (100.000, "nessun
  codice") e abc391_e 1/3 sbagliato; token medi 37.281, mediana 36.576, min 14.218, max 100.000, al tetto 1; 37 min.
  Acceptance 2,20; corruzione 0/18; OOM 0; ple_cmp ok.
- Tool call (qwen3_coder, 50): 33/50 (edits_annidati 4/10, anyof_e_mappa 10/10, anyof_null 8/10, fence_spiega 1/10 con
  8 tool call fantasma, fence_poi_agisci 10/10).
- Perche': stessi pesi e stesso draft FP8 della base, piu' contesto (UVA) e passo piu' corto (C1, F98). Esito identico alla
  base FP8 di ieri (16/18: 1 al tetto + 1 sbagliato) con il 15% di token medi in meno. Incerto: 18 tentativi.

### K3 — K1 + G (replay stato GDN esatto)
- Patch: K1 + G (`--use-replayssm`, `VLLM_GDN_REPLAY_LIB=/mnt/llmunity-models/fase2-build/g/build_120/gdn_replay_C_120.so`,
  `PYTHONPATH=/mnt/llmunity-models/vllm-fase2g`). Harness 1000 step bit a bit (g_run.log). Sulla base FP8: +89.927 token
  (+11,5%) a 13 GB. Replay ESATTO: run_gref (BANCO #179) G == base + top-k + `--block-size 3392` bit a bit a 1 utente
  (8 prompt x 128 token) e 4 utenti (lista di 4). Unica differenza dalla base: pagina/chunk 3392 (classe M', DIVERSO
  leggero: gate prima divergenza + KLD + 6 difficili x3).
  G = vLLM #58863 + #59366 con 3 modifiche FORGIA, senza le quali non e' esatto: record con decay = exp(g) gia'
  calcolato dal kernel (non g); commit sequenziale h = fma(delta, k, h * decay) al posto della forma chiusa upstream
  (differisce di 1 ulp, provato in g_harness_fla.log: 0/1000); kernel replay CUDA da estensione separata gdn_replay_C
  (build fedele: native == .so di oggi 1000/1000). Step puri = fuso CUDA, step misti = verify Triton (== FLA di oggi).
- Prima misura (BANCO #207, run_kv_k3.log, vspeed_k3-kv13p5.log, steps_k3-kv13p5.csv; LIB + G, C1 non ancora, temp 1,0):
  KV 13,5 GB = 909.971 token = 4 x 227,5k con MTP BF16 (base 783.886), loading 76,94 GiB, stress senza OOM e ora tutto
  dentro la KV (turno 2 TTFT 1,1-2,0 s contro 13-37 s in coda); 1 utente brevi 155,4-168,3, 245k TTFT 21,4 s,
  138,8 (t1) / 135,4 (t2) tok/s; 4 utenti t2 media 84,1; 4 pesanti media 94,6; acceptance 2,84; decode 1/2/3/4 utenti
  18,92/21,95/25,26/27,61 ms; step misti (chunk 3392) ~285-294 ms. Incerto: una corsa (4 utenti 84 contro 87 di K1).
- Comando completo (journal della corsa finale):
  ```
  Environment: PYTHONPATH=/mnt/llmunity-models/vllm-fase2g
    VLLM_GDN_REPLAY_LIB=/mnt/llmunity-models/fase2-build/g/build_120/gdn_replay_C_120.so
    VLLM_QSA_DET_TOPK_LIB=/mnt/llmunity-models/fase2-build/topk/build/det_topk_C.so
    VLLM_QWEN4EXP_PLE_FILE_UVA=1 VLLM_QWEN4EXP_DRAFT_VOCAB=98304   (unit: LimitMEMLOCK=infinity, MemoryMax=12G)
  vllm serve /mnt/llmunity-models/d0xin-fp8ple-ramview ... (stessi argomenti di K1) --kv-cache-memory-bytes 14495514624
    --speculative-config {"method":"mtp","num_speculative_tokens":3}
    --cpu-offload-gb 3 --cpu-offload-params visual embed_tokens --use-replayssm
  ```
  Controlli al boot: "GDN RecoverSSM speculative verify active (spec_query_len 4)", "GDN decode kernel: cuda", "Total CPU
  offloaded parameters: 2.02", loading 76,94 GiB, pagina di attenzione 3392.
- KV finale: 909.971 token = 4 x 227,5k (stress 832k tutto dentro la KV, niente coda).
- Velocita' della combinazione finale (run_vs_K3fin.log, 09:10-09:18): brevi 171,7-186,5 tok/s; 245k TTFT 19,2 s, 151,6
  tok/s; 4 utenti t2 85,4/98,2/99,9/116,6 = **100,0** (base ~93, +7,5%); pesanti 87,1/103,1/85,8/88,5 = 91,1; acceptance
  2,88; decode 1/2/3/4 utenti 17,04/19,87/23,04/25,45 ms; step misti (chunk 3392) 257-262 ms; stress 832k tutto dentro la
  KV: turno 2 TTFT 0,9-1,5 s e 80-121 tok/s (K1/K2 lo mettono in coda, 17-47 s); immagine 3,99 s / 0,11 s.
  Incerto: 1 utente a 245k sotto la base in entrambe le misure (151,6 e 138,8 tok/s contro 169). Il passo di decode a 1
  utente pero' e' identico a K1 (17,04 contro 17,01 ms, steps_K3fin/K1fin.csv): la differenza e' l'acceptance di un solo
  campione da 512 token a temperatura 1,0, non un costo di G (RICERCA #281).
- 6 difficili x3 a 100k: **17/18 (94,4%)**: medium 6/6, hard 11/12 (abc399_e 1/3 sbagliato); token medi 37.387,
  mediana 39.468, min 12.279, max 55.919, al tetto 0; 41 min. Acceptance 2,21; corruzione 0/18; OOM 0; ple_cmp ok.
  Ieri MTP BF16: 17/18 (37,7k medi): identico.
- Tool call (qwen3_coder, 50): 32/50 (stesso schema di K1: fence_spiega 0/10 per #57541).
- Perche': il replay e' esatto (gate bit a bit a pagina 3392), il chunk 3392 e' l'unica differenza dalla base (M', gate
  leggero dentro il rumore 1/4 utenti) e i 6 difficili non mostrano perdita. Produzione: albero vllm-fase2gp (= fase2g + fix
  fence) con VLLM_QWEN3_FENCE_FIX=1 se il gate fence passa (va rifatto su fase2gp, LEAD #246).

## 3. Leve misurate e scartate (con il perche')
- Chunk reale = 3.200, non 4096: in mamba_cache_mode "align" il prefill si spezza a multipli di pagina (scheduler.py
  r.473-475). Valori utili: 6416 / 9616 (pagina 3200), 6800 / 10192 (pagina 3392 con G). Esito (BANCO #220, run_k2c6416.log, K2): chunk 6416 = step misto da 6.400 token in 440 ms,
  prefill +10% (14,5k contro 13,2k tok/s) ma stallo del decode raddoppiato; 4 utenti 93,7/39,0/94,1. SCARTATO: si resta
  a 4096 (chunk reale 3.200).
- C1 (PLE via UVA dagli shard tmpfs, `VLLM_QWEN4EXP_PLE_FILE_UVA=1` con `--engram-config {"host_file_gather":true}`):
  ESATTO, in tutti e tre i candidati (decisione LEAD #180; ple_cmp dopo ogni corsa finale). run_c1 (BANCO #179): bit a bit contro baseref su 8 prompt x 512 token, ple_cmp 10/10 ok; gap host
  degli step misti 30,6 -> 1,6 ms, periodo misto 332 -> 305 ms (-8%); gap decode 1,66 -> ~1,3 ms; 4 utenti 82,5-84,6 ->
  84,7-86,8 tok/s; RAM in piu' 0. Perche': toglie stream.synchronize() + loop Python di os.preadv per riga
  (nvidia/ngram_embedding.py r.150-185). Registrazione r+b flag 0 (ReadOnly non supportato su questa GPU/x86):
  ple_cmp.sh dopo ogni avvio.
- num_spec 2/4 (BANCO #211 su K1, #220 su K2; client 4 utenti medio a 32/128/180k, temp 1,0, con LIB + C1):
  K1 MTP 3 95,0 / MTP 2 94,0 / MTP 4 crollo a 180k; K2 MTP 3 97,9 / MTP 2 96,0 / MTP 4 92,7. MTP 2 accorcia il passo ma
  perde acceptance (2,40-2,42); MTP 4 allunga il draft (~6,3 ms) e in align costa un blocco mamba per richiesta
  (KV K1 764.382 contro 799.157: preemption a 4 x 180k, RICERCA #213). Scelto MTP 3 per tutti.
- Strada 2 (KV in RAM, vLLM PR #59209 HiSparse per QSA): non usata. Vuole prefix caching spento e perde -16,6% a bassa
  concorrenza con 0 miss (https://github.com/vllm-project/vllm/pull/59209); inoltre --use-replayssm e' incompatibile
  con i KV connector (config/vllm.py r.3483). La strada 1 basta per 160-210k.
- Leva B (MoE qf_moe, classe DIVERSO): non necessaria finche' i candidati reggono la velocita'; tetto stimato ~8% del
  passo a 4 utenti (MoE vicino alla banda con ~120 esperti distinti per layer, ipotesi RICERCA #108).
- Autotune FlashInfer MoE: resta spento (flashinfer #4841, corruzione con tattiche gemm1/gemm2 autotunate).
- Finalize MoE non fuso: inutile, il finalize e' gia' separato e deterministico (nsys).

## 4. Rischi e bug noti per la produzione (non risolti qui, decisione utente)
- vLLM issue #47194 (aperta) https://github.com/vllm-project/vllm/issues/47194 : ibridi + prefix caching + MTP 3,
  tool call che trapelano nel testo. Coperto dal test tool call dei candidati; nessun fix upstream.
- Parser tool: vLLM issue #57541 (fix PR #57553 non mergiata) https://github.com/vllm-project/vllm/issues/57541 :
  "<function=..." dentro blocchi ``` diventa una tool call. RIPRODOTTO su K1 (BANCO #231, qwen3_coder): 32/50;
  fence_spiega 0/10 con tool call fantasma 8/10 e testo inghiottito 7/10; fence_poi_agisci 10/10; anyof_e_mappa 10/10;
  anyof_null 9/10; edits_annidati 3/10 (il modello usa run_command prima di apply_edits: comportamento, non difetto).
  Vale anche per la produzione di oggi. Nel fork qwen3_coder e qwen3_xml sono lo stesso Qwen3EngineToolParser
  (tool_parsers/__init__.py r.185-191): cambiare parser non serve. Correzione: backport di vLLM PR #57553
  (https://github.com/vllm-project/vllm/pull/57553, Python puro, fence CommonMark solo negli stati CONTENT/REASONING),
  approvato (LEAD #240), worktree fase2p, env VLLM_QWEN3_FENCE_FIX=1; gate: 50 tool call su K2 + caso "``` mai chiusa
  nel ragionamento poi tool call vera". Esito (run_fence.log, K2 su vllm-fase2p, 6 casi x 10, 09:19-09:34 UTC):
  **GATE FALLITO, fix NON adottato** (LEAD #297; env VLLM_QWEN3_FENCE_FIX spenta, alberi fase2p/fase2gp = fase2/fase2g).
  FIX=0 43/60, FIX=1 34/60: fence_spiega identico 1/10 (stessi 8 fantasma: con top-k deterministico e seed fisso il
  testo generato e' lo stesso, cambia solo il parser); fence_aperta_poi_tool 10/10 -> 1/10 (``` aperta nel testo e mai
  chiusa: il fix nasconde la tool call vera); gli altri 3 casi invariati. Meccanismo dei fantasma (RICERCA #299, dal
  codice): vllm/parser/qwen3.py r.140-144, transizione implicita REASONING + <tool_call> -> chiamata vera: quando il
  modello, ragionando sul blocco, riscrive <tool_call> dentro il <think>, il parser chiude il ragionamento e apre una
  chiamata. Nessuna regola sulle fence lo tocca. Prova su CPU (FORGIA #304, ghost_probe.py, parser del fork):
  meccanismo CONFERMATO. Rimedio pronto ma NON ADOTTATO (spento): `VLLM_QWEN3_NO_REASONING_TOOL=1` in fase2p/fase2gp
  toglie solo la transizione (REASONING, TOOL_START): fantasma sparito, chiamata vera dopo </think> invariata, ma si perde
  una chiamata emessa senza chiudere </think> (7 test upstream "implicit end" falliscono, atteso). Serve un gate GPU (il
  modello chiama mai senza </think>? conteggio su 50+ tool call reali) prima di accenderla: decisione utente.
  Rischio APERTO per Copilot agent (vale anche per la produzione di oggi).
- Copilot e ragionamento: microsoft/vscode #338819 https://github.com/microsoft/vscode/issues/338819 : Copilot rimanda
  reasoning_content solo se il delta ha cot_id/reasoning_opaque/signature. Il template Unsloth rende il ragionamento di
  tutti i turni (preserve_thinking default true): senza echo la storia agent ha "<think></think>" vuoti. Proxy:
  rinomina reasoning -> reasoning_content + cot_id (lo prepara il LEAD).
- vLLM PR #58784 (merged 28/9, fork del 27/9: assente): serve solo per il draft probabilistico, non usato.

## 5. Raccomandazione

### Tabella finale (stesso vspeed della base, temp 1,0; 6 difficili x3 a 100k con il protocollo di ieri)

| | base di oggi | K1 | K2 | **K3** |
|---|---|---|---|---|
| MTP | FP8 | BF16 | FP8 | BF16 |
| KV (token) | 783.886 (4x196k) | 799.157 (4x199,8k) | 903.506 (4x225,9k) | **909.971 (4x227,5k)** |
| brevi 1 utente tok/s | 156-162 | 171,0-186,2 | 172,6-178,8 | 171,7-186,5 |
| 245k: TTFT / tok/s | 21,0 s / 169 | 19,5 s / 172,2 | 19,4 s / 168,0 | 19,2 s / 151,6 (*) |
| 4 utenti t2 media | ~93 | 94,0 | 92,5 | **100,0** |
| 4 pesanti t2 media | 84-94 | 86,9 | 96,4 | 91,1 |
| decode 4 utenti ms/step | 26,95 | 25,68 | 25,55 | 25,45 |
| stress 4 x 195-228k | in coda, no OOM | in coda, no OOM | in coda, no OOM | **tutto dentro, no OOM** |
| 6 difficili x3 a 100k | 16/18 (43,8k) | 15/18 (35,0k) | 16/18 (37,3k, 1 al tetto) | **17/18 (37,4k)** |
| tool call /50 (qwen3_coder) | n.m. | 32 | 33 | 32 |
| classe accuratezza | base | esatto/esatto in distr. | esatto/esatto in distr. | + G esatto, chunk 3392 (M' leggero) |

(*) passo a 1 utente identico a K1 (17,04 contro 17,01 ms): e' l'acceptance di un solo campione da 512 token.

### Vincitore proposto: K3 (decide l'utente)
- Perche': unico che porta 4 utenti oltre i 210k (4 x 227,5k) con l'MTP BF16 preferito; il piu' veloce a 4 utenti
  (100,0 contro ~93 della base, +7,5%; decode 25,45 ms/step), brevi +10%, TTFT a 245k -9%; lo stress 4 x 195-228k entra
  tutto in KV (turno 2 in 0,9-1,5 s invece della coda); 17/18 sui 6 difficili, come l'MTP BF16 di ieri, nessun tentativo
  al tetto, nessuna corruzione. In piu' corregge il non determinismo del top-k di oggi (§1).
- Patch dentro (tutte spente di default, una per env/flag): top-k LIB (correzione), C.2 embed UVA, vision UVA, C1, G,
  F98. Classe: tutte esatte o esatte in distribuzione; l'unica differenza numerica dalla base e' il chunk di prefill 3392
  (pagina di G), gate leggero dentro il rumore 1/4 utenti e 6 difficili 17/18.
- Comando (journal della corsa finale, albero vllm-fase2g; per la produzione vllm-fase2gp = fase2g + due patch del
  parser entrambe spente, VLLM_QWEN3_FENCE_FIX e VLLM_QWEN3_NO_REASONING_TOOL: con le env assenti e' identico):
  ```
  [Service] LimitMEMLOCK=infinity  MemoryMax=12G  MemorySwapMax=0
  Environment: HOME=/home/ec2-user CUDA_HOME=/usr/local/cuda HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    PYTHONPATH=/mnt/llmunity-models/vllm-fase2gp
    VLLM_GDN_REPLAY_LIB=/mnt/llmunity-models/fase2-build/g/build_120/gdn_replay_C_120.so
    VLLM_QSA_DET_TOPK_LIB=/mnt/llmunity-models/fase2-build/topk/build/det_topk_C.so
    VLLM_QWEN4EXP_PLE_FILE_UVA=1 VLLM_QWEN4EXP_DRAFT_VOCAB=98304
  /mnt/llmunity-models/vllm-venv/bin/vllm serve /mnt/llmunity-models/d0xin-fp8ple-ramview --served-model-name qwen x
    --load-format safetensors --tensor-parallel-size 1 --max-model-len 262144 --max-num-seqs 4
    --max-num-batched-tokens 4096 --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 14495514624 --kv-cache-dtype fp8
    --engram-config {"host_file_gather":true} --reasoning-parser qwen3
    --chat-template /mnt/llmunity-models/prod-tests/chat-template-qwen3.8-unsloth.jinja --enable-auto-tool-choice
    --tool-call-parser qwen3_coder --compilation-config {"cudagraph_mode":"full_decode_only"}
    --no-enable-flashinfer-autotune --speculative-config {"method":"mtp","num_speculative_tokens":3}
    --cpu-offload-gb 3 --cpu-offload-params visual embed_tokens --use-replayssm
  ```
  Prima dell'avvio: PLE in tmpfs (`mountpoint -q /mnt/llmunity-ple-ram || sudo mount /mnt/llmunity-ple-ram`, poi
  ensure_d0xin_ple_ram.py). Controlli nel log: "GDN RecoverSSM speculative verify active", "GDN decode kernel: cuda",
  "Total CPU offloaded parameters: 2.02", "GPU KV cache size: 909,971 tokens". Dopo ogni riavvio: ple_cmp.sh (C1
  registra la tmpfs in r+b).
- Manutenzione: gdn_replay_C_120.so e det_topk_C.so sono compilate contro il torch del venv di oggi (2.13 cu132, sm_120):
  se il venv cambia vanno ricompilate (build.py in fase2-build/g e fase2-build/topk, ~20 s l'una) e vanno rifatti i due
  harness (g_run.sh, topk_chain.sh).
- Rischio operativo (FORGIA #310, lsblk): /mnt/llmunity-models e' "Amazon EC2 NVMe Instance Storage" (xfs, 1,7 TB), la
  root e' EBS 250 GB: a uno stop si perdono fork, worktree fase2*, le due .so, venv e modelli (come la PLE). Le patch
  sono tutte in patches/ della stanza (diff + build.py + harness in script/) e ricostruibili; proposta (decisione LEAD):
  copia su EBS (/home/ec2-user, 230 GB liberi) di fase2-build (9 MB) e dei file modificati di vllm-fase2gp; venv (7,1 GB)
  e viste si ricostruiscono come oggi per la PLE (RICERCA #311). Con la regola StopEC2 attiva K3 sparisce a ogni stop.
- Rollback (combo di oggi, albero principale invariato su ple-host-file-gather, niente PYTHONPATH ne' env nuove): vista
  d0xin-mtpfp8-ramview, --kv-cache-memory-bytes 13958643712, senza --cpu-offload-*, senza --use-replayssm (comando in
  RISPOSTE_LEAD.md §2 / vspeed.sh).
- Alternative: K2 se si preferisce il draft FP8 della base (4 x 226k, 16/18, 4 utenti pari alla base); K1 se non si
  vuole G (4 x 200k, 15/18 dentro il rumore).
- Rischi aperti: tool call fantasma del parser (§4, anche oggi), #47194, Copilot reasoning/cot_id (proxy del LEAD).
  Incerto: tutte le velocita' sono una corsa per configurazione a temperatura 1,0 (rumore ~+-5% sui 4 utenti); i 6
  difficili sono 18 tentativi.
