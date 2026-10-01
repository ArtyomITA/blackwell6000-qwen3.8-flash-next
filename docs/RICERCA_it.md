# RICERCA — fonti verificate (cantiere SM120, 1/10/2026)

Ogni voce: fatto, link, stato alla data di consultazione, impatto sul piano. "Ipotesi" = non verificato.

## Strada 2 (KV in RAM per la QSA)
- **vLLM PR #59209 "[Feature] Support HiSparse for QSA"** — https://github.com/vllm-project/vllm/pull/59209
  - Aperta 29/9/2026, non mergiata, branch `lHrHenry233:qsa-hisparse`, 21 file, nessun csrc nuovo visto
    (vllm/v1/hisparse/{binding,coordinator,layout,runtime}.py, qwen4_exp/common/qsa_cache.py, qwen4_exp/nvidia/qsa.py,
    qwen4_exp/nvidia/ops/qsa.py, single_type_kv_cache_manager.py, kv_cache_interface.py, config/attention.py,
    config/vllm.py, test).
  - Attivazione: `--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":8}}'`.
  - Requisiti dichiarati: runner V2, hybrid KV cache manager acceso, **prefix caching spento** (`--no-enable-prefix-caching`).
  - Test: GSM8K 197/200 on/off. MTP k=3 non testato esplicitamente (solo unit "MTP selection reuse").
  - Hardware: solo GB300 / "L20B", TP2. SM120 mai.
  - Velocita': -2..-21% a concorrenza alta; a C32 -16,59% con 0 miss ("low-concurrency overhead unresolved").
  - Impatto: decisione LEAD #11, strada 2 = ultima risorsa (prefix caching vitale per Copilot).
- **vLLM PR #53592** (HiSparse DeepSeek V4) — https://github.com/vllm-project/vllm/pull/53592 — DRAFT, regressione ad alta
  concorrenza; split-page in hisparse_kernels.cu. #59209 dichiara solo infra comune.
- **vLLM PR #57145** (OffloadingConnector, skip scratch groups) — https://github.com/vllm-project/vllm/pull/57145 —
  merged 17/9/2026; abilita `--kv-offloading-size` su Flash-Next. E' offload della prefix cache (blocchi evicti in CPU),
  non allarga la KV viva: non serve a 4x210k.
- **mochgolf/qsa-hisparse** (SGLang, 2x RTX 4090 48 GB, TP2, 256k/richiesta, page 64) —
  https://github.com/mochgolf/qsa-hisparse — niente MTP, niente numeri di velocita' attuali.
- Vincolo nel fork (config/vllm.py r.3483): `--use-replayssm` incompatibile con KV connector: G e strada 2 oggi esclusivi.

## Strada 1
- **C.2 embed_tokens via UVA**: il fork ha gia' UVAOffloader (vllm/model_executor/offloader/uva.py r.100-125) con filtro
  `--cpu-offload-params` per segmento; le torri ci passano via interfaces.py r.385-391. embed_tokens e' fuori da
  make_layers: basta una chiamata `wrap_modules` (patch FORGIA patches/C2_embed_uva.diff). Flag:
  `--cpu-offload-gb 3 --cpu-offload-params visual,embed_tokens`.
- **G, replay GDN = RecoverSSM**:
  - **vLLM PR #58863** "RecoverSSM for Qwen GDN and the PLE short conv" — https://github.com/vllm-project/vllm/pull/58863
    — aperta, da rebase (30/9). `--use-replayssm`; num_speculative_blocks = 0 (~36k token liberati per sequenza; H200
    pool 1.402.982 -> 1.489.533 token con record +6% di pagina). Verify Triton bit-identico al fused nativo; commit
    "entro 1 ulp BF16 dal riferimento FP32". Testato MTP K=3/5, align + prefix caching, FULL graph; GB10, RTX PRO 6000,
    H200. Vincolo: FULL decode vuole max_num_seqs <= num_blocks. Velocita': c=4 -5,2% per ciclo di verify (GB10).
  - **vLLM PR #59366** (draft, sopra #58863) — https://github.com/vllm-project/vllm/pull/59366 — replay nel kernel CUDA
    fuso (`fused_gdn_decode_post_conv_mtp_replay`) per gli step puri; misti restano Triton. "Bit-identical to native
    snapshot mode"; 128 test passati su RTX PRO 6000. Tocca csrc.
  - Nel fork: infra RecoverSSM gia' presente ma solo per Kimi-K3 KDA (config/vllm.py r.3420-3490); richiede
    `--mamba-backend triton` (da verificare se cambia i kernel GDN prefill/decode loggati oggi).

  - Analisi FORGIA #27 sul codice dei diff: con replayssm #58863 forza il decode GDN Triton (DIVERSO negli step puri);
    commit in forma chiusa != aggiornamento sequenziale del fuso (mul+fma): stato committed non bit a bit. Variante
    esatta proposta da FORGIA, approvata a tappe (LEAD #29).

## Spec decode
- **vLLM PR #58784** (MRV2: rifiuta slot draft mai proposti) — https://github.com/vllm-project/vllm/pull/58784 —
  MERGED 28/9/2026 (fork del 27/9: assente). Prerequisito per P. Con draft argmax il rischio per noi e' basso
  (q one-hot: rejection corretto in distribuzione).
- **vLLM issue #47194** (prefix caching + MTP3 su ibridi: tool call che trapelano, needle a 0/10) —
  https://github.com/vllm-project/vllm/issues/47194 — APERTA, nessun fix linkato; workaround upstream = MTP spento.
  Test agent con prefix caching on/off obbligatorio.
- **F, draft a vocabolario ridotto**:
  - MiaAI-Lab PR #58 — https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/58 — Flash-Next, lista
    files/draft_vocab_en_code_47k.txt (47.149 ID), +8,4% medio (codice +9,3%), acceptance invariata; S=4 134,8->146,0.
  - vLLM issue #58578 — https://github.com/vllm-project/vllm/issues/58578 — aperta, nessuna PR; stesso disegno
    (slice lm_head + draft_id_to_target_id, formato SGLang --speculative-token-map).
  - Fork: checkpoint ha solo lm_head.weight (nessuna testa MTP propria): il draft condivide il lm_head del target
    (llm_base_proposer.py r.1508-1535), quindi F = copia separata (+0,24 GB a 47k ID).
  - tonyd2wild DGX-Spark — https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark — default MTP3 con draft
    ridotto ai 65.536 ID piu' bassi (nessuna mappa), acceptance 64%, 3,56 accettati/step; MTP4 leggera regressione.
    Variante approvata (LEAD #46): vista lm_head.weight[:N], d2t non serve, +0 GB.
  - Stesso repo, PLE da NVMe con pread su thread pool: Spark 32,5 contro 43,8 tok/s con tabella in memoria (-26%).
  - Misura RICERCA #191 (tokenizer del modello, 993.643 token di content LCB + convs, /tmp/tokcov2_ricerca.py sulla
    macchina): quota ID < N: 65.536 0,979; 81.920 0,990; 90.112 0,994; 94.208 0,998; 98.304 0,998; nessun token tra
    98.304 e 131.072. "```" = ID 71.093 escluso da N=65.536 (spiega acceptance 2,81 -> 2,58 di BANCO #185). Proposta
    N=98.304. Limite: ragionamento non salvato nei jsonl.
  - Acceptance nel ragionamento lungo (6 difficili a 100k, temp 1,0; journal, media pesata sui token draftati,
    RICERCA #274): fail100k-bf16 senza F 2,201 (0,602/0,369/0,230); fail100k-fp8 2,182 (0,598/0,361/0,222); lcb-K1 con
    F98 2,208 (0,604/0,372/0,232). F98 non toglie acceptance.
- **Autotune FlashInfer MoE**: flashinfer #4841 — https://github.com/flashinfer-ai/flashinfer/issues/4841 — chiusa senza
  fix kernel; corruzione 4-19% per boot solo con gemm1/gemm2 autotunati; guadagno full autotune +8% (173,5 vs 158,4).
  Resta spento.

## Kernel MoE (sezione F, solo se lo step resta > 30 ms)
- flashinfer issue #5561 — https://github.com/flashinfer-ai/flashinfer/issues/5561 — Flash-Next NVFP4 (E=512, H=2560,
  I=640, top-10, W4A4) su SM120: kernel di riferimento Apache-2.0 M=1/4/8 = 23,0/24,6/36,1 us contro miglior FlashInfer
  32,5/47,2/71,2 us. Tutti 0,14-0,16 L2 relativo vs fp32. Classe DIVERSO.
- flashinfer PR #5168 — https://github.com/flashinfer-ai/flashinfer/pull/5168 — aperta, solo b12x (escluso per noi).

## Tool call / parser (gate G.3, H.3)
- vLLM issue #57541 — https://github.com/vllm-project/vllm/issues/57541 — aperta, fix PR #57553 non mergiata:
  "<function=..." dentro blocchi ``` diventa tool call vera, riprodotto su Qwen3.8-Flash-Next-NVFP4.
- Riprodotto su K1 (BANCO #231): fence_spiega 0/10, 8/10 tool call fantasma. Nel fork qwen3_coder e qwen3_xml = stesso
  Qwen3EngineToolParser (tool_parsers/__init__.py r.185-191). Fix vLLM PR #57553 (aperta, 27/9) —
  https://github.com/vllm-project/vllm/pull/57553 — Python puro (incremental_lexer, parser_engine_config,
  streaming_parser_engine, qwen3.py + test), fence CommonMark solo in CONTENT/REASONING; backport approvato (LEAD #240).
- Gate fence (BANCO #293): il backport #57553 NON aiuta (fence_spiega identico 1/10) e rompe "``` aperta nel testo poi
  tool call vera" (10/10 -> 1/10): fix fuori (LEAD #297). Meccanismo probabile dei fantasma (RICERCA #299): fork
  vllm/parser/qwen3.py r.140-144, transizione implicita (REASONING, TOOL_START) -> TOOL_PREAMBLE: un <tool_call> scritto
  nel ragionamento chiude il think e apre una chiamata vera. CONFERMATO su CPU (FORGIA #304, ghost_probe.py): rimedio
  pronto spento VLLM_QWEN3_NO_REASONING_TOOL=1 (toglie la transizione; perde le chiamate emesse senza </think>):
  NON adottato, serve gate GPU, decisione utente.
- HF Qwen3-Coder-Next discussione 17 — https://huggingface.co/Qwen/Qwen3-Coder-Next/discussions/17 — qwen3_coder su
  input lunghi: flusso infinito "!!!!" (token 0); risolto con qwen3_xml. Token 0 e' anche la firma di #58784 e della
  corruzione autotune: nel test di corruzione contare le sequenze di token 0.

## Copilot (H.2)
- microsoft/vscode issue #338819 — https://github.com/microsoft/vscode/issues/338819 — aperta 30/9/2026 (VS Code
  1.138-1.139.1): Copilot legge reasoning_content ma lo rimanda nei turni dopo solo se il delta ha cot_id,
  reasoning_opaque o signature. Kimi K3/SGLang: ragionamento 68 -> 3 token nei turni successivi; con echo 175/303/244.
  Workaround: proxy aggiunge cot_id.
- Template prod-tests/chat-template-qwen3.8-unsloth.jinja r.114-120: preserve_thinking non definito = true, rende il
  ragionamento in tutti i turni assistant; senza echo restano "<think>\n\n</think>" vuoti. Prefix cache: si ricalcola
  solo l'ultimo turno; rischio principale = qualita' (decisione utente, LEAD #49).

## Step misti e prefill (LEAD #68)
- Community stessa GPU: MarcoPizeta prefill freddo 28,1k tok/s piatto 8k-128k (128k in 4,66 s), immagine vLLM
  qwen38-flash-next, MTP3, PLE INT4 offload (lossy). Noi: 245k in 21,0 s = 11,7k tok/s; step misto 3.200 token 250-270 ms.
- Fork, scheduler.py r.349-350 e r.473-475: in mamba_cache_mode "align" il chunk di prefill e' troncato al multiplo di
  block_size (3.200): con --max-num-batched-tokens 4096 il chunk reale e' 3.200. 6416 = 2 blocchi + 16 token decode.
- Fork, nvidia/ngram_embedding.py r.150-185 (stage_rows): dopo stream.synchronize(), loop Python di os.preadv una riga
  unica alla volta, per ogni modulo PLE, ogni step: candidato principale del gap host (C1 lo toglie).
- Contro-prova (RICERCA #81): SGLang discussion #36891 — https://github.com/sgl-project/sglang/discussions/36891 —
  RTX PRO 6000, prefill 64k 10,1k tok/s, ~490k 7,9k tok/s; decode C1 171, C4 427 aggregati. gabrielolympie —
  https://github.com/gabrielolympie/sglang-flashnext-sm120 — prefill 2,5k ~10,4k tok/s; 1 utente 231, 4 utenti 620-657
  aggregati ma con leve lossy (accettazione MTP rilassata 0,3, dense W8A16 fp8, HC e lm_head fp8). Il nostro prefill
  e' nella media SM120; il 28k di MarcoPizeta e' un'eccezione.
- vLLM PR #54915 (indexer logits compatti, merged 4/9/2026, bit-exact) — https://github.com/vllm-project/vllm/pull/54915
  — gia' nel fork (qsa_indexer.py r.583-609).
- vLLM PR #56240 (FlashInfer PrimTS QSA, draft) — https://github.com/vllm-project/vllm/pull/56240 — SM100/103, non SM120;
  prefill +6-8%, accuratezza non equivalente ancora.
- PD-multiplexing GreenContext (SGLang) — https://www.lmsys.org/blog/2025-09-28-pdmux/ — niente spec decode, non in vLLM.
- vLLM PR #49827 (GDN batch-invariant, aperta) — https://github.com/vllm-project/vllm/pull/49827 — solo SM90+ non
  quantizzato; conferma che oggi passi misti (FLA) e puri (CUDA fuso) danno numeri diversi.
- cudaHostRegisterReadOnly: valido solo se cudaDevAttrHostRegisterReadOnlySupported = 1
  (https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html). Kernel 6.18: pin in scrittura su
  tmpfs ammesso (https://lwn.net/Articles/930966/). Registrazione ~74 ms per 2 GiB (vLLM RFC #42361).

## Determinismo (BANCO #87, LEAD #90)
- Fork flashinfer_cutlass_moe.py r.365-389: flashinfer_cutlass_fused_moe chiamata senza use_fused_finalize; FlashInfer
  0.7.0 fused_moe/core.py r.987 default use_fused_finalize=True. Fused finalize con atomiche = non deterministico
  run-to-run (flashinfer issue #5221 https://github.com/flashinfer-ai/flashinfer/issues/5221 per b12x, PR #5375
  fixed-order; SGLang PR #40105 https://github.com/sgl-project/sglang/pull/40105 merged 18/9/2026 lo spegne di default,
  deterministic inference lo forza off; costo CUTLASS +7,82% latenza MoE a 16k token su B300). Causa probabile (non
  ancora provata) del rumore tra avvii. Proposta: env di gate use_fused_finalize=False.
  **SMENTITA (RICERCA #104, da nsys BANCO #101):** la finalize fusa scatta solo se la tattica GEMM2 ha
  epilogue_fusion_type == FINALIZE (cutlass_fused_moe_kernels.cuh r.4576-4592); con autotune spento nsys mostra
  finalizeMoeRoutingKernel separato (deterministico). Sospetti successivi: torch.topk dell'indexer a pari merito,
  algoritmo cuBLASLt scelto in base al workspace al boot.
- **CAUSA del non-determinismo (RICERCA #119-#120): top-k dell'indexer QSA.** Fork ops/qsa_indexer.py r.471-498: su SM120
  _topk usa sempre torch.ops._C.persistent_topk (posizioni via atomicAdd, buffer candidati troncabili, nessun sort);
  chiamato da decode (r.563) e prefill (r.633). Fonti:
  - vLLM PR #55122 (aperta, 25/9/2026) — https://github.com/vllm-project/vllm/pull/55122 — persistent_topk restituisce
    ordine e insiemi diversi a input identici; fix det_select_row, env VLLM_QSA_DET_TOPK=1, kernel 0,67-1,00x, csrc.
  - docai.hu — https://docai.hu/en/blog/qwen38-flash-next-nondeterministic-vllm-kernel — temp 0 instabile su 13/50 task,
    logit fino a 3 nat; overlay Python torch.topk(sorted=False) + 2 sort stabili: 0/50 instabili, punteggio 97 -> 98,
    prefill 1,35x piu' lento (GB10).
  - vLLM PR #57815 — https://github.com/vllm-project/vllm/pull/57815 — top-k "auto" cambia kernel col numero di righe.
  - PROVATO da BANCO #139 (det2): base 32k/200k fino a 5,54 nat al token 0 nello stesso avvio; con overlay Python
    VLLM_QSA_DET_TOPK_PY=1 (FORGIA #125) top-20 identici 4/4 in 2 avvii. Costo overlay prefill 32k +6,4%, 200k +27%.
  - #55122 verificata su SM121 (limite smem 99 KiB per blocco, ramo FilteredTopK irraggiungibile): SM120 stesso limite
    (ipotesi), quindi atteso kernel 0,67-1,00x; il costo 1,1-2,7x riguarda solo H100/A100.
- Errore mio #15: --cpu-offload-params vuole valori separati da spazio ("visual embed_tokens"), non virgola
  (trovato da FORGIA #88).

## Conti di memoria
- Pool effettivo ~17,8 KB/token (12 GiB / 722.804 = 17.826 B; 13 GiB / 783.886 = 17.807 B), non 14.144 B: lo stato GDN
  e' ammortizzato nelle pagine. embed UVA +1,27 GB = +~71k token.

## Community (numeri di confronto)
- MarcoPizeta/flash-next-rtxpro6000-bench (1x RTX PRO 6000, 64 GB RAM, PLE INT4 offload, MTP 3, max-model-len 135k):
  4 utenti a 32k = 150,7 tok/s aggregati; 1 utente 8k 140,5 tok/s. https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench
  — la nostra combo (4 utenti 27-121k, media ~93 a testa) e' molto sopra.
