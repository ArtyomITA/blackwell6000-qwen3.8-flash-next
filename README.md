<div align="center">

# Blackwell 6000 + qwen 3.8 flash next

**Qwen3.8 Flash-Next and its Swift 1.5 fine-tune on a single RTX PRO 6000 Blackwell with vLLM.**
4 users with 227k tokens of context each, about 100 tok/s per user, exact kernels and a fix for a real determinism bug.

[![vLLM](https://img.shields.io/badge/vLLM-Trosfy_fork-0f766e)](https://github.com/Trosfy/vllm)
[![GPU](https://img.shields.io/badge/GPU-RTX_PRO_6000_%28SM120%29-76b900?logo=nvidia&logoColor=white)](#requirements)
[![Model](https://img.shields.io/badge/model-Qwen3.8_Flash--Next_%7C_Swift_1.5_%28NVFP4%29-7c3aed)](#requirements)
[![Context](https://img.shields.io/badge/context-4x227k_tokens-2563eb)](#results)
[![Speed](https://img.shields.io/badge/decode-100_tok%2Fs_per_user_x4-2563eb)](#results)
[![Peak](https://img.shields.io/badge/peak-186.5_tok%2Fs_single_user-2563eb)](#public-numbers-on-the-same-gpu)
[![Coding](https://img.shields.io/badge/hard_coding-17%2F18-16a34a)](#results)
[![MTP heads](https://img.shields.io/badge/MTP_heads-Hugging_Face-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/adriandj3/Swift-1.5-Qwen3.8-Flash-Next-MTP-for-GGUF-and-NVFP4)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

[Results](#results) · [What K3 changes](#what-k3-changes) · [Install](#install) · [Tests](#tests-and-gates) · [Known issues](#known-issues) · [Copilot](#github-copilot-vs-code-in-front-of-it)

| **186.5 tok/s** | **100 tok/s** | **909,971** | **17/18** | **+0.5%** |
|:---:|:---:|:---:|:---:|:---:|
| peak, single user, lossless | per user, 4 users at once | KV tokens, 4 × 227.5k | hard LiveCodeBench v6, 100k budget | prefill cost of the determinism fix |

</div>

## What this is

Serving **Qwen3.8 Flash-Next** and its **Swift 1.5** fine-tune (NVFP4 weights, FP8 PLE table) with vLLM on **one RTX PRO
6000 Blackwell (96 GB, SM120)**, an AWS g7e.2xlarge. The patches work at the architecture level, so they apply to both:
Swift 1.5 has the same layers, the same PLE table and the same MTP head as the base model. All the K3 numbers below were
measured with Swift 1.5. Every optimization here is exact (bit-identical output against the baseline), and
one change fixes a real determinism bug in the sparse-attention indexer that was there before.

The final configuration is called **K3**. It was chosen from three candidates measured under the same protocol, and it
is the production configuration on that machine (October 2026), serving a small team through GitHub Copilot.

Everything here is a patch on top of an existing vLLM fork plus two small CUDA extensions. No model weights are
included. The MTP draft head we used (BF16), the FP8 variant and a GGUF for llama.cpp are on Hugging Face:
[adriandj3/Swift-1.5-Qwen3.8-Flash-Next-MTP-for-GGUF-and-NVFP4](https://huggingface.co/adriandj3/Swift-1.5-Qwen3.8-Flash-Next-MTP-for-GGUF-and-NVFP4).

| folder | what is inside |
|---|---|
| [`patches/`](patches/) | the full K3 patch for the vLLM fork, plus the parser changes we tried and did not adopt |
| [`kernels/`](kernels/) | the two CUDA extensions: deterministic top-k and GDN state replay |
| [`model/`](model/) | scripts that stage the 48 GB PLE table in RAM and check it |
| [`deploy/`](deploy/) | systemd units, OpenResty proxy for Copilot, VS Code entry, AWS EBS script |
| [`tests/`](tests/) | the exactness gates and benchmarks, as they were run |
| [`docs/`](docs/) | full results and research notes (Italian) |

## Results

Baseline = the same machine, fork and model before these patches: MTP experts in FP8, vision in VRAM, 13 GB of FP8 KV.

| | baseline | K3 |
|---|---|---|
| KV capacity | 783,886 tokens (4 × 196k) | **909,971 tokens (4 × 227.5k)** |
| 4 users, tok/s per user (second turn, 32-180k context) | ~93 | **100.0** (85.4 / 98.2 / 99.9 / 116.6) |
| 4 users, heavy prompts | 84-94 | 91.1 |
| 1 user, short prompts | 156-162 | 171.7-186.5 |
| TTFT, 245k-token prompt | 21.0 s | 19.2 s |
| decode step, 1 / 4 users | n/a | 17.04 / 25.45 ms |
| MTP acceptance (3 draft tokens) | n/a | 2.88 |
| stress, 4 × 195-228k at once | requests queue (17-47 s) | all resident, second-turn TTFT 0.9-1.5 s |
| 6 hard LiveCodeBench v6 problems × 3 attempts, 100k budget, temp 1.0 | 16/18, 43.8k mean tokens | **17/18**, 37.4k mean tokens (median 39.5k, max 55.9k, 0 at cap) |
| tool-call test (50 calls, `qwen3_coder` parser) | n/a | 32/50 (see Known issues) |

Caveats: one speed run per configuration (run-to-run noise ±5%), and 18 coding attempts per configuration. With 1 user
and a 245k prompt, K3 measured 151.6 tok/s against 169 for the baseline. That was a single 512-token sample at
temperature 1.0: the 1-user decode step is identical to the configuration without G (17.04 vs 17.01 ms), so the gap is
MTP acceptance, not a cost of the patch.

Full numbers, commands and logs for the three candidates (Italian): [docs/FINALE_it.md](docs/FINALE_it.md).

### Public numbers on the same GPU

Other people have published decode speeds for this model on an RTX PRO 6000. Engines, checkpoints, context sizes and
methods all differ, so read this as orientation, not as a ranking.

| setup | engine | 1 user (tok/s) | 4 users, per user (tok/s) | KV / context | notes |
|---|---|---:|---:|---|---|
| **K3, this repo** | vLLM (Trosfy fork) + these patches | **171.7-186.5** | **100.0** | 910k tokens, 4 × 227.5k | lossless, temperature 1.0, outputs bit-identical to the unpatched baseline |
| [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) | SGLang fork | 231 (median) | 155-164 | 572k tokens | 1-user figure with the MTP acceptance threshold relaxed to 0.3 (exact only at temperature 0), FP8 weights at runtime |
| [SGLang discussion #36891](https://github.com/sgl-project/sglang/discussions/36891) | SGLang | 171 | ~107 | n/a | |
| [d0xin FP8 PLE card](https://huggingface.co/d0xin/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-FP8PLE) | SGLang (Pennyroyal) | 133 (median, fixed 4096-token generation) | n/a | 262k native | 174.7 effective tok/s on an agentic workload |
| [MarcoPizeta/flash-next-rtxpro6000-bench](https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench) | vLLM | 140.5 at 8k | ~38 at 32k | n/a | |

Among the setups that publish it, K3 has the largest KV budget (910k tokens, all four users resident in VRAM), and its
single-user speed is at the top of the lossless numbers we found. The only higher figure (231) relaxes MTP acceptance.

## What K3 changes

| # | change | how to turn it on | accuracy class | gate |
|---|---|---|---|---|
| 1 | **Deterministic top-k in the QSA indexer** (backport of [vllm#55122](https://github.com/vllm-project/vllm/pull/55122)) | `VLLM_QSA_DET_TOPK_LIB=…/det_topk_C.so` | quality fix | indices identical to a slow stable-sort reference, 16/16 cases; server bit-identical across restarts |
| 2 | Vision tower and `embed_tokens` in host RAM via UVA | `--cpu-offload-gb 3 --cpu-offload-params visual embed_tokens` | exact | embedding op bit-identical on 10⁶ ids |
| 3 | **C1**: PLE rows read through UVA straight from the tmpfs shards | `--engram-config '{"host_file_gather":true}'` + `VLLM_QWEN4EXP_PLE_FILE_UVA=1` | exact | 10⁶ rows equal to `preadv`; server bit-identical |
| 4 | **F98**: MTP draft head limited to the first 98,304 vocabulary ids | `VLLM_QWEN4EXP_DRAFT_VOCAB=98304` | exact in distribution | greedy output bit-identical |
| 5 | **G**: exact GDN state replay for speculative decoding (port of [vllm#58863](https://github.com/vllm-project/vllm/pull/58863) + [#59366](https://github.com/vllm-project/vllm/pull/59366)) | `--use-replayssm` + `VLLM_GDN_REPLAY_LIB=…/gdn_replay_C_120.so` | exact replay | 1000-step harness bit-identical; server bit-identical against the baseline run with `--block-size 3392` |

On top of these: MTP 3 tokens with the model's own BF16 MTP head, FP8 KV cache, `--max-num-batched-tokens 4096`.

Notes on each change:

1. **Top-k (quality fix).** `persistent_topk`, the only top-k path on SM120, places results with `atomicAdd`. Identical
   inputs can therefore end up with a different set or order of selected KV blocks. We measured it in one server, with the same prompt and a cold
   prefill: the logprob of the first generated token varied by up to 2.38 nat at 32k and 5.54 nat at 200k. A public
   write-up reports the same effect on Flash-Next (GB10, fixed there with an exact `torch.topk` path instead):
   [docai.hu](https://docai.hu/en/blog/qwen38-flash-next-nondeterministic-vllm-kernel). We did not run an accuracy A/B
   with and without the fix; every K run already has it. The backport is loaded as a separate extension. It costs +0.5% prefill at 200k, and decode stays within noise.
2. **Vision and embeddings via UVA.** The values must be **space-separated**: with `visual,embed_tokens` (comma) nothing
   is offloaded. This frees about 2 GiB of VRAM, which goes to the KV cache.
3. **C1.** Replaces a per-row `os.preadv` loop plus `stream.synchronize()` with a Triton gather over host memory
   registered with `cudaHostRegister`. ReadOnly registration is not supported on this GPU, so flag 0 is used. Run
   `verify_d0xin_ple_copies.py` to confirm the table is untouched. Effect: host gap of mixed prefill/decode steps
   30.6 → 1.6 ms.
4. **F98.** The draft only proposes; the target verifies, so the output distribution does not change. N = 65,536 was
   too small: the token ```` ``` ```` has id 71,093.
5. **G.** Three changes to the upstream PRs. With them the replay is bit-identical to the per-token kernel in our
   tests (1000-step harness, and greedy server output against the baseline):
   - record `decay = exp(g)` exactly as the kernel computes it, instead of `g`;
   - commit sequentially, `h = fma(delta, k, h * decay)`, instead of the closed-form commit used upstream (its authors
     describe it as within one BF16 ulp of the FP32 reference; we did not measure that form ourselves);
   - keep the fused CUDA decode kernel for pure decode steps (#58863 alone switches decode to Triton), loading the
     replay-enabled kernel from a separate extension.

   G raises the attention page to 3392 tokens, so the real prefill chunk is 3392 instead of 3200. That is the only
   difference from the baseline. Result: +11.5% KV.

Measured and rejected:

| option | why it was rejected |
|---|---|
| prefill chunk 6416 | +10% prefill, but decode stalls doubled |
| MTP 2 | lower acceptance |
| MTP 4 | longer draft, and one extra mamba block per request: less KV |
| KV offload to host RAM ([vllm#59209](https://github.com/vllm-project/vllm/pull/59209)) | needs prefix caching off, −16.6% at low concurrency, incompatible with `--use-replayssm` |
| FlashInfer MoE autotune | corrupts output (flashinfer#4841) |
| fence-aware tool parser ([vllm#57553](https://github.com/vllm-project/vllm/pull/57553)) | gate failed: an unclosed fence hides the real tool call |

## Requirements

- GPU: RTX PRO 6000 Blackwell (SM120, 96 GB). Host: 64 GB RAM, swap off. The PLE table takes 48 GB of RAM in a noswap
  tmpfs.
- Our stack: Python 3.12, torch 2.13.0+cu132, FlashInfer 0.7.0, Triton 3.7.1, NVIDIA driver 595.91, CUDA 13.
- vLLM: fork [Trosfy/vllm](https://github.com/Trosfy/vllm), branch `ple-host-file-gather`, commit `8b2069b62`. This fork
  adds Qwen3.8 Flash-Next, PLE `host_file_gather` and the `qwen3_coder` parser.
- Model: [`d0xin/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-FP8PLE`](https://huggingface.co/d0xin/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-FP8PLE)
  (FP8 PLE derivative of `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`). Check the model license: Swift requires a
  commercial license from UkisAI above a revenue threshold.

## Install

1. **Build the fork and apply the patch.** The patch only touches Python files and tests. You can apply it to the built
   tree itself, or (our setup) to a git worktree used through `PYTHONPATH`, with the compiled `vllm/*.so` symlinked from
   the built tree.

   ```bash
   git clone https://github.com/Trosfy/vllm.git vllm-src
   ```
   ```bash
   cd vllm-src && git checkout 8b2069b62
   ```
   Build and install it into your venv as the fork documents (editable install, SM120). Then:
   ```bash
   git apply /path/to/this-repo/patches/k3-vllm.diff
   ```

2. **Build the two extensions.** About 20 s each. Do not run them while vLLM is running; on a 64 GB host keep
   `MAX_JOBS=1`.

   ```bash
   VLLM_SRC=/path/to/vllm-src MAX_JOBS=1 python kernels/det_topk/build.py
   ```
   ```bash
   VLLM_SRC=/path/to/vllm-src MAX_JOBS=1 python kernels/gdn_replay/build.py 120
   ```
   Results: `kernels/det_topk/build/det_topk_C.so` and `kernels/gdn_replay/build_120/gdn_replay_C_120.so`. Rebuild them
   whenever torch changes.

3. **Model and PLE in RAM.** Download the model, mount a noswap tmpfs and stage the 10 PLE shards. The scripts in
   `model/` have their paths at the top of each file; edit them for your layout.

   ```bash
   hf download d0xin/Swift-1.5-Qwen3.8-Flash-Next-NVFP4-FP8PLE --local-dir /mnt/llmunity-models/d0xin-fp8ple
   ```
   ```bash
   echo 'tmpfs /mnt/llmunity-ple-ram tmpfs rw,noswap,size=50G,uid=1000,gid=1000,mode=0700 0 0' | sudo tee -a /etc/fstab
   ```
   ```bash
   sudo swapoff -a && sudo mount /mnt/llmunity-ple-ram
   ```
   ```bash
   python model/stage_d0xin_ple_tmpfs.py
   ```
   The first run copies the shards (48 GB), writes a sha256 manifest, and builds the "ramview": a directory of symlinks
   where the PLE points to RAM and everything else points to disk. On every later boot, `model/ensure_d0xin_ple_ram.py`
   copies only the missing shards and checks their hashes; the systemd unit `llmunity-ple-stage.service` runs it.

4. **Serve.** Use [deploy/systemd/llmunity-vllm-prod.service](deploy/systemd/llmunity-vllm-prod.service); it needs
   `LimitMEMLOCK=infinity` for the pinned host memory. The core of it:

   ```bash
   PYTHONPATH=/path/to/patched-vllm \
   VLLM_QSA_DET_TOPK_LIB=/path/to/det_topk_C.so VLLM_GDN_REPLAY_LIB=/path/to/gdn_replay_C_120.so \
   VLLM_QWEN4EXP_PLE_FILE_UVA=1 VLLM_QWEN4EXP_DRAFT_VOCAB=98304 \
   vllm serve /mnt/llmunity-models/d0xin-fp8ple-ramview --served-model-name qwen \
     --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.90 \
     --kv-cache-memory-bytes 14495514624 --kv-cache-dtype fp8 \
     --engram-config '{"host_file_gather":true}' --reasoning-parser qwen3 \
     --chat-template deploy/chat-template-qwen3.8-unsloth.jinja \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --compilation-config '{"cudagraph_mode":"full_decode_only"}' --no-enable-flashinfer-autotune \
     --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
     --cpu-offload-gb 3 --cpu-offload-params visual embed_tokens --use-replayssm
   ```

   Check the boot log for these lines:
   - `Total CPU offloaded parameters: 2.02`
   - `GDN RecoverSSM speculative verify active (spec_query_len 4)`
   - `GDN decode kernel: cuda`
   - a KV cache of 909,971 tokens

## Tests and gates

[tests/](tests/) holds the scripts exactly as they were run during the measurements. They carry our machine's paths and
Italian comments, so treat them as a reference rather than a test suite:

| script | what it checks |
|---|---|
| `gate_gen.py` + `gate_cmp.py` | greedy tokens and top-20 target logprobs, bit-identical between two servers, 8 prompts of 120-245k |
| `gate_gen4.py` | the same check with 4 users arriving together |
| `g_harness_full.py`, `g_harness_fla.py` | GDN replay over 1000 steps against the reference kernels |
| `c1_harness.py` | UVA gather against `preadv` on 10⁶ PLE rows |
| `emb_harness.py` | embeddings read via UVA against embeddings in VRAM |
| `topk_selftest.py`, `kernels/det_topk/topk_cmp.py` | deterministic top-k against a stable-sort reference |
| `det2.py` | determinism inside one server (same prompt, cold prefill) |
| `toolcall_gate.py` | the 50-call tool-call test |
| `vspeed_g.sh`, `sweep.py`, `banda.py` | speed benchmark, decode sweep, memory bandwidth |
| `lcb_cand.sh` | the coding test |
| `ghost_probe*.py` | CPU reproduction of the phantom tool calls |

## Known issues

- **Phantom tool calls.** The Qwen3 parser turns a `<tool_call>` that the model writes *inside* `<think>` into a real call:
  there is an implicit `REASONING → TOOL` transition in `vllm/parser/qwen3.py`. This exists without our patches too.
  A switch that removes that transition is in
  [patches/not-adopted/](patches/not-adopted/qwen3-parser-fence-and-reasoning-tool.diff)
  (`VLLM_QWEN3_NO_REASONING_TOOL=1`). It is not adopted: it would drop real calls emitted before `</think>`, so it needs
  a GPU gate first.
- [vllm#47194](https://github.com/vllm-project/vllm/issues/47194): hybrid models + prefix caching + MTP can leak tool
  calls into the text. Open upstream.
- The thinking budget is large. At the template's default effort (`xhigh`), hard problems take 35-60k reasoning tokens.

## GitHub Copilot (VS Code) in front of it

[deploy/proxy/](deploy/proxy/) is an OpenResty config with two Lua filters:
- `force_modelfile_params.lua` forces the sampling parameters: temperature 1.0, top_p 0.95, top_k 20, penalties 0.
- `reasoning_cot.lua` adds a `cot_id` to every reasoning delta. Without an id, Copilot up to VS Code 1.140 does not send
  the reasoning back in later tool-call rounds ([microsoft/vscode#338819](https://github.com/microsoft/vscode/issues/338819),
  fixed for 1.141). It also renames `reasoning` to `reasoning_content`; VS Code reads both names, so that part is
  harmless but not required.

[deploy/copilot/chatLanguageModels.example.json](deploy/copilot/chatLanguageModels.example.json) is the matching VS Code
entry:
- `customendpoint`, `maxInputTokens` 170000 + `maxOutputTokens` 50000, which stays under the 227k per slot;
- reasoning effort `low` / `medium` / `high`.

The bundled chat template (Unsloth's Qwen3.8 template) maps `high` to `xhigh`; the model's own template rejects `high`.
`xhigh` can think for 7-12 minutes on hard problems; `medium` is the everyday setting.

## AWS g7e note

The g7e instance store (where the model and the venv usually live) is **wiped at every stop**.
[deploy/systemd/salva-su-ebs.sh](deploy/systemd/salva-su-ebs.sh) copies what K3 needs (~135 GB) to the root EBS volume
once, verifies the copy, and switches `/etc/fstab` to bind-mount it at the same path. After that, a stop/start just
reloads from EBS.

## Credits and license

- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) and the upstream PRs this work ports or adapts: #55122,
  #58863, #59366, #57553.
- The [Trosfy/vllm](https://github.com/Trosfy/vllm) fork.
- d0xin for the FP8 PLE checkpoint, and UkisAI for Swift.
- Unsloth for the chat template.
- docai.hu for the write-up on Flash-Next non-determinism.

The code and patches here are released under Apache-2.0 ([LICENSE](LICENSE)). The CUDA sources in `kernels/` keep
their vLLM copyright headers. Model weights are not included and come under their own licenses. Notes in `docs/` are in
Italian.
