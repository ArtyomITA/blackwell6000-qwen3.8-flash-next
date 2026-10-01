# Determinismo nello stesso avvio: stesso prompt 5 volte con cache_salt diverso (prefill a freddo), greedy, 8 token,
# top-20 logprob. Prompt corto (2k), medio (32k), lungo (200k). Stampa per ripetizione la differenza max dei logprob
# del token 0 rispetto alla prima e se i token generati coincidono.
# Uso: det2.py <porta> <vista> <tag>
import json, sys, time, urllib.request

from transformers import AutoTokenizer

port, view, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3]
C = "/mnt/llmunity-models/prod-tests/convs"
tok = AutoTokenizer.from_pretrained(view)
ids = tok.encode(open(f"{C}/c196k_a.txt", encoding="utf-8").read(), add_special_tokens=False)
allout = {}
for L in (2000, 32000, 200000):
    outs = []; secs = []
    for r in range(5):
        body = {"model": "qwen", "prompt": ids[:L], "max_tokens": 8, "temperature": 0.0, "logprobs": 20,
                "return_tokens_as_token_ids": True, "cache_salt": f"{tag}-{L}-{r}"}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        lp = json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"][0]["logprobs"]
        secs.append(time.time() - t0); outs.append(lp)
    print(f"{tag} L={L} secondi per richiesta (prefill a freddo + 8 token): {[round(x, 2) for x in secs]}", flush=True)
    allout[L] = outs
    a = outs[0]
    for r, b in enumerate(outs[1:], 1):
        keys = set(a["top_logprobs"][0]) & set(b["top_logprobs"][0])
        d = max(abs(a["top_logprobs"][0][k] - b["top_logprobs"][0][k]) for k in keys) if keys else float("nan")
        print(f"{tag} L={L} rip {r}: token0 diff max {d:.5f}, top0 identici {a['top_logprobs'][0] == b['top_logprobs'][0]}, "
              f"token uguali {a['tokens'] == b['tokens']}, top-20 di tutti gli 8 token identici {a['top_logprobs'] == b['top_logprobs']}", flush=True)
json.dump(allout, open(f"/mnt/llmunity-models/fase2-tests/banco/det2_{tag}.json", "w"))
print("DET2_FINE", tag)
