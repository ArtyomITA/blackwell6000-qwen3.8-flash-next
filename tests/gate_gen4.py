# Variante a 4 utenti con arrivo identico: UNA richiesta /v1/completions con prompt = lista di 4 (c115k, c196k_a,
# c196k_d, c196k_b tagliato a 160k), greedy, top-20 logprob, cache_salt (prefill a freddo). Stesso formato di gate_gen.py.
# Uso: GATE_SALT=x gate_gen4.py <porta> <vista> <out.json> [max_tokens=128]
import json, os, sys, urllib.request

from transformers import AutoTokenizer

port, view, out_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 128
C = "/mnt/llmunity-models/prod-tests/convs"
tok = AutoTokenizer.from_pretrained(view)
enc = lambda n: tok.encode(open(f"{C}/{n}.txt", encoding="utf-8").read(), add_special_tokens=False)
names = ["c115k", "c196k_a", "c196k_d", "c196k_b-160k"]
prompts = [enc("c115k"), enc("c196k_a"), enc("c196k_d"), enc("c196k_b")[:160000]]
body = {"model": "qwen", "prompt": prompts, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 20,
        "return_tokens_as_token_ids": True, "cache_salt": os.environ.get("GATE_SALT", "g4") + "-4u"}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
j = json.loads(urllib.request.urlopen(req, timeout=7200).read())
res = {}
for ch in j["choices"]:
    lp = ch["logprobs"]; i = ch["index"]
    res[names[i]] = {"prompt_len": len(prompts[i]), "tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"],
                     "top": lp["top_logprobs"]}
json.dump(res, open(out_path, "w"))
print("GATE_GEN4_FINE", out_path, {n: len(r["tokens"]) for n, r in res.items()})
