# G.1/G.2: generazione greedy con top-20 logprob del target per ogni token, su 8 prompt da 120-245k.
# Uso: gate_gen.py <porta> <vista> <out.json> [utenti_in_parallelo=1] [max_tokens=512]
# Il confronto tra due uscite lo fa gate_cmp.py (bit a bit, prima divergenza, margine, KLD).
import json, os, sys, threading, urllib.request

from transformers import AutoTokenizer

port, view, out_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
par = int(sys.argv[4]) if len(sys.argv) > 4 else 1
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 512
C = "/mnt/llmunity-models/prod-tests/convs"
tok = AutoTokenizer.from_pretrained(view)
enc = {n: tok.encode(open(f"{C}/{n}.txt", encoding="utf-8").read(), add_special_tokens=False)
       for n in ("c115k", "c196k_a", "c196k_b", "c196k_c", "c196k_d", "c240k")}
# 6 conversazioni intere (finiscono con il turno dell'assistente) + 2 tagli a meta' conversazione.
prompts = {n: ids for n, ids in enc.items()}
prompts["c196k_a@128k"] = enc["c196k_a"][:128000]
prompts["c196k_b@160k"] = enc["c196k_b"][:160000]
res = {}


def run(name):
    body = {"model": "qwen", "prompt": prompts[name], "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": 20, "ignore_eos": False, "return_tokens_as_token_ids": True}
    if os.environ.get("GATE_SALT"):  # prefill sempre a freddo: niente prefix hit
        body["cache_salt"] = os.environ["GATE_SALT"] + name.replace("@", "-")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    lp = j["choices"][0]["logprobs"]
    res[name] = {"prompt_len": len(prompts[name]), "tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"],
                 "top": lp["top_logprobs"]}


names = list(prompts)
for i in range(0, len(names), par):
    th = [threading.Thread(target=run, args=(n,)) for n in names[i:i + par]]
    [t.start() for t in th]; [t.join() for t in th]
json.dump(res, open(out_path, "w"))
print("GATE_GEN_FINE", out_path, {n: len(r["tokens"]) for n, r in res.items()})
