# Gate ESATTO (PLAN v3 §2): greedy + logprob del target, confronto bit a bit tra due avvii (patch spenta/accesa).
# run:     gate_exact.py run <porta> <out.json> <utenti 1|4> <conv1.txt> [conv2.txt ...]
#          temp 0, max_tokens 256, logprobs 5, /v1/completions con prompt grezzo; utenti=4 manda i prompt a gruppi di 4
#          in UNA richiesta con lista (stesso arrivo = stessi step tra corse), utenti=1 uno alla volta.
#          Per il prefill a freddo usare server con prefix cache salata o prompt nuovi.
# compare: gate_exact.py compare <a.json> <b.json>
#          per prompt: token identici, prima divergenza, logprob identici (==) sul prefisso comune. Exit 0 solo se tutto ==.
import json, sys, urllib.request


def post(port, prompt):
    body = {"model": "qwen", "prompt": prompt, "max_tokens": 256, "temperature": 0.0, "logprobs": 5, "seed": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ch = json.loads(urllib.request.urlopen(req, timeout=7200).read())["choices"][0]
    lp = ch.get("logprobs") or {}
    return {"text": ch["text"], "tokens": lp.get("tokens"), "token_logprobs": lp.get("token_logprobs"),
            "top_logprobs": lp.get("top_logprobs")}


def post_batch(port, prompts):
    # Una sola richiesta con lista di prompt: stesso arrivo, stessa sequenza di step tra le corse (RICERCA #130).
    body = {"model": "qwen", "prompt": prompts, "max_tokens": 256, "temperature": 0.0, "logprobs": 5, "seed": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chs = sorted(json.loads(urllib.request.urlopen(req, timeout=7200).read())["choices"], key=lambda c: c["index"])
    return [{"text": c["text"], "tokens": (c.get("logprobs") or {}).get("tokens"),
             "token_logprobs": (c.get("logprobs") or {}).get("token_logprobs"),
             "top_logprobs": (c.get("logprobs") or {}).get("top_logprobs")} for c in chs]


def run(port, out, users, paths):
    res = {}
    for i in range(0, len(paths), users):
        group = paths[i:i + users]
        if users == 1:
            res[group[0]] = post(port, open(group[0], encoding="utf-8").read())
        else:
            for p, r in zip(group, post_batch(port, [open(p, encoding="utf-8").read() for p in group])):
                res[p] = r
    json.dump(res, open(out, "w"))
    print(f"salvati {len(res)} prompt in {out}")


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ok = True
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            print(f"{k}: manca in uno dei due"); ok = False; continue
        tx, ty = x["tokens"] or list(x["text"]), y["tokens"] or list(y["text"])
        div = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), None)
        if div is None and len(tx) != len(ty):
            div = min(len(tx), len(ty))
        n = len(tx) if div is None else div
        lx, ly = x["token_logprobs"], y["token_logprobs"]
        lp_eq = lx is not None and ly is not None and lx[:n] == ly[:n] and x["top_logprobs"][:n] == y["top_logprobs"][:n]
        maxd = max((abs(p - q) for p, q in zip(lx[:n], ly[:n]) if p is not None and q is not None), default=0.0) if lx and ly else None
        same = div is None and lp_eq
        ok &= same
        print(f"{k.split('/')[-1]}: token {'identici' if div is None else f'divergono al {div}'} ({len(tx)}/{len(ty)}), "
              f"logprob {'== bit a bit' if lp_eq else f'diversi, max |d| {maxd}'}")
    print("GATE ESATTO:", "PASSATO" if ok else "FALLITO")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5:])
    else:
        compare(sys.argv[2], sys.argv[3])
