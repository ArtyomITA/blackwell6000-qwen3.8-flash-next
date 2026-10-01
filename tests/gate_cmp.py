# Confronto di due uscite di gate_gen.py (A = riferimento, B = candidato).
# Per prompt: token identici? top-20 logprob identici bit a bit? prima divergenza greedy, margine top1-top2 di A
# in quel punto, KLD(A||B) mediata sul prefisso comune (top-20 rinormalizzati, unione delle chiavi con floor).
# Uso: gate_cmp.py <A.json> <B.json>
import json, math, sys

A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))


def kld(p, q, floor=-30.0):
    keys = set(p) | set(q)
    pa = {k: math.exp(p.get(k, floor)) for k in keys}
    qa = {k: math.exp(q.get(k, floor)) for k in keys}
    sp, sq = sum(pa.values()), sum(qa.values())
    return sum(pa[k] / sp * (math.log(pa[k] / sp) - math.log(qa[k] / sq)) for k in keys if pa[k] > 0)


tot_bit = True
for name in A:
    a, b = A[name], B.get(name)
    if b is None:
        print(f"{name}: manca in B"); tot_bit = False; continue
    n = min(len(a["tokens"]), len(b["tokens"]))
    div = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
    same_tokens = div is None and len(a["tokens"]) == len(b["tokens"])
    common = n if div is None else div
    bit = same_tokens and all(a["top"][i] == b["top"][i] for i in range(n))
    tot_bit &= bit
    ks = [kld(a["top"][i], b["top"][i]) for i in range(common)]
    first_lp = next((i for i in range(common) if a["top"][i] != b["top"][i]), None)
    msg = f"{name} ({a['prompt_len']} tok): {'BIT A BIT' if bit else 'diversi'}, token comuni {common}/{len(a['tokens'])}"
    if first_lp is not None:
        msg += f", primo logprob diverso al token {first_lp}"
    if div is not None:
        top = sorted(a["top"][div].values(), reverse=True)
        msg += f", prima divergenza greedy al token {div}, margine top1-top2 di A {top[0] - top[1]:.4f}"
    if ks:
        msg += f", KLD media {sum(ks) / len(ks):.2e} max {max(ks):.2e}"
    print(msg)
print("ESITO:", "BIT A BIT su tutti i prompt" if tot_bit else "NON bit a bit")
