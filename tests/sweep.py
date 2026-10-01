# B.4: sweep decode puro. Per ogni contesto L e numero di utenti U manda U richieste insieme su prompt diversi
# (c196k_a-d troncati a L token, prefill gia' in prefix cache), poi legge il CSV della patch B.3 (VLLM_STEP_TIMING_FILE)
# nella finestra della prova: ms/step mediani (target, draft, periodo) degli step FULL graph con U richieste.
# Uso: sweep.py <porta> <csv_step> <vista> <L1,L2,...> <U1,U2,...> [max_tokens=256] [temp=1.0]
import json, statistics, sys, threading, time, urllib.request

from transformers import AutoTokenizer

port, csv_path, view = int(sys.argv[1]), sys.argv[2], sys.argv[3]
Ls = [int(x) for x in sys.argv[4].split(",")]
Us = [int(x) for x in sys.argv[5].split(",")]
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 256
temp = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
C = "/mnt/llmunity-models/prod-tests/convs"
tok = AutoTokenizer.from_pretrained(view)
convs = [tok.encode(open(f"{C}/c196k_{x}.txt", encoding="utf-8").read(), add_special_tokens=False) for x in "abcd"]


def post(ids, n, out, i):
    body = {"model": "qwen", "prompt": ids, "max_tokens": n, "temperature": temp, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; k = 0
    for line in urllib.request.urlopen(req, timeout=7200):
        line = line.decode().strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            first = first or time.time(); k += 1
    out[i] = (n - 1) / (time.time() - first) if first else 0.0


def steps(t0, t1, u):
    rows = []
    for line in open(csv_path):
        t, cg, ntok, nreq, fw, dr, per = line.strip().split(",")
        if t0 <= float(t) <= t1 and cg == "1" and int(nreq) == u:
            rows.append((int(ntok), float(fw), float(dr), float(per)))
    return rows


print("L,U,steps,tok_per_req_step,fwd_ms,draft_ms,period_ms,gap_ms,tok_s_client_med,tok_s_from_steps")
for L in Ls:
    prompts = [c[:L] for c in convs]
    assert all(len(p) == L for p in prompts), "conversazione piu' corta di L"
    for p in prompts:  # prefill in prefix cache, uno alla volta
        post(p, 1, {}, 0)
    for u in Us:
        out = {}
        t0 = time.time()
        th = [threading.Thread(target=post, args=(prompts[i], max_tokens, out, i)) for i in range(u)]
        [t.start() for t in th]; [t.join() for t in th]
        time.sleep(2)  # gli ultimi step si scrivono quando parte lo step dopo
        post(prompts[0][:64], 8, {}, 0)
        time.sleep(1)
        r = steps(t0, time.time(), u)
        if not r:
            print(f"{L},{u},0,,,,,,{statistics.median(out.values()):.1f},"); continue
        fw = statistics.median(x[1] for x in r); dr = statistics.median(x[2] for x in r)
        per = statistics.median(x[3] for x in r)
        tpr = u * max_tokens / len(r) / u  # token accettati per richiesta per step (circa)
        print(f"{L},{u},{len(r)},{tpr:.2f},{fw:.2f},{dr:.2f},{per:.2f},{per - fw - dr:.2f},"
              f"{statistics.median(out.values()):.1f},{1000 * tpr / per:.1f}", flush=True)
print("SWEEP_FINE")
