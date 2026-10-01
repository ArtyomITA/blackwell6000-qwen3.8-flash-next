# B.8: replay stile agent. 4 utenti partono da 60/90/120/150k token (c196k_a-d), poi 8 turni ciascuno:
# +1-6k token di "risultato tool" (pezzi di c240k), 400 token generati a temp 1.0. echo=1 rimanda il testo generato
# (ragionamento compreso) nel turno dopo, echo=0 lo sostituisce con un ragionamento vuoto (Copilot senza cot_id).
# Dal CSV della patch B.3: quota di step misti (numero e tempo), ms/step; per turno TTFT e tok/s in decode.
# Uso: agent_replay.py <porta> <csv_step> <vista> <echo 0|1> [turni=8]
import json, random, statistics, sys, threading, time, urllib.request

from transformers import AutoTokenizer

port, csv_path, view, echo = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
turns = int(sys.argv[5]) if len(sys.argv) > 5 else 8
C = "/mnt/llmunity-models/prod-tests/convs"
tok = AutoTokenizer.from_pretrained(view)
pool = tok.encode(open(f"{C}/c240k.txt", encoding="utf-8").read(), add_special_tokens=False)
starts = {x: tok.decode(tok.encode(open(f"{C}/c196k_{x}.txt", encoding="utf-8").read(), add_special_tokens=False)[:L])
          for x, L in zip("abcd", (60000, 90000, 120000, 150000))}
rows = []
lock = threading.Lock()


def post(prompt):
    body = {"model": "qwen", "prompt": prompt, "max_tokens": 400, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = {}
    for line in urllib.request.urlopen(req, timeout=7200):
        line = line.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        j = json.loads(line[5:])
        usage = j.get("usage") or usage
        for ch in j.get("choices", []):
            if ch.get("text"):
                first = first or time.time(); text += ch["text"]
    end = time.time(); n = usage.get("completion_tokens", 0)
    return text, usage.get("prompt_tokens"), (first or end) - t0, (n - 1) / (end - first) if first and end > first else 0


def user(x, seed):
    rnd = random.Random(seed)
    prompt = starts[x]
    for t in range(turns):
        text, ptok, ttft, tps = post(prompt)
        with lock:
            rows.append((x, t, ptok, ttft, tps))
        n = rnd.randint(1000, 6000); o = rnd.randint(0, len(pool) - n)
        tool = tok.decode(pool[o:o + n])
        said = text if echo else "\n</think>\n\n"
        prompt += (said + "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + tool +
                   "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")


t0 = time.time()
th = [threading.Thread(target=user, args=(x, i)) for i, x in enumerate("abcd")]
[t.start() for t in th]; [t.join() for t in th]
time.sleep(2); t1 = time.time()
print(f"echo={int(echo)} durata {t1 - t0:.0f} s")
print("utente,turno,prompt_tok,ttft_s,tok_s")
for r in sorted(rows):
    print(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f},{r[4]:.1f}")
print(f"TTFT mediano turni>=1 {statistics.median(r[3] for r in rows if r[1] >= 1):.2f} s, "
      f"tok/s mediano {statistics.median(r[4] for r in rows):.1f}")
mix = pure = 0; tmix = tpure = 0.0
for line in open(csv_path):
    t, cg, ntok, nreq, fw, dr, per = line.strip().split(",")
    if t0 <= float(t) <= t1 and float(per) < 1000:
        if cg == "1":
            pure += 1; tpure += float(per)
        else:
            mix += 1; tmix += float(per)
if mix + pure:
    print(f"step puri {pure} ({tpure / 1000:.1f} s), misti {mix} ({tmix / 1000:.1f} s): misti = {100 * mix / (mix + pure):.1f}% "
          f"degli step, {100 * tmix / (tmix + tpure):.1f}% del tempo GPU")
print("AGENT_FINE")
