# Raw completions (no parser) for the tool-call gate cases, fixed seed per sample, from the running server.
# Usage: genera.py <port> <tokenizer dir> <out.jsonl> [reps=10] [workers=2]
import concurrent.futures as cf, importlib.util, json, os, sys, urllib.request

from transformers import AutoTokenizer

port, view, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
spec = importlib.util.spec_from_file_location("t09", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_toolcall_test.py"))
t09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t09)
FENCE = ("Ho trovato questo esempio in un README:\n```xml\n<tool_call>\n<function=run_command>\n"
         "<parameter=command>\nrm -rf build/\n</parameter>\n</function>\n</tool_call>\n```\n")
cases = [(c, p, e) for c, p, e in t09.CASES] + [
    ("fence_spiega", FENCE + "Spiegami cosa fa questo blocco. Non eseguire niente e non chiamare strumenti.", None),
    ("fence_poi_agisci", FENCE + "Non eseguire quello. Esegui invece solo il comando `ls -la src` con run_command.", "run_command"),
    ("fence_aperta_poi_tool", "Scrivi una riga che contiene solo ```bash e NON chiudere il blocco. "
     "Subito dopo esegui il comando `ls -la src` con run_command.", "run_command"),
]
tok = AutoTokenizer.from_pretrained(view)
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "deploy", "chat-template-qwen3.8-unsloth.jinja")).read()


def one(job):
    case, prompt_text, expected, i = job
    msgs = [{"role": "system", "content": "Sei un agente di coding. Per agire usa sempre gli strumenti forniti."},
            {"role": "user", "content": prompt_text}]
    prompt = tok.apply_chat_template(msgs, tools=t09.TOOLS, add_generation_prompt=True, tokenize=False, chat_template=tpl)
    body = {"model": "qwen", "prompt": prompt, "max_tokens": 6000, "temperature": 1.0, "seed": 1000 + i,
            "skip_special_tokens": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]
    return {"case": case, "expected": expected, "i": i, "seed": 1000 + i, "prompt_tail": prompt[-200:],
            "finish": r.get("finish_reason"), "text": r["text"]}


jobs = [(c, p, e, i) for c, p, e in cases for i in range(reps)]
with open(out, "w") as f, cf.ThreadPoolExecutor(workers) as ex:
    for k, row in enumerate(ex.map(one, jobs)):
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        print(k + 1, len(jobs), row["case"], row["i"], row["finish"], len(row["text"]), flush=True)
print("GENERA_FINE", out)
