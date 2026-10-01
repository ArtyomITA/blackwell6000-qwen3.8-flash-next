# Testo GREZZO delle risposte a fence_spiega (stesso prompt di toolcall_gate, stesso template e stessi tools), via
# /v1/completions: dove scrive il modello <tool_call>/<function=...>: dentro o fuori dai ```? 10 campioni.
# Uso: probe_fence.py <porta> <vista> <out.jsonl>
import importlib.util, json, re, sys, urllib.request

from transformers import AutoTokenizer

port, view, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("tg", "/mnt/llmunity-models/prod-tests/09_toolcall_test.py")
t09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t09)
FENCE = ("Ho trovato questo esempio in un README:\n```xml\n<tool_call>\n<function=run_command>\n"
         "<parameter=command>\nrm -rf build/\n</parameter>\n</function>\n</tool_call>\n```\n")
msgs = [{"role": "system", "content": "Sei un agente di coding. Per agire usa sempre gli strumenti forniti."},
        {"role": "user", "content": FENCE + "Spiegami cosa fa questo blocco. Non eseguire niente e non chiamare strumenti."}]
tok = AutoTokenizer.from_pretrained(view)
tpl = open("/mnt/llmunity-models/prod-tests/chat-template-qwen3.8-unsloth.jinja").read()
prompt = tok.apply_chat_template(msgs, tools=t09.TOOLS, add_generation_prompt=True, tokenize=False, chat_template=tpl)
dentro = fuori = 0
with open(out, "w") as f:
    for i in range(10):
        body = {"model": "qwen", "prompt": prompt, "max_tokens": 6000, "temperature": 1.0, "skip_special_tokens": False}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        txt = json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["text"]
        f.write(json.dumps({"i": i, "text": txt}, ensure_ascii=False) + "\n")
        answer = txt.split("</think>")[-1]
        for m in re.finditer(r"<tool_call>|<function=", answer):
            # dentro un fence se il numero di ``` prima della posizione e' dispari
            if answer[:m.start()].count("```") % 2:
                dentro += 1
            else:
                fuori += 1
print(f"PROBE_FENCE: marcatori tool nella risposta (dopo </think>) dentro ``` = {dentro}, fuori = {fuori}; testi in {out}")
