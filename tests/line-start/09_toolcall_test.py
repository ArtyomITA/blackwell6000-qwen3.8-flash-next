#!/usr/bin/env python3
"""TEST 09 - correttezza dei tool call con argomenti complessi (oggetti annidati, array di oggetti, anyOf, enum),
nello stile delle richieste di un agente di coding. Conta quanti tool call arrivano ben formati e con i tipi giusti.
Solo libreria standard. Uso: python3 09_toolcall_test.py --port 18080 --label vecchio --reps 10"""
import argparse, json, sys, urllib.error, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "apply_edits", "description": "Applica modifiche a file del progetto.",
     "parameters": {"type": "object", "required": ["edits", "options"], "properties": {
         "edits": {"type": "array", "items": {"type": "object", "required": ["path", "range", "new_text"], "properties": {
             "path": {"type": "string"},
             "range": {"type": "object", "required": ["start", "end"], "properties": {
                 "start": {"type": "integer"}, "end": {"type": "integer"}}},
             "new_text": {"type": "string"}}}},
         "options": {"type": "object", "required": ["dry_run", "mode"], "properties": {
             "dry_run": {"type": "boolean"}, "mode": {"type": "string", "enum": ["replace", "insert", "delete"]}}}}}}},
    {"type": "function", "function": {"name": "run_command", "description": "Esegue un comando nel terminale.",
     "parameters": {"type": "object", "required": ["command"], "properties": {
         "command": {"type": "string"},
         "timeout": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
         "env": {"type": "object", "additionalProperties": {"type": "string"}}}}}},
]
CASES = [
    ("edits_annidati", "Nel file src/app.py sostituisci le righe da 10 a 12 con il testo `return total * 2` e nel file "
     "tests/test_app.py sostituisci le righe da 3 a 3 con `assert run() == 4`. Usa mode replace e dry_run false. "
     "Fai una sola chiamata a apply_edits con entrambe le modifiche.", "apply_edits"),
    ("anyof_e_mappa", "Esegui il comando `pytest -q tests/` con timeout 120 e variabili d'ambiente CI=1 e PYTHONPATH=src.",
     "run_command"),
    ("anyof_null", "Esegui il comando `git status --short` senza timeout (timeout nullo) e senza variabili d'ambiente.",
     "run_command"),
]


def check(case, name, args):
    errs = []
    if not isinstance(args, dict):
        return ["argomenti non sono un oggetto"]
    if name == "apply_edits":
        e = args.get("edits"); o = args.get("options")
        if not isinstance(e, list): errs.append("edits non e una lista (%s)" % type(e).__name__)
        else:
            if case == "edits_annidati" and len(e) != 2: errs.append("edits ha %d elementi invece di 2" % len(e))
            for it in e:
                if not isinstance(it, dict): errs.append("elemento di edits non e un oggetto"); continue
                r = it.get("range")
                if not isinstance(r, dict): errs.append("range non e un oggetto (%s)" % type(r).__name__)
                elif not all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool) for k in ("start", "end")):
                    errs.append("range.start/end non interi")
        if not isinstance(o, dict): errs.append("options non e un oggetto (%s)" % type(o).__name__)
        else:
            if not isinstance(o.get("dry_run"), bool): errs.append("dry_run non booleano")
            if o.get("mode") not in ("replace", "insert", "delete"): errs.append("mode fuori enum")
    if name == "run_command":
        if not isinstance(args.get("command"), str): errs.append("command non stringa")
        t = args.get("timeout", None)
        if case == "anyof_e_mappa":
            if not (isinstance(t, int) and not isinstance(t, bool)): errs.append("timeout non intero (%r)" % (t,))
            env = args.get("env")
            if not isinstance(env, dict): errs.append("env non e un oggetto (%s)" % type(env).__name__)
            elif env.get("CI") != "1": errs.append("env.CI diverso da '1' (%r)" % (env.get("CI"),))
        if case == "anyof_null" and t is not None: errs.append("timeout doveva essere null (%r)" % (t,))
    return errs


def call(a, prompt):
    body = {"model": "qwen", "stream": False, "max_tokens": 6000, "temperature": 1.0, "tools": TOOLS,
            "tool_choice": "auto", "messages": [
                {"role": "system", "content": "Sei un agente di coding. Per agire usa sempre gli strumenti forniti."},
                {"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://{a.host}:{a.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"errore_http": e.code, "msg": e.read().decode("utf-8", "replace")[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--label", default=""); ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    for case, prompt, expected in CASES:
        ok = 0; problems = {}
        for _ in range(a.reps):
            d = call(a, prompt)
            if "errore_http" in d:
                k = "HTTP %s: %s" % (d["errore_http"], d["msg"][:120]); problems[k] = problems.get(k, 0) + 1; continue
            m = d["choices"][0]["message"]; tcs = m.get("tool_calls") or []
            if not tcs:
                k = "nessun tool_call (finish=%s, testo=%r)" % (d["choices"][0].get("finish_reason"), (m.get("content") or "")[:80])
                problems[k] = problems.get(k, 0) + 1; continue
            tc = tcs[0]["function"]
            try: args = json.loads(tc.get("arguments") or "{}")
            except Exception: problems["arguments non e JSON valido"] = problems.get("arguments non e JSON valido", 0) + 1; continue
            errs = ([] if tc.get("name") == expected else ["strumento sbagliato: %s" % tc.get("name")]) + check(case, tc.get("name"), args)
            if errs:
                for e in errs: problems[e] = problems.get(e, 0) + 1
            else: ok += 1
        print(json.dumps({"label": a.label, "caso": case, "corretti": ok, "su": a.reps, "problemi": problems}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
