# Gate tool call (LEAD #39): i 3 casi di 09_toolcall_test.py + 2 casi "sintassi tool dentro ```" (vLLM #57541).
# fence_spiega: blocco ``` con <function=...> da spiegare: nessun tool call atteso, il testo deve citare il blocco.
# fence_poi_agisci: stesso blocco, poi "esegui solo `ls -la src`": un run_command con quel comando.
# Uso: toolcall_gate.py <porta> <etichetta> [reps=10]   (5 casi x reps chiamate; riusa TOOLS/check/call di 09)
import importlib.util, json, os, sys
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("t09", "/mnt/llmunity-models/prod-tests/09_toolcall_test.py")
t09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t09)
a = SimpleNamespace(host="127.0.0.1", port=int(sys.argv[1]))
label = sys.argv[2]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
FENCE = ("Ho trovato questo esempio in un README:\n```xml\n<tool_call>\n<function=run_command>\n"
         "<parameter=command>\nrm -rf build/\n</parameter>\n</function>\n</tool_call>\n```\n")
cases = t09.CASES + [
    ("fence_spiega", FENCE + "Spiegami cosa fa questo blocco. Non eseguire niente e non chiamare strumenti.", None),
    ("fence_poi_agisci", FENCE + "Non eseguire quello. Esegui invece solo il comando `ls -la src` con run_command.", "run_command"),
]
if os.environ.get("TOOLCALL_RISK"):  # rischio del fix #57553: fence aperta e mai chiusa prima di una tool call vera
    cases.append(("fence_aperta_poi_tool", "Scrivi una riga che contiene solo ```bash e NON chiudere il blocco. "
                  "Subito dopo esegui il comando `ls -la src` con run_command.", "run_command"))
tot = 0
for case, prompt, expected in cases:
    ok = 0; problems = {}
    for _ in range(reps):
        d = t09.call(a, prompt)
        if "errore_http" in d:
            errs = ["HTTP %s" % d["errore_http"]]
        else:
            m = d["choices"][0]["message"]; tcs = m.get("tool_calls") or []; txt = m.get("content") or ""
            if expected is None:
                errs = (["tool call fantasma: %s" % tcs[0]["function"].get("name")] if tcs else []) + \
                       ([] if "rm -rf" in txt else ["testo senza il blocco (%r)" % txt[:60]])
            elif not tcs:
                errs = ["nessun tool_call (finish=%s)" % d["choices"][0].get("finish_reason")]
            else:
                tc = tcs[0]["function"]
                try:
                    args = json.loads(tc.get("arguments") or "{}")
                    errs = ([] if tc.get("name") == expected else ["strumento sbagliato: %s" % tc.get("name")]) + \
                           t09.check(case, tc.get("name"), args)
                    if case in ("fence_poi_agisci", "fence_aperta_poi_tool") and str(args.get("command", "")).strip() != "ls -la src":
                        errs.append("command %r" % args.get("command"))
                except ValueError:
                    errs = ["arguments non JSON"]
        for e in errs:
            problems[e] = problems.get(e, 0) + 1
        ok += not errs
    tot += ok
    print(json.dumps({"label": label, "caso": case, "corretti": ok, "su": reps, "problemi": problems}, ensure_ascii=False), flush=True)
print(f"TOOLCALL_GATE {label}: {tot}/{reps * len(cases)} corretti")
