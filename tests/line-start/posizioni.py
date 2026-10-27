# Where does the model write <tool_call>? reasoning vs answer, line start vs mid-line, inside ``` or not.
import collections, json, re, sys

c = collections.Counter()
esempi = []
for l in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(l)
    t = r["text"]
    end = t.find("</think>")
    for m in re.finditer(r"<tool_call>", t):
        i = m.start()
        where = "reasoning" if end == -1 or i < end else "answer"
        before = t[:i]
        start = before == "" or before.endswith("\n") or before.endswith("</think>")
        seg = t[end + 8:i] if where == "answer" else t[:i]
        fence = seg.count("```") % 2 == 1
        key = (r["case"], where, "line_start" if start else "mid_line", "in_fence" if fence else "no_fence")
        c[key] += 1
        if not start and len(esempi) < 12:
            esempi.append((r["case"], r["i"], where, repr(t[max(0, i - 50):i + 12])))
for k, v in sorted(c.items()):
    print(v, *k)
print("--- mid-line examples")
for e in esempi:
    print(*e)
