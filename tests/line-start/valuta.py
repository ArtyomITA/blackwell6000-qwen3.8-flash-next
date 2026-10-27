# Parse the same raw completions with one vLLM source tree (non-streaming and streaming), CPU only.
# Usage: PYTHONPATH=<src> valuta.py <grezzi.jsonl> <label> [linestart]
#   linestart: simulate "a tool call may only start at the beginning of a line" by breaking every
#   <tool_call> that is not at line start (or right after </think>) before parsing.
import importlib.util, json, os, re, sys, types
from unittest.mock import MagicMock

_pytest = types.ModuleType("pytest")  # conftest only needs the decorator; the production venv has no pytest
_pytest.fixture = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)
sys.modules.setdefault("pytest", _pytest)

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import TOOL_CALL_END, TOOL_CALL_START, qwen3_config

path, label = sys.argv[1], sys.argv[2]
linestart = len(sys.argv) > 3 and sys.argv[3] == "linestart"
SCOPE = sys.argv[4] if len(sys.argv) > 4 else "all"  # all | reasoning | answer: where the rule applies
spec = importlib.util.spec_from_file_location("t09", os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_toolcall_test.py"))
t09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t09)
tools = []
for t in t09.TOOLS:
    m = MagicMock(); m.function.name = t["function"]["name"]; tools.append(m)
req = MagicMock(); req.tools = tools
tok = make_mock_tokenizer({TOOL_CALL_START: 100, TOOL_CALL_END: 101})
THINK_END = "</think>"


def rompi(text):
    out, last = [], 0
    for m in re.finditer(re.escape(TOOL_CALL_START) + "|<function=", text):  # both openers start a call
        before = text[:m.start()]
        if before.endswith("\n") or before.endswith(THINK_END) or before == "":
            continue
        in_reasoning = THINK_END not in before
        if (SCOPE == "reasoning" and not in_reasoning) or (SCOPE == "answer" and in_reasoning):
            continue
        out.append(text[last:m.start()] + m.group(0)[:-1] + "​" + m.group(0)[-1])
        last = m.end()
    return "".join(out) + text[last:]


def parse(text):
    p = ParserEngine(tok, tools=tools, parser_engine_config=qwen3_config(thinking=True))
    r = p.extract_tool_calls(text, req)
    calls = []
    for c in r.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except ValueError:
            args = {"_raw": c.function.arguments}
        calls.append({"name": c.function.name, "args": args})
    return calls, r.content or ""


righe = [json.loads(l) for l in open(path, encoding="utf-8")]
per_case = {}
for r in righe:
    text = rompi(r["text"]) if linestart else r["text"]
    try:
        calls, content = parse(text)
        err = None
    except Exception as e:  # report, do not hide
        calls, content, err = [], "", f"{type(e).__name__}: {e}"[:200]
    exp = r["expected"]
    if exp is None:
        ok = not calls and "rm -rf" in content
        phantom = len(calls)
        lost = 0
    else:
        good = [c for c in calls if c["name"] == exp]
        ok = bool(good) and not err
        if r["case"] in ("fence_poi_agisci", "fence_aperta_poi_tool"):
            ok = ok and str(good[0]["args"].get("command", "")).strip() == "ls -la src"
        phantom = len([c for c in calls if c["name"] != exp])
        lost = 0 if good else 1
    s = per_case.setdefault(r["case"], {"ok": 0, "n": 0, "phantom_calls": 0, "real_call_lost": 0, "errors": 0})
    s["n"] += 1; s["ok"] += ok; s["phantom_calls"] += phantom; s["real_call_lost"] += lost; s["errors"] += err is not None
tot = sum(s["ok"] for s in per_case.values())
print(json.dumps({"label": label, "linestart": linestart, "total_ok": tot, "n": len(righe), "per_case": per_case}))
