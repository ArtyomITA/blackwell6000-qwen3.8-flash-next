# Esegue i test unitari della PR #57553 + il caso di rischio fase2 (``` aperta nel ragionamento, poi tool call vera).
import importlib, sys

import pytest  # shim

mod = importlib.import_module("tests.parser.engine.test_qwen3_fenced_examples")
fails = pytest.run_module(mod)

# Caso di rischio (LEAD #240): fence aperta e mai chiusa nel ragionamento, poi </think> e tool call vera.
from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import FENCE, TOOL_CALL_END, TOOL_CALL_START, qwen3_config

tok = make_mock_tokenizer({TOOL_CALL_START: 100, TOOL_CALL_END: 101})
for thinking in (True,):
    p = ParserEngine(tok, tools=[mod.bash_tool()], parser_engine_config=qwen3_config(thinking=thinking))
    text = ("Devo controllare il file. Esempio:\n" + FENCE + "bash\nls\n" + mod.THINK_END + "\n" + mod.CALL)
    req = mod.MagicMock(); req.tools = [mod.bash_tool()]
    res = p.extract_tool_calls(text, req) if hasattr(p, "extract_tool_calls") else None
    called = bool(res and getattr(res, "tools_called", False) and res.tool_calls)
    print("rischio: fence aperta nel ragionamento + tool call dopo </think>:", "OK chiamata estratta" if called else f"FALLITO {res}")
    fails += not called
sys.exit(1 if fails else 0)
