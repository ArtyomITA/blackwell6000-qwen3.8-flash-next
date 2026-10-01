# Come ghost_probe.py ma con la env VLLM_QWEN3_NO_REASONING_TOOL letta dal codice del worktree.
import os
from unittest.mock import MagicMock

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import TOOL_CALL_END, TOOL_CALL_START, qwen3_config

THINK_END = "<" + "/think" + ">"
CALL = ("<tool_call>\n<function=run_command>\n<parameter=command>rm -rf build</parameter>\n</function>\n</tool_call>")
cases = {
    "tool_call nel ragionamento, poi spiegazione": "Il blocco mostra " + CALL + " come esempio." + THINK_END + "\nSpiegazione.",
    "chiamata vera dopo </think>": "Devo eseguire il comando." + THINK_END + "\n" + CALL,
    "chiamata vera senza </think> (modello che non chiude)": "Devo eseguire il comando.\n" + CALL,
}
tok = make_mock_tokenizer({TOOL_CALL_START: 100, TOOL_CALL_END: 101})
tool = MagicMock(); tool.function.name = "run_command"
req = MagicMock(); req.tools = [tool]
for name, text in cases.items():
    r = ParserEngine(tok, tools=[tool], parser_engine_config=qwen3_config(thinking=True)).extract_tool_calls(text, req)
    print(f"env={os.environ.get('VLLM_QWEN3_NO_REASONING_TOOL', '0')} | {name}: tool_calls={[c.function.name for c in (r.tool_calls or [])]}")
