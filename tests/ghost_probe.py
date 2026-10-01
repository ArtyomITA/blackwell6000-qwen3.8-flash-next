# Prova CPU (RICERCA #299): una <tool_call> scritta DENTRO il ragionamento (senza fence) diventa una chiamata vera?
# Con e senza la transizione implicita (REASONING, TOOL_START) del parser Qwen3 del fork.
import dataclasses
from unittest.mock import MagicMock

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.qwen3 import TOOL_CALL_END, TOOL_CALL_START, qwen3_config

THINK_END = "<" + "/think" + ">"
CALL = ("<tool_call>\n<function=run_command>\n<parameter=command>rm -rf build</parameter>\n</function>\n</tool_call>")
cases = {
    "tool_call nel ragionamento, poi spiegazione": "Il blocco mostra " + CALL + " come esempio di chiamata." + THINK_END + "\nSpiegazione: il comando cancella build.",
    "chiamata vera dopo </think>": "Devo eseguire il comando." + THINK_END + "\n" + CALL,
}
tok = make_mock_tokenizer({TOOL_CALL_START: 100, TOOL_CALL_END: 101})
tool = MagicMock(); tool.function.name = "run_command"
req = MagicMock(); req.tools = [tool]
for implicit in (True, False):
    cfg = qwen3_config(thinking=True)
    if not implicit:
        trans = {k: v for k, v in cfg.transitions.items()
                 if not (k[0] == ParserState.REASONING and v.next_state in (ParserState.TOOL_PREAMBLE, ParserState.TOOL_NAME))}
        cfg = dataclasses.replace(cfg, transitions=trans)
    for name, text in cases.items():
        p = ParserEngine(tok, tools=[tool], parser_engine_config=cfg)
        r = p.extract_tool_calls(text, req)
        names = [c.function.name for c in (r.tool_calls or [])]
        print(f"transizione implicita {'SI' if implicit else 'NO'} | {name}: tool_calls={names} content={(r.content or '')[:60]!r}")
