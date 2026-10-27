# Phantom tool calls and a line-start rule (Qwen3 parser in vLLM)

By **Adrian Ciprian Iancu**. Data for the discussion in [vllm-project/vllm#57553](https://github.com/vllm-project/vllm/pull/57553): when the model explains a tool-call example, the Qwen3 parser in vLLM turns the quoted markup into real tool calls.

## What is here

| file | what |
|---|---|
| `09_toolcall_test.py` | the tools and the three plain tool-call cases used by the production tests |
| `genera.py` | 60 raw completions (no parser) from the running server: 6 cases x 10, temperature 1.0, fixed seed per sample, same chat template and tools as the production tests |
| `raw_completions.jsonl` | the 60 texts, generated on 27 Oct 2026 by Swift 1.5 Qwen3.8 Flash-Next NVFP4 on one RTX PRO 6000 Blackwell (sm_120) |
| `valuta.py` | parses the same texts offline (non-streaming `ParserEngine`, qwen3 config with thinking) with one vLLM source tree; `linestart` simulates a rule where a call may only open at the start of a line, for both `<tool_call>` and `<function=` |
| `posizioni.py` | where the model writes `<tool_call>`: reasoning or answer, start of line or mid-line, inside a code fence or not |

## Results

| case (10 runs each) | main e3883b7 | #57553 a682a8b | #59821 8a9dfb8 | main + line-start rule |
|---|---|---|---|---|
| explain a fenced tool-call example, no tools (`fence_spiega`) | 1/10, 14 phantom calls | 1/10, 14 | 1/10, 14 | 8/10, 0 |
| fenced example, then run `ls -la src` (`fence_poi_agisci`) | 10/10 | 10/10 | 10/10 | 10/10 |
| open ```bash left unclosed, then a real call (`fence_aperta_poi_tool`) | 10/10 | 0/10 | 10/10 | 10/10 |
| two plain tool-call cases (`anyof_e_mappa`, `anyof_null`) | 20/20 | 20/20 | 20/20 | 20/20 |

- All 51 real calls start at the beginning of a line; all 27 quoted `<tool_call>` are mid-line (13 in the reasoning, 14 in the answer).
- The parser also opens a call on a bare `<function=` quoted mid-line, so the rule must cover both openers: with `<tool_call>` alone 8 phantom calls remained.
- `edits_annidati` is left out of the comparison: the model first calls `run_command` to read the files in 7 of 10 runs, with every parser.
- The two remaining `fence_spiega` failures with the rule are not tool calls: the answer did not quote the command.
- `fence_aperta_poi_tool` 0/10 with #57553 is its intended behaviour (an unclosed fence keeps what follows as text): in all 10 runs the real call came after the open fence. The PR's first commit e24c0af gives the same 0/10 on these texts.

## Run

```bash
python genera.py 30002 <tokenizer dir> raw_completions.jsonl 10 2
cd <vllm checkout> && PYTHONPATH=. python ../valuta.py ../raw_completions.jsonl main
cd <vllm checkout> && PYTHONPATH=. python ../valuta.py ../raw_completions.jsonl main+linestart linestart
python posizioni.py raw_completions.jsonl
```

`genera.py` and `valuta.py` read the test cases and tools from `09_toolcall_test.py` in this folder and the chat template from `deploy/`.
