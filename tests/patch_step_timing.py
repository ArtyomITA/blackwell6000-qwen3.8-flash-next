# Applica la patch B.3 (ms/step per ogni step reale) al worktree fase2.
# Attivazione: VLLM_STEP_TIMING_FILE=<percorso csv>. Spenta di default: senza la variabile il codice si comporta come prima.
# Scrive file nuovi e poi mv (inode nuovo), come chiesto dal DOSSIER.
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmunity-models/vllm-fase2"


def patch(rel, pairs):
    path = os.path.join(root, rel)
    src = open(path).read()
    for old, new in pairs:
        assert src.count(old) == 1, (rel, old[:60], src.count(old))
        src = src.replace(old, new)
    tmp = path + ".new"
    open(tmp, "w").write(src)
    os.replace(tmp, path)


patch("vllm/v1/worker/gpu/async_utils.py", [
    ("import contextlib\n", "import collections\nimport contextlib\nimport os\nimport time\n"),
    (
        "    def __init__(self):\n        self._collecting = False\n",
        "    def __init__(self):\n"
        "        # fase2 B.3: con VLLM_STEP_TIMING_FILE ogni step reale finisce in un CSV\n"
        "        # (t, full_cg, num_tokens, num_reqs, forward_ms, drafter_ms, period_ms).\n"
        "        # ponytail: in questo modo collect() non riceve gli step (adaptive verification spenta da noi).\n"
        "        path = os.environ.get(\"VLLM_STEP_TIMING_FILE\")\n"
        "        self._file = open(path, \"a\", buffering=1) if path else None\n"
        "        self._pending: collections.deque = collections.deque()\n"
        "        self._collecting = self._file is not None\n",
    ),
    (
        "        finally:\n            self._collecting = False\n",
        "        finally:\n            self._collecting = self._file is not None\n",
    ),
    (
        "            self._step.drafter_end.record()\n            self._timed.append((self._step, self._batch))\n            self._step = None\n",
        "            self._step.drafter_end.record()\n"
        "            if self._file is None:\n"
        "                self._timed.append((self._step, self._batch))\n"
        "            else:\n"
        "                self._pending.append((self._step, self._batch))\n"
        "                self._drain()\n"
        "            self._step = None\n"
        "\n"
        "    def _drain(self) -> None:\n"
        "        # Senza sync: uno step si scrive quando l'inizio dello step dopo e' gia' passato sulla GPU.\n"
        "        q = self._pending\n"
        "        while len(q) >= 2 and q[1][0].forward_start.query() and q[0][0].drafter_end.query():\n"
        "            ev, (cg, ntok, nreq) = q.popleft()\n"
        "            self._file.write(\n"
        "                f\"{time.time():.3f},{int(cg)},{ntok},{nreq},\"\n"
        "                f\"{ev.forward_start.elapsed_time(ev.forward_end):.3f},\"\n"
        "                f\"{ev.drafter_start.elapsed_time(ev.drafter_end):.3f},\"\n"
        "                f\"{ev.forward_start.elapsed_time(q[0][0].forward_start):.3f}\\n\"\n"
        "            )\n",
    ),
])

patch("vllm/v1/worker/gpu/model_runner.py", [
    (
        "        self.kv_connector.finish_forward()\n\n        if self.is_last_pp_rank:\n",
        "        self.kv_connector.finish_forward()\n        self.step_timing.forward_end()\n\n        if self.is_last_pp_rank:\n",
    ),
    (
        "            with use_workspace_lane(self._draft_workspace_lane):\n                draft_tokens = self.speculator.propose(\n",
        "            self.step_timing.drafter_start()\n"
        "            with use_workspace_lane(self._draft_workspace_lane):\n                draft_tokens = self.speculator.propose(\n",
    ),
    (
        "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
        "            self.step_timing.drafter_end()\n"
        "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
    ),
])
print("patch B.3 applicata in", root)
