# Kernel #55122 (_det_topk.persistent_topk) contro overlay Python (_det_topk del worktree): indici identici?
# Logit con molti pari merito, larghezze diverse (percorsi decode/medio/radix), visible casuali, 16 e 1024 righe.
import os, sys, time

import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import _TOPK_WORKSPACE_BYTES, _det_topk

torch.ops.load_library(sys.argv[1])
torch.manual_seed(0)
K, ok = 512, True
ws = torch.empty(_TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
for rows in (16, 1024):
    for n in (300, 8192, 30000, 61440):
        for ties in (True, False):
            lg = (torch.randint(0, 64, (rows, n), device="cuda").float() if ties
                  else torch.randn(rows, n, device="cuda"))
            vis = torch.randint(1, n + 1, (rows,), device="cuda", dtype=torch.int32); vis[0] = n
            a = torch.empty(rows, K, dtype=torch.int32, device="cuda"); b = torch.empty_like(a)
            _det_topk(lg, vis, K, a)
            torch.ops._det_topk.persistent_topk(lg, vis, b, ws, K, n)
            same = torch.equal(a, b); ok &= same
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5): _det_topk(lg, vis, K, a)
            torch.cuda.synchronize(); t1 = time.time()
            for _ in range(5): torch.ops._det_topk.persistent_topk(lg, vis, b, ws, K, n)
            torch.cuda.synchronize(); t2 = time.time()
            print(f"righe {rows} n {n} pari_merito {ties}: {'IDENTICI' if same else 'DIVERSI ' + str((a != b).any(1).sum().item()) + ' righe'}"
                  f" | overlay {(t1 - t) / 5 * 1e3:.2f} ms, kernel {(t2 - t1) / 5 * 1e3:.2f} ms")
print("TOPK kernel == overlay:", "PASSATO" if ok else "FALLITO")
sys.exit(0 if ok else 1)
