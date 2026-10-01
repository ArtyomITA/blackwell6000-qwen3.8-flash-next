# Self-test overlay: confronto con riferimento lento (sort stabile completo) su casi con pari merito e visible < k.
import torch
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import _det_topk

def ref(logits, vis, k):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    for r in range(logits.shape[0]):
        v = int(vis[r]); row = logits[r, :v].double().cpu()
        if v <= k:
            out[r, :v] = torch.arange(v, dtype=torch.int32); continue
        order = sorted(range(v), key=lambda c: (-row[c].item(), c))[:k]
        out[r] = torch.tensor(sorted(order), dtype=torch.int32)
    return out

import sys
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda"
torch.manual_seed(0)
for rows, n, k in ((16, 3000, 512), (40, 700, 512), (5, 300, 512)):
    lg = torch.randint(0, 50, (rows, n), device=DEV).float()   # tanti pari merito
    vis = torch.randint(1, n + 1, (rows,), device=DEV, dtype=torch.int32); vis[0] = n
    bi = torch.empty(rows, k, dtype=torch.int32, device=DEV)
    _det_topk(lg, vis, k, bi)
    a = bi.cpu(); b = ref(lg.cpu(), vis.cpu(), k)
    assert torch.equal(a, b), (rows, n, k, (a != b).sum())
    bi2 = torch.empty_like(bi); _det_topk(lg, vis, k, bi2); assert torch.equal(bi, bi2)
print("topk overlay self-test OK")
