# Gate op C.2: embedding BF16 reale (embed_tokens del checkpoint) letta via UVA da RAM pinned contro la stessa in GPU,
# stesso modulo VocabParallelEmbedding/F.embedding. 10^6 id casuali + bordi. Bit a bit. GPU sotto flock.
import json, sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

view = "/mnt/llmunity-models/d0xin-fp8ple-ramview"
name = "model.language_model.embed_tokens.weight"
fname = json.load(open(f"{view}/model.safetensors.index.json"))["weight_map"][name]
with safe_open(f"{view}/{fname}", "pt") as f:
    w = f.get_tensor(name)
print("embed", tuple(w.shape), w.dtype)
gpu = w.cuda()
uva = get_accelerator_view_from_cpu_tensor(w.pin_memory())
del w
g = torch.Generator(device="cuda").manual_seed(0)
ok = True
for n in (16, 4096, 1_000_000):
    ids = torch.randint(0, gpu.shape[0], (n,), device="cuda", generator=g)
    ids[:2] = torch.tensor([0, gpu.shape[0] - 1], device="cuda")
    a, b = F.embedding(ids, gpu), F.embedding(ids, uva)
    same = torch.equal(a, b); ok &= same
    print(f"{n} id: {'bit a bit' if same else 'DIVERSI'}")
print("GATE C.2 op:", "PASSATO" if ok else "FALLITO")
sys.exit(0 if ok else 1)
