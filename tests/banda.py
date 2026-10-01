# B.2: banda DRAM GPU (copia D2D), host<->GPU (pinned) e gather sparso di righe 512 B / 1 KB
# con l'op hisparse_gather_compact gia' compilata nel fork. Uso: python banda.py [host_gib=2]
# Va lanciato sotto flock /mnt/llmunity-models/gpu.lock (usa ~6 GB di GPU e host_gib di RAM pinned).
import sys

import torch

import vllm._custom_ops  # noqa: F401  carica le librerie con le op _C_cache_ops

dev = torch.device("cuda")
host_gib = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps  # ms per chiamata


print(torch.cuda.get_device_name(), "| torch", torch.__version__, "| host pinned GiB", host_gib)

# 1) D2D: 2 GiB letti + 2 GiB scritti per copia.
n = 2 << 30
src = torch.empty(n, dtype=torch.uint8, device=dev)
dst = torch.empty_like(src)
ms = timed(lambda: dst.copy_(src))
print(f"D2D copia 2 GiB: {ms:.3f} ms, {2 * n / ms / 1e6:.0f} GB/s (lettura+scrittura)")
del src, dst

# 2) H2D / D2H da memoria pinned.
nb = int(host_gib * (1 << 30))
host = torch.empty(nb, dtype=torch.uint8, pin_memory=True)
host.random_(0, 256)
gpu = torch.empty(nb, dtype=torch.uint8, device=dev)
ms = timed(lambda: gpu.copy_(host, non_blocking=True), reps=5)
print(f"H2D {host_gib} GiB pinned: {ms:.2f} ms, {nb / ms / 1e6:.1f} GB/s")
ms = timed(lambda: host.copy_(gpu, non_blocking=True), reps=5)
print(f"D2H {host_gib} GiB pinned: {ms:.2f} ms, {nb / ms / 1e6:.1f} GB/s")
del gpu

# 3) gather sparso: 48 righe di richiesta = 4 utenti x 12 layer QSA, fino a 2048 righe ciascuna (top-k QSA).
op = torch.ops._C_cache_ops.hisparse_gather_compact
g = torch.Generator(device="cpu").manual_seed(0)
for row_bytes in (512, 1024):
    hc = host.view(-1, row_bytes)
    host_rows = hc.shape[0]
    reqs, top_k, bs = 48, 2048, 64
    hot = torch.zeros((reqs * top_k) // bs, bs, row_bytes, dtype=torch.uint8, device=dev)
    gi = torch.randint(0, host_rows, (reqs, top_k), generator=g, dtype=torch.int32).to(dev)
    hi = torch.arange(reqs * top_k, dtype=torch.int32, device=dev).view(reqs, top_k)
    for miss in (64, 266, 512, 2048):  # 266 ~ 13% di 2048 (miss LRU del paper HiSparse)
        cnt = torch.full((reqs,), miss, dtype=torch.int32, device=dev)
        ms = timed(lambda: op(hc, hot, gi, hi, cnt))
        mb = reqs * miss * row_bytes / 1e6
        print(f"gather {row_bytes} B x {reqs}x{miss}: {ms:.3f} ms, {mb:.1f} MB, {mb / ms:.2f} GB/s")
    # verifica bit a bit dell'ultimo gather (miss = 2048).
    want = hc[gi.flatten().cpu().long()]
    got = hot.view(-1, row_bytes).cpu()
    assert torch.equal(got, want), "gather sbagliato"
    print(f"gather {row_bytes} B: righe identiche bit a bit ({want.shape[0]})")
    del hot
print("BANDA_FINE")
