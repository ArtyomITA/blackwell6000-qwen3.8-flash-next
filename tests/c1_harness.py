# Harness C1 (PLE via UVA dagli shard tmpfs): 10^6 righe casuali lette con preadv (come host_file_gather di oggi)
# contro il kernel Triton del worktree fase2 che le legge dalla GPU via UVA (cudaHostRegister sui file). Bit a bit.
# Registrazione r+b flag 0 (ReadOnly non supportato qui), guardia MemAvailable < 2 GB, unregister in finally; cmp dopo.
# Uso: python c1_harness.py [vista] [righe]   (GPU sotto flock; registra ~52 GB di tmpfs gia' in RAM, niente copia)
import json, mmap, os, struct, sys, time, warnings

import numpy as np
import torch

from vllm.models.qwen4_exp.nvidia.ngram_embedding import _lookup_ple_rows_from_file_uva_kernel
from vllm.triton_utils import triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

view = sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmunity-models/d0xin-fp8ple-ramview"
n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
wmap = json.load(open(f"{view}/model.safetensors.index.json"))["weight_map"]
shards = {}  # indice -> (file reale, offset assoluto, righe, byte per riga)
for name, fname in wmap.items():
    if ".ngram_embedding.shard_" in name and name.endswith(".weight"):
        path = os.path.realpath(f"{view}/{fname}")
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(hlen))[name]
        lo, hi = meta["data_offsets"]
        rows, dim = meta["shape"]
        shards[int(name.split("shard_")[1].split(".")[0])] = (path, 8 + hlen + lo, rows, dim)
idx = sorted(shards)
assert idx == list(range(len(idx))), "shard non contigui"
shard_size, row_bytes = shards[0][2], shards[0][3]
vocab = sum(v[2] for v in shards.values())
print(f"{len(shards)} shard, {shard_size} righe/shard, {row_bytes} B/riga, vocab {vocab}")

def mem_avail_gb():
    return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20


t0 = time.time(); views, owners = {}, []
ok = False
try:
    for path in sorted({v[0] for v in shards.values()}):
        # ReadOnly non supportato (attributo 0, err 801), PROT_READ rifiutato (err 1): r+b come il fork; nessuna scrittura.
        with open(path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED)
        owner = np.frombuffer(mm, dtype=np.uint8)
        r = torch.cuda.cudart().cudaHostRegister(owner.ctypes.data, owner.nbytes, 0)
        assert r.value == 0, f"cudaHostRegister {path}: {r}"
        owners.append(owner)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            views[path] = get_accelerator_view_from_cpu_tensor(torch.from_numpy(owner)).data_ptr()
        if mem_avail_gb() < 2:
            raise SystemExit(f"GUARDIA RAM: MemAvailable {mem_avail_gb():.1f} GB")
    print(f"registrati {len(views)} file in {time.time() - t0:.1f} s, MemAvailable {mem_avail_gb():.1f} GB")
    bases = torch.tensor([views[shards[i][0]] + shards[i][1] for i in idx], dtype=torch.int64, device="cuda")

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (n_rows,), generator=g)
    ids[:4] = torch.tensor([0, vocab - 1, shard_size - 1, shard_size])  # bordi
    out = torch.empty(n_rows, row_bytes, dtype=torch.uint8, device="cuda")
    ids_gpu = ids.cuda()
    _lookup_ple_rows_from_file_uva_kernel[(n_rows,)](bases, ids_gpu, out, row_bytes, shard_size, vocab,
                                                     BLOCK_D=triton.next_power_of_2(row_bytes))
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(10):
        _lookup_ple_rows_from_file_uva_kernel[(n_rows,)](bases, ids_gpu, out, row_bytes, shard_size, vocab,
                                                         BLOCK_D=triton.next_power_of_2(row_bytes))
    torch.cuda.synchronize()
    dt = (time.time() - t1) / 10
    print(f"gather UVA {n_rows} righe: {dt * 1e3:.2f} ms = {n_rows * row_bytes / dt / 1e9:.1f} GB/s")

    fds = {p: os.open(p, os.O_RDONLY) for p in views}
    ref = np.empty((n_rows, row_bytes), dtype=np.uint8)
    for j, i in enumerate(ids.tolist()):
        s = i // shard_size
        path, off, _, _ = shards[s]
        ref[j] = np.frombuffer(os.pread(fds[path], row_bytes, off + (i - s * shard_size) * row_bytes), dtype=np.uint8)
    ok = torch.equal(out.cpu(), torch.from_numpy(ref))
    bad = (out.cpu() != torch.from_numpy(ref)).any(1).sum().item()
    print(f"C1 righe bit a bit: {'PASSATO' if ok else 'FALLITO'} ({n_rows - bad}/{n_rows})")
finally:
    for owner in owners:
        torch.cuda.cudart().cudaHostUnregister(owner.ctypes.data)
sys.exit(0 if ok else 1)
