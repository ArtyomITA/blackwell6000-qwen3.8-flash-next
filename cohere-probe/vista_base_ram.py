"""Model view of nvidia/Qwen3.8-Flash-Next-NVFP4 with the PLE table in the noswap tmpfs, like K3's d0xin view.

The NVIDIA checkpoint keeps the FP8 PLE table (47.7 GiB) and the MTP head (2.3 GiB) in one file. vLLM's
VLLM_QWEN4EXP_PLE_FILE_UVA path needs the PLE bytes in RAM, but the tmpfs is 50G: so we split the file by raw
byte ranges (no tensor is ever loaded in RAM) into
  <RAM>/model-ple-base.safetensors   ngram_embedding tensors (tmpfs)
  <VIEW>/model-mtp.safetensors       everything else (EBS)
and build <VIEW> = symlinks to the other shards + a rewritten model.safetensors.index.json.
Run it with the Swift PLE shards already removed from the tmpfs. A sampled byte comparison checks the copy.
"""
import json
import os
import random
import struct
from pathlib import Path

SRC = Path("/mnt/llmunity-models/models/nvidia-flash-next-nvfp4")
COMBO = SRC / "model-fp8-mtp-ple.safetensors"
RAM = Path("/mnt/llmunity-ple-ram")
VIEW = Path("/mnt/llmunity-models/models/nvidia-base-ramview")
PLE_NAME, MTP_NAME = "model-ple-base.safetensors", "model-mtp.safetensors"
CHUNK = 64 * 1024**2


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def scrivi(src_fd, base, tensori, out_path):
    """Write a safetensors file with `tensori` (name -> entry of the source header), copying raw bytes."""
    nuovo, off = {"__metadata__": {"format": "pt"}}, 0
    for k, v in tensori:
        a, b = v["data_offsets"]
        nuovo[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + b - a]}
        off += b - a
    hb = json.dumps(nuovo, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(hb)) + hb)
        for k, v in tensori:
            a, b = v["data_offsets"]
            pos = base + a
            while pos < base + b:
                n = min(CHUNK, base + b - pos)
                out.write(os.pread(src_fd, n, pos))
                os.posix_fadvise(src_fd, pos, n, os.POSIX_FADV_DONTNEED)  # keep the EBS page cache out of RAM
                pos += n
    return nuovo


def controlla(src_fd, base, out_path, tensori, campioni=64):
    h, obase = header(out_path)
    rng = random.Random(0)
    with open(out_path, "rb") as f:
        for k, v in rng.sample(tensori, min(campioni, len(tensori))):
            a, b = v["data_offsets"]
            oa, ob = h[k]["data_offsets"]
            assert ob - oa == b - a, k
            for p in (0, (b - a) // 2, max(0, b - a - 4096)):
                n = min(4096, b - a - p)
                assert os.pread(src_fd, n, base + a + p) == os.pread(f.fileno(), n, obase + oa + p), k


def main():
    assert not list(RAM.glob("model-plefp8-*")), "remove the Swift PLE shards from the tmpfs first"
    h, base = header(COMBO)
    h.pop("__metadata__", None)
    ple = sorted(((k, v) for k, v in h.items() if "ngram_embedding" in k), key=lambda kv: kv[1]["data_offsets"][0])
    mtp = sorted(((k, v) for k, v in h.items() if "ngram_embedding" not in k), key=lambda kv: kv[1]["data_offsets"][0])
    VIEW.mkdir(parents=True, exist_ok=True)
    fd = os.open(COMBO, os.O_RDONLY)
    try:
        scrivi(fd, base, ple, RAM / PLE_NAME)
        controlla(fd, base, RAM / PLE_NAME, ple)
        scrivi(fd, base, mtp, VIEW / MTP_NAME)
        controlla(fd, base, VIEW / MTP_NAME, mtp)
    finally:
        os.close(fd)
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    for k, _ in ple:
        idx["weight_map"][k] = PLE_NAME
    for k, _ in mtp:
        idx["weight_map"][k] = MTP_NAME
    assert COMBO.name not in idx["weight_map"].values()
    (VIEW / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))
    for e in SRC.iterdir():
        if e.name in (COMBO.name, "model.safetensors.index.json") or e.name.startswith("."):
            continue
        link = VIEW / e.name
        if not link.is_symlink():
            link.symlink_to(e)
    link = VIEW / PLE_NAME
    if not link.is_symlink():
        link.symlink_to(RAM / PLE_NAME)
    print("vista pronta:", VIEW, "PLE", len(ple), "tensori,", round((RAM / PLE_NAME).stat().st_size / 2**30, 2), "GiB in tmpfs;",
          "MTP", len(mtp), "tensori,", round((VIEW / MTP_NAME).stat().st_size / 2**30, 2), "GiB")


if __name__ == "__main__":
    main()
