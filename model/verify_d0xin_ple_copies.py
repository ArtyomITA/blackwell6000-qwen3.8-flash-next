"""Verify the persistent source and disposable RAM staging copy before unmount."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/llmunity-models")
source = ROOT / "d0xin-fp8ple"
ram = Path("/mnt/llmunity-ple-ram")
manifest = json.loads((ROOT / "backend-results/d0xin-ple-tmpfs-stage.json").read_text())
if not ram.is_mount():
    raise RuntimeError("RAM staging mount missing")
for entry in manifest["shards"]:
    for kind, base in (("source", source), ("ram", ram)):
        path = base / entry["name"]
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"{kind} size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"{kind} hash mismatch: {path}")
    print(entry["name"] + " SHA256_OK", flush=True)
print("ALL_10_SOURCE_AND_RAM_SHARDS_IDENTICAL", flush=True)
