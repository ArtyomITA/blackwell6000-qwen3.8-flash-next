"""Stage only missing PLE shards into noswap tmpfs before serving d0xin."""

import hashlib
import json
import os
from pathlib import Path

SOURCE = Path("/mnt/llmunity-models/d0xin-fp8ple")
RAM = Path("/mnt/llmunity-ple-ram")
VIEW = Path("/mnt/llmunity-models/d0xin-fp8ple-ramview")
MANIFEST = Path("/mnt/llmunity-models/backend-results/d0xin-ple-tmpfs-stage.json")
FLOOR = 7 * 1024**3
CHUNK = 8 * 1024**2


def available():
    line = next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    return int(line.split()[1]) * 1024


def main():
    mount = next((x for x in Path("/proc/mounts").read_text().splitlines() if x.split()[1] == str(RAM)), None)
    if not mount or mount.split()[2] != "tmpfs" or "noswap" not in mount.split()[3].split(","):
        raise RuntimeError("PLE mount must be tmpfs,noswap")
    if Path("/proc/swaps").read_text().splitlines()[1:]:
        raise RuntimeError("swap must be disabled")
    expected = {x["name"]: x for x in json.loads(MANIFEST.read_text())["shards"]}
    shards = sorted(SOURCE.glob("model-plefp8-*.safetensors"))
    if len(shards) != 10 or {x.name for x in shards} != set(expected):
        raise RuntimeError("unexpected PLE shard inventory")
    for shard in shards:
        dest = RAM / shard.name
        entry = expected[shard.name]
        if shard.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"source size changed: {shard.name}")
        if not dest.exists():
            if available() - shard.stat().st_size < FLOOR:
                raise RuntimeError("insufficient available RAM for shard staging")
            digest = hashlib.sha256()
            with shard.open("rb") as inp, dest.open("xb") as out:
                while block := inp.read(CHUNK):
                    if available() - len(block) < FLOOR:
                        raise RuntimeError("host RAM floor during staging")
                    out.write(block)
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"hash mismatch: {shard.name}")
        if dest.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"RAM shard size mismatch: {shard.name}")
    VIEW.mkdir(exist_ok=True)
    for entry in SOURCE.iterdir():
        target = RAM / entry.name if entry.name in expected else entry
        link = VIEW / entry.name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"model view target changed: {link}")
        elif link.exists():
            raise RuntimeError(f"model view contains unexpected file: {link}")
        else:
            link.symlink_to(target)
    print(f"PLE_RAM_READY shards={len(shards)} available_gib={available()/1024**3:.3f}", flush=True)


if __name__ == "__main__":
    main()
