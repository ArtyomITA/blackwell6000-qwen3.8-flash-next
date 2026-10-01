"""Copy d0xin's FP8 PLE shards into noswap tmpfs with a host-RAM floor."""

import hashlib
import json
import os
from pathlib import Path

SOURCE = Path("/mnt/llmunity-models/d0xin-fp8ple")
RAM = Path("/mnt/llmunity-ple-ram")
VIEW = Path("/mnt/llmunity-models/d0xin-fp8ple-ramview")
MIN_AVAILABLE = 7 * 1024**3
CHUNK = 8 * 1024**2


def available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def main():
    mount = os.statvfs(RAM)
    if not RAM.is_dir() or not os.path.ismount(RAM):
        raise RuntimeError("PLE RAM directory is not a mount")
    mounts = Path("/proc/mounts").read_text()
    mount_line = next((line for line in mounts.splitlines() if line.split()[1] == str(RAM)), "")
    if not mount_line or mount_line.split()[2] != "tmpfs" or "noswap" not in mount_line.split()[3].split(","):
        raise RuntimeError("PLE mount must be tmpfs,noswap")
    if Path("/proc/swaps").read_text().splitlines()[1:]:
        raise RuntimeError("swap must be empty")
    shards = sorted(SOURCE.glob("model-plefp8-*.safetensors"))
    if len(shards) != 10:
        raise RuntimeError(f"expected 10 PLE shards, found {len(shards)}")
    total = sum(shard.stat().st_size for shard in shards)
    if total > mount.f_bavail * mount.f_frsize:
        raise RuntimeError("tmpfs free space smaller than PLE shards")
    if available_bytes() - total < MIN_AVAILABLE:
        raise RuntimeError("not enough host RAM for PLE staging floor")
    records = []
    for shard in shards:
        destination = RAM / shard.name
        if destination.exists():
            raise FileExistsError(destination)
        digest = hashlib.sha256()
        with shard.open("rb") as inp, destination.open("xb") as out:
            while True:
                block = inp.read(CHUNK)
                if not block:
                    break
                if available_bytes() - len(block) < MIN_AVAILABLE:
                    raise RuntimeError(f"host RAM floor while copying {shard.name}")
                out.write(block)
                digest.update(block)
        if destination.stat().st_size != shard.stat().st_size:
            raise RuntimeError(f"size mismatch {shard.name}")
        record = {"name": shard.name, "bytes": destination.stat().st_size, "sha256": digest.hexdigest(), "host_available_gib": round(available_bytes() / 1024**3, 3)}
        records.append(record)
        print(json.dumps(record), flush=True)
    VIEW.mkdir(exist_ok=False)
    for entry in SOURCE.iterdir():
        target = RAM / entry.name if entry.name.startswith("model-plefp8-") and entry.suffix == ".safetensors" else entry
        (VIEW / entry.name).symlink_to(target)
    result = {"source": str(SOURCE), "view": str(VIEW), "ram_mount": str(RAM), "shards": records, "total_bytes": total, "host_available_gib": round(available_bytes() / 1024**3, 3), "swap_entries": 0}
    output = Path("/mnt/llmunity-models/backend-results/d0xin-ple-tmpfs-stage.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT=" + str(output), flush=True)


if __name__ == "__main__":
    main()
