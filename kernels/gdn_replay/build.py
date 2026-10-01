# Build dell'estensione gdn_replay_C (solo per l'harness G). Lanciare con systemd-run MemoryMax=4G, MAX_JOBS=1,
# SOLO senza vLLM attivo. ARCH: "120" (default) o "120f" per confrontare il SASS con la .so di oggi.
import os, sys
from torch.utils.cpp_extension import load

arch = sys.argv[1] if len(sys.argv) > 1 else "120"
here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(os.environ.get("VLLM_SRC", "/mnt/llmunity-models/vllm-fase2"), "csrc")  # sorgenti del fork vLLM
mod = load(
    name=f"gdn_replay_C_{arch}",
    sources=[f"{here}/binding.cpp", f"{here}/fused_gdn_decode_kernel.cu"],
    extra_include_paths=[csrc, f"{csrc}/libtorch_stable/gdn"],
    extra_cflags=["-O3", "-std=c++20", "-DUSE_CUDA", "-DVLLM_ENABLE_FUSED_GDN_DECODE=1"],
    extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                       "-DUSE_CUDA", "-DVLLM_ENABLE_FUSED_GDN_DECODE=1"],
    build_directory=f"{here}/build_{arch}",
    is_python_module=False,
    verbose=True,
)
print("BUILD OK", arch)
