# Build di _det_topk (backport #55122). systemd-run MemoryMax=4G, MAX_JOBS=1, solo senza vLLM attivo.
import os
from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(os.environ.get("VLLM_SRC", "/mnt/llmunity-models/vllm-fase2"), "csrc")  # sorgenti del fork vLLM
load(name="det_topk_C", sources=[f"{here}/binding.cpp", f"{here}/csrc/libtorch_stable/topk.cu"],
     extra_include_paths=[f"{here}/csrc/libtorch_stable", f"{csrc}/libtorch_stable", csrc],
     extra_cflags=["-O3", "-std=c++20", "-DUSE_CUDA"],
     extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_120,code=sm_120", "-DUSE_CUDA"],
     build_directory=f"{here}/build", is_python_module=False, verbose=True)
print("BUILD OK")
