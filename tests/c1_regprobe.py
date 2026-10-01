# Sonda C1: quali flag di cudaHostRegister funzionano su un mmap di tmpfs (64 MiB del primo file PLE), con codice CUDA.
import ctypes, glob, mmap, os

import numpy as np
import torch

torch.cuda.init()
lib = next(glob.iglob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "**", "libcudart.so*"), recursive=True), None)
rt = ctypes.CDLL(lib) if lib else None
if rt:
    v = ctypes.c_int()
    rt.cudaDeviceGetAttribute(ctypes.byref(v), 113, 0)  # cudaDevAttrHostRegisterReadOnlySupported
    print("HostRegisterReadOnlySupported =", v.value, "| lib", lib)
    v2 = ctypes.c_int(); rt.cudaDeviceGetAttribute(ctypes.byref(v2), 107, 0)  # CanUseHostPointerForRegisteredMem? (debug)
path = sorted(glob.glob("/mnt/llmunity-ple-ram/model-plefp8-*.safetensors"))[0]
n = 64 << 20
for mode, prot, flags in (("rb", mmap.PROT_READ, 8), ("rb", mmap.PROT_READ, 10), ("rb", mmap.PROT_READ, 0), ("r+b", mmap.PROT_READ | mmap.PROT_WRITE, 0)):
    with open(path, mode) as f:
        mm = mmap.mmap(f.fileno(), n, flags=mmap.MAP_SHARED, prot=prot)
    a = np.frombuffer(mm, dtype=np.uint8)
    if rt:
        err = rt.cudaHostRegister(ctypes.c_void_p(a.ctypes.data), ctypes.c_size_t(n), ctypes.c_uint(flags))
        name = ctypes.c_char_p(rt.cudaGetErrorString(err) if False else None)
        rt.cudaGetErrorString.restype = ctypes.c_char_p
        print(f"{mode} prot={prot} flags={flags}: err={err} {rt.cudaGetErrorString(err).decode()}")
        if err == 0:
            rt.cudaHostUnregister(ctypes.c_void_p(a.ctypes.data))
        rt.cudaGetLastError()
    del a; mm.close()
