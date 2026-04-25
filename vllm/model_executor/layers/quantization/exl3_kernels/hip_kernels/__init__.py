"""
HIP C++ extension kernels for ROCm.

JIT-compiled via torch.utils.cpp_extension.load() on first import.
Compiled binary is cached in ~/.cache/exl3_hip_kernels/.
"""

import os
from pathlib import Path
from torch.utils.cpp_extension import load

_dir = Path(__file__).parent

# Detect GPU arch at runtime (gfx1100=RDNA3 XTX, gfx1201=RDNA4 R9700).
# Without detection, kernels crash with GPF when architecture mismatches.
import subprocess as _sp
def _detect_arch():
    forced = os.environ.get("EXL3_GFX_ARCH")
    if forced:
        return forced
    try:
        out = _sp.check_output(["rocminfo"], text=True, stderr=_sp.DEVNULL)
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Name:") and "gfx" in s:
                name = s.split("Name:")[1].strip()
                if name.startswith("gfx"):
                    return name
    except Exception:
        pass
    return "gfx1100"
_arch = _detect_arch()
_prev_arch = os.environ.get("PYTORCH_ROCM_ARCH")
os.environ["PYTORCH_ROCM_ARCH"] = _arch

try:
    hip_ext = load(
        name=f"exl3_hip_kernels_{_arch}",
        sources=[
            str(_dir / "binding.cpp"),
            str(_dir / "rms_norm_kernel.cu"),
            str(_dir / "activation_kernel.cu"),
            str(_dir / "hadamard_kernel.cu"),
            str(_dir / "kv_cache_kernel.cu"),
            str(_dir / "exl3_gemm_kernel.cu"),
        ],
        extra_cuda_cflags=[
            f"--offload-arch={_arch}",
            "-O3",
            "-I/opt/rocm/include",
        ],
        build_directory=os.path.expanduser("~/.cache/exl3_hip_kernels"),
        verbose=False,
    )
finally:
    # Restore previous PYTORCH_ROCM_ARCH
    if _prev_arch is None:
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
    else:
        os.environ["PYTORCH_ROCM_ARCH"] = _prev_arch
