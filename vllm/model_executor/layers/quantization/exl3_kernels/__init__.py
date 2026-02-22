# SPDX-License-Identifier: Apache-2.0
"""EXL3 Triton kernels for fused dequant+GEMM and Hadamard transform.

Registers exl3_gemm as a torch custom op so that torch.compile / Dynamo
treats it as an opaque leaf (no tracing into autotuner, numpy tables, etc).
"""

import torch

from vllm.model_executor.layers.quantization.exl3_kernels.hadamard import (
    had_r_128,
)
from vllm.model_executor.layers.quantization.exl3_kernels.triton_kernel import (
    exl3_gemm as _exl3_gemm_impl,
)
from vllm.utils.torch_utils import direct_register_custom_op


# ---------------------------------------------------------------------------
# Register exl3_gemm as a custom op for torch.compile compatibility.
# ---------------------------------------------------------------------------

def _exl3_gemm_op(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_i32: torch.Tensor,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Wrapper matching custom_op signature (no optional args)."""
    return _exl3_gemm_impl(A, B_packed, bits=bits, cb=cb, B_i32=B_i32)


def _exl3_gemm_fake(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_i32: torch.Tensor,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Fake impl for Dynamo abstract interpretation — returns correct shape."""
    M = A.shape[0]
    N = B_packed.shape[1] * 16
    return torch.empty((M, N), dtype=torch.float16, device=A.device)


direct_register_custom_op(
    op_name="exl3_gemm",
    op_func=_exl3_gemm_op,
    mutates_args=[],
    fake_impl=_exl3_gemm_fake,
)


def exl3_gemm(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    bits: int = 4,
    cb: int = 0,
    out: torch.Tensor | None = None,
    B_i32: torch.Tensor | None = None,
    split_k: int = 0,
) -> torch.Tensor:
    """EXL3 fused dequant + GEMM.

    When torch.compile is active, dispatches through the registered custom op.
    Otherwise calls the implementation directly for full flexibility.
    """
    if B_i32 is None:
        B_i32 = B_packed.view(torch.int32)
    # Always use the custom op path — it's a no-overhead dispatch and
    # ensures Dynamo compatibility in all modes.
    return torch.ops.vllm.exl3_gemm(A, B_packed, B_i32, bits, cb)


__all__ = ["exl3_gemm", "had_r_128"]
