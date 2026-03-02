# SPDX-License-Identifier: Apache-2.0
"""EXL3 Triton kernels for fused dequant+GEMM and Hadamard transform.

Registers exl3_gemm as a torch custom op so that torch.compile / Dynamo
treats it as an opaque leaf (no tracing into autotuner, numpy tables, etc).
"""

import os
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.exl3_kernels.hadamard import (
    _batched_had_r_128_triton,
    batched_had_r_128,
    had_r_128,
)
from vllm.model_executor.layers.quantization.exl3_kernels.triton_kernel import (
    exl3_fused_moe_gemm as _exl3_fused_moe_gemm_impl,
    exl3_fused_moe_gemm_had as _exl3_fused_moe_gemm_had_impl,
    exl3_gemm as _exl3_gemm_impl,
    exl3_multi_gemm as _exl3_multi_gemm_impl,
    exl3_trellis_to_fp16,
    get_bit_tables,
)
from vllm.utils.torch_utils import direct_register_custom_op


# ---------------------------------------------------------------------------
# HIP rocWMMA fused MoE GEMM extension (Phase 3)
# ---------------------------------------------------------------------------
# Toggle: EXL3_HIP_MOE_GEMM=1 (default ON) or =0 to force Triton
_USE_HIP_MOE_GEMM = os.environ.get("EXL3_HIP_MOE_GEMM", "1") == "1"

_HAS_HIP_MOE = False
_hip_ext = None

_HAS_HIP_MOE_FP16 = False

# V4 VALU kernel: register-only M=1 decode, no WMMA overhead
# EXL3_HIP_GEMM_V4=1 (default ON) or =0 to force MoE-based dense path
_USE_HIP_V4 = os.environ.get("EXL3_HIP_GEMM_V4", "1") == "1"
_HAS_HIP_V4 = False
_HAS_HIP_BATCHED_V4 = False

# v3 pipelined kernel toggle: EXL3_HIP_GEMM_V3=0 (default OFF) or =1 to enable
# v3 dense GEMM regresses attention layers (+0.7 tok/s when disabled).
_USE_HIP_V3 = os.environ.get("EXL3_HIP_GEMM_V3", "0") == "1"
_HAS_HIP_V3 = False
_HAS_HIP_MOE_V3 = False

# DPP-fused batched Hadamard-128: EXL3_HIP_HAD=1 (default ON) or =0 to force Triton
_USE_HIP_HAD = os.environ.get("EXL3_HIP_HAD", "1") == "1"
_HAS_HIP_MOE_GEMM_HAD = False

# BLOCK_M=64 prefill kernel: EXL3_PREFILL_M64=1 (default ON) or =0 to disable
_USE_PREFILL_M64 = os.environ.get("EXL3_PREFILL_M64", "1") == "1"
_HAS_HIP_MOE_M64 = False

_HAS_HIP_BATCHED_HAD = False
_HAS_HIP_DUAL_HAD = False

if _USE_HIP_MOE_GEMM:
    try:
        from vllm.model_executor.layers.quantization.exl3_kernels.hip_kernels \
            import hip_ext as _hip_ext
        _HAS_HIP_MOE = (
            _hip_ext is not None
            and hasattr(_hip_ext, 'exl3_fused_moe_gemm')
        )
        _HAS_HIP_MOE_FP16 = (
            _hip_ext is not None
            and hasattr(_hip_ext, 'exl3_fused_moe_gemm_fp16')
        )
        if _USE_HIP_V4:
            _HAS_HIP_V4 = (
                _hip_ext is not None
                and hasattr(_hip_ext, 'exl3_gemm_v4')
            )
            _HAS_HIP_BATCHED_V4 = (
                _hip_ext is not None
                and hasattr(_hip_ext, 'exl3_batched_gemm_v4')
            )
        if _USE_HIP_V3:
            _HAS_HIP_V3 = (
                _hip_ext is not None
                and hasattr(_hip_ext, 'exl3_gemm_v3')
            )
            _HAS_HIP_MOE_V3 = (
                _hip_ext is not None
                and hasattr(_hip_ext, 'exl3_fused_moe_gemm_v3')
            )
        _HAS_HIP_MOE_GEMM_HAD = (
            _hip_ext is not None
            and hasattr(_hip_ext, 'exl3_fused_moe_gemm_had')
        )
        if _USE_PREFILL_M64:
            _HAS_HIP_MOE_M64 = (
                _hip_ext is not None
                and hasattr(_hip_ext, 'exl3_fused_moe_gemm_m64')
            )
        if _USE_HIP_HAD:
            _HAS_HIP_BATCHED_HAD = (
                _hip_ext is not None
                and hasattr(_hip_ext, 'batched_had_r_128')
            )
            _HAS_HIP_DUAL_HAD = (
                _hip_ext is not None
                and hasattr(_hip_ext, 'batched_dual_had_r_128')
            )
    except (ImportError, RuntimeError):
        pass

# v3 lock buffer cache: {(device, needed): tensor}
_hip_v3_lock_buf = {}


def _get_v3_lock_buf(grid_m, grid_n64, device):
    """Get or allocate pre-zeroed lock buffer for v3 lock-based split-K."""
    needed = grid_m * grid_n64
    key = (device, needed)
    buf = _hip_v3_lock_buf.get(key)
    if buf is None or buf.numel() < needed:
        _hip_v3_lock_buf[key] = torch.zeros(
            needed, dtype=torch.int32, device=device)
        buf = _hip_v3_lock_buf[key]
    buf[:needed].zero_()
    return buf[:needed]


# Cached split-K partial buffer for HIP MoE path: {(device, split_k, N): tensor}
# Key includes N to avoid non-contiguous slices when gate (N=512) and down
# (N=2048) share a buffer — .contiguous() on a non-contiguous slice creates
# a copy every call (96× per step = significant overhead in CUDA graphs).
_hip_moe_splitk_buf = {}


def _get_hip_moe_splitk_buf(split_k, EM, N, device):
    """Get or allocate cached C_partial buffer for HIP MoE split-K."""
    key = (device, split_k, N)
    buf = _hip_moe_splitk_buf.get(key)
    if buf is None or buf.shape[1] < EM:
        _hip_moe_splitk_buf[key] = torch.empty(
            (split_k, max(EM, 1), N),
            dtype=torch.float16, device=device)
        buf = _hip_moe_splitk_buf[key]
    return buf


# Cached output buffer for HIP MoE GEMM: {(device, N): tensor}
# Avoids torch.empty allocation 144x/step (48 layers × 3 projections).
_hip_moe_output_buf = {}


def _get_hip_moe_output_buf(EM, N, device):
    """Get or allocate cached output C buffer for HIP MoE GEMM."""
    key = (device, N)
    buf = _hip_moe_output_buf.get(key)
    if buf is None or buf.shape[0] < EM:
        _hip_moe_output_buf[key] = torch.empty(
            (max(EM, 1), N), dtype=torch.float16, device=device)
        buf = _hip_moe_output_buf[key]
    return buf[:EM]


def _hip_moe_auto_split_k(num_m_blocks, num_k_tiles, is_decode=False):
    """Auto split-K selection for HIP MoE GEMM.

    Decode (few M-blocks): higher split-K (max 8) for K-parallelism.
    Prefill (many M-blocks): lower split-K (max 4), M-parallelism suffices.
    """
    if num_m_blocks <= 16 and num_k_tiles >= 16:
        max_sk = 8 if is_decode else 4
        return min(max_sk, num_k_tiles)
    return 1


# ---------------------------------------------------------------------------
# HIP v2 dense GEMM: reuses MoE kernel with 1 expert for M=1 decode.
# 4.4x faster than Triton dense GEMM (17µs vs 76µs).
# Toggle: EXL3_HIP_DENSE_GEMM=1 (default ON when HIP MoE available)
# ---------------------------------------------------------------------------
# Default OFF: only 5µs/kernel faster in CUDA graph replay (28.8→23.7µs),
# total savings ~0.6ms/step — not worth the code complexity.
_USE_HIP_DENSE_GEMM = os.environ.get("EXL3_HIP_DENSE_GEMM", "1") == "1"

# Cached routing tensors for dense GEMM (single M-block, expert 0)
_hip_dense_routing: dict = {}  # {device: (expert_ids, num_tokens_post_padded)}

BLOCK_M_DENSE = 16


def _get_hip_dense_routing(device):
    """Get cached single-expert routing tensors for HIP v2 dense GEMM."""
    r = _hip_dense_routing.get(device)
    if r is None:
        expert_ids = torch.zeros(1, dtype=torch.int32, device=device)
        num_post = torch.tensor([BLOCK_M_DENSE], dtype=torch.int32, device=device)
        r = (expert_ids, num_post)
        _hip_dense_routing[device] = r
    return r


# Cached padded input buffer for dense GEMM: {(device, K): tensor}
_hip_dense_input_buf: dict = {}


def _get_hip_dense_input_buf(K, device):
    """Get cached BLOCK_M×K padded input buffer for HIP v2 dense GEMM."""
    key = (device, K)
    buf = _hip_dense_input_buf.get(key)
    if buf is None:
        buf = torch.zeros(BLOCK_M_DENSE, K, dtype=torch.float16, device=device)
        _hip_dense_input_buf[key] = buf
    return buf


# Cached split-K partial buffer for dense GEMM: {(device, split_k, N): tensor}
_hip_dense_splitk_buf: dict = {}


def _get_hip_dense_splitk_buf(split_k, N, device):
    """Get or allocate cached C_partial for HIP v2 dense split-K."""
    key = (device, split_k, N)
    buf = _hip_dense_splitk_buf.get(key)
    if buf is None:
        buf = torch.empty(
            (split_k, BLOCK_M_DENSE, N), dtype=torch.float16, device=device)
        _hip_dense_splitk_buf[key] = buf
    return buf


# Cached split-K partial buffer for V4 GEMM: {(device, split_k, N): tensor}
_hip_v4_splitk_buf: dict = {}


def _get_hip_v4_splitk_buf(split_k, N, device):
    """Get or allocate cached C_partial for HIP V4 split-K (M=1)."""
    key = (device, split_k, N)
    buf = _hip_v4_splitk_buf.get(key)
    if buf is None:
        buf = torch.empty(
            (split_k, 1, N), dtype=torch.float16, device=device)
        _hip_v4_splitk_buf[key] = buf
    return buf


# Cached split-K partial buffer for batched V4: {(device, split_k, MN): tensor}
_hip_batched_v4_splitk_buf: dict = {}


def _get_hip_batched_v4_splitk_buf(split_k, num_outputs, N, device):
    """Get or allocate cached C_partial for batched V4 split-K."""
    MN = num_outputs * N
    key = (device, split_k, MN)
    buf = _hip_batched_v4_splitk_buf.get(key)
    if buf is None:
        buf = torch.empty(
            (split_k, MN), dtype=torch.float16, device=device)
        _hip_batched_v4_splitk_buf[key] = buf
    return buf


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
    M, K = A.shape
    N = B_packed.shape[1] * 16

    # HIP V4 VALU kernel: register-only M=1 decode, no WMMA overhead.
    # ~1.3-1.8x faster than V2 MoE-based dense path.
    if (_HAS_HIP_V4 and M == 1 and cb in (0, 1)
            and K % 16 == 0 and N % 16 == 0):
        word_idx, next_word_idx, shift_tbl = get_bit_tables(bits, A.device)
        num_k_tiles = K // 16
        split_k = min(8, num_k_tiles)

        C = torch.empty(1, N, dtype=torch.float16, device=A.device)

        if split_k > 1:
            C_partial = _get_hip_v4_splitk_buf(split_k, N, A.device)
        else:
            C_partial = torch.empty(
                1, 1, 1, dtype=torch.float16, device=A.device)

        _hip_ext.exl3_gemm_v4(
            A, B_i32, C,
            word_idx, next_word_idx, shift_tbl,
            bits, cb, split_k, C_partial,
        )
        return C

    # HIP v2 dense GEMM: reuse MoE kernel with 1 expert, BLOCK_M=16
    # Only for M=1 decode (most common case) and cb=0,1
    if (_USE_HIP_DENSE_GEMM and _HAS_HIP_MOE and M == 1 and cb in (0, 1)
            and K % 16 == 0 and N % 16 == 0):
        word_idx, next_word_idx, shift_tbl = get_bit_tables(bits, A.device)
        expert_ids, num_post = _get_hip_dense_routing(A.device)

        # Pad input from (1, K) to (BLOCK_M, K) — kernel expects BLOCK_M rows
        A_padded = _get_hip_dense_input_buf(K, A.device)
        A_padded[0].copy_(A[0])

        # Reshape B from (tiles_k, tiles_n, WPT) to (1, tiles_k, tiles_n, WPT)
        B_e = B_i32.unsqueeze(0)

        num_k_tiles = K // 16
        split_k = _hip_moe_auto_split_k(1, num_k_tiles, is_decode=True)
        split_k = min(split_k, num_k_tiles)

        C = torch.empty(BLOCK_M_DENSE, N, dtype=torch.float16,
                        device=A.device)

        if split_k > 1:
            # Dedicated small buffer — MoE buffers may have larger EM_max
            # which makes slices non-contiguous
            C_partial = _get_hip_dense_splitk_buf(split_k, N, A.device)
            C_partial.zero_()
        else:
            C_partial = torch.empty(
                1, 1, 1, dtype=torch.float16, device=A.device)

        _hip_ext.exl3_fused_moe_gemm(
            A_padded, B_e, C,
            expert_ids, num_post,
            word_idx, next_word_idx, shift_tbl,
            BLOCK_M_DENSE, bits, cb, split_k, C_partial,
        )
        return C[:1]  # Return only first row

    # v3 pipelined HIP kernel: 4-wave (N=64 per block), lock-based split-K
    if _HAS_HIP_V3 and N % 64 == 0 and cb in (0, 1):
        C = torch.empty(M, N, dtype=torch.float16, device=A.device)
        word_idx, next_word_idx, shift_tbl = get_bit_tables(bits, A.device)

        num_k_tiles = K // 16
        split_k = min(4, num_k_tiles) if num_k_tiles >= 16 else 1
        grid_m = (M + 15) // 16
        grid_n64 = (N + 63) // 64
        locks = _get_v3_lock_buf(grid_m, grid_n64, A.device)

        _hip_ext.exl3_gemm_v3(
            A.contiguous(), B_i32, C,
            word_idx, next_word_idx, shift_tbl,
            locks, bits, cb, split_k,
        )
        return C
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


# ---------------------------------------------------------------------------
# Register exl3_multi_gemm as a custom op for torch.compile compatibility.
# Batches multiple matching-N GEMMs into one launch for merged layers.
# ---------------------------------------------------------------------------

def _exl3_multi_gemm_op(
    A_batched: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    num_outputs: int,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Wrapper matching custom_op signature."""
    M = A_batched.shape[1]
    K = A_batched.shape[2]
    N = B_stacked_i32.shape[2] * 16

    # Batched V4: single-launch HIP kernel for M=1 decode
    if (_HAS_HIP_BATCHED_V4 and M == 1 and cb in (0, 1)
            and K % 16 == 0 and N % 16 == 0):
        word_idx, next_word_idx, shift_tbl = get_bit_tables(
            bits, A_batched.device)
        num_k_tiles = K // 16
        split_k = min(8, num_k_tiles)
        C = torch.empty(
            num_outputs, 1, N, dtype=torch.float16, device=A_batched.device)
        if split_k > 1:
            C_partial = _get_hip_batched_v4_splitk_buf(
                split_k, num_outputs, N, A_batched.device)
        else:
            C_partial = torch.empty(
                1, 1, dtype=torch.float16, device=A_batched.device)
        _hip_ext.exl3_batched_gemm_v4(
            A_batched, B_stacked_i32, C,
            word_idx, next_word_idx, shift_tbl,
            num_outputs, bits, cb, split_k, C_partial)
        return C

    return _exl3_multi_gemm_impl(
        A_batched, B_stacked_i32, num_outputs, bits=bits, cb=cb,
    )


def _exl3_multi_gemm_fake(
    A_batched: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    num_outputs: int,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Fake impl for Dynamo — returns correct shape."""
    M = A_batched.shape[1]
    N = B_stacked_i32.shape[2] * 16
    return torch.empty(
        (num_outputs, M, N), dtype=torch.float16, device=A_batched.device)


direct_register_custom_op(
    op_name="exl3_multi_gemm",
    op_func=_exl3_multi_gemm_op,
    mutates_args=[],
    fake_impl=_exl3_multi_gemm_fake,
)


def exl3_multi_gemm(
    A_batched: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    num_outputs: int,
    bits: int = 4,
    cb: int = 0,
) -> torch.Tensor:
    """Batched EXL3 fused dequant + GEMM for multiple matching-N projections.

    Args:
        A_batched: (num_outputs, M, K) float16 input activations.
        B_stacked_i32: (num_outputs, tiles_k, tiles_n, wpt) int32 packed weights.
        num_outputs: Number of output matrices.
        bits: Bits per weight (1-8).
        cb: Codebook variant (0=default, 1=MCG, 2=MUL1).

    Returns:
        C: (num_outputs, M, N) float16.
    """
    return torch.ops.vllm.exl3_multi_gemm(
        A_batched, B_stacked_i32, num_outputs, bits, cb)


# ---------------------------------------------------------------------------
# Register exl3_fused_moe_gemm as a custom op for torch.compile compatibility.
# ---------------------------------------------------------------------------

def _exl3_fused_moe_gemm_hip(
    A: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM_max: int,
    bits: int,
    cb: int = 0,
    B_fp16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch fused MoE GEMM to HIP rocWMMA kernel.

    If B_fp16 is provided and the FP16 kernel is available, uses the
    FP16 path (no dequant, just load+MMA). Otherwise falls back to the
    trellis dequant path.
    """
    K = A.shape[1]
    num_m_blocks = expert_ids.shape[0]
    num_k_tiles = K // 16

    # FP16 path: B_fp16 is (E, K, N) fp16
    if B_fp16 is not None and _HAS_HIP_MOE_FP16:
        N = B_fp16.shape[2]

        is_prefill = num_m_blocks > 64

        # Auto split-K
        split_k = _hip_moe_auto_split_k(num_m_blocks, num_k_tiles,
                                         is_decode=not is_prefill)
        split_k = min(split_k, num_k_tiles)
        if is_prefill:
            C = torch.zeros(EM_max, N, dtype=torch.float16, device=A.device)
        else:
            C = _get_hip_moe_output_buf(EM_max, N, A.device)

        if split_k > 1:
            C_partial = _get_hip_moe_splitk_buf(split_k, EM_max, N, A.device)
            C_partial = C_partial[:split_k, :EM_max, :N]
            C_partial.zero_()
        else:
            C_partial = torch.empty(
                1, 1, 1, dtype=torch.float16, device=A.device)

        _hip_ext.exl3_fused_moe_gemm_fp16(
            A, B_fp16, C,
            expert_ids, num_tokens_post_padded,
            EM_max, split_k, C_partial,
        )
        return C

    # Dequant path: B_stacked_i32 is (E, tiles_k, tiles_n, WPT//2)
    N = B_stacked_i32.shape[2] * 16  # tiles_n * 16

    # BLOCK_M=64 prefill path: 4× dequant reuse via M64 kernel variant.
    # Uses M_FACTOR=4 internally — dequants B once, reuses for 4 A sub-tiles.
    # Best for small-to-medium prefills; v2 wins at large prefills (more parallelism).
    if _HAS_HIP_MOE_M64 and 32 <= num_m_blocks <= 400:
        return _exl3_fused_moe_gemm_hip_m64(
            A, B_stacked_i32, expert_ids, num_tokens_post_padded,
            EM_max, bits, cb)

    # Get bit extraction tables
    word_idx, next_word_idx, shift_tbl = get_bit_tables(bits, A.device)

    # v3 MoE disabled: benchmarks show v2 is 1.3-1.8x faster than v3 for MoE
    # (v3's 4-wave sync overhead hurts fused MoE more than it helps)

    # v2 path
    # Prefill (many blocks): fresh torch.zeros — avoids stale data, OS zeroed pages cheap
    # Decode (few blocks): cached buffer — avoids allocation overhead in CUDA graphs
    is_prefill = num_m_blocks > 64

    # Auto split-K: decode gets max 8, prefill gets max 4
    split_k = _hip_moe_auto_split_k(num_m_blocks, num_k_tiles,
                                     is_decode=not is_prefill)
    split_k = min(split_k, num_k_tiles)
    if is_prefill:
        C = torch.zeros(EM_max, N, dtype=torch.float16, device=A.device)
    else:
        C = _get_hip_moe_output_buf(EM_max, N, A.device)

    if split_k > 1:
        C_partial = _get_hip_moe_splitk_buf(split_k, EM_max, N, A.device)
        C_partial = C_partial[:split_k, :EM_max, :N]
        C_partial.zero_()
    else:
        C_partial = torch.empty(1, 1, 1, dtype=torch.float16, device=A.device)

    _hip_ext.exl3_fused_moe_gemm(
        A, B_stacked_i32, C,
        expert_ids, num_tokens_post_padded,
        word_idx, next_word_idx, shift_tbl,
        EM_max, bits, cb,
        split_k, C_partial,
    )
    return C


def _exl3_fused_moe_gemm_hip_m64(
    A: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM_max: int,
    bits: int,
    cb: int = 0,
) -> torch.Tensor:
    """Dispatch fused MoE GEMM to HIP BLOCK_M=32 prefill kernel.

    Same as _exl3_fused_moe_gemm_hip but uses the M32 kernel variant
    that processes 2 M-sub-tiles per block, dequanting B once and
    reusing for 2 A loads. ~2× less dequant work for prefill.
    """
    K = A.shape[1]
    N = B_stacked_i32.shape[2] * 16
    num_m_blocks = expert_ids.shape[0]
    num_k_tiles = K // 16

    word_idx, next_word_idx, shift_tbl = get_bit_tables(bits, A.device)

    # For prefill with large M, split_k=1 is usually optimal
    num_m_blocks_super = (num_m_blocks + 1) // 2  # M_FACTOR=2
    if num_m_blocks_super <= 4 and num_k_tiles >= 16:
        split_k = min(4, num_k_tiles)
    else:
        split_k = 1

    C = torch.zeros(EM_max, N, dtype=torch.float16, device=A.device)

    if split_k > 1:
        C_partial = _get_hip_moe_splitk_buf(split_k, EM_max, N, A.device)
        C_partial = C_partial[:split_k, :EM_max, :N]
        C_partial.zero_()
    else:
        C_partial = torch.empty(1, 1, 1, dtype=torch.float16, device=A.device)

    _hip_ext.exl3_fused_moe_gemm_m64(
        A, B_stacked_i32, C,
        expert_ids, num_tokens_post_padded,
        word_idx, next_word_idx, shift_tbl,
        EM_max, bits, cb,
        split_k, C_partial,
    )
    return C


def _exl3_fused_moe_gemm_op(
    A: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    B_fp16: torch.Tensor,
    EM_max: int,
    N_out: int,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Wrapper matching custom_op signature (no optional args).

    Dispatches to HIP when available and cb==0, otherwise Triton.
    B_fp16: (E, K, N) fp16 pre-dequanted weights, or empty tensor if unavailable.
    N_out: output dimension (needed for fake impl when B_fp16 is used).
    """
    # B_fp16.numel() > 0 means FP16 weights are available
    has_fp16 = B_fp16.numel() > 0
    if _HAS_HIP_MOE and cb in (0, 1):
        return _exl3_fused_moe_gemm_hip(
            A, B_stacked_i32, expert_ids,
            num_tokens_post_padded, EM_max, bits, cb,
            B_fp16=B_fp16 if has_fp16 else None,
        )
    return _exl3_fused_moe_gemm_impl(
        A, B_stacked_i32.view(torch.int16),
        sorted_token_ids, expert_ids,
        num_tokens_post_padded, EM_max=EM_max,
        bits=bits, cb=cb, B_i32=B_stacked_i32,
    )


def _exl3_fused_moe_gemm_fake(
    A: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    B_fp16: torch.Tensor,
    EM_max: int,
    N_out: int,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Fake impl for Dynamo — returns correct shape."""
    if B_fp16.numel() > 0:
        N = N_out
    else:
        N = B_stacked_i32.shape[2] * 16  # tiles_n * 16
    return torch.empty(
        (EM_max, N),
        dtype=torch.float16, device=A.device)


direct_register_custom_op(
    op_name="exl3_fused_moe_gemm",
    op_func=_exl3_fused_moe_gemm_op,
    mutates_args=[],
    fake_impl=_exl3_fused_moe_gemm_fake,
)


def exl3_fused_moe_gemm(
    A: torch.Tensor,
    B_stacked: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: "int | torch.Tensor",
    EM_max: int = 0,
    bits: int = 4,
    cb: int = 0,
    num_valid_tokens: int | None = None,
    split_k: int = 0,
    B_i32: torch.Tensor | None = None,
    B_fp16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused multi-expert EXL3 dequant + GEMM for MoE layers.

    If B_fp16 is provided (E, K, N fp16), uses FP16 kernel (no dequant).
    """
    if B_i32 is None:
        B_i32 = B_stacked.view(torch.int32)
    # Ensure tensor for custom op
    if isinstance(num_tokens_post_padded, int):
        num_post_pad_t = torch.tensor(
            [num_tokens_post_padded], dtype=torch.int32, device=A.device)
        if EM_max <= 0:
            EM_max = num_tokens_post_padded
    else:
        num_post_pad_t = num_tokens_post_padded.to(torch.int32)
        if EM_max <= 0:
            EM_max = num_tokens_post_padded.item()
    # B_fp16: pass through or empty tensor (custom ops can't have Optional)
    if B_fp16 is None:
        B_fp16_t = torch.empty(0, dtype=torch.float16, device=A.device)
        N_out = B_i32.shape[2] * 16
    else:
        B_fp16_t = B_fp16
        N_out = B_fp16.shape[2]
    return torch.ops.vllm.exl3_fused_moe_gemm(
        A, B_i32, sorted_token_ids, expert_ids,
        num_post_pad_t, B_fp16_t, EM_max, N_out, bits, cb,
    )


def exl3_fused_moe_gemm_prefill(
    A: torch.Tensor,
    B_stacked: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: "int | torch.Tensor",
    EM_max: int = 0,
    bits: int = 4,
    cb: int = 0,
    B_i32: torch.Tensor | None = None,
    block_m: int = 64,
) -> torch.Tensor:
    """Prefill-optimized fused MoE GEMM: Triton with larger BLOCK_M.

    Bypasses custom op (no graph capture needed) and HIP (BLOCK_M=16 hardcoded).
    Larger BLOCK_M amortizes trellis dequant across more A rows per B tile load.
    BLOCK_M=64 → 4x less dequant ALU vs BLOCK_M=16.
    """
    return _exl3_fused_moe_gemm_impl(
        A, B_stacked, sorted_token_ids, expert_ids,
        num_tokens_post_padded, EM_max=EM_max, bits=bits, cb=cb,
        B_i32=B_i32, block_m=block_m, split_k=1,
    )


# ---------------------------------------------------------------------------
# Register exl3_fused_moe_gemm_had as a custom op for torch.compile.
# ---------------------------------------------------------------------------

_h16_cache = {}

def _get_h16_matrix(device):
    """Get or create the normalized H_16 Hadamard matrix (16×16 fp16)."""
    if device not in _h16_cache:
        # H_16 = (1/√16) * Hadamard(16)
        # Build via Sylvester construction: H_1 = [1], H_2n = [[H_n, H_n], [H_n, -H_n]]
        import numpy as np
        h = np.array([[1.0]])
        for _ in range(4):  # 4 doublings: 1→2→4→8→16
            h = np.block([[h, h], [h, -h]])
        h = h / 4.0  # 1/√16 = 1/4
        _h16_cache[device] = torch.tensor(
            h, dtype=torch.float16, device=device).contiguous()
    return _h16_cache[device]


def _exl3_fused_moe_gemm_had_hip(
    A: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    svh_stacked: torch.Tensor,
    EM_max: int,
    bits: int,
    cb: int = 0,
) -> torch.Tensor:
    """Dispatch fused MoE GEMM+Had to HIP kernel."""
    K = A.shape[1]
    N = B_stacked_i32.shape[2] * 16  # tiles_n * 16
    num_m_blocks = expert_ids.shape[0]
    num_k_tiles = K // 16

    word_idx, next_word_idx, shift_tbl = get_bit_tables(bits, A.device)
    H16 = _get_h16_matrix(A.device)

    is_prefill = num_m_blocks > 64

    # Auto split-K: decode gets max 8, prefill gets max 4
    split_k = _hip_moe_auto_split_k(num_m_blocks, num_k_tiles,
                                     is_decode=not is_prefill)
    split_k = min(split_k, num_k_tiles)

    if is_prefill:
        C = torch.zeros(EM_max, N, dtype=torch.float16, device=A.device)
    else:
        C = _get_hip_moe_output_buf(EM_max, N, A.device)

    has_svh = 1 if svh_stacked.numel() > 0 else 0

    if split_k > 1:
        C_partial = _get_hip_moe_splitk_buf(split_k, EM_max, N, A.device)
        C_partial = C_partial[:split_k, :EM_max, :N]
        C_partial.zero_()
    else:
        C_partial = torch.empty(1, 1, 1, dtype=torch.float16, device=A.device)

    _hip_ext.exl3_fused_moe_gemm_had(
        A, B_stacked_i32, C,
        expert_ids, num_tokens_post_padded,
        word_idx, next_word_idx, shift_tbl,
        H16, svh_stacked,
        EM_max, bits, cb,
        split_k, has_svh, C_partial,
    )
    return C


def _exl3_fused_moe_gemm_had_op(
    A: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    svh_stacked: torch.Tensor,
    EM_max: int,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Wrapper matching custom_op signature (no optional args).

    Dispatches to HIP fused GEMM+Had when available and cb==0 and N%128==0,
    otherwise falls back to Triton.
    """
    N = B_stacked_i32.shape[2] * 16
    if _HAS_HIP_MOE_GEMM_HAD and cb in (0, 1) and N % 128 == 0:
        return _exl3_fused_moe_gemm_had_hip(
            A, B_stacked_i32, expert_ids,
            num_tokens_post_padded, svh_stacked,
            EM_max, bits, cb,
        )
    return _exl3_fused_moe_gemm_had_impl(
        A, B_stacked_i32.view(torch.int16),
        sorted_token_ids, expert_ids,
        num_tokens_post_padded, svh_stacked,
        EM_max=EM_max, bits=bits, cb=cb,
        B_i32=B_stacked_i32,
    )


def _exl3_fused_moe_gemm_had_fake(
    A: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    svh_stacked: torch.Tensor,
    EM_max: int,
    bits: int,
    cb: int,
) -> torch.Tensor:
    """Fake impl for Dynamo — returns correct shape."""
    N = B_stacked_i32.shape[2] * 16  # tiles_n * 16
    return torch.empty(
        (EM_max, N),
        dtype=torch.float16, device=A.device)


direct_register_custom_op(
    op_name="exl3_fused_moe_gemm_had",
    op_func=_exl3_fused_moe_gemm_had_op,
    mutates_args=[],
    fake_impl=_exl3_fused_moe_gemm_had_fake,
)


def exl3_fused_moe_gemm_had(
    A: torch.Tensor,
    B_stacked: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: "int | torch.Tensor",
    svh_stacked: torch.Tensor,
    EM_max: int = 0,
    bits: int = 4,
    cb: int = 0,
    num_valid_tokens: int | None = None,
    split_k: int = 0,
    B_i32: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused multi-expert EXL3 dequant + GEMM + Had-128 + per-expert SVH."""
    if B_i32 is None:
        B_i32 = B_stacked.view(torch.int32)
    # Ensure tensor for custom op
    if isinstance(num_tokens_post_padded, int):
        num_post_pad_t = torch.tensor(
            [num_tokens_post_padded], dtype=torch.int32, device=A.device)
        if EM_max <= 0:
            EM_max = num_tokens_post_padded
    else:
        num_post_pad_t = num_tokens_post_padded.to(torch.int32)
        if EM_max <= 0:
            EM_max = num_tokens_post_padded.item()
    return torch.ops.vllm.exl3_fused_moe_gemm_had(
        A, B_i32, sorted_token_ids, expert_ids,
        num_post_pad_t, svh_stacked, EM_max, bits, cb,
    )


# ---------------------------------------------------------------------------
# Register batched_had_r_128 as a custom op for torch.compile compatibility.
# ---------------------------------------------------------------------------

def _batched_had_r_128_op(
    x_sorted: torch.Tensor,
    scale_stacked: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    pre_int: int,
) -> torch.Tensor:
    """Wrapper matching custom_op signature (bool as int).

    Dispatches to HIP DPP-fused kernel when available (EXL3_HIP_HAD=1),
    otherwise falls back to Triton.
    """
    M = x_sorted.size(0)
    eid = expert_ids_expanded[:M].contiguous() if expert_ids_expanded.size(0) != M else expert_ids_expanded
    if _HAS_HIP_BATCHED_HAD and eid.size(0) == M:
        out = torch.empty_like(x_sorted)
        _hip_ext.batched_had_r_128(
            x_sorted, scale_stacked, out,
            eid.int(), pre_int)
        return out
    return _batched_had_r_128_triton(
        x_sorted, scale_stacked, eid, pre=bool(pre_int))


def _batched_had_r_128_fake(
    x_sorted: torch.Tensor,
    scale_stacked: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    pre_int: int,
) -> torch.Tensor:
    """Fake impl for Dynamo — returns correct shape."""
    return torch.empty_like(x_sorted)


direct_register_custom_op(
    op_name="batched_had_r_128",
    op_func=_batched_had_r_128_op,
    mutates_args=[],
    fake_impl=_batched_had_r_128_fake,
)


# ---------------------------------------------------------------------------
# Register batched_dual_had_r_128 as a custom op for torch.compile.
# Fuses gate+up input Had: same x_sorted, 2 scales → 2 outputs.
# ---------------------------------------------------------------------------

def _batched_dual_had_r_128_op(
    x_sorted: torch.Tensor,
    scale0_stacked: torch.Tensor,
    scale1_stacked: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    pre_int: int,
) -> list[torch.Tensor]:
    """Dual-output batched Had-128: one butterfly, two scale+store passes.

    Dispatches to HIP kernel when available, otherwise falls back to
    two separate batched Had calls.
    """
    # Scales should be pre-split contiguous at load time (w13_suh_gate/up).
    # Add safety .contiguous() only if needed (no-op for already contiguous).
    s0 = scale0_stacked if scale0_stacked.is_contiguous() else scale0_stacked.contiguous()
    s1 = scale1_stacked if scale1_stacked.is_contiguous() else scale1_stacked.contiguous()
    M = x_sorted.size(0)
    if _HAS_HIP_DUAL_HAD and expert_ids_expanded.size(0) == M:
        out0 = torch.empty_like(x_sorted)
        out1 = torch.empty_like(x_sorted)
        _hip_ext.batched_dual_had_r_128(
            x_sorted, s0, s1,
            out0, out1,
            expert_ids_expanded.int(), pre_int)
        return [out0, out1]
    # Fallback: two separate Had calls (also handles eid size mismatch)
    out0 = _batched_had_r_128_op(
        x_sorted, s0, expert_ids_expanded, pre_int)
    out1 = _batched_had_r_128_op(
        x_sorted, s1, expert_ids_expanded, pre_int)
    return [out0, out1]


def _batched_dual_had_r_128_fake(
    x_sorted: torch.Tensor,
    scale0_stacked: torch.Tensor,
    scale1_stacked: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    pre_int: int,
) -> list[torch.Tensor]:
    """Fake impl for Dynamo — returns correct shapes."""
    return [torch.empty_like(x_sorted), torch.empty_like(x_sorted)]


direct_register_custom_op(
    op_name="batched_dual_had_r_128",
    op_func=_batched_dual_had_r_128_op,
    mutates_args=[],
    fake_impl=_batched_dual_had_r_128_fake,
)


# ---------------------------------------------------------------------------
# Register exl3_fused_moe_gate_up as a compound custom op.
# Fuses: dual_had(input) → gate_gemm → up_gemm → silu_mul into 1 op.
# NOTE: Superseded by V7 exl3_moe_had_gemm_had compound op (below), which
# achieves the same graph node reduction (4 per layer) with simpler per-
# projection ops. Kept for backwards compatibility / fallback.
# ---------------------------------------------------------------------------

def _exl3_fused_moe_gate_up_op(
    x_sorted: torch.Tensor,
    w13_suh_gate: torch.Tensor,
    w13_suh_up: torch.Tensor,
    w1_trellis_i32: torch.Tensor,
    w3_trellis_i32: torch.Tensor,
    w1_svh: torch.Tensor,
    w3_svh: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    identity_ids: torch.Tensor,
    eid_blocks: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM_max: int,
    N_gate: int,
    N_up: int,
    bits: int,
) -> torch.Tensor:
    """Compound op: dual_had → gate_gemm → gate_output_had →
    up_gemm → up_output_had → silu_mul.

    Returns the hidden state (silu(gate_h) * up_h) ready for down projection.
    Uses _hip_ext directly when available to avoid torch.ops dispatch overhead.
    """
    M = x_sorted.size(0)

    if _HAS_HIP_DUAL_HAD and expert_ids_expanded.size(0) == M:
        # --- HIP fast path: all _hip_ext calls, no framework dispatch ---
        eid_int = expert_ids_expanded.int()

        # 1. Dual input Had: reads x_sorted once, writes xh_gate + xh_up
        xh_gate = torch.empty_like(x_sorted)
        xh_up = torch.empty_like(x_sorted)
        _hip_ext.batched_dual_had_r_128(
            x_sorted, w13_suh_gate, w13_suh_up,
            xh_gate, xh_up, eid_int, 1)  # pre=True

        # 2. Gate GEMM (direct HIP)
        gate = _exl3_fused_moe_gemm_hip(
            xh_gate, w1_trellis_i32, eid_blocks,
            num_tokens_post_padded, EM_max, bits)

        # 3. Gate output Had (direct HIP)
        gate_h = torch.empty(M, N_gate, dtype=torch.float16,
                             device=x_sorted.device)
        _hip_ext.batched_had_r_128(
            gate, w1_svh, gate_h, eid_int, 0)

        # 4. Up GEMM (direct HIP)
        up = _exl3_fused_moe_gemm_hip(
            xh_up, w3_trellis_i32, eid_blocks,
            num_tokens_post_padded, EM_max, bits)

        # 5. Up output Had (direct HIP)
        up_h = torch.empty(M, N_up, dtype=torch.float16,
                           device=x_sorted.device)
        _hip_ext.batched_had_r_128(
            up, w3_svh, up_h, eid_int, 0)

        # 6. SiLU(gate) * up
        return F.silu(gate_h) * up_h

    # --- Fallback: dispatch through torch.ops.vllm ---
    xh_gate, xh_up = torch.ops.vllm.batched_dual_had_r_128(
        x_sorted, w13_suh_gate, w13_suh_up,
        expert_ids_expanded, 1)

    gate = torch.ops.vllm.exl3_fused_moe_gemm(
        xh_gate, w1_trellis_i32, identity_ids, eid_blocks,
        num_tokens_post_padded,
        torch.empty(0, dtype=torch.float16, device=x_sorted.device),
        EM_max, N_gate, bits, 0)
    gate_h = torch.ops.vllm.batched_had_r_128(
        gate, w1_svh, expert_ids_expanded, 0)

    up = torch.ops.vllm.exl3_fused_moe_gemm(
        xh_up, w3_trellis_i32, identity_ids, eid_blocks,
        num_tokens_post_padded,
        torch.empty(0, dtype=torch.float16, device=x_sorted.device),
        EM_max, N_up, bits, 0)
    up_h = torch.ops.vllm.batched_had_r_128(
        up, w3_svh, expert_ids_expanded, 0)

    return F.silu(gate_h) * up_h


def _exl3_fused_moe_gate_up_fake(
    x_sorted: torch.Tensor,
    w13_suh_gate: torch.Tensor,
    w13_suh_up: torch.Tensor,
    w1_trellis_i32: torch.Tensor,
    w3_trellis_i32: torch.Tensor,
    w1_svh: torch.Tensor,
    w3_svh: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    identity_ids: torch.Tensor,
    eid_blocks: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM_max: int,
    N_gate: int,
    N_up: int,
    bits: int,
) -> torch.Tensor:
    """Fake impl for Dynamo — returns correct shape."""
    return torch.empty(
        (EM_max, N_gate),
        dtype=torch.float16, device=x_sorted.device)


direct_register_custom_op(
    op_name="exl3_fused_moe_gate_up",
    op_func=_exl3_fused_moe_gate_up_op,
    mutates_args=[],
    fake_impl=_exl3_fused_moe_gate_up_fake,
)


def exl3_fused_moe_gate_up(
    x_sorted: torch.Tensor,
    w13_suh_gate: torch.Tensor,
    w13_suh_up: torch.Tensor,
    w1_trellis_i32: torch.Tensor,
    w3_trellis_i32: torch.Tensor,
    w1_svh: torch.Tensor,
    w3_svh: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    identity_ids: torch.Tensor,
    eid_blocks: torch.Tensor,
    num_tokens_post_padded: "int | torch.Tensor",
    EM_max: int,
    bits: int = 4,
) -> torch.Tensor:
    """Fused gate+up projection: dual_had → 2×GEMM → 2×output_had → silu_mul.

    Single custom op = 1 graph node instead of 7 (dual_had + 2×gemm + 2×had + silu).
    """
    if isinstance(num_tokens_post_padded, int):
        num_post_pad_t = torch.tensor(
            [num_tokens_post_padded], dtype=torch.int32, device=x_sorted.device)
    else:
        num_post_pad_t = num_tokens_post_padded.to(torch.int32)

    N_gate = w1_trellis_i32.shape[2] * 16
    N_up = w3_trellis_i32.shape[2] * 16

    return torch.ops.vllm.exl3_fused_moe_gate_up(
        x_sorted, w13_suh_gate, w13_suh_up,
        w1_trellis_i32, w3_trellis_i32,
        w1_svh, w3_svh,
        expert_ids_expanded, identity_ids, eid_blocks,
        num_post_pad_t, EM_max, N_gate, N_up, bits,
    )


# ---------------------------------------------------------------------------
# Register exl3_moe_had_gemm_had as a compound custom op (V7).
# Wraps Had_in → GEMM → Had_out into 1 opaque graph node, launching 3 GPU
# kernels back-to-back on the HIP stream with no framework dispatch overhead.
# ---------------------------------------------------------------------------

# Toggle: EXL3_MOE_COMPOUND=0 (default OFF — superseded by V2)
_USE_MOE_COMPOUND = os.environ.get("EXL3_MOE_COMPOUND", "0") == "1"


def _exl3_moe_had_gemm_had_op(
    x_sorted: torch.Tensor,           # (EM_max, K_in)
    suh_stacked: torch.Tensor,        # (E, K_in) input Had scale
    B_stacked_i32: torch.Tensor,      # (E, tiles_k, tiles_n, WPT//2) weights
    svh_stacked: torch.Tensor,        # (E, N_out) output Had scale
    eid_per_token: torch.Tensor,      # (EM_max,) expert ID per row (Had)
    eid_blocks: torch.Tensor,         # (num_m_blocks,) expert ID per block (GEMM)
    num_tokens_post_padded: torch.Tensor,
    B_fp16: torch.Tensor,             # (E, K, N) fp16 or empty
    EM_max: int,
    N_out: int,
    bits: int,
) -> torch.Tensor:
    """Compound op: Had_in → GEMM → Had_out in 1 graph node.

    Launches 3 GPU kernels back-to-back on the HIP stream.
    When HIP batched Had is available, calls _hip_ext directly to avoid
    torch.ops.vllm dispatch overhead between kernels.
    """
    has_fp16 = B_fp16.numel() > 0
    M = x_sorted.size(0)
    eid_len = eid_per_token.size(0)

    if _HAS_HIP_BATCHED_HAD and not has_fp16 and eid_len == M:
        # --- HIP fast path: direct _hip_ext calls, no framework dispatch ---
        eid_int = eid_per_token.int()

        # 1. Input Hadamard: x_sorted → xh
        xh = torch.empty_like(x_sorted)
        _hip_ext.batched_had_r_128(
            x_sorted, suh_stacked, xh,
            eid_int, 1)  # pre=True

        # 2. GEMM: xh → C (reuses existing HIP helper)
        C = _exl3_fused_moe_gemm_hip(
            xh, B_stacked_i32, eid_blocks,
            num_tokens_post_padded, EM_max, bits)

        # 3. Output Hadamard: C → out
        out = torch.empty(M, N_out, dtype=torch.float16,
                          device=x_sorted.device)
        _hip_ext.batched_had_r_128(
            C, svh_stacked, out,
            eid_int, 0)  # pre=False (post-scale)

        return out

    # --- Fallback: dispatch through torch.ops.vllm (Triton or HIP GEMM) ---
    identity_ids = torch.arange(
        M, device=x_sorted.device, dtype=torch.int32)
    eid_sliced = eid_per_token[:M]

    # 1. Input Hadamard
    xh = torch.ops.vllm.batched_had_r_128(
        x_sorted, suh_stacked, eid_sliced, 1)

    # 2. GEMM
    C = torch.ops.vllm.exl3_fused_moe_gemm(
        xh, B_stacked_i32, identity_ids, eid_blocks,
        num_tokens_post_padded,
        B_fp16 if has_fp16 else torch.empty(
            0, dtype=torch.float16, device=x_sorted.device),
        EM_max, N_out, bits, 0)

    if has_fp16:
        # FP16 path: Had is baked into weights, no output Had needed
        return C

    # 3. Output Hadamard
    out = torch.ops.vllm.batched_had_r_128(
        C, svh_stacked, eid_sliced, 0)

    return out


def _exl3_moe_had_gemm_had_fake(
    x_sorted: torch.Tensor,
    suh_stacked: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    svh_stacked: torch.Tensor,
    eid_per_token: torch.Tensor,
    eid_blocks: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    B_fp16: torch.Tensor,
    EM_max: int,
    N_out: int,
    bits: int,
) -> torch.Tensor:
    """Fake impl for Dynamo — returns correct shape."""
    return torch.empty(
        (EM_max, N_out),
        dtype=torch.float16, device=x_sorted.device)


direct_register_custom_op(
    op_name="exl3_moe_had_gemm_had",
    op_func=_exl3_moe_had_gemm_had_op,
    mutates_args=[],
    fake_impl=_exl3_moe_had_gemm_had_fake,
)


def exl3_moe_had_gemm_had(
    x_sorted: torch.Tensor,
    suh_stacked: torch.Tensor,
    B_stacked: torch.Tensor,
    svh_stacked: torch.Tensor,
    eid_per_token: torch.Tensor,
    eid_blocks: torch.Tensor,
    num_tokens_post_padded: "int | torch.Tensor",
    EM_max: int,
    bits: int = 4,
    B_i32: torch.Tensor | None = None,
    B_fp16: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compound Had→GEMM→Had: 1 graph node, 3 GPU kernels.

    Back-to-back kernel launching — queues Had_in, GEMM, Had_out on the
    HIP stream inside a single opaque custom op.
    """
    if B_i32 is None:
        B_i32 = B_stacked.view(torch.int32)
    if isinstance(num_tokens_post_padded, int):
        num_post_pad_t = torch.tensor(
            [num_tokens_post_padded], dtype=torch.int32,
            device=x_sorted.device)
    else:
        num_post_pad_t = num_tokens_post_padded.to(torch.int32)

    if B_fp16 is None:
        B_fp16_t = torch.empty(0, dtype=torch.float16, device=x_sorted.device)
        N_out = B_i32.shape[2] * 16
    else:
        B_fp16_t = B_fp16
        N_out = B_fp16.shape[2]

    return torch.ops.vllm.exl3_moe_had_gemm_had(
        x_sorted, suh_stacked, B_i32, svh_stacked,
        eid_per_token, eid_blocks,
        num_post_pad_t, B_fp16_t, EM_max, N_out, bits,
    )


# ---------------------------------------------------------------------------
# Zero-allocation compound Had→GEMM→Had (V2).
#
# Same kernel sequence as V7 exl3_moe_had_gemm_had, but ALL scratch buffers
# (xh, C, output) are passed IN from the caller — zero torch.empty inside
# the opaque op.  This lets torch.compile's memory planner own the buffers
# so they live in the CUDA graph memory pool instead of hitting real
# cudaMalloc on every graph replay.
#
# Toggle: EXL3_MOE_COMPOUND_V2=1 (default ON when HIP batched Had available)
# ---------------------------------------------------------------------------

_USE_MOE_COMPOUND_V2 = os.environ.get("EXL3_MOE_COMPOUND_V2", "1") == "1"

# Scratch buffer cache: {(name, device, cols): tensor}
# Grows if rows increase (decode→prefill), never shrinks.
_moe_scratch_bufs = {}


def _get_moe_scratch_buf(name: str, rows: int, cols: int, device):
    """Get or allocate a cached scratch buffer for compound MoE ops.

    Keyed by (name, device, cols). Grows row-wise if needed.
    Allocated via torch.empty in Python (visible to torch.compile memory
    planner → replayed from CUDA graph memory pool).
    """
    key = (name, device, cols)
    buf = _moe_scratch_bufs.get(key)
    if buf is None or buf.shape[0] < rows:
        _moe_scratch_bufs[key] = torch.empty(
            (max(rows, 1), cols), dtype=torch.float16, device=device)
        buf = _moe_scratch_bufs[key]
    return buf[:rows]


def _exl3_moe_had_gemm_had_v2_op(
    x_sorted: torch.Tensor,           # (EM_max, K_in)
    suh_stacked: torch.Tensor,        # (E, K_in) input Had scale
    B_stacked_i32: torch.Tensor,      # (E, tiles_k, tiles_n, WPT//2) weights
    svh_stacked: torch.Tensor,        # (E, N_out) output Had scale
    eid_per_token: torch.Tensor,      # (EM_max,) expert ID per row (Had)
    eid_blocks: torch.Tensor,         # (num_m_blocks,) expert ID per block (GEMM)
    num_tokens_post_padded: torch.Tensor,
    buf_xh: torch.Tensor,            # scratch: (EM_max, K_in) — mutated
    buf_C: torch.Tensor,             # scratch: (EM_max, N_out) — mutated
    buf_C_partial: torch.Tensor,     # scratch: (split_k, EM_max, N_out) — mutated
    output: torch.Tensor,            # result: (EM_max, N_out) — mutated
    EM_max: int,
    N_out: int,
    bits: int,
    split_k: int,
) -> None:
    """V2 compound op: Had_in → GEMM → Had_out, zero internal allocations.

    All scratch buffers are passed in and mutated in-place.
    Launches 3 GPU kernels (+ 1 reduce if split_k > 1) back-to-back.
    """
    if _HAS_HIP_BATCHED_HAD and _HAS_HIP_MOE:
        # --- HIP fast path: direct kernel calls, no Python dispatch ---
        word_idx, next_word_idx, shift_tbl = get_bit_tables(
            bits, x_sorted.device)

        # 1. Input Hadamard: x_sorted → buf_xh
        _hip_ext.batched_had_r_128(
            x_sorted, suh_stacked, buf_xh,
            eid_per_token, 1)  # pre=True

        # 2. GEMM: buf_xh → buf_C (direct kernel call, no wrapper allocation)
        _hip_ext.exl3_fused_moe_gemm(
            buf_xh, B_stacked_i32, buf_C,
            eid_blocks, num_tokens_post_padded,
            word_idx, next_word_idx, shift_tbl,
            EM_max, bits, 0,  # cb=0
            split_k, buf_C_partial,
        )

        # 3. Output Hadamard: buf_C → output
        _hip_ext.batched_had_r_128(
            buf_C, svh_stacked, output,
            eid_per_token, 0)  # pre=False (post-scale)
        return

    # --- Fallback: Triton path ---
    M = x_sorted.size(0)
    eid_sliced = eid_per_token[:M]

    # 1. Input Hadamard
    xh = _batched_had_r_128_triton(x_sorted, suh_stacked, eid_sliced, True)
    buf_xh.copy_(xh)

    # 2. GEMM (use Triton fused MoE)
    C = _exl3_fused_moe_gemm_impl(
        buf_xh, B_stacked_i32.view(torch.float16),
        eid_blocks, num_tokens_post_padded, EM_max, bits)
    buf_C[:C.shape[0], :C.shape[1]].copy_(C)

    # 3. Output Hadamard
    out = _batched_had_r_128_triton(buf_C, svh_stacked, eid_sliced, False)
    output.copy_(out)


def _exl3_moe_had_gemm_had_v2_fake(
    x_sorted: torch.Tensor,
    suh_stacked: torch.Tensor,
    B_stacked_i32: torch.Tensor,
    svh_stacked: torch.Tensor,
    eid_per_token: torch.Tensor,
    eid_blocks: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    buf_xh: torch.Tensor,
    buf_C: torch.Tensor,
    buf_C_partial: torch.Tensor,
    output: torch.Tensor,
    EM_max: int,
    N_out: int,
    bits: int,
    split_k: int,
) -> None:
    """Fake impl for Dynamo — mutates output in-place, returns None."""
    # output is mutated in-place; Dynamo tracks via mutates_args
    return


direct_register_custom_op(
    op_name="exl3_moe_had_gemm_had_v2",
    op_func=_exl3_moe_had_gemm_had_v2_op,
    mutates_args=["buf_xh", "buf_C", "buf_C_partial", "output"],
    fake_impl=_exl3_moe_had_gemm_had_v2_fake,
)


def exl3_moe_had_gemm_had_v2(
    x_sorted: torch.Tensor,
    suh_stacked: torch.Tensor,
    B_stacked: torch.Tensor,
    svh_stacked: torch.Tensor,
    eid_per_token: torch.Tensor,
    eid_blocks: torch.Tensor,
    num_tokens_post_padded: "int | torch.Tensor",
    buf_xh: torch.Tensor,
    buf_C: torch.Tensor,
    buf_C_partial: torch.Tensor,
    output: torch.Tensor,
    EM_max: int,
    bits: int = 4,
    split_k: int = 0,
    B_i32: torch.Tensor | None = None,
) -> torch.Tensor:
    """V2 compound Had→GEMM→Had: zero internal allocations.

    All scratch buffers (buf_xh, buf_C, buf_C_partial, output) are
    caller-allocated and passed in. Returns output (same tensor, mutated
    in-place).
    """
    if B_i32 is None:
        B_i32 = B_stacked.view(torch.int32)
    if isinstance(num_tokens_post_padded, int):
        num_post_pad_t = torch.tensor(
            [num_tokens_post_padded], dtype=torch.int32,
            device=x_sorted.device)
    else:
        num_post_pad_t = num_tokens_post_padded.to(torch.int32)

    N_out = output.shape[1]

    torch.ops.vllm.exl3_moe_had_gemm_had_v2(
        x_sorted, suh_stacked, B_i32, svh_stacked,
        eid_per_token, eid_blocks,
        num_post_pad_t, buf_xh, buf_C, buf_C_partial, output,
        EM_max, N_out, bits, split_k,
    )
    return output


__all__ = ["exl3_gemm", "exl3_fused_moe_gemm", "exl3_fused_moe_gemm_had",
           "had_r_128", "batched_had_r_128", "exl3_fused_moe_gate_up",
           "exl3_moe_had_gemm_had", "exl3_moe_had_gemm_had_v2"]
