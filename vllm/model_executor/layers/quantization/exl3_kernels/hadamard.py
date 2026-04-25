"""
PyTorch and Triton implementations of the 128-dim Hadamard transform used by EXL3.

Provides the same interface as exllamav3_ext.had_r_128:
    had_r_128(x, out, pre_scale, post_scale, r_scale)

Where:
    - x: input tensor, shape (..., dim) where dim is divisible by 128
    - out: output tensor (same shape as x, can be x for in-place)
    - pre_scale: optional fp16 tensor of sign flips, applied before Hadamard
    - post_scale: optional fp16 tensor of sign flips, applied after Hadamard
    - r_scale: scalar multiplier (typically 1.0; Hadamard already includes 1/sqrt(128))

Uses fp16 matmul for speed on GPU. The 128x128 Hadamard matrix is cached per device.

batched_had_r_128() has a Triton kernel path (default) that fuses gather+scale+Had
into a single kernel launch, saving ~576 launches/step over the PyTorch 3-op path.
Set EXL3_TRITON_HAD=0 to fall back to the PyTorch path.
"""

import os
import torch
import math
import triton
import triton.language as tl


def _build_hadamard_128() -> torch.Tensor:
    """Build the 128x128 normalized Hadamard matrix (1/sqrt(128) scaling)."""
    H = torch.tensor([[1.0]])
    for _ in range(7):  # 2^7 = 128
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(128.0)


# Lazily cached Hadamard matrix per device (fp16 for speed)
_had128_cache = {}


def _get_had128(device: torch.device) -> torch.Tensor:
    if device not in _had128_cache:
        _had128_cache[device] = _build_hadamard_128().to(device=device, dtype=torch.float16)
    return _had128_cache[device]


# H_16 Hadamard matrix cache (for Triton batched Had-128)
_h16_cache = {}


def _get_h16(device):
    """Build and cache the 16x16 normalized Hadamard matrix (entries +/-1/sqrt(16)) as fp16."""
    if device not in _h16_cache:
        H = torch.tensor([[1.0]], dtype=torch.float32)
        for _ in range(4):  # 2^4 = 16
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        H = H / (16.0 ** 0.5)
        _h16_cache[device] = H.to(device=device, dtype=torch.float16)
    return _h16_cache[device]


def had_r_128(
    x: torch.Tensor,
    out: torch.Tensor,
    pre_scale: torch.Tensor | None,
    post_scale: torch.Tensor | None,
    r_scale: float,
) -> None:
    """
    Apply blockwise 128-dim Hadamard transform with optional pre/post sign-flip scales.

    Matches the ExLlamaV3 ext.had_r_128(x, xh, suh, svh, scale) interface.

    Args:
        x: Input tensor, shape (..., dim) where dim % 128 == 0. dtype=float16.
        out: Output tensor, same shape as x. Can alias x for in-place operation.
        pre_scale: Sign-flip scales applied BEFORE Hadamard, shape (dim,), fp16. Or None.
        post_scale: Sign-flip scales applied AFTER Hadamard, shape (dim,), fp16. Or None.
        r_scale: Additional scalar multiplier (typically 1.0).
    """
    orig_shape = x.shape
    dim = x.shape[-1]
    assert dim % 128 == 0, f"Last dimension must be divisible by 128, got {dim}"

    H = _get_had128(x.device)

    # Work in fp16 for speed; convert input if needed
    x_flat = x.reshape(-1, dim).half()

    # Pre-scale (sign flips before Hadamard)
    if pre_scale is not None:
        x_flat = x_flat * pre_scale.unsqueeze(0)

    # Blockwise Hadamard via matmul: (batch*num_blocks, 128) @ (128, 128)
    result = torch.mm(x_flat.reshape(-1, 128), H.T)

    # Apply r_scale
    if r_scale != 1.0:
        result = result * r_scale

    # Post-scale (sign flips after Hadamard)
    result = result.reshape(-1, dim)
    if post_scale is not None:
        result = result * post_scale.unsqueeze(0)

    out.copy_(result.reshape(orig_shape))


# =============================================================================
# Triton kernel for batched Had-128: fused gather + scale + Hadamard
#
# Had-128 = H_8 (x) H_16 (Kronecker product):
#   1. Reshape 128 cols -> 8 groups of 16
#   2. H_16: matmul each group by H_16 (includes 1/sqrt(16))
#   3. H_8: butterfly across 8 groups (3 rounds), then * 1/sqrt(8)
#   Total scaling = 1/sqrt(128)
#
# Grid: (M, dim // 128) — one program per row per 128-col block
# =============================================================================

@triton.jit
def _batched_had_r_128_kernel(
    X_ptr, Scale_ptr, Out_ptr, EID_ptr, H16_ptr,
    M,
    stride_xm, stride_xd,
    stride_sm, stride_sd,
    stride_om, stride_od,
    stride_h16_r, stride_h16_c,
    PRE: tl.constexpr,
):
    pid_m = tl.program_id(0)   # row index
    pid_n = tl.program_id(1)   # which 128-col block

    if pid_m >= M:
        return

    base_col = pid_n * 128
    offs16 = tl.arange(0, 16)

    # Gather expert ID for this row
    eid = tl.load(EID_ptr + pid_m)
    eid = tl.maximum(eid, 0)  # clamp padding rows

    # Load 8 groups of 16 from X
    x_base = X_ptr + pid_m * stride_xm + base_col * stride_xd
    v0 = tl.load(x_base + (0 * 16 + offs16) * stride_xd).to(tl.float32)
    v1 = tl.load(x_base + (1 * 16 + offs16) * stride_xd).to(tl.float32)
    v2 = tl.load(x_base + (2 * 16 + offs16) * stride_xd).to(tl.float32)
    v3 = tl.load(x_base + (3 * 16 + offs16) * stride_xd).to(tl.float32)
    v4 = tl.load(x_base + (4 * 16 + offs16) * stride_xd).to(tl.float32)
    v5 = tl.load(x_base + (5 * 16 + offs16) * stride_xd).to(tl.float32)
    v6 = tl.load(x_base + (6 * 16 + offs16) * stride_xd).to(tl.float32)
    v7 = tl.load(x_base + (7 * 16 + offs16) * stride_xd).to(tl.float32)

    # Load scale for this expert
    s_base = Scale_ptr + eid * stride_sm + base_col * stride_sd
    s0 = tl.load(s_base + (0 * 16 + offs16) * stride_sd).to(tl.float32)
    s1 = tl.load(s_base + (1 * 16 + offs16) * stride_sd).to(tl.float32)
    s2 = tl.load(s_base + (2 * 16 + offs16) * stride_sd).to(tl.float32)
    s3 = tl.load(s_base + (3 * 16 + offs16) * stride_sd).to(tl.float32)
    s4 = tl.load(s_base + (4 * 16 + offs16) * stride_sd).to(tl.float32)
    s5 = tl.load(s_base + (5 * 16 + offs16) * stride_sd).to(tl.float32)
    s6 = tl.load(s_base + (6 * 16 + offs16) * stride_sd).to(tl.float32)
    s7 = tl.load(s_base + (7 * 16 + offs16) * stride_sd).to(tl.float32)

    # Pre mode: scale before Hadamard
    if PRE:
        v0 = v0 * s0
        v1 = v1 * s1
        v2 = v2 * s2
        v3 = v3 * s3
        v4 = v4 * s4
        v5 = v5 * s5
        v6 = v6 * s6
        v7 = v7 * s7

    # Load H_16 matrix (16x16)
    h16_r = tl.arange(0, 16)[:, None]
    h16_c = tl.arange(0, 16)[None, :]
    H16 = tl.load(H16_ptr + h16_r * stride_h16_r + h16_c * stride_h16_c)

    # Step 1: H_16 within each group — (1,16) @ (16,16) -> (1,16)
    v0 = tl.dot(v0[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)
    v1 = tl.dot(v1[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)
    v2 = tl.dot(v2[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)
    v3 = tl.dot(v3[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)
    v4 = tl.dot(v4[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)
    v5 = tl.dot(v5[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)
    v6 = tl.dot(v6[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)
    v7 = tl.dot(v7[None, :].to(tl.float16), H16).to(tl.float32).reshape(16)

    # Step 2: H_8 butterfly across 8 groups (3 rounds, in fp32)
    # Round 1
    t0 = v0 + v1
    t1 = v0 - v1
    t2 = v2 + v3
    t3 = v2 - v3
    t4 = v4 + v5
    t5 = v4 - v5
    t6 = v6 + v7
    t7 = v6 - v7
    # Round 2
    u0 = t0 + t2
    u1 = t1 + t3
    u2 = t0 - t2
    u3 = t1 - t3
    u4 = t4 + t6
    u5 = t5 + t7
    u6 = t4 - t6
    u7 = t5 - t7
    # Round 3
    v0 = u0 + u4
    v1 = u1 + u5
    v2 = u2 + u6
    v3 = u3 + u7
    v4 = u0 - u4
    v5 = u1 - u5
    v6 = u2 - u6
    v7 = u3 - u7

    # Scale: 1/sqrt(8) for H_8 (H_16 already includes 1/sqrt(16))
    inv_sqrt8: tl.constexpr = 0.35355339059327373
    v0 = v0 * inv_sqrt8
    v1 = v1 * inv_sqrt8
    v2 = v2 * inv_sqrt8
    v3 = v3 * inv_sqrt8
    v4 = v4 * inv_sqrt8
    v5 = v5 * inv_sqrt8
    v6 = v6 * inv_sqrt8
    v7 = v7 * inv_sqrt8

    # Post mode: scale after Hadamard
    if not PRE:
        v0 = v0 * s0
        v1 = v1 * s1
        v2 = v2 * s2
        v3 = v3 * s3
        v4 = v4 * s4
        v5 = v5 * s5
        v6 = v6 * s6
        v7 = v7 * s7

    # Store
    o_base = Out_ptr + pid_m * stride_om + base_col * stride_od
    tl.store(o_base + (0 * 16 + offs16) * stride_od, v0.to(tl.float16))
    tl.store(o_base + (1 * 16 + offs16) * stride_od, v1.to(tl.float16))
    tl.store(o_base + (2 * 16 + offs16) * stride_od, v2.to(tl.float16))
    tl.store(o_base + (3 * 16 + offs16) * stride_od, v3.to(tl.float16))
    tl.store(o_base + (4 * 16 + offs16) * stride_od, v4.to(tl.float16))
    tl.store(o_base + (5 * 16 + offs16) * stride_od, v5.to(tl.float16))
    tl.store(o_base + (6 * 16 + offs16) * stride_od, v6.to(tl.float16))
    tl.store(o_base + (7 * 16 + offs16) * stride_od, v7.to(tl.float16))


def _batched_had_r_128_triton(
    x_sorted: torch.Tensor,
    scale_stacked: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    pre: bool = True,
) -> torch.Tensor:
    """Triton implementation of batched Had-128: 1 kernel launch instead of 3."""
    M, dim = x_sorted.shape
    assert dim % 128 == 0

    H16 = _get_h16(x_sorted.device)
    out = torch.empty_like(x_sorted)

    grid = (M, dim // 128)
    _batched_had_r_128_kernel[grid](
        x_sorted, scale_stacked, out, expert_ids_expanded, H16,
        M,
        x_sorted.stride(0), x_sorted.stride(1),
        scale_stacked.stride(0), scale_stacked.stride(1),
        out.stride(0), out.stride(1),
        H16.stride(0), H16.stride(1),
        PRE=pre,
        num_warps=1,
        num_stages=1,
    )
    return out


# =============================================================================
# batched_had_r_128: public API with Triton/PyTorch toggle
# =============================================================================

_USE_TRITON_HAD = os.environ.get("EXL3_TRITON_HAD", "1") == "1"


def _batched_had_r_128_pytorch(
    x_sorted: torch.Tensor,
    scale_stacked: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    pre: bool = True,
) -> torch.Tensor:
    """PyTorch reference: 3 kernel launches (gather + scale + mm)."""
    dim = x_sorted.shape[-1]
    H = _get_had128(x_sorted.device)

    scale_per_token = scale_stacked[expert_ids_expanded]

    if pre:
        x_scaled = x_sorted * scale_per_token
        result = torch.mm(x_scaled.reshape(-1, 128), H.T)
        return result.reshape_as(x_sorted)
    else:
        result = torch.mm(x_sorted.reshape(-1, 128), H.T)
        result = result.reshape_as(x_sorted)
        return result * scale_per_token


def batched_had_r_128(
    x_sorted: torch.Tensor,
    scale_stacked: torch.Tensor,
    expert_ids_expanded: torch.Tensor,
    pre: bool = True,
) -> torch.Tensor:
    """
    Batched Hadamard-128 with expert-indexed scales for fused MoE.

    Instead of looping per-expert to apply had_r_128 with per-expert suh/svh,
    this gathers the correct scale per token and applies Hadamard in one batch.

    Uses a Triton kernel (1 launch) by default. Set EXL3_TRITON_HAD=0 for
    the PyTorch fallback (3 launches).

    Args:
        x_sorted: (EM, dim) float16 — sorted tokens (padded).
        scale_stacked: (E, dim) float16 — per-expert scales (suh or svh).
        expert_ids_expanded: (EM,) int32/int64 — expert ID per token row.
        pre: If True, scale is applied before Hadamard (suh).
             If False, scale is applied after Hadamard (svh).

    Returns:
        result: (EM, dim) float16
    """
    if _USE_TRITON_HAD:
        return _batched_had_r_128_triton(
            x_sorted, scale_stacked, expert_ids_expanded, pre)
    else:
        return _batched_had_r_128_pytorch(
            x_sorted, scale_stacked, expert_ids_expanded, pre)
