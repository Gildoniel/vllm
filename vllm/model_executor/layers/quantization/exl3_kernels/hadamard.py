"""
PyTorch implementation of the 128-dim Hadamard transform used by EXL3.

Provides the same interface as exllamav3_ext.had_r_128:
    had_r_128(x, out, pre_scale, post_scale, r_scale)

Where:
    - x: input tensor, shape (..., dim) where dim is divisible by 128
    - out: output tensor (same shape as x, can be x for in-place)
    - pre_scale: optional fp16 tensor of sign flips, applied before Hadamard
    - post_scale: optional fp16 tensor of sign flips, applied after Hadamard
    - r_scale: scalar multiplier (typically 1.0; Hadamard already includes 1/sqrt(128))

Uses fp16 matmul for speed on GPU. The 128x128 Hadamard matrix is cached per device.
"""

import torch
import math


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

    Args:
        x_sorted: (EM, dim) float16 — sorted tokens (padded).
        scale_stacked: (E, dim) float16 — per-expert scales (suh or svh).
        expert_ids_expanded: (EM,) int32/int64 — expert ID per token row.
        pre: If True, scale is applied before Hadamard (suh).
             If False, scale is applied after Hadamard (svh).

    Returns:
        result: (EM, dim) float16
    """
    dim = x_sorted.shape[-1]
    H = _get_had128(x_sorted.device)

    # Gather per-token scale from stacked expert scales: 1 kernel
    scale_per_token = scale_stacked[expert_ids_expanded]  # (EM, dim)

    if pre:
        # Pre-scale (suh): multiply then Hadamard
        x_scaled = x_sorted * scale_per_token
        result = torch.mm(x_scaled.reshape(-1, 128), H.T)
        return result.reshape_as(x_sorted)
    else:
        # Post-scale (svh): Hadamard then multiply
        result = torch.mm(x_sorted.reshape(-1, 128), H.T)
        result = result.reshape_as(x_sorted)
        return result * scale_per_token
