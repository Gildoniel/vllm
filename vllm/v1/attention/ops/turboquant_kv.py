"""TurboQuant KV Cache — encode/decode kernels.

Encode: FP16 KV → WHT rotation → Lloyd-Max quantize → uint8 index storage
Decode: uint8 indices → centroid lookup → inverse WHT → FP16 KV

Phase 1: PyTorch implementation (WHT via matmul, quantize via searchsorted)
Phase 2: Fused Triton kernels + 3-bit packing (TODO)

Reference: Google TurboQuant (ICLR 2026, arXiv:2504.19874)
"""

import torch
import math

from vllm.v1.attention.ops.turboquant_centroids import (
    get_centroids,
    get_boundaries,
)

# ---------------------------------------------------------------------------
# Hadamard matrix (self-contained, no vLLM imports needed)
# ---------------------------------------------------------------------------

_had_cache: dict = {}


def _get_had_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Build and cache the normalized Hadamard matrix for given dimension."""
    key = (dim, device)
    if key not in _had_cache:
        n_stages = int(math.log2(dim))
        assert 2 ** n_stages == dim, f"dim must be power of 2, got {dim}"
        H = torch.tensor([[1.0]])
        for _ in range(n_stages):
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        H = H / math.sqrt(dim)
        _had_cache[key] = H.to(device=device, dtype=torch.float16)
    return _had_cache[key]


# ---------------------------------------------------------------------------
# Encode: FP16 KV → TurboQuant (indices + norms)
# ---------------------------------------------------------------------------

def turboquant_encode(
    kv: torch.Tensor,           # [num_tokens, num_heads, head_dim], FP16
    bits: int = 3,              # 3 or 4
    had_fn=None,                # callable: had_r_128(x, out, None, None, 1.0)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode KV vectors to TurboQuant format.

    Args:
        kv: Input KV tensor [num_tokens, num_heads, head_dim] in FP16.
        bits: Quantization bits (3 or 4).
        had_fn: Optional external Had-128 function (e.g. DPP-fused HIP kernel).
                If None, uses matmul with cached Hadamard matrix.

    Returns:
        indices: [num_tokens, num_heads, head_dim] uint8 (1 index per byte)
        norms: [num_tokens, num_heads] FP16
    """
    num_tokens, num_heads, head_dim = kv.shape
    assert head_dim in (128, 256), f"head_dim must be 128 or 256, got {head_dim}"

    device = kv.device

    # Step 1: Compute L2 norms
    kv_f32 = kv.float()
    norms = kv_f32.norm(dim=-1).half()  # [num_tokens, num_heads]

    # Step 2: Normalize to unit vectors
    kv_normed = kv_f32 / (norms.float().unsqueeze(-1) + 1e-12)

    # Step 3: WHT rotation (self-inverse: WHT(WHT(x)) = x)
    kv_rotated = kv_normed.half()
    if had_fn is not None:
        out = torch.empty_like(kv_rotated)
        had_fn(kv_rotated, out, None, None, 1.0)
        kv_rotated = out
    else:
        H = _get_had_matrix(head_dim, device)
        orig_shape = kv_rotated.shape
        kv_rotated = (kv_rotated.reshape(-1, head_dim) @ H.T).reshape(orig_shape)

    # Step 4: Quantize — searchsorted against Lloyd-Max boundaries
    boundaries = get_boundaries(bits, head_dim, device=device, dtype=torch.float32)
    indices = torch.searchsorted(boundaries, kv_rotated.float().reshape(-1)).reshape(
        num_tokens, num_heads, head_dim
    ).to(torch.uint8)

    return indices, norms


# ---------------------------------------------------------------------------
# Decode: TurboQuant (indices + norms) → FP16 KV
# ---------------------------------------------------------------------------

def turboquant_decode(
    indices: torch.Tensor,      # [num_tokens, num_heads, head_dim] uint8
    norms: torch.Tensor,        # [num_tokens, num_heads] FP16
    bits: int = 3,
    head_dim: int = 128,
    had_fn=None,
) -> torch.Tensor:
    """Decode TurboQuant indices back to FP16 KV vectors.

    Args:
        indices: Quantization indices [num_tokens, num_heads, head_dim] uint8.
        norms: L2 norms [num_tokens, num_heads] FP16.
        bits: Quantization bits (3 or 4).
        head_dim: Head dimension (128 or 256).
        had_fn: Optional external Had function. If None, uses matmul.

    Returns:
        kv: [num_tokens, num_heads, head_dim] FP16
    """
    device = indices.device

    # Step 1: Centroid lookup
    centroids = get_centroids(bits, head_dim, device=device, dtype=torch.float32)
    y = centroids[indices.long()]  # [num_tokens, num_heads, head_dim]

    # Step 2: Inverse WHT (WHT is self-inverse)
    y_half = y.half()
    if had_fn is not None:
        out = torch.empty_like(y_half)
        had_fn(y_half, out, None, None, 1.0)
        y_half = out
    else:
        H = _get_had_matrix(head_dim, device)
        orig_shape = y_half.shape
        y_half = (y_half.reshape(-1, head_dim) @ H.T).reshape(orig_shape)

    # Step 3: Scale by norm
    kv = y_half.float() * norms.float().unsqueeze(-1)

    return kv.half()
