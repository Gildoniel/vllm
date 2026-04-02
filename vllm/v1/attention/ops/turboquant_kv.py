"""TurboQuant KV Cache — encode/decode with bit packing + Triton kernels.

Encode: FP16 KV → WHT rotation → Lloyd-Max quantize → packed bit storage
Decode: packed bits → centroid lookup → inverse WHT → FP16 KV

Memory layout per token per KV head:
  3-bit (d=128): [norm FP16 (2B)] [128 × 3-bit packed (48B)] = 50 bytes (5.1x vs FP16)
  4-bit (d=128): [norm FP16 (2B)] [128 × 4-bit packed (64B)] = 66 bytes (3.9x vs FP16)

WHT rotation is done externally (matmul fallback or DPP-fused HIP kernel).
Triton kernels handle quantize+pack (encode) and unpack+lookup (decode).

Reference: Google TurboQuant (ICLR 2026, arXiv:2504.19874)
"""

import torch
import math

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_centroids import (
    get_centroids,
    get_boundaries,
)

# ---------------------------------------------------------------------------
# Hadamard matrix (self-contained, no heavy vLLM imports)
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
# Triton: Fused quantize + 4-bit pack kernel
# ---------------------------------------------------------------------------

@triton.jit
def _tq_encode_pack4_kernel(
    # WHT-rotated input: [num_tokens, num_heads, head_dim] FP16
    rotated_ptr,
    # Output packed: [num_tokens, num_heads, head_dim // 2] uint8
    packed_ptr,
    # Output norms: [num_tokens, num_heads] FP16 (pre-computed, just copy)
    # Boundaries: (15,) float32
    boundaries_ptr,
    # Strides
    rot_stride_token: tl.int64,
    rot_stride_head: tl.int64,
    pack_stride_token: tl.int64,
    pack_stride_head: tl.int64,
    # Constants
    HEAD_DIM: tl.constexpr,         # 128
    PACKED_DIM: tl.constexpr,       # 64 (= HEAD_DIM // 2)
):
    """Quantize WHT-rotated values and pack into 4-bit nibbles.

    Each output byte = lo_index | (hi_index << 4).
    Grid: (num_tokens, num_heads)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    base = token_idx * rot_stride_token + head_idx * rot_stride_head

    # Load rotated vector
    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(rotated_ptr + base + offs).to(tl.float32)

    # Quantize: count boundaries exceeded (searchsorted equivalent)
    indices = tl.zeros((HEAD_DIM,), dtype=tl.int32)
    for j in tl.static_range(15):  # 16 levels → 15 boundaries
        b_j = tl.load(boundaries_ptr + j)
        indices += (x > b_j).to(tl.int32)

    # Pack pairs into nibbles: byte[k] = indices[2k] | (indices[2k+1] << 4)
    even_offs = tl.arange(0, PACKED_DIM) * 2
    odd_offs = even_offs + 1
    even_idx = tl.load(rotated_ptr + base + even_offs)  # dummy load for gather
    # Can't gather from register vector in Triton — recompute for even/odd

    # Recompute indices for even positions
    x_even = tl.load(rotated_ptr + base + even_offs).to(tl.float32)
    idx_even = tl.zeros((PACKED_DIM,), dtype=tl.int32)
    for j in tl.static_range(15):
        b_j = tl.load(boundaries_ptr + j)
        idx_even += (x_even > b_j).to(tl.int32)

    # Recompute indices for odd positions
    x_odd = tl.load(rotated_ptr + base + odd_offs).to(tl.float32)
    idx_odd = tl.zeros((PACKED_DIM,), dtype=tl.int32)
    for j in tl.static_range(15):
        b_j = tl.load(boundaries_ptr + j)
        idx_odd += (x_odd > b_j).to(tl.int32)

    packed = (idx_even | (idx_odd << 4)).to(tl.uint8)

    pack_base = token_idx * pack_stride_token + head_idx * pack_stride_head
    tl.store(packed_ptr + pack_base + tl.arange(0, PACKED_DIM), packed)


@triton.jit
def _tq_encode_pack3_kernel(
    # WHT-rotated input: [num_tokens, num_heads, head_dim] FP16
    rotated_ptr,
    # Output packed: [num_tokens, num_heads, head_dim * 3 // 8] uint8
    packed_ptr,
    # Boundaries: (7,) float32
    boundaries_ptr,
    # Strides
    rot_stride_token: tl.int64,
    rot_stride_head: tl.int64,
    pack_stride_token: tl.int64,
    pack_stride_head: tl.int64,
    # Constants
    HEAD_DIM: tl.constexpr,         # 128
    PACKED_DIM: tl.constexpr,       # 48 (= HEAD_DIM * 3 // 8)
):
    """Quantize WHT-rotated values and pack into 3-bit groups.

    Packing: groups of 8 indices (each 3-bit, 0-7) → 3 bytes (24 bits).
    Byte layout per group of 8 indices (a,b,c,d,e,f,g,h):
      byte0 = a | (b << 3) | (c_lo << 6)       # c_lo = c & 0x3
      byte1 = (c_hi) | (d << 1) | (e << 4) | (f_lo << 7)  # c_hi = c >> 2
      byte2 = (f_hi) | (g << 1) | (h << 4)     # f_hi = f >> 1, f_lo = f & 1

    Grid: (num_tokens, num_heads)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    base = token_idx * rot_stride_token + head_idx * rot_stride_head
    pack_base = token_idx * pack_stride_token + head_idx * pack_stride_head

    # Process in groups of 8 values → 3 output bytes
    N_GROUPS: tl.constexpr = HEAD_DIM // 8  # 16 groups for d=128

    for g in tl.static_range(N_GROUPS):
        g_offs = tl.arange(0, 8) + g * 8
        x = tl.load(rotated_ptr + base + g_offs).to(tl.float32)

        # Quantize: 7 boundaries → indices 0..7
        idx = tl.zeros((8,), dtype=tl.int32)
        for j in tl.static_range(7):
            b_j = tl.load(boundaries_ptr + j)
            idx += (x > b_j).to(tl.int32)

        # Extract individual indices via masking
        # idx is a vector of 8 — extract each element
        i0 = tl.sum(tl.where(tl.arange(0, 8) == 0, idx, 0))
        i1 = tl.sum(tl.where(tl.arange(0, 8) == 1, idx, 0))
        i2 = tl.sum(tl.where(tl.arange(0, 8) == 2, idx, 0))
        i3 = tl.sum(tl.where(tl.arange(0, 8) == 3, idx, 0))
        i4 = tl.sum(tl.where(tl.arange(0, 8) == 4, idx, 0))
        i5 = tl.sum(tl.where(tl.arange(0, 8) == 5, idx, 0))
        i6 = tl.sum(tl.where(tl.arange(0, 8) == 6, idx, 0))
        i7 = tl.sum(tl.where(tl.arange(0, 8) == 7, idx, 0))

        # Pack 8 × 3-bit into 3 bytes
        byte0 = i0 | (i1 << 3) | ((i2 & 0x3) << 6)
        byte1 = (i2 >> 2) | (i3 << 1) | (i4 << 4) | ((i5 & 0x1) << 7)
        byte2 = (i5 >> 1) | (i6 << 2) | (i7 << 5)

        out_off = pack_base + g * 3
        tl.store(packed_ptr + out_off, byte0.to(tl.uint8))
        tl.store(packed_ptr + out_off + 1, byte1.to(tl.uint8))
        tl.store(packed_ptr + out_off + 2, byte2.to(tl.uint8))


# ---------------------------------------------------------------------------
# Triton: Fused unpack + centroid lookup kernels
# ---------------------------------------------------------------------------

@triton.jit
def _tq_decode_unpack4_kernel(
    # Input packed: [num_tokens, num_heads, head_dim // 2] uint8
    packed_ptr,
    # Input norms: [num_tokens, num_heads] FP16
    norms_ptr,
    # Output: [num_tokens, num_heads, head_dim] FP16
    out_ptr,
    # Centroids: (16,) float32
    centroids_ptr,
    # Strides
    pack_stride_token: tl.int64,
    pack_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    # Constants
    HEAD_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,
):
    """Unpack 4-bit nibbles, lookup centroids, scale by norm.

    Inverse WHT is done externally. Output is in WHT-rotated space × norm.
    Grid: (num_tokens, num_heads)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pack_base = token_idx * pack_stride_token + head_idx * pack_stride_head

    # Load packed bytes
    pack_offs = tl.arange(0, PACKED_DIM)
    packed = tl.load(packed_ptr + pack_base + pack_offs).to(tl.int32)

    # Unpack: lo nibble and hi nibble
    idx_even = packed & 0xF          # low 4 bits
    idx_odd = (packed >> 4) & 0xF    # high 4 bits

    # Centroid lookup via tl.where cascade (16 levels)
    y_even = tl.zeros((PACKED_DIM,), dtype=tl.float32)
    y_odd = tl.zeros((PACKED_DIM,), dtype=tl.float32)
    for c in tl.static_range(16):
        c_val = tl.load(centroids_ptr + c).to(tl.float32)
        y_even = tl.where(idx_even == c, c_val, y_even)
        y_odd = tl.where(idx_odd == c, c_val, y_odd)

    # Load norm and scale
    norm = tl.load(norms_ptr + token_idx * norm_stride_token + head_idx).to(tl.float32)
    y_even = y_even * norm
    y_odd = y_odd * norm

    # Interleave and store: out[2k] = y_even[k], out[2k+1] = y_odd[k]
    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    even_offs = tl.arange(0, PACKED_DIM) * 2
    odd_offs = even_offs + 1
    tl.store(out_ptr + out_base + even_offs, y_even.to(tl.float16))
    tl.store(out_ptr + out_base + odd_offs, y_odd.to(tl.float16))


@triton.jit
def _tq_decode_unpack3_kernel(
    # Input packed: [num_tokens, num_heads, head_dim * 3 // 8] uint8
    packed_ptr,
    # Input norms: [num_tokens, num_heads] FP16
    norms_ptr,
    # Output: [num_tokens, num_heads, head_dim] FP16
    out_ptr,
    # Centroids: (8,) float32
    centroids_ptr,
    # Strides
    pack_stride_token: tl.int64,
    pack_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    # Constants
    HEAD_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,
):
    """Unpack 3-bit groups, lookup centroids, scale by norm.

    Grid: (num_tokens, num_heads)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pack_base = token_idx * pack_stride_token + head_idx * pack_stride_head
    out_base = token_idx * out_stride_token + head_idx * out_stride_head

    norm = tl.load(norms_ptr + token_idx * norm_stride_token + head_idx).to(tl.float32)

    # Preload centroids into registers
    c0 = tl.load(centroids_ptr + 0).to(tl.float32)
    c1 = tl.load(centroids_ptr + 1).to(tl.float32)
    c2 = tl.load(centroids_ptr + 2).to(tl.float32)
    c3 = tl.load(centroids_ptr + 3).to(tl.float32)
    c4 = tl.load(centroids_ptr + 4).to(tl.float32)
    c5 = tl.load(centroids_ptr + 5).to(tl.float32)
    c6 = tl.load(centroids_ptr + 6).to(tl.float32)
    c7 = tl.load(centroids_ptr + 7).to(tl.float32)

    N_GROUPS: tl.constexpr = HEAD_DIM // 8

    for g in tl.static_range(N_GROUPS):
        # Load 3 bytes for this group
        b0 = tl.load(packed_ptr + pack_base + g * 3).to(tl.int32)
        b1 = tl.load(packed_ptr + pack_base + g * 3 + 1).to(tl.int32)
        b2 = tl.load(packed_ptr + pack_base + g * 3 + 2).to(tl.int32)

        # Unpack 8 × 3-bit indices
        i0 = b0 & 0x7
        i1 = (b0 >> 3) & 0x7
        i2 = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        i3 = (b1 >> 1) & 0x7
        i4 = (b1 >> 4) & 0x7
        i5 = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        i6 = (b2 >> 2) & 0x7
        i7 = (b2 >> 5) & 0x7

        # Centroid lookup + norm scale, store 8 output values
        off = out_base + g * 8

        v0 = tl.where(i0 == 0, c0, tl.where(i0 == 1, c1, tl.where(i0 == 2, c2, tl.where(i0 == 3, c3, tl.where(i0 == 4, c4, tl.where(i0 == 5, c5, tl.where(i0 == 6, c6, c7)))))))
        v1 = tl.where(i1 == 0, c0, tl.where(i1 == 1, c1, tl.where(i1 == 2, c2, tl.where(i1 == 3, c3, tl.where(i1 == 4, c4, tl.where(i1 == 5, c5, tl.where(i1 == 6, c6, c7)))))))
        v2 = tl.where(i2 == 0, c0, tl.where(i2 == 1, c1, tl.where(i2 == 2, c2, tl.where(i2 == 3, c3, tl.where(i2 == 4, c4, tl.where(i2 == 5, c5, tl.where(i2 == 6, c6, c7)))))))
        v3 = tl.where(i3 == 0, c0, tl.where(i3 == 1, c1, tl.where(i3 == 2, c2, tl.where(i3 == 3, c3, tl.where(i3 == 4, c4, tl.where(i3 == 5, c5, tl.where(i3 == 6, c6, c7)))))))
        v4 = tl.where(i4 == 0, c0, tl.where(i4 == 1, c1, tl.where(i4 == 2, c2, tl.where(i4 == 3, c3, tl.where(i4 == 4, c4, tl.where(i4 == 5, c5, tl.where(i4 == 6, c6, c7)))))))
        v5 = tl.where(i5 == 0, c0, tl.where(i5 == 1, c1, tl.where(i5 == 2, c2, tl.where(i5 == 3, c3, tl.where(i5 == 4, c4, tl.where(i5 == 5, c5, tl.where(i5 == 6, c6, c7)))))))
        v6 = tl.where(i6 == 0, c0, tl.where(i6 == 1, c1, tl.where(i6 == 2, c2, tl.where(i6 == 3, c3, tl.where(i6 == 4, c4, tl.where(i6 == 5, c5, tl.where(i6 == 6, c6, c7)))))))
        v7 = tl.where(i7 == 0, c0, tl.where(i7 == 1, c1, tl.where(i7 == 2, c2, tl.where(i7 == 3, c3, tl.where(i7 == 4, c4, tl.where(i7 == 5, c5, tl.where(i7 == 6, c6, c7)))))))

        tl.store(out_ptr + off + 0, (v0 * norm).to(tl.float16))
        tl.store(out_ptr + off + 1, (v1 * norm).to(tl.float16))
        tl.store(out_ptr + off + 2, (v2 * norm).to(tl.float16))
        tl.store(out_ptr + off + 3, (v3 * norm).to(tl.float16))
        tl.store(out_ptr + off + 4, (v4 * norm).to(tl.float16))
        tl.store(out_ptr + off + 5, (v5 * norm).to(tl.float16))
        tl.store(out_ptr + off + 6, (v6 * norm).to(tl.float16))
        tl.store(out_ptr + off + 7, (v7 * norm).to(tl.float16))


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _apply_wht(x_half: torch.Tensor, head_dim: int, had_fn=None) -> torch.Tensor:
    """Apply WHT rotation (self-inverse). Input/output: FP16."""
    if had_fn is not None:
        out = torch.empty_like(x_half)
        had_fn(x_half, out, None, None, 1.0)
        return out
    H = _get_had_matrix(head_dim, x_half.device)
    shape = x_half.shape
    return (x_half.reshape(-1, head_dim) @ H.T).reshape(shape)


def turboquant_encode(
    kv: torch.Tensor,           # [num_tokens, num_heads, head_dim], FP16
    bits: int = 3,              # 3 or 4
    had_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode KV vectors to TurboQuant packed format.

    Returns:
        packed: [num_tokens, num_heads, packed_dim] uint8
                packed_dim = head_dim * bits // 8
        norms: [num_tokens, num_heads] FP16
    """
    num_tokens, num_heads, head_dim = kv.shape
    assert head_dim in (128, 256), f"head_dim must be 128 or 256, got {head_dim}"
    device = kv.device

    # Step 1: Compute L2 norms
    kv_f32 = kv.float()
    norms = kv_f32.norm(dim=-1).half()

    # Step 2: Normalize to unit vectors
    kv_normed = kv_f32 / (norms.float().unsqueeze(-1) + 1e-12)

    # Step 3: WHT rotation
    kv_rotated = _apply_wht(kv_normed.half(), head_dim, had_fn)

    # Step 4: Quantize + pack via Triton kernel
    boundaries = get_boundaries(bits, head_dim, device=device, dtype=torch.float32)
    packed_dim = head_dim * bits // 8

    packed = torch.empty(num_tokens, num_heads, packed_dim,
                         device=device, dtype=torch.uint8)

    grid = (num_tokens, num_heads)
    # RDNA3 tuning: 2 warps, 2 stages
    num_warps = 2
    num_stages = 2

    if bits == 4:
        _tq_encode_pack4_kernel[grid](
            kv_rotated, packed, boundaries,
            kv_rotated.stride(0), kv_rotated.stride(1),
            packed.stride(0), packed.stride(1),
            HEAD_DIM=head_dim, PACKED_DIM=packed_dim,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:  # 3-bit
        _tq_encode_pack3_kernel[grid](
            kv_rotated, packed, boundaries,
            kv_rotated.stride(0), kv_rotated.stride(1),
            packed.stride(0), packed.stride(1),
            HEAD_DIM=head_dim, PACKED_DIM=packed_dim,
            num_warps=num_warps, num_stages=num_stages,
        )

    return packed, norms


def turboquant_decode(
    packed: torch.Tensor,       # [num_tokens, num_heads, packed_dim] uint8
    norms: torch.Tensor,        # [num_tokens, num_heads] FP16
    bits: int = 3,
    head_dim: int = 128,
    had_fn=None,
) -> torch.Tensor:
    """Decode TurboQuant packed data back to FP16 KV vectors.

    Returns:
        kv: [num_tokens, num_heads, head_dim] FP16
    """
    num_tokens, num_heads, packed_dim = packed.shape
    device = packed.device
    centroids = get_centroids(bits, head_dim, device=device, dtype=torch.float32)

    # Output buffer (WHT-rotated space, scaled by norm)
    out_rotated = torch.empty(num_tokens, num_heads, head_dim,
                              device=device, dtype=torch.float16)

    grid = (num_tokens, num_heads)
    num_warps = 2
    num_stages = 2

    if bits == 4:
        _tq_decode_unpack4_kernel[grid](
            packed, norms, out_rotated, centroids,
            packed.stride(0), packed.stride(1),
            norms.stride(0),
            out_rotated.stride(0), out_rotated.stride(1),
            HEAD_DIM=head_dim, PACKED_DIM=packed_dim,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:  # 3-bit
        _tq_decode_unpack3_kernel[grid](
            packed, norms, out_rotated, centroids,
            packed.stride(0), packed.stride(1),
            norms.stride(0),
            out_rotated.stride(0), out_rotated.stride(1),
            HEAD_DIM=head_dim, PACKED_DIM=packed_dim,
            num_warps=num_warps, num_stages=num_stages,
        )

    # Inverse WHT (WHT is self-inverse)
    kv = _apply_wht(out_rotated, head_dim, had_fn)
    return kv


# ---------------------------------------------------------------------------
# 3.5-bit: split channels — first half 4-bit, second half 3-bit
# Memory: d=128 → 64*4/8 + 64*3/8 = 32 + 24 = 56 bytes + 2B norm = 58B
# Compression: 256/58 = 4.4x (between 3-bit 5.1x and 4-bit 3.9x)
# ---------------------------------------------------------------------------

def turboquant_encode_35(
    kv: torch.Tensor,           # [num_tokens, num_heads, head_dim], FP16
    had_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode KV to 3.5-bit TurboQuant: first half 4-bit, second half 3-bit.

    Returns:
        packed: [num_tokens, num_heads, packed_dim] uint8
                packed_dim = head_dim//2*4//8 + head_dim//2*3//8
                           = head_dim//4 + head_dim*3//16
                For d=128: 32 + 24 = 56 bytes
        norms: [num_tokens, num_heads] FP16
    """
    num_tokens, num_heads, head_dim = kv.shape
    device = kv.device
    half_dim = head_dim // 2

    # Shared: norm + WHT rotation
    kv_f32 = kv.float()
    norms = kv_f32.norm(dim=-1).half()
    kv_normed = kv_f32 / (norms.float().unsqueeze(-1) + 1e-12)
    kv_rotated = _apply_wht(kv_normed.half(), head_dim, had_fn)

    # Split into two halves
    rot_hi = kv_rotated[..., :half_dim].contiguous()   # 4-bit
    rot_lo = kv_rotated[..., half_dim:].contiguous()    # 3-bit

    # Encode each half
    b4 = get_boundaries(4, head_dim, device=device, dtype=torch.float32)
    b3 = get_boundaries(3, head_dim, device=device, dtype=torch.float32)

    packed_4bit_dim = half_dim // 2   # 32 for d=128
    packed_3bit_dim = half_dim * 3 // 8  # 24 for d=128

    packed_4bit = torch.empty(num_tokens, num_heads, packed_4bit_dim,
                              device=device, dtype=torch.uint8)
    packed_3bit = torch.empty(num_tokens, num_heads, packed_3bit_dim,
                              device=device, dtype=torch.uint8)

    grid = (num_tokens, num_heads)
    nw, ns = 2, 2

    _tq_encode_pack4_kernel[grid](
        rot_hi, packed_4bit, b4,
        rot_hi.stride(0), rot_hi.stride(1),
        packed_4bit.stride(0), packed_4bit.stride(1),
        HEAD_DIM=half_dim, PACKED_DIM=packed_4bit_dim,
        num_warps=nw, num_stages=ns,
    )
    _tq_encode_pack3_kernel[grid](
        rot_lo, packed_3bit, b3,
        rot_lo.stride(0), rot_lo.stride(1),
        packed_3bit.stride(0), packed_3bit.stride(1),
        HEAD_DIM=half_dim, PACKED_DIM=packed_3bit_dim,
        num_warps=nw, num_stages=ns,
    )

    packed = torch.cat([packed_4bit, packed_3bit], dim=-1)
    return packed, norms


def turboquant_decode_35(
    packed: torch.Tensor,       # [num_tokens, num_heads, packed_dim] uint8
    norms: torch.Tensor,        # [num_tokens, num_heads] FP16
    head_dim: int = 128,
    had_fn=None,
) -> torch.Tensor:
    """Decode 3.5-bit TurboQuant packed data back to FP16 KV vectors."""
    num_tokens, num_heads, total_packed = packed.shape
    device = packed.device
    half_dim = head_dim // 2

    packed_4bit_dim = half_dim // 2
    packed_3bit_dim = half_dim * 3 // 8

    packed_4bit = packed[..., :packed_4bit_dim].contiguous()
    packed_3bit = packed[..., packed_4bit_dim:].contiguous()

    c4 = get_centroids(4, head_dim, device=device, dtype=torch.float32)
    c3 = get_centroids(3, head_dim, device=device, dtype=torch.float32)

    # Decode each half (output is in WHT-rotated space, NOT scaled by norm yet)
    # We need a modified decode that doesn't scale by norm — do it after concat.
    # Use norms=ones temporarily, then apply real norm after WHT.

    ones = torch.ones_like(norms)
    out_hi = torch.empty(num_tokens, num_heads, half_dim, device=device, dtype=torch.float16)
    out_lo = torch.empty(num_tokens, num_heads, half_dim, device=device, dtype=torch.float16)

    grid = (num_tokens, num_heads)
    nw, ns = 2, 2

    _tq_decode_unpack4_kernel[grid](
        packed_4bit, ones, out_hi, c4,
        packed_4bit.stride(0), packed_4bit.stride(1),
        ones.stride(0),
        out_hi.stride(0), out_hi.stride(1),
        HEAD_DIM=half_dim, PACKED_DIM=packed_4bit_dim,
        num_warps=nw, num_stages=ns,
    )
    _tq_decode_unpack3_kernel[grid](
        packed_3bit, ones, out_lo, c3,
        packed_3bit.stride(0), packed_3bit.stride(1),
        ones.stride(0),
        out_lo.stride(0), out_lo.stride(1),
        HEAD_DIM=half_dim, PACKED_DIM=packed_3bit_dim,
        num_warps=nw, num_stages=ns,
    )

    # Concatenate halves back
    out_rotated = torch.cat([out_hi, out_lo], dim=-1)

    # Inverse WHT
    kv = _apply_wht(out_rotated, head_dim, had_fn)

    # Scale by norm
    kv = kv.float() * norms.float().unsqueeze(-1)
    return kv.half()


# ---------------------------------------------------------------------------
# Phase 4: Fused Q·K^T score in WHT space (skip K dequant)
# ---------------------------------------------------------------------------
# score = norm_k * dot(WHT(q_normed), centroids[k_indices])
# Rotate query once, dot directly with centroid lookups — no inverse WHT.
# For GQA: q_heads_per_kv = num_q_heads // num_kv_heads

@triton.jit
def _tq_fused_qk_score_4bit_kernel(
    # q_rot: [num_q_heads, head_dim] FP16 — WHT-rotated query (already scaled)
    q_rot_ptr,
    # packed_k: [num_kv_tokens, num_kv_heads, packed_dim] uint8
    packed_k_ptr,
    # norms_k: [num_kv_tokens, num_kv_heads] FP16
    norms_k_ptr,
    # scores_out: [num_q_heads, num_kv_tokens] FP32
    scores_ptr,
    # centroids: (16,) FP32
    centroids_ptr,
    # Strides
    q_stride_head: tl.int64,
    pk_stride_token: tl.int64,
    pk_stride_head: tl.int64,
    nk_stride_token: tl.int64,
    sc_stride_head: tl.int64,
    # Sizes
    num_kv_tokens: tl.int32,
    # Constants
    HEAD_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,       # HEAD_DIM // 2
    Q_HEADS_PER_KV: tl.constexpr,   # GQA ratio
    BLOCK_KV: tl.constexpr,         # tokens per program
):
    """Fused Q·K^T score for 4-bit TQ cache — tiled over KV tokens.

    Grid: (cdiv(num_kv_tokens, BLOCK_KV), num_q_heads)
    Each program computes BLOCK_KV scores.
    """
    block_id = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // Q_HEADS_PER_KV

    kv_start = block_id * BLOCK_KV

    # Load q_rot vector ONCE for this Q head — reuse across all BLOCK_KV tokens
    q_offs = tl.arange(0, PACKED_DIM)
    q_even = tl.load(q_rot_ptr + q_head * q_stride_head + q_offs * 2).to(tl.float32)
    q_odd = tl.load(q_rot_ptr + q_head * q_stride_head + q_offs * 2 + 1).to(tl.float32)

    # Preload all 16 centroids into registers
    c_vals = tl.zeros((16,), dtype=tl.float32)
    for c in tl.static_range(16):
        c_vals = tl.where(tl.arange(0, 16) == c, tl.load(centroids_ptr + c).to(tl.float32), c_vals)

    # Process BLOCK_KV tokens
    for t in tl.static_range(BLOCK_KV):
        kv_token = kv_start + t
        if kv_token < num_kv_tokens:
            pk_base = kv_token * pk_stride_token + kv_head * pk_stride_head
            packed = tl.load(packed_k_ptr + pk_base + q_offs).to(tl.int32)
            idx_even = packed & 0xF
            idx_odd = (packed >> 4) & 0xF

            # Vectorized centroid lookup
            val_even = tl.zeros((PACKED_DIM,), dtype=tl.float32)
            val_odd = tl.zeros((PACKED_DIM,), dtype=tl.float32)
            for c in tl.static_range(16):
                c_val = tl.load(centroids_ptr + c).to(tl.float32)
                val_even = tl.where(idx_even == c, c_val, val_even)
                val_odd = tl.where(idx_odd == c, c_val, val_odd)

            dot = tl.sum(q_even * val_even) + tl.sum(q_odd * val_odd)
            norm = tl.load(norms_k_ptr + kv_token * nk_stride_token + kv_head).to(tl.float32)
            tl.store(scores_ptr + q_head * sc_stride_head + kv_token, dot * norm)


@triton.jit
def _tq_fused_qk_score_3bit_kernel(
    q_rot_ptr,
    packed_k_ptr,
    norms_k_ptr,
    scores_ptr,
    centroids_ptr,
    q_stride_head: tl.int64,
    pk_stride_token: tl.int64,
    pk_stride_head: tl.int64,
    nk_stride_token: tl.int64,
    sc_stride_head: tl.int64,
    num_kv_tokens: tl.int32,
    HEAD_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,       # HEAD_DIM * 3 // 8
    Q_HEADS_PER_KV: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Fused Q·K^T score for 3-bit TQ cache — tiled over KV tokens.
    Grid: (cdiv(num_kv_tokens, BLOCK_KV), num_q_heads)
    """
    block_id = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // Q_HEADS_PER_KV
    kv_start = block_id * BLOCK_KV

    # Preload centroids
    c0 = tl.load(centroids_ptr + 0).to(tl.float32)
    c1 = tl.load(centroids_ptr + 1).to(tl.float32)
    c2 = tl.load(centroids_ptr + 2).to(tl.float32)
    c3 = tl.load(centroids_ptr + 3).to(tl.float32)
    c4 = tl.load(centroids_ptr + 4).to(tl.float32)
    c5 = tl.load(centroids_ptr + 5).to(tl.float32)
    c6 = tl.load(centroids_ptr + 6).to(tl.float32)
    c7 = tl.load(centroids_ptr + 7).to(tl.float32)

    N_GROUPS: tl.constexpr = HEAD_DIM // 8

    # Preload q_rot for this head
    # We load per-group inside the token loop to keep register pressure low

    for t in tl.static_range(BLOCK_KV):
        kv_token = kv_start + t
        if kv_token < num_kv_tokens:
            pk_base = kv_token * pk_stride_token + kv_head * pk_stride_head
            norm = tl.load(norms_k_ptr + kv_token * nk_stride_token + kv_head).to(tl.float32)

            dot: tl.float32 = 0.0
            for g in tl.static_range(N_GROUPS):
                b0 = tl.load(packed_k_ptr + pk_base + g * 3).to(tl.int32)
                b1 = tl.load(packed_k_ptr + pk_base + g * 3 + 1).to(tl.int32)
                b2 = tl.load(packed_k_ptr + pk_base + g * 3 + 2).to(tl.int32)

                i0 = b0 & 0x7
                i1 = (b0 >> 3) & 0x7
                i2 = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
                i3 = (b1 >> 1) & 0x7
                i4 = (b1 >> 4) & 0x7
                i5 = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
                i6 = (b2 >> 2) & 0x7
                i7 = (b2 >> 5) & 0x7

                v0 = tl.where(i0 == 0, c0, tl.where(i0 == 1, c1, tl.where(i0 == 2, c2, tl.where(i0 == 3, c3, tl.where(i0 == 4, c4, tl.where(i0 == 5, c5, tl.where(i0 == 6, c6, c7)))))))
                v1 = tl.where(i1 == 0, c0, tl.where(i1 == 1, c1, tl.where(i1 == 2, c2, tl.where(i1 == 3, c3, tl.where(i1 == 4, c4, tl.where(i1 == 5, c5, tl.where(i1 == 6, c6, c7)))))))
                v2 = tl.where(i2 == 0, c0, tl.where(i2 == 1, c1, tl.where(i2 == 2, c2, tl.where(i2 == 3, c3, tl.where(i2 == 4, c4, tl.where(i2 == 5, c5, tl.where(i2 == 6, c6, c7)))))))
                v3 = tl.where(i3 == 0, c0, tl.where(i3 == 1, c1, tl.where(i3 == 2, c2, tl.where(i3 == 3, c3, tl.where(i3 == 4, c4, tl.where(i3 == 5, c5, tl.where(i3 == 6, c6, c7)))))))
                v4 = tl.where(i4 == 0, c0, tl.where(i4 == 1, c1, tl.where(i4 == 2, c2, tl.where(i4 == 3, c3, tl.where(i4 == 4, c4, tl.where(i4 == 5, c5, tl.where(i4 == 6, c6, c7)))))))
                v5 = tl.where(i5 == 0, c0, tl.where(i5 == 1, c1, tl.where(i5 == 2, c2, tl.where(i5 == 3, c3, tl.where(i5 == 4, c4, tl.where(i5 == 5, c5, tl.where(i5 == 6, c6, c7)))))))
                v6 = tl.where(i6 == 0, c0, tl.where(i6 == 1, c1, tl.where(i6 == 2, c2, tl.where(i6 == 3, c3, tl.where(i6 == 4, c4, tl.where(i6 == 5, c5, tl.where(i6 == 6, c6, c7)))))))
                v7 = tl.where(i7 == 0, c0, tl.where(i7 == 1, c1, tl.where(i7 == 2, c2, tl.where(i7 == 3, c3, tl.where(i7 == 4, c4, tl.where(i7 == 5, c5, tl.where(i7 == 6, c6, c7)))))))

                q_base = q_head * q_stride_head + g * 8
                q0 = tl.load(q_rot_ptr + q_base + 0).to(tl.float32)
                q1 = tl.load(q_rot_ptr + q_base + 1).to(tl.float32)
                q2 = tl.load(q_rot_ptr + q_base + 2).to(tl.float32)
                q3 = tl.load(q_rot_ptr + q_base + 3).to(tl.float32)
                q4 = tl.load(q_rot_ptr + q_base + 4).to(tl.float32)
                q5 = tl.load(q_rot_ptr + q_base + 5).to(tl.float32)
                q6 = tl.load(q_rot_ptr + q_base + 6).to(tl.float32)
                q7 = tl.load(q_rot_ptr + q_base + 7).to(tl.float32)

                dot += q0*v0 + q1*v1 + q2*v2 + q3*v3 + q4*v4 + q5*v5 + q6*v6 + q7*v7

            tl.store(scores_ptr + q_head * sc_stride_head + kv_token, dot * norm)


def turboquant_fused_qk_scores(
    query: torch.Tensor,            # [num_q_heads, head_dim] FP16
    packed_k: torch.Tensor,         # [num_kv_tokens, num_kv_heads, packed_dim] uint8
    norms_k: torch.Tensor,          # [num_kv_tokens, num_kv_heads] FP16
    bits: int = 3,
    head_dim: int = 128,
    scale: float = None,
    had_fn=None,
) -> torch.Tensor:
    """Compute Q·K^T attention scores directly from TQ-packed K cache.

    Rotates query with WHT once, then dots against centroid lookups.
    Avoids K dequant (inverse WHT + FP16 materialization).

    Args:
        query: [num_q_heads, head_dim] — single decode token query
        packed_k: [num_kv_tokens, num_kv_heads, packed_dim] — packed K cache
        norms_k: [num_kv_tokens, num_kv_heads] — K norms
        bits: 3, 4, or 35
        head_dim: head dimension (128 or 256)
        scale: attention scale (default: 1/sqrt(head_dim))

    Returns:
        scores: [num_q_heads, num_kv_tokens] FP32
    """
    num_q_heads = query.shape[0]
    num_kv_tokens, num_kv_heads = norms_k.shape
    device = query.device

    if scale is None:
        scale = head_dim ** -0.5

    # Rotate query with WHT
    q_rot = _apply_wht(query.unsqueeze(0), head_dim, had_fn).squeeze(0)
    # Apply attention scale to rotated query
    q_rot = (q_rot.float() * scale).half()

    q_heads_per_kv = num_q_heads // num_kv_heads
    scores = torch.empty(num_q_heads, num_kv_tokens, device=device, dtype=torch.float32)

    BLOCK_KV = 16  # tokens per program — good balance for RDNA3/4
    nw, ns = 2, 2

    if bits == 4:
        packed_dim = head_dim // 2
        grid = (triton.cdiv(num_kv_tokens, BLOCK_KV), num_q_heads)
        _tq_fused_qk_score_4bit_kernel[grid](
            q_rot, packed_k, norms_k, scores,
            get_centroids(4, head_dim, device, torch.float32),
            q_rot.stride(0),
            packed_k.stride(0), packed_k.stride(1),
            norms_k.stride(0),
            scores.stride(0),
            num_kv_tokens,
            HEAD_DIM=head_dim, PACKED_DIM=packed_dim,
            Q_HEADS_PER_KV=q_heads_per_kv,
            BLOCK_KV=BLOCK_KV,
            num_warps=nw, num_stages=ns,
        )
    elif bits == 3:
        packed_dim = head_dim * 3 // 8
        grid = (triton.cdiv(num_kv_tokens, BLOCK_KV), num_q_heads)
        _tq_fused_qk_score_3bit_kernel[grid](
            q_rot, packed_k, norms_k, scores,
            get_centroids(3, head_dim, device, torch.float32),
            q_rot.stride(0),
            packed_k.stride(0), packed_k.stride(1),
            norms_k.stride(0),
            scores.stride(0),
            num_kv_tokens,
            HEAD_DIM=head_dim, PACKED_DIM=packed_dim,
            Q_HEADS_PER_KV=q_heads_per_kv,
            BLOCK_KV=BLOCK_KV,
            num_warps=nw, num_stages=ns,
        )
    elif bits == 35:
        scores = _turboquant_fused_qk_scores_35(
            q_rot, packed_k, norms_k, head_dim, q_heads_per_kv)
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return scores


def _turboquant_fused_qk_scores_35(
    q_rot: torch.Tensor,            # [num_q_heads, head_dim] FP16 (already scaled)
    packed_k: torch.Tensor,         # [num_kv_tokens, num_kv_heads, packed_dim] uint8
    norms_k: torch.Tensor,          # [num_kv_tokens, num_kv_heads] FP16
    head_dim: int,
    q_heads_per_kv: int,
) -> torch.Tensor:
    """3.5-bit fused Q·K^T: first half 4-bit, second half 3-bit."""
    num_q_heads = q_rot.shape[0]
    num_kv_tokens, num_kv_heads = norms_k.shape
    device = q_rot.device
    half_dim = head_dim // 2

    packed_4bit_dim = half_dim // 2  # 32
    packed_3bit_dim = half_dim * 3 // 8  # 24

    # Split packed K into 4-bit and 3-bit halves
    packed_4bit = packed_k[..., :packed_4bit_dim].contiguous()
    packed_3bit = packed_k[..., packed_4bit_dim:].contiguous()

    # Split q_rot into halves matching WHT-rotated K
    q_rot_hi = q_rot[:, :half_dim].contiguous()   # matches 4-bit half
    q_rot_lo = q_rot[:, half_dim:].contiguous()    # matches 3-bit half

    c4 = get_centroids(4, head_dim, device, torch.float32)
    c3 = get_centroids(3, head_dim, device, torch.float32)

    # Score = norm_k * (dot_hi + dot_lo)
    # Use separate kernels for each half, accumulate
    scores_hi = torch.empty(num_q_heads, num_kv_tokens, device=device, dtype=torch.float32)
    scores_lo = torch.empty(num_q_heads, num_kv_tokens, device=device, dtype=torch.float32)

    BLOCK_KV = 16
    nw, ns = 2, 2
    grid = (triton.cdiv(num_kv_tokens, BLOCK_KV), num_q_heads)

    # 4-bit half: score_hi = norm * dot(q_hi, centroids[k_4bit])
    _tq_fused_qk_score_4bit_kernel[grid](
        q_rot_hi, packed_4bit,
        norms_k,
        scores_hi,
        c4,
        q_rot_hi.stride(0),
        packed_4bit.stride(0), packed_4bit.stride(1),
        norms_k.stride(0),
        scores_hi.stride(0),
        num_kv_tokens,
        HEAD_DIM=half_dim, PACKED_DIM=packed_4bit_dim,
        Q_HEADS_PER_KV=q_heads_per_kv,
        BLOCK_KV=BLOCK_KV,
        num_warps=nw, num_stages=ns,
    )

    # 3-bit half: score_lo = norm * dot(q_lo, centroids[k_3bit])
    # Sum: score_hi + score_lo = norm * (dot_hi + dot_lo) = norm * dot(q_rot, full_centroid)
    _tq_fused_qk_score_3bit_kernel[grid](
        q_rot_lo, packed_3bit,
        norms_k,
        scores_lo,
        c3,
        q_rot_lo.stride(0),
        packed_3bit.stride(0), packed_3bit.stride(1),
        norms_k.stride(0),
        scores_lo.stride(0),
        num_kv_tokens,
        HEAD_DIM=half_dim, PACKED_DIM=packed_3bit_dim,
        Q_HEADS_PER_KV=q_heads_per_kv,
        BLOCK_KV=BLOCK_KV,
        num_warps=nw, num_stages=ns,
    )

    return scores_hi + scores_lo


# ---------------------------------------------------------------------------
# Unpacked fallback (Phase 1 compatibility, used if Triton unavailable)
# ---------------------------------------------------------------------------

def turboquant_encode_unpacked(
    kv: torch.Tensor, bits: int = 3, had_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode without bit packing (1 byte per index). For testing."""
    num_tokens, num_heads, head_dim = kv.shape
    device = kv.device

    kv_f32 = kv.float()
    norms = kv_f32.norm(dim=-1).half()
    kv_normed = kv_f32 / (norms.float().unsqueeze(-1) + 1e-12)
    kv_rotated = _apply_wht(kv_normed.half(), head_dim, had_fn)

    boundaries = get_boundaries(bits, head_dim, device=device, dtype=torch.float32)
    indices = torch.searchsorted(boundaries, kv_rotated.float().reshape(-1)).reshape(
        num_tokens, num_heads, head_dim
    ).to(torch.uint8)
    return indices, norms


def turboquant_decode_unpacked(
    indices: torch.Tensor, norms: torch.Tensor,
    bits: int = 3, head_dim: int = 128, had_fn=None,
) -> torch.Tensor:
    """Decode from unpacked indices (1 byte per index). For testing."""
    device = indices.device
    centroids = get_centroids(bits, head_dim, device=device, dtype=torch.float32)
    y = centroids[indices.long()]
    y_half = _apply_wht(y.half(), head_dim, had_fn)
    return (y_half.float() * norms.float().unsqueeze(-1)).half()
