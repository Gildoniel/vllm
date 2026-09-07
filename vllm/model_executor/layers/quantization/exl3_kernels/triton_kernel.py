"""
EXL3 Triton kernel: fused dequantization + GEMM for all codebooks and bitwidths.

    C = A @ dequant(B)

where B is stored in EXL3 trellis-coded quantization format.

Supports:
  - Codebooks: cb=0 (default 3inst), cb=1 (MCG), cb=2 (MUL1)
  - Bitwidths: 1-8 bpw
  - Cross-platform: NVIDIA (CUDA) and AMD (ROCm) via Triton

Target: ROCm / RDNA3 (RX 7900 XTX) with WMMA support.
"""

import torch
import triton
import triton.language as tl
import numpy as np
import os
import json
import time
from pathlib import Path


# =============================================================================
# Tensor core permutation (from exllamav3 quantize.py)
# =============================================================================

def tensor_core_perm() -> np.ndarray:
    """
    Generate the 256-element tensor core permutation.
    perm[packed_pos] = matrix_pos (row * 16 + col)

    This maps from the packed bitstream order (used by NVIDIA tensor cores
    and the EXL3 trellis encoding) to standard row-major 16x16 order.
    """
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1 = r0 + 1
        r2 = r0 + 8
        r3 = r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = r1 * 16 + c0
        perm[t * 8 + 2] = r2 * 16 + c0
        perm[t * 8 + 3] = r3 * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c1
        perm[t * 8 + 5] = r1 * 16 + c1
        perm[t * 8 + 6] = r2 * 16 + c1
        perm[t * 8 + 7] = r3 * 16 + c1
    return np.array(perm, dtype=np.int32)


def tensor_core_inv_perm() -> np.ndarray:
    """
    Inverse permutation: inv_perm[matrix_pos] = packed_pos.
    Given a (row, col) in the 16x16 tile (as row*16+col),
    returns the position in the packed bitstream.
    """
    perm = tensor_core_perm()
    inv = np.empty(256, dtype=np.int32)
    inv[perm] = np.arange(256, dtype=np.int32)
    return inv


_inv_perm_np = tensor_core_inv_perm()  # precomputed at import time
_inv_perm_cache = {}


def get_inv_perm(device) -> torch.Tensor:
    """Get cached inverse permutation tensor on the given device."""
    if device not in _inv_perm_cache:
        _inv_perm_cache[device] = torch.tensor(
            _inv_perm_np, device=device, dtype=torch.int32)
    return _inv_perm_cache[device]


# =============================================================================
# Precompute bit extraction tables (CPU-side, per bitwidth)
# =============================================================================
# All tables are precomputed at import time for bits 1-8.
# This avoids numpy calls inside the forward graph so that torch.compile
# (Dynamo) can trace through get_bit_tables without hitting Unsupported ops.

def _compute_bit_extraction_tables(bits: int) -> tuple:
    """
    For each of the 256 matrix positions in a 16x16 tile, compute:
      - word_idx: which uint32 word to load (lo word of funnel shift)
      - next_word_idx: hi word of funnel shift
      - shift: funnel shift amount

    Computed analytically from the CUDA dq() formula in exl3_dq.cuh:
      For packed position t:
        b0 = (t + 257) * bits - 16   (start bit of 16-bit extraction window)
        b1 = b0 + 16                 (end bit)
        i0 = b0 // 32               (word containing start bit)
        i1 = (b1 - 1) // 32         (word containing end bit)
        s0 = (i1 + 1) * 32 - b1     (right-shift amount)
      Extraction: ((ptr[i0] << 32) | ptr[i1]) >> s0
      Triton mapping: lo = ptr[i1], hi = ptr[i0]

    Returns numpy arrays of shape (256,) as int32.
    """
    inv_perm = _inv_perm_np
    num_uint32 = bits * 256 // 32  # words per tile

    word_idx_arr = np.zeros(256, dtype=np.int32)
    next_word_idx_arr = np.zeros(256, dtype=np.int32)
    shift_arr = np.zeros(256, dtype=np.int32)

    for mat_pos in range(256):
        packed_pos = int(inv_perm[mat_pos])

        # CUDA dq() generic formula
        b0 = (packed_pos + 257) * bits - 16
        b1 = b0 + 16
        i0 = b0 // 32
        i1 = (b1 - 1) // 32
        s0 = (i1 + 1) * 32 - b1

        # Triton funnel shift: ((hi << 32) | lo) >> shift
        # CUDA: fshift(ptr[i1], ptr[i0], s0) = ((ptr[i0] << 32) | ptr[i1]) >> s0
        # So: lo = ptr[i1], hi = ptr[i0]
        word_idx_arr[mat_pos] = i1 % num_uint32      # lo word
        next_word_idx_arr[mat_pos] = i0 % num_uint32  # hi word
        shift_arr[mat_pos] = s0

    return word_idx_arr, next_word_idx_arr, shift_arr


# Precompute all tables at import time (bits 1-8) as numpy.
# Device tensors are created lazily but the numpy->torch conversion
# is just torch.tensor() with a concrete array — Dynamo-safe.
_bit_tables_np: dict[int, tuple] = {}
for _bits in range(1, 9):
    _bit_tables_np[_bits] = _compute_bit_extraction_tables(_bits)

_bit_tables_cache: dict[tuple, tuple] = {}


def get_bit_tables(bits: int, device) -> tuple:
    """Get cached bit extraction tables for given bitwidth on device."""
    key = (bits, device)
    if key not in _bit_tables_cache:
        wi, ni, si = _bit_tables_np[bits]
        _bit_tables_cache[key] = (
            torch.tensor(wi, device=device, dtype=torch.int32),
            torch.tensor(ni, device=device, dtype=torch.int32),
            torch.tensor(si, device=device, dtype=torch.int32),
        )
    return _bit_tables_cache[key]


# =============================================================================
# Triton kernel: generalized fused dequant + GEMM
# =============================================================================

@triton.jit
def _exl3_gemm_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    # Dimensions
    M, N, K,
    # Strides (in elements)
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Tile layout
    tiles_n,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,  # = 256 * BPW // 32 = 8 * BPW
    CB: tl.constexpr,              # codebook: 0, 1, or 2
):
    """
    Fused EXL3 dequantization + GEMM for any bitwidth and codebook.

    Grid: (cdiv(M, BLOCK_M), N // 16)
    Each program computes a (BLOCK_M, 16) output tile.

    B layout: (K//16, N//16, WORDS_PER_TILE) int32 — packed trellis data.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Accumulator: (BLOCK_M, 16) in float32
    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    # --- Load precomputed bit extraction tables ---
    # Tables are (16, 16) when reshaped from the 256-element flat arrays
    k_local = tl.arange(0, 16)[:, None]   # (16, 1)
    n_local = tl.arange(0, 16)[None, :]   # (1, 16)
    mat_pos = k_local * 16 + n_local      # (16, 16) flat position [0, 256)

    word_idx = tl.load(word_idx_ptr + mat_pos)           # (16, 16)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)  # (16, 16)
    shift = tl.load(shift_ptr + mat_pos)                  # (16, 16)
    shift_hi = (32 - shift) & 31

    # --- K-dimension loop: one 16x16 B tile per iteration ---
    num_k_tiles = K // 16
    for tk in range(num_k_tiles):
        # Load A tile: (BLOCK_M, 16) float16
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_m = offs_m[:, None] < M
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        # Load packed B words for this tile
        b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE
        lo = tl.load(B_ptr + b_base + word_idx)          # (16, 16) int32
        hi = tl.load(B_ptr + b_base + next_word_idx)     # (16, 16) int32

        # --- Funnel shift: extract 16-bit index from bitstream ---
        # Triton >> on int32 is arithmetic (sign-extending), so mask carefully.
        lo_part = (lo >> shift) & ((1 << shift_hi) - 1) & 0xFFFF
        hi_part = (hi << shift_hi) & 0xFFFF
        index = tl.where(shift > 0, lo_part | hi_part, lo & 0xFFFF)

        # --- Codebook decode ---
        if CB == 0:
            # cb=0: x = index * 89226354 + 64248484; LOP3; fp16 add
            x = index * 89226354 + 64248484
            x_lo = (x & 0x8FFF) ^ 0x3B60
            x_hi = ((x >> 16) & 0x8FFF) ^ 0x3B60
            x = (x_hi << 16) | x_lo
            low_bits = (x & 0xFFFF).to(tl.int16)
            high_bits = ((x >> 16) & 0xFFFF).to(tl.int16)
            low_f16 = low_bits.to(tl.float16, bitcast=True)
            high_f16 = high_bits.to(tl.float16, bitcast=True)
            weight = low_f16 + high_f16

        elif CB == 1:
            # cb=1 (MCG): x = index * 0xCBAC1FED (no additive constant); LOP3; fp16 add
            # 0xCBAC1FED = 3417055213 as uint32, as int32 = -877912083
            x = index * (-877912083)
            x_lo = (x & 0x8FFF) ^ 0x3B60
            x_hi = ((x >> 16) & 0x8FFF) ^ 0x3B60
            x = (x_hi << 16) | x_lo
            low_bits = (x & 0xFFFF).to(tl.int16)
            high_bits = ((x >> 16) & 0xFFFF).to(tl.int16)
            low_f16 = low_bits.to(tl.float16, bitcast=True)
            high_f16 = high_bits.to(tl.float16, bitcast=True)
            weight = low_f16 + high_f16

        elif CB == 2:
            # cb=2 (MUL1): x = index * 0x83DCD12D; byte sum + 0x6400; FMA
            # 0x83DCD12D = 2212286765 as uint32, as int32 = -2082680531
            x = index * (-2082680531)
            # Byte sum: sum of 4 bytes (treating x as uint32 bytes)
            b0 = x & 0xFF
            b1 = (x >> 8) & 0xFF
            b2 = (x >> 16) & 0xFF
            b3 = (x >> 24) & 0xFF
            # For b3: x >> 24 is arithmetic on int32, sign-extends.
            # We need unsigned byte value. Mask the input first.
            # Actually (x >> 24) & 0xFF already gives the unsigned byte.
            vsum = b0 + b1 + b2 + b3
            # Add accumulator 0x6400 and reinterpret as fp16
            vsum = vsum + 0x6400
            vsum_i16 = (vsum & 0xFFFF).to(tl.int16)
            h = vsum_i16.to(tl.float16, bitcast=True)
            # FMA: result = h * k_inv + k_bias
            # k_inv = 0x1EEE as fp16, k_bias = 0xC931 as fp16
            k_inv_i16 = tl.full(h.shape, 0x1EEE, dtype=tl.int16)
            k_bias_i16 = tl.full(h.shape, -14031, dtype=tl.int16)  # 0xC931 as signed int16
            k_inv_f16 = k_inv_i16.to(tl.float16, bitcast=True)
            k_bias_f16 = k_bias_i16.to(tl.float16, bitcast=True)
            weight = h * k_inv_f16 + k_bias_f16

        # --- Matrix multiply: acc += a_tile @ weight ---
        acc += tl.dot(a_tile, weight)

    # --- Store C tile ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_m = offs_m[:, None] < M
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m)


# =============================================================================
# Optimized kernel v2: K-unrolling + multi-column output (BLOCK_N)
#
# CUDA reference uses BLOCK_M=16, BLOCK_N=32, BLOCK_K=128.
# This kernel processes BLOCK_N output columns and unrolls the K-loop
# by K_UNROLL factor (effective BLOCK_K = 16 * K_UNROLL).
# =============================================================================

@triton.jit
def _dequant_tile(lo, hi, shift, shift_hi, CB: tl.constexpr):
    """Dequantize a single 16x16 B tile. Returns (16, 16) float16 weights."""
    lo_part = (lo >> shift) & ((1 << shift_hi) - 1) & 0xFFFF
    hi_part = (hi << shift_hi) & 0xFFFF
    index = tl.where(shift > 0, lo_part | hi_part, lo & 0xFFFF)

    if CB == 0:
        x = index * 89226354 + 64248484
        x_lo = (x & 0x8FFF) ^ 0x3B60
        x_hi = ((x >> 16) & 0x8FFF) ^ 0x3B60
        x = (x_hi << 16) | x_lo
        low_bits = (x & 0xFFFF).to(tl.int16)
        high_bits = ((x >> 16) & 0xFFFF).to(tl.int16)
        weight = low_bits.to(tl.float16, bitcast=True) + high_bits.to(tl.float16, bitcast=True)
    elif CB == 1:
        x = index * (-877912083)
        x_lo = (x & 0x8FFF) ^ 0x3B60
        x_hi = ((x >> 16) & 0x8FFF) ^ 0x3B60
        x = (x_hi << 16) | x_lo
        low_bits = (x & 0xFFFF).to(tl.int16)
        high_bits = ((x >> 16) & 0xFFFF).to(tl.int16)
        weight = low_bits.to(tl.float16, bitcast=True) + high_bits.to(tl.float16, bitcast=True)
    elif CB == 2:
        x = index * (-2082680531)
        b0 = x & 0xFF
        b1 = (x >> 8) & 0xFF
        b2 = (x >> 16) & 0xFF
        b3 = (x >> 24) & 0xFF
        vsum = b0 + b1 + b2 + b3 + 0x6400
        h = (vsum & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True)
        k_inv = tl.full(h.shape, 0x1EEE, dtype=tl.int16).to(tl.float16, bitcast=True)
        k_bias = tl.full(h.shape, -14031, dtype=tl.int16).to(tl.float16, bitcast=True)
        weight = h * k_inv + k_bias
    return weight


@triton.jit
def _exl3_gemm_kernel_v2(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    # Dimensions
    M, N, K,
    # Strides (in elements)
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Tile layout
    tiles_n,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,       # output columns per program (16 or 32)
    K_UNROLL: tl.constexpr,      # K-tiles per outer iteration (1, 2, 4, 8)
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
):
    """
    Optimized EXL3 dequant + GEMM with K-unrolling and multi-column output.

    Grid: (cdiv(M, BLOCK_M), N // BLOCK_N)
    Each program computes a (BLOCK_M, BLOCK_N) output tile.
    Effective BLOCK_K = 16 * K_UNROLL.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # --- Load precomputed bit extraction tables (same for all tiles) ---
    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local  # (16, 16)

    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    # M-offsets (constant across K-loop)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M

    # --- Multi-column accumulators ---
    COLS: tl.constexpr = BLOCK_N // 16  # 1 or 2

    if COLS == 1:
        acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    # --- K-dimension loop with unrolling ---
    num_k_tiles = K // 16
    num_outer = num_k_tiles // K_UNROLL
    remainder = num_k_tiles % K_UNROLL

    for tk_outer in range(num_outer):
        for tk_inner in tl.static_range(K_UNROLL):
            tk = tk_outer * K_UNROLL + tk_inner

            # Load A tile: (BLOCK_M, 16)
            offs_k = tk * 16 + tl.arange(0, 16)
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

            # Column 0: dequant B[tk, pid_n * COLS + 0]
            b_col0 = pid_n * COLS
            b_base0 = (tk * tiles_n + b_col0) * WORDS_PER_TILE
            lo0 = tl.load(B_ptr + b_base0 + word_idx)
            hi0 = tl.load(B_ptr + b_base0 + next_word_idx)
            w0 = _dequant_tile(lo0, hi0, shift, shift_hi, CB)
            acc0 += tl.dot(a_tile, w0)

            # Column 1 (if BLOCK_N=32): dequant B[tk, pid_n * COLS + 1]
            if COLS == 2:
                b_col1 = pid_n * COLS + 1
                b_base1 = (tk * tiles_n + b_col1) * WORDS_PER_TILE
                lo1 = tl.load(B_ptr + b_base1 + word_idx)
                hi1 = tl.load(B_ptr + b_base1 + next_word_idx)
                w1 = _dequant_tile(lo1, hi1, shift, shift_hi, CB)
                acc1 += tl.dot(a_tile, w1)

    # Handle remainder K-tiles (when K not divisible by K_UNROLL * 16)
    for tk_r in range(remainder):
        tk = num_outer * K_UNROLL + tk_r
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        b_col0 = pid_n * COLS
        b_base0 = (tk * tiles_n + b_col0) * WORDS_PER_TILE
        lo0 = tl.load(B_ptr + b_base0 + word_idx)
        hi0 = tl.load(B_ptr + b_base0 + next_word_idx)
        w0 = _dequant_tile(lo0, hi0, shift, shift_hi, CB)
        acc0 += tl.dot(a_tile, w0)

        if COLS == 2:
            b_col1 = pid_n * COLS + 1
            b_base1 = (tk * tiles_n + b_col1) * WORDS_PER_TILE
            lo1 = tl.load(B_ptr + b_base1 + word_idx)
            hi1 = tl.load(B_ptr + b_base1 + next_word_idx)
            w1 = _dequant_tile(lo1, hi1, shift, shift_hi, CB)
            acc1 += tl.dot(a_tile, w1)

    # --- Store output ---
    offs_n0 = pid_n * BLOCK_N + tl.arange(0, 16)
    c_ptrs0 = C_ptr + offs_m[:, None] * stride_cm + offs_n0[None, :] * stride_cn
    tl.store(c_ptrs0, acc0.to(tl.float16), mask=mask_m)

    if COLS == 2:
        offs_n1 = pid_n * BLOCK_N + 16 + tl.arange(0, 16)
        c_ptrs1 = C_ptr + offs_m[:, None] * stride_cm + offs_n1[None, :] * stride_cn
        tl.store(c_ptrs1, acc1.to(tl.float16), mask=mask_m)


# =============================================================================
# Split-K kernel: better GPU utilization for M=1 decode
#
# For small M (decode), the grid has too few blocks to saturate the GPU.
# Split-K splits the K dimension across multiple thread blocks and reduces.
# On 7900 XTX: 2-3x speedup for down_proj, 1.3-2x for other layers.
# =============================================================================

@triton.jit
def _exl3_gemm_splitk_kernel(
    A_ptr, B_ptr, C_partial_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_cp_split, stride_cp_m, stride_cp_n,
    tiles_n,
    tiles_per_split,
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Split-K: each program handles K/SPLIT_K tiles, writes to C_partial[split, M, N]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local
    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M
    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    tk_start = pid_k * tiles_per_split
    tk_end = tl.minimum(tk_start + tiles_per_split, K // 16)

    for tk in range(tk_start, tk_end):
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE
        lo = tl.load(B_ptr + b_base + word_idx)
        hi = tl.load(B_ptr + b_base + next_word_idx)
        w = _dequant_tile(lo, hi, shift, shift_hi, CB)
        acc += tl.dot(a_tile, w)

    offs_n = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = (C_partial_ptr
              + pid_k * stride_cp_split
              + offs_m[:, None] * stride_cp_m
              + offs_n[None, :] * stride_cp_n)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m)


# =============================================================================
# Legacy kernel (kept for backward compatibility in tests)
# =============================================================================

@triton.jit
def _exl3_gemm_4bit_cb0_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr, inv_perm_ptr,
    # Dimensions
    M, N, K,
    # Strides (in elements)
    stride_am, stride_ak,
    stride_cm, stride_cn,
    # Tile layout
    tiles_n,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
):
    """
    Fused EXL3 dequantization + GEMM for 4bpw cb=0 (legacy kernel).

    Grid: (cdiv(M, BLOCK_M), N // 16)
    Each program computes a (BLOCK_M, 16) output tile.

    B layout: (K//16, N//16, 32) int32 — packed 4-bit trellis data.
    Each 32-word block encodes a 16x16 tile of 4-bit weights.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Accumulator: (BLOCK_M, 16) in float32
    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    # --- Precompute inverse permutation lookups (constant across K loop) ---
    # 2D matrix positions within a 16x16 tile
    k_local = tl.arange(0, 16)[:, None]   # (16, 1)
    n_local = tl.arange(0, 16)[None, :]   # (1, 16)
    mat_pos = k_local * 16 + n_local      # (16, 16) flat position [0, 256)

    # inv_perm[matrix_pos] = packed_pos in bitstream
    packed_pos = tl.load(inv_perm_ptr + mat_pos)  # (16, 16)

    # Bit extraction params for 4-bit.
    # CUDA dq8_aligned_4bits: i1 = t_offset/8, b=ptr[i1], a=ptr[(i1-1)%32]
    # indices[j] extracted at shift = (7 - j) * 4 from word i1
    # Funnel shift: ((a << 32) | b) >> shift, so lo=b=ptr[i1], hi=a=ptr[(i1-1)%32]
    word_idx = packed_pos >> 3          # = t_group = lane_id
    j_within = packed_pos & 7
    shift = (7 - j_within) * 4
    next_word_idx = (word_idx - 1) & 31  # previous word for funnel shift hi
    shift_hi = (32 - shift) & 31

    # --- K-dimension loop: one 16x16 B tile per iteration ---
    num_k_tiles = K // 16
    for tk in range(num_k_tiles):
        # Load A tile: (BLOCK_M, 16) float16
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_m = offs_m[:, None] < M
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        # Load packed B words for this tile
        b_base = (tk * tiles_n + pid_n) * 32
        lo = tl.load(B_ptr + b_base + word_idx)
        hi = tl.load(B_ptr + b_base + next_word_idx)

        # --- Funnel shift: extract 16-bit index from 1024-bit bitstream ---
        lo_part = (lo >> shift) & ((1 << shift_hi) - 1) & 0xFFFF
        hi_part = (hi << shift_hi) & 0xFFFF
        index = tl.where(shift > 0, lo_part | hi_part, lo & 0xFFFF)

        # --- decode_3inst cb=0 ---
        x = index * 89226354 + 64248484
        x_lo = (x & 0x8FFF) ^ 0x3B60
        x_hi = ((x >> 16) & 0x8FFF) ^ 0x3B60
        x = (x_hi << 16) | x_lo
        low_bits = (x & 0xFFFF).to(tl.int16)
        high_bits = ((x >> 16) & 0xFFFF).to(tl.int16)
        low_f16 = low_bits.to(tl.float16, bitcast=True)
        high_f16 = high_bits.to(tl.float16, bitcast=True)
        weight = low_f16 + high_f16

        # --- Matrix multiply: acc += a_tile @ weight ---
        acc += tl.dot(a_tile, weight)

    # --- Store C tile ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_m = offs_m[:, None] < M
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m)


# =============================================================================
# H_16 Hadamard matrix cache (for fused GEMM + Had-128 epilogue)
# =============================================================================

_h16_cache = {}

def _get_h16(device):
    """Build and cache the 16x16 normalized Hadamard matrix (entries ±1/√16) as fp16."""
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


# =============================================================================
# Fused GEMM + Hadamard-128 epilogue (non-split-K)
#
# Output Hadamard-128 = H_8 ⊗ H_16 (Kronecker product).
# Each thread block accumulates 8 × (BLOCK_M, 16) groups across 128 output cols.
# Epilogue: H_16 matmul per group, then H_8 butterfly across groups, svh flip.
# =============================================================================

@triton.jit
def _exl3_gemm_had128_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    H16_ptr,       # 16x16 normalized Hadamard matrix (fp16)
    svh_ptr,       # sign-flip vector, shape (N,), or null
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    stride_h16_r, stride_h16_c,
    # Tile layout
    tiles_n,       # = N // 16
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
    HAS_SVH: tl.constexpr,
):
    """
    Fused EXL3 dequant + GEMM + Hadamard-128 epilogue.

    Grid: (cdiv(M, BLOCK_M), N // 128)
    Each program computes a (BLOCK_M, 128) output tile with fused H_128.
    """
    pid_m = tl.program_id(0)
    pid_n128 = tl.program_id(1)  # which 128-col block

    # Load bit extraction tables
    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local
    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M

    # 8 accumulators, one per 16-col group within the 128-col block
    acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    # Base N-tile index for this 128-col block
    base_n_tile = pid_n128 * 8  # 8 tiles of 16 cols each = 128 cols

    # K-loop
    num_k_tiles = K // 16
    for tk in range(num_k_tiles):
        # Load A tile
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        # Dequant and accumulate for each of the 8 B tile columns
        for g in tl.static_range(8):
            b_col = base_n_tile + g
            b_base = (tk * tiles_n + b_col) * WORDS_PER_TILE
            lo = tl.load(B_ptr + b_base + word_idx)
            hi = tl.load(B_ptr + b_base + next_word_idx)
            w = _dequant_tile(lo, hi, shift, shift_hi, CB)

            if g == 0:
                acc0 += tl.dot(a_tile, w)
            elif g == 1:
                acc1 += tl.dot(a_tile, w)
            elif g == 2:
                acc2 += tl.dot(a_tile, w)
            elif g == 3:
                acc3 += tl.dot(a_tile, w)
            elif g == 4:
                acc4 += tl.dot(a_tile, w)
            elif g == 5:
                acc5 += tl.dot(a_tile, w)
            elif g == 6:
                acc6 += tl.dot(a_tile, w)
            elif g == 7:
                acc7 += tl.dot(a_tile, w)

    # --- Epilogue: H_128 = H_8 ⊗ H_16 ---

    # Load H_16 matrix (16x16, fp16)
    h16_r = tl.arange(0, 16)[:, None]
    h16_c = tl.arange(0, 16)[None, :]
    H16 = tl.load(H16_ptr + h16_r * stride_h16_r + h16_c * stride_h16_c)  # (16, 16) fp16

    # Step 1: H_16 within each group (matmul acc_i @ H16)
    acc0 = tl.dot(acc0.to(tl.float16), H16).to(tl.float32)
    acc1 = tl.dot(acc1.to(tl.float16), H16).to(tl.float32)
    acc2 = tl.dot(acc2.to(tl.float16), H16).to(tl.float32)
    acc3 = tl.dot(acc3.to(tl.float16), H16).to(tl.float32)
    acc4 = tl.dot(acc4.to(tl.float16), H16).to(tl.float32)
    acc5 = tl.dot(acc5.to(tl.float16), H16).to(tl.float32)
    acc6 = tl.dot(acc6.to(tl.float16), H16).to(tl.float32)
    acc7 = tl.dot(acc7.to(tl.float16), H16).to(tl.float32)

    # Step 2: H_8 butterfly across 8 groups (3 rounds, in fp32)
    # Round 1
    t0 = acc0 + acc1
    t1 = acc0 - acc1
    t2 = acc2 + acc3
    t3 = acc2 - acc3
    t4 = acc4 + acc5
    t5 = acc4 - acc5
    t6 = acc6 + acc7
    t7 = acc6 - acc7
    # Round 2
    s0 = t0 + t2
    s1 = t1 + t3
    s2 = t0 - t2
    s3 = t1 - t3
    s4 = t4 + t6
    s5 = t5 + t7
    s6 = t4 - t6
    s7 = t5 - t7
    # Round 3
    acc0 = s0 + s4
    acc1 = s1 + s5
    acc2 = s2 + s6
    acc3 = s3 + s7
    acc4 = s0 - s4
    acc5 = s1 - s5
    acc6 = s2 - s6
    acc7 = s3 - s7

    # Scale: 1/sqrt(8) for H_8 (H_16 already has 1/sqrt(16) baked in, total = 1/sqrt(128))
    inv_sqrt8: tl.constexpr = 0.35355339059327373
    acc0 = acc0 * inv_sqrt8
    acc1 = acc1 * inv_sqrt8
    acc2 = acc2 * inv_sqrt8
    acc3 = acc3 * inv_sqrt8
    acc4 = acc4 * inv_sqrt8
    acc5 = acc5 * inv_sqrt8
    acc6 = acc6 * inv_sqrt8
    acc7 = acc7 * inv_sqrt8

    # Step 3: svh sign-flip per group
    if HAS_SVH:
        base_n = pid_n128 * 128
        svh_offs = tl.arange(0, 16)[None, :]
        svh0 = tl.load(svh_ptr + base_n + 0 * 16 + svh_offs).to(tl.float32)
        svh1 = tl.load(svh_ptr + base_n + 1 * 16 + svh_offs).to(tl.float32)
        svh2 = tl.load(svh_ptr + base_n + 2 * 16 + svh_offs).to(tl.float32)
        svh3 = tl.load(svh_ptr + base_n + 3 * 16 + svh_offs).to(tl.float32)
        svh4 = tl.load(svh_ptr + base_n + 4 * 16 + svh_offs).to(tl.float32)
        svh5 = tl.load(svh_ptr + base_n + 5 * 16 + svh_offs).to(tl.float32)
        svh6 = tl.load(svh_ptr + base_n + 6 * 16 + svh_offs).to(tl.float32)
        svh7 = tl.load(svh_ptr + base_n + 7 * 16 + svh_offs).to(tl.float32)
        acc0 = acc0 * svh0
        acc1 = acc1 * svh1
        acc2 = acc2 * svh2
        acc3 = acc3 * svh3
        acc4 = acc4 * svh4
        acc5 = acc5 * svh5
        acc6 = acc6 * svh6
        acc7 = acc7 * svh7

    # Store all 8 groups
    for g in tl.static_range(8):
        offs_n = pid_n128 * 128 + g * 16 + tl.arange(0, 16)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        if g == 0:
            tl.store(c_ptrs, acc0.to(tl.float16), mask=mask_m)
        elif g == 1:
            tl.store(c_ptrs, acc1.to(tl.float16), mask=mask_m)
        elif g == 2:
            tl.store(c_ptrs, acc2.to(tl.float16), mask=mask_m)
        elif g == 3:
            tl.store(c_ptrs, acc3.to(tl.float16), mask=mask_m)
        elif g == 4:
            tl.store(c_ptrs, acc4.to(tl.float16), mask=mask_m)
        elif g == 5:
            tl.store(c_ptrs, acc5.to(tl.float16), mask=mask_m)
        elif g == 6:
            tl.store(c_ptrs, acc6.to(tl.float16), mask=mask_m)
        elif g == 7:
            tl.store(c_ptrs, acc7.to(tl.float16), mask=mask_m)


# =============================================================================
# Split-K variant for fused GEMM + Had-128
# Writes partial results WITHOUT Hadamard (reduce kernel applies it).
# =============================================================================

@triton.jit
def _exl3_gemm_splitk_had128_kernel(
    A_ptr, B_ptr, C_partial_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_cp_split, stride_cp_m, stride_cp_n,
    tiles_n,
    tiles_per_split,
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Split-K partial GEMM for 128-wide output blocks (no Hadamard in epilogue).

    Grid: (cdiv(M, BLOCK_M), N // 128, split_k)
    Writes 8 × 16-col partial results to C_partial[pid_k, M, 128_cols].
    """
    pid_m = tl.program_id(0)
    pid_n128 = tl.program_id(1)
    pid_k = tl.program_id(2)

    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local
    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M

    acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    base_n_tile = pid_n128 * 8

    tk_start = pid_k * tiles_per_split
    tk_end = tl.minimum(tk_start + tiles_per_split, K // 16)

    for tk in range(tk_start, tk_end):
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        for g in tl.static_range(8):
            b_col = base_n_tile + g
            b_base = (tk * tiles_n + b_col) * WORDS_PER_TILE
            lo = tl.load(B_ptr + b_base + word_idx)
            hi = tl.load(B_ptr + b_base + next_word_idx)
            w = _dequant_tile(lo, hi, shift, shift_hi, CB)

            if g == 0:
                acc0 += tl.dot(a_tile, w)
            elif g == 1:
                acc1 += tl.dot(a_tile, w)
            elif g == 2:
                acc2 += tl.dot(a_tile, w)
            elif g == 3:
                acc3 += tl.dot(a_tile, w)
            elif g == 4:
                acc4 += tl.dot(a_tile, w)
            elif g == 5:
                acc5 += tl.dot(a_tile, w)
            elif g == 6:
                acc6 += tl.dot(a_tile, w)
            elif g == 7:
                acc7 += tl.dot(a_tile, w)

    # Store partial results (no Hadamard — the reduce kernel applies it)
    for g in tl.static_range(8):
        offs_n = pid_n128 * 128 + g * 16 + tl.arange(0, 16)
        c_ptrs = (C_partial_ptr
                  + pid_k * stride_cp_split
                  + offs_m[:, None] * stride_cp_m
                  + offs_n[None, :] * stride_cp_n)
        if g == 0:
            tl.store(c_ptrs, acc0.to(tl.float16), mask=mask_m)
        elif g == 1:
            tl.store(c_ptrs, acc1.to(tl.float16), mask=mask_m)
        elif g == 2:
            tl.store(c_ptrs, acc2.to(tl.float16), mask=mask_m)
        elif g == 3:
            tl.store(c_ptrs, acc3.to(tl.float16), mask=mask_m)
        elif g == 4:
            tl.store(c_ptrs, acc4.to(tl.float16), mask=mask_m)
        elif g == 5:
            tl.store(c_ptrs, acc5.to(tl.float16), mask=mask_m)
        elif g == 6:
            tl.store(c_ptrs, acc6.to(tl.float16), mask=mask_m)
        elif g == 7:
            tl.store(c_ptrs, acc7.to(tl.float16), mask=mask_m)


# =============================================================================
# Reduce + Had-128 kernel (fused reduction of split-K partials + Hadamard)
# =============================================================================

@triton.jit
def _reduce_had128_kernel(
    C_partial_ptr, C_ptr,
    H16_ptr, svh_ptr,
    M, N,
    stride_cp_split, stride_cp_m, stride_cp_n,
    stride_cm, stride_cn,
    stride_h16_r, stride_h16_c,
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_SVH: tl.constexpr,
):
    """
    Fused reduce + H_128 + svh for split-K partial results.

    Grid: (cdiv(M, BLOCK_M), N // 128)
    Loads SPLIT_K partials, sums them, applies H_128 epilogue, stores.
    """
    pid_m = tl.program_id(0)
    pid_n128 = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M

    # Sum partials for each of the 8 groups
    acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    for sk in range(SPLIT_K):
        for g in tl.static_range(8):
            offs_n = pid_n128 * 128 + g * 16 + tl.arange(0, 16)
            cp_ptrs = (C_partial_ptr
                       + sk * stride_cp_split
                       + offs_m[:, None] * stride_cp_m
                       + offs_n[None, :] * stride_cp_n)
            partial = tl.load(cp_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            if g == 0:
                acc0 += partial
            elif g == 1:
                acc1 += partial
            elif g == 2:
                acc2 += partial
            elif g == 3:
                acc3 += partial
            elif g == 4:
                acc4 += partial
            elif g == 5:
                acc5 += partial
            elif g == 6:
                acc6 += partial
            elif g == 7:
                acc7 += partial

    # --- H_128 epilogue (same as non-split-K version) ---

    # Load H_16
    h16_r = tl.arange(0, 16)[:, None]
    h16_c = tl.arange(0, 16)[None, :]
    H16 = tl.load(H16_ptr + h16_r * stride_h16_r + h16_c * stride_h16_c)

    # Step 1: H_16 per group
    acc0 = tl.dot(acc0.to(tl.float16), H16).to(tl.float32)
    acc1 = tl.dot(acc1.to(tl.float16), H16).to(tl.float32)
    acc2 = tl.dot(acc2.to(tl.float16), H16).to(tl.float32)
    acc3 = tl.dot(acc3.to(tl.float16), H16).to(tl.float32)
    acc4 = tl.dot(acc4.to(tl.float16), H16).to(tl.float32)
    acc5 = tl.dot(acc5.to(tl.float16), H16).to(tl.float32)
    acc6 = tl.dot(acc6.to(tl.float16), H16).to(tl.float32)
    acc7 = tl.dot(acc7.to(tl.float16), H16).to(tl.float32)

    # Step 2: H_8 butterfly
    t0 = acc0 + acc1
    t1 = acc0 - acc1
    t2 = acc2 + acc3
    t3 = acc2 - acc3
    t4 = acc4 + acc5
    t5 = acc4 - acc5
    t6 = acc6 + acc7
    t7 = acc6 - acc7

    s0 = t0 + t2
    s1 = t1 + t3
    s2 = t0 - t2
    s3 = t1 - t3
    s4 = t4 + t6
    s5 = t5 + t7
    s6 = t4 - t6
    s7 = t5 - t7

    acc0 = s0 + s4
    acc1 = s1 + s5
    acc2 = s2 + s6
    acc3 = s3 + s7
    acc4 = s0 - s4
    acc5 = s1 - s5
    acc6 = s2 - s6
    acc7 = s3 - s7

    inv_sqrt8: tl.constexpr = 0.35355339059327373
    acc0 = acc0 * inv_sqrt8
    acc1 = acc1 * inv_sqrt8
    acc2 = acc2 * inv_sqrt8
    acc3 = acc3 * inv_sqrt8
    acc4 = acc4 * inv_sqrt8
    acc5 = acc5 * inv_sqrt8
    acc6 = acc6 * inv_sqrt8
    acc7 = acc7 * inv_sqrt8

    # Step 3: svh sign-flip
    if HAS_SVH:
        base_n = pid_n128 * 128
        svh_offs = tl.arange(0, 16)[None, :]
        svh0 = tl.load(svh_ptr + base_n + 0 * 16 + svh_offs).to(tl.float32)
        svh1 = tl.load(svh_ptr + base_n + 1 * 16 + svh_offs).to(tl.float32)
        svh2 = tl.load(svh_ptr + base_n + 2 * 16 + svh_offs).to(tl.float32)
        svh3 = tl.load(svh_ptr + base_n + 3 * 16 + svh_offs).to(tl.float32)
        svh4 = tl.load(svh_ptr + base_n + 4 * 16 + svh_offs).to(tl.float32)
        svh5 = tl.load(svh_ptr + base_n + 5 * 16 + svh_offs).to(tl.float32)
        svh6 = tl.load(svh_ptr + base_n + 6 * 16 + svh_offs).to(tl.float32)
        svh7 = tl.load(svh_ptr + base_n + 7 * 16 + svh_offs).to(tl.float32)
        acc0 = acc0 * svh0
        acc1 = acc1 * svh1
        acc2 = acc2 * svh2
        acc3 = acc3 * svh3
        acc4 = acc4 * svh4
        acc5 = acc5 * svh5
        acc6 = acc6 * svh6
        acc7 = acc7 * svh7

    # Store
    for g in tl.static_range(8):
        offs_n = pid_n128 * 128 + g * 16 + tl.arange(0, 16)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        if g == 0:
            tl.store(c_ptrs, acc0.to(tl.float16), mask=mask_m)
        elif g == 1:
            tl.store(c_ptrs, acc1.to(tl.float16), mask=mask_m)
        elif g == 2:
            tl.store(c_ptrs, acc2.to(tl.float16), mask=mask_m)
        elif g == 3:
            tl.store(c_ptrs, acc3.to(tl.float16), mask=mask_m)
        elif g == 4:
            tl.store(c_ptrs, acc4.to(tl.float16), mask=mask_m)
        elif g == 5:
            tl.store(c_ptrs, acc5.to(tl.float16), mask=mask_m)
        elif g == 6:
            tl.store(c_ptrs, acc6.to(tl.float16), mask=mask_m)
        elif g == 7:
            tl.store(c_ptrs, acc7.to(tl.float16), mask=mask_m)


# =============================================================================
# Python wrappers
# =============================================================================

def _get_splitk_buf(split_k, M, N, device):
    """Allocate a per-call partial buffer for split-K reduction.

    Previously this was a cached buffer keyed by (device, split_k), but that
    caused a race condition under concurrent async CUDA kernel dispatches:
    two requests arriving via different streams would stomp on the same
    buffer before their reductions completed, producing garbled output
    (repeating '!' tokens). torch's caching allocator makes per-call
    allocation essentially free.
    """
    return torch.empty((split_k, max(M, 1), max(N, 1)), dtype=torch.float16, device=device)


# =============================================================================
# Split-K auto-tuning
# =============================================================================

# In-memory cache: (M, K, N, bits, cb, device_name) → (split_k, num_warps, num_stages, BLOCK_M)
_tune_cache = {}

# Disk cache path: env var or default
_TUNE_CACHE_FILE = os.environ.get(
    "EXL3_SPLITK_CACHE",
    os.path.join(str(Path.home()), ".cache", "exl3_triton", "splitk_cache.json"),
)

# Disable auto-tuning via env var (fallback to static heuristic)
_AUTOTUNE_DISABLED = os.environ.get("EXL3_SPLITK_AUTOTUNE", "1") == "0"

# Default kernel config
_DEFAULT_CONFIG = (1, 2, 2, 16)  # (split_k, num_warps, num_stages, BLOCK_M)

def _normalize_config(val):
    """Convert cache entry to (split_k, num_warps, num_stages, BLOCK_M) tuple.
    Backwards-compatible: old int entries become (sk, 2, 2, 16)."""
    if isinstance(val, int):
        return (val, 2, 2, 16)
    if isinstance(val, (list, tuple)) and len(val) == 4:
        return tuple(int(x) for x in val)
    return _DEFAULT_CONFIG


def _heuristic_config(M, tiles_k, tiles_n):
    """Static heuristic for kernel config (fallback when auto-tuning is disabled).
    Returns (split_k, num_warps, num_stages, BLOCK_M)."""
    total_tiles = tiles_k * tiles_n
    if M <= 16 and total_tiles > 32768 and tiles_k >= 16:
        split_k = min(8, tiles_k)
        while split_k > 1 and tiles_k % split_k != 0:
            split_k -= 1
        return (split_k, 2, 2, 16)
    return _DEFAULT_CONFIG


def _get_tune_candidates(tiles_k, M):
    """Return list of (split_k, num_warps, num_stages, BLOCK_M) configs to benchmark."""
    if M > 16:
        # Prefill: no split-K, only vary warps
        return [_DEFAULT_CONFIG]

    # Decode (M <= 16): vary split_k, num_warps, num_stages, BLOCK_M
    configs = []
    for sk in [1, 2, 4, 8]:
        if sk > tiles_k or tiles_k % sk != 0:
            continue
        for warps in [1, 2]:
            for stages in [2, 3]:
                for block_m in [16, 32]:
                    configs.append((sk, warps, stages, block_m))
    return configs if configs else [_DEFAULT_CONFIG]


def _load_disk_cache():
    """Load tune cache from disk into memory.
    Backwards-compatible: old int entries become (sk, 2, 2, 16)."""
    try:
        with open(_TUNE_CACHE_FILE, "r") as f:
            disk_data = json.load(f)
        for key_str, val in disk_data.items():
            parts = key_str.split("|")
            if len(parts) == 6:
                device_name, M, K, N, bits, cb = parts
                cache_key = (int(M), int(K), int(N), int(bits), int(cb), device_name)
                _tune_cache[cache_key] = _normalize_config(val)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass


def _save_to_disk_cache(cache_key, config):
    """Save a single entry to the disk cache.
    config is (split_k, num_warps, num_stages, BLOCK_M)."""
    M, K, N, bits, cb, device_name = cache_key
    key_str = f"{device_name}|{M}|{K}|{N}|{bits}|{cb}"

    # Load existing disk cache
    disk_data = {}
    try:
        with open(_TUNE_CACHE_FILE, "r") as f:
            disk_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    disk_data[key_str] = list(config)

    os.makedirs(os.path.dirname(_TUNE_CACHE_FILE), exist_ok=True)
    with open(_TUNE_CACHE_FILE, "w") as f:
        json.dump(disk_data, f, indent=2)


def clear_splitk_cache(disk=True):
    """Clear the auto-tune cache (in-memory and optionally disk)."""
    _tune_cache.clear()
    if disk:
        try:
            os.remove(_TUNE_CACHE_FILE)
        except FileNotFoundError:
            pass


def autotune_splitk(M, K, N, bits, cb, device, warmup=3, iters=15, save_to_disk=True):
    """
    Benchmark kernel configs and return the best (split_k, num_warps, num_stages, BLOCK_M).

    Checks in-memory cache → disk cache → benchmark.
    Tests split_k, num_warps, num_stages, BLOCK_M combinations for M<=16 decode.
    """
    device_name = torch.cuda.get_device_name(device)
    cache_key = (M, K, N, bits, cb, device_name)

    # Check in-memory cache
    cached = _tune_cache.get(cache_key)
    if cached is not None:
        return cached

    # Check disk cache (lazy load on first miss)
    if not _tune_cache:
        _load_disk_cache()
        cached = _tune_cache.get(cache_key)
        if cached is not None:
            return cached

    tiles_k = K // 16
    tiles_n = N // 16
    candidates = _get_tune_candidates(tiles_k, M)

    if len(candidates) == 1:
        config = candidates[0]
        _tune_cache[cache_key] = config
        if save_to_disk:
            _save_to_disk_cache(cache_key, config)
        return config

    # Create random test tensors
    A_test = torch.randn(M, K, dtype=torch.float16, device=device)
    B_test = torch.randint(
        -32768, 32767, (tiles_k, tiles_n, 16 * bits),
        dtype=torch.int16, device=device,
    )
    B_i32_test = B_test.view(torch.int32)
    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, device)
    WORDS_PER_TILE = 8 * bits

    best_config = _DEFAULT_CONFIG
    best_time = float("inf")

    for config in candidates:
        sk, warps, stages, block_m = config
        try:
            if sk > 1:
                tiles_per_split = triton.cdiv(tiles_k, sk)
                buf = _get_splitk_buf(sk, M, N, device)
                C_partial = buf[:sk, :M, :N]
                grid = (triton.cdiv(M, block_m), tiles_n, sk)

                # Warmup
                for _ in range(warmup):
                    _exl3_gemm_splitk_kernel[grid](
                        A_test, B_i32_test, C_partial,
                        word_idx_t, next_word_idx_t, shift_t,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
                        tiles_n, tiles_per_split,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        SPLIT_K=sk,
                        num_warps=warps, num_stages=stages,
                    )
                    C_partial.sum(dim=0)
                torch.cuda.synchronize(device)

                # Timed iterations
                times = []
                for _ in range(iters):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    _exl3_gemm_splitk_kernel[grid](
                        A_test, B_i32_test, C_partial,
                        word_idx_t, next_word_idx_t, shift_t,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
                        tiles_n, tiles_per_split,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        SPLIT_K=sk,
                        num_warps=warps, num_stages=stages,
                    )
                    C_partial.sum(dim=0)
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - t0)
            else:
                C = torch.empty((M, N), dtype=torch.float16, device=device)
                grid = (triton.cdiv(M, block_m), tiles_n)

                # Warmup
                for _ in range(warmup):
                    _exl3_gemm_kernel[grid](
                        A_test, B_i32_test, C,
                        word_idx_t, next_word_idx_t, shift_t,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C.stride(0), C.stride(1),
                        tiles_n,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        num_warps=warps, num_stages=stages,
                    )
                torch.cuda.synchronize(device)

                # Timed iterations
                times = []
                for _ in range(iters):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    _exl3_gemm_kernel[grid](
                        A_test, B_i32_test, C,
                        word_idx_t, next_word_idx_t, shift_t,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C.stride(0), C.stride(1),
                        tiles_n,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        num_warps=warps, num_stages=stages,
                    )
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - t0)

            median_time = sorted(times)[len(times) // 2]
            if median_time < best_time:
                best_time = median_time
                best_config = config
        except Exception:
            pass  # Some configs may crash (e.g. stages=3 on some shapes)

    _tune_cache[cache_key] = best_config
    if save_to_disk:
        _save_to_disk_cache(cache_key, best_config)

    return best_config


# Load disk cache eagerly at import time
_load_disk_cache()


def exl3_gemm(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    bits: int = 4,
    cb: int = 0,
    out: torch.Tensor = None,
    B_i32: torch.Tensor = None,
    split_k: int = 0,
) -> torch.Tensor:
    """
    EXL3 fused dequant + GEMM for any bitwidth and codebook.

    Args:
        A: Input activations, float16, shape (M, K). K must be divisible by 16.
        B_packed: Packed EXL3 weights, int16, shape (K//16, N//16, 16*bits).
        bits: Bits per weight (1-8).
        cb: Codebook variant (0=default, 1=MCG, 2=MUL1).
        out: Optional pre-allocated output tensor, float16, shape (M, N).
        B_i32: Optional pre-computed int32 view of B_packed (avoids view() call).
        split_k: Split-K factor for M=1 decode optimization. 0 = auto-select.

    Returns:
        C: Output, float16, shape (M, N)
    """
    assert A.dtype == torch.float16, f"A must be float16, got {A.dtype}"
    assert A.is_contiguous()
    assert 1 <= bits <= 8, f"bits must be 1-8, got {bits}"
    assert cb in (0, 1, 2), f"cb must be 0, 1, or 2, got {cb}"

    M, K = A.shape
    tiles_k, tiles_n, packed_per_tile = B_packed.shape
    N = tiles_n * 16

    # View B as int32 for bit manipulation (no copy)
    if B_i32 is None:
        B_i32 = B_packed.view(torch.int32)

    # Get precomputed bit extraction tables
    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, A.device)

    WORDS_PER_TILE = 8 * bits

    # Auto-select kernel config: (split_k, num_warps, num_stages, BLOCK_M)
    if split_k == 0:
        if _AUTOTUNE_DISABLED:
            config = _heuristic_config(M, tiles_k, tiles_n)
        else:
            device_name = torch.cuda.get_device_name(A.device)
            cache_key = (M, K, N, bits, cb, device_name)
            cached = _tune_cache.get(cache_key)
            if cached is not None:
                config = cached
            else:
                config = autotune_splitk(M, K, N, bits, cb, A.device)
        split_k, num_warps, num_stages, BLOCK_M = config
    else:
        # Explicit split_k provided, use default kernel params
        num_warps, num_stages, BLOCK_M = 2, 2, 16

    if split_k > 1:
        # Split-K path: better GPU utilization for small M
        tiles_per_split = triton.cdiv(tiles_k, split_k)
        partial_buf = _get_splitk_buf(split_k, M, N, A.device)
        C_partial = partial_buf[:split_k, :M, :N]

        grid = (triton.cdiv(M, BLOCK_M), tiles_n, split_k)
        _exl3_gemm_splitk_kernel[grid](
            A, B_i32, C_partial,
            word_idx_t, next_word_idx_t, shift_t,
            M, N, K,
            A.stride(0), A.stride(1),
            C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
            tiles_n, tiles_per_split,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            SPLIT_K=split_k,
            num_warps=num_warps, num_stages=num_stages,
        )

        # Reduce: sum partial results in fp16 (no precision loss for ≤8 splits)
        if out is not None:
            torch.sum(C_partial, dim=0, out=out)
            return out
        else:
            return C_partial.sum(dim=0)

    else:
        # Standard path: single kernel
        if out is not None:
            C = out
        else:
            C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        grid = (triton.cdiv(M, BLOCK_M), tiles_n)
        _exl3_gemm_kernel[grid](
            A, B_i32, C,
            word_idx_t, next_word_idx_t, shift_t,
            M, N, K,
            A.stride(0), A.stride(1),
            C.stride(0), C.stride(1),
            tiles_n,
            BLOCK_M=BLOCK_M,
            WORDS_PER_TILE=WORDS_PER_TILE,
            CB=cb,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return C


def exl3_gemm_4bit_cb0(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    inv_perm: torch.Tensor = None,
) -> torch.Tensor:
    """
    EXL3 4-bit cb=0 fused dequant + GEMM (legacy wrapper).

    Args:
        A: Input activations, float16, shape (M, K).
           K must be divisible by 16.
        B_packed: Packed EXL3 weights, int16, shape (K//16, N//16, 64).
           Each tile contains 64 int16 values = 32 uint32 words = 1024 bits
           encoding 256 4-bit weights.
        inv_perm: Inverse tensor core permutation, int32, shape (256,).
           Maps matrix position to packed position. Auto-computed if None.

    Returns:
        C: Output, float16, shape (M, N)
    """
    assert A.dtype == torch.float16, f"A must be float16, got {A.dtype}"
    assert B_packed.dtype == torch.int16, f"B must be int16, got {B_packed.dtype}"
    assert A.is_contiguous()
    assert B_packed.is_contiguous()

    M, K = A.shape
    tiles_k, tiles_n, packed_per_tile = B_packed.shape
    N = tiles_n * 16

    assert K == tiles_k * 16, f"K mismatch: A has K={K}, B implies K={tiles_k * 16}"
    assert packed_per_tile == 64, f"Expected 64 int16 per tile for 4-bit, got {packed_per_tile}"

    # View B as int32 for bit manipulation (no copy)
    B_i32 = B_packed.view(torch.int32)  # (K//16, N//16, 32)

    if inv_perm is None:
        inv_perm = get_inv_perm(A.device)

    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    BLOCK_M = 16
    grid = (triton.cdiv(M, BLOCK_M), tiles_n)

    _exl3_gemm_4bit_cb0_kernel[grid](
        A, B_i32, C, inv_perm,
        M, N, K,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        tiles_n,
        BLOCK_M=BLOCK_M,
    )

    return C


# =============================================================================
# Fused GEMM + Hadamard-128 auto-tuning
# =============================================================================

# In-memory cache: (M, K, N, bits, cb, device_name, "had") → (split_k, num_warps, num_stages, BLOCK_M)
_had_tune_cache = {}

# Disk cache path for fused Had kernels
_HAD_TUNE_CACHE_FILE = os.environ.get(
    "EXL3_HAD_CACHE",
    os.path.join(str(Path.home()), ".cache", "exl3_triton", "had_cache.json"),
)


def _load_had_disk_cache():
    """Load fused Had tune cache from disk."""
    try:
        with open(_HAD_TUNE_CACHE_FILE, "r") as f:
            disk_data = json.load(f)
        for key_str, val in disk_data.items():
            parts = key_str.split("|")
            if len(parts) == 6:
                device_name, M, K, N, bits, cb = parts
                cache_key = (int(M), int(K), int(N), int(bits), int(cb), device_name)
                _had_tune_cache[cache_key] = _normalize_config(val)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass


def _save_had_to_disk_cache(cache_key, config):
    """Save a single entry to the fused Had disk cache."""
    M, K, N, bits, cb, device_name = cache_key
    key_str = f"{device_name}|{M}|{K}|{N}|{bits}|{cb}"

    disk_data = {}
    try:
        with open(_HAD_TUNE_CACHE_FILE, "r") as f:
            disk_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    disk_data[key_str] = list(config)

    os.makedirs(os.path.dirname(_HAD_TUNE_CACHE_FILE), exist_ok=True)
    with open(_HAD_TUNE_CACHE_FILE, "w") as f:
        json.dump(disk_data, f, indent=2)


def autotune_had(M, K, N, bits, cb, device, warmup=3, iters=15, save_to_disk=True):
    """
    Benchmark fused GEMM+Had kernel configs and return the best
    (split_k, num_warps, num_stages, BLOCK_M).
    """
    device_name = torch.cuda.get_device_name(device)
    cache_key = (M, K, N, bits, cb, device_name)

    cached = _had_tune_cache.get(cache_key)
    if cached is not None:
        return cached

    # Lazy load from disk
    if not _had_tune_cache:
        _load_had_disk_cache()
        cached = _had_tune_cache.get(cache_key)
        if cached is not None:
            return cached

    tiles_k = K // 16
    tiles_n = N // 16
    tiles_n128 = N // 128
    candidates = _get_tune_candidates(tiles_k, M)

    if len(candidates) == 1:
        config = candidates[0]
        _had_tune_cache[cache_key] = config
        if save_to_disk:
            _save_had_to_disk_cache(cache_key, config)
        return config

    # Create test tensors
    A_test = torch.randn(M, K, dtype=torch.float16, device=device)
    B_test = torch.randint(
        -32768, 32767, (tiles_k, tiles_n, 16 * bits),
        dtype=torch.int16, device=device,
    )
    B_i32_test = B_test.view(torch.int32)
    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, device)
    H16 = _get_h16(device)
    WORDS_PER_TILE = 8 * bits

    best_config = _DEFAULT_CONFIG
    best_time = float("inf")

    for config in candidates:
        sk, warps, stages, block_m = config
        try:
            C = torch.empty((M, N), dtype=torch.float16, device=device)
            if sk > 1:
                tiles_per_split = triton.cdiv(tiles_k, sk)
                buf = _get_splitk_buf(sk, M, N, device)
                C_partial = buf[:sk, :M, :N]

                grid_sk = (triton.cdiv(M, block_m), tiles_n128, sk)
                grid_red = (triton.cdiv(M, block_m), tiles_n128)

                for _ in range(warmup):
                    _exl3_gemm_splitk_had128_kernel[grid_sk](
                        A_test, B_i32_test, C_partial,
                        word_idx_t, next_word_idx_t, shift_t,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
                        tiles_n, tiles_per_split,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        SPLIT_K=sk,
                        num_warps=warps, num_stages=stages,
                    )
                    _reduce_had128_kernel[grid_red](
                        C_partial, C,
                        H16, None,
                        M, N,
                        C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
                        C.stride(0), C.stride(1),
                        H16.stride(0), H16.stride(1),
                        BLOCK_M=block_m, SPLIT_K=sk, HAS_SVH=False,
                        num_warps=warps, num_stages=stages,
                    )
                torch.cuda.synchronize(device)

                times = []
                for _ in range(iters):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    _exl3_gemm_splitk_had128_kernel[grid_sk](
                        A_test, B_i32_test, C_partial,
                        word_idx_t, next_word_idx_t, shift_t,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
                        tiles_n, tiles_per_split,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        SPLIT_K=sk,
                        num_warps=warps, num_stages=stages,
                    )
                    _reduce_had128_kernel[grid_red](
                        C_partial, C,
                        H16, None,
                        M, N,
                        C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
                        C.stride(0), C.stride(1),
                        H16.stride(0), H16.stride(1),
                        BLOCK_M=block_m, SPLIT_K=sk, HAS_SVH=False,
                        num_warps=warps, num_stages=stages,
                    )
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - t0)
            else:
                grid = (triton.cdiv(M, block_m), tiles_n128)

                for _ in range(warmup):
                    _exl3_gemm_had128_kernel[grid](
                        A_test, B_i32_test, C,
                        word_idx_t, next_word_idx_t, shift_t,
                        H16, None,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C.stride(0), C.stride(1),
                        H16.stride(0), H16.stride(1),
                        tiles_n,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        HAS_SVH=False,
                        num_warps=warps, num_stages=stages,
                    )
                torch.cuda.synchronize(device)

                times = []
                for _ in range(iters):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    _exl3_gemm_had128_kernel[grid](
                        A_test, B_i32_test, C,
                        word_idx_t, next_word_idx_t, shift_t,
                        H16, None,
                        M, N, K,
                        A_test.stride(0), A_test.stride(1),
                        C.stride(0), C.stride(1),
                        H16.stride(0), H16.stride(1),
                        tiles_n,
                        BLOCK_M=block_m, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
                        HAS_SVH=False,
                        num_warps=warps, num_stages=stages,
                    )
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - t0)

            median_time = sorted(times)[len(times) // 2]
            if median_time < best_time:
                best_time = median_time
                best_config = config
        except Exception:
            pass

    _had_tune_cache[cache_key] = best_config
    if save_to_disk:
        _save_had_to_disk_cache(cache_key, best_config)

    return best_config


# Load had disk cache eagerly at import time
_load_had_disk_cache()


def exl3_gemm_had(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    bits: int = 4,
    cb: int = 0,
    svh: torch.Tensor = None,
    out: torch.Tensor = None,
    B_i32: torch.Tensor = None,
    split_k: int = 0,
) -> torch.Tensor:
    """
    Fused EXL3 dequant + GEMM + output Hadamard-128 + svh sign-flip.

    Equivalent to: had_r_128(exl3_gemm(A, B, bits, cb), svh) but in one
    fused operation (fewer kernel launches and memory round-trips).

    Args:
        A: Input activations, float16, shape (M, K). K must be divisible by 16.
        B_packed: Packed EXL3 weights, int16, shape (K//16, N//16, 16*bits).
        bits: Bits per weight (1-8).
        cb: Codebook variant (0=default, 1=MCG, 2=MUL1).
        svh: Sign-flip vector for output Hadamard, shape (N,), fp16. None = no sign flip.
        out: Optional pre-allocated output tensor, float16, shape (M, N).
        B_i32: Optional pre-computed int32 view of B_packed.
        split_k: Split-K factor. 0 = auto-select.

    Returns:
        C: Output, float16, shape (M, N)
    """
    assert A.dtype == torch.float16, f"A must be float16, got {A.dtype}"
    assert A.is_contiguous()
    assert 1 <= bits <= 8, f"bits must be 1-8, got {bits}"
    assert cb in (0, 1, 2), f"cb must be 0, 1, or 2, got {cb}"

    M, K = A.shape
    tiles_k, tiles_n, packed_per_tile = B_packed.shape
    N = tiles_n * 16

    # Fallback: if N not divisible by 128, use separate GEMM + Hadamard
    if N % 128 != 0:
        C = exl3_gemm(A, B_packed, bits=bits, cb=cb, out=out, B_i32=B_i32, split_k=split_k)
        if svh is not None:
            C *= svh.unsqueeze(0)
        return C

    if B_i32 is None:
        B_i32 = B_packed.view(torch.int32)

    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, A.device)
    H16 = _get_h16(A.device)
    WORDS_PER_TILE = 8 * bits

    tiles_n128 = N // 128
    HAS_SVH = svh is not None

    # Auto-select kernel config
    if split_k == 0:
        if _AUTOTUNE_DISABLED:
            config = _heuristic_config(M, tiles_k, tiles_n)
        else:
            device_name = torch.cuda.get_device_name(A.device)
            cache_key = (M, K, N, bits, cb, device_name)
            cached = _had_tune_cache.get(cache_key)
            if cached is not None:
                config = cached
            else:
                config = autotune_had(M, K, N, bits, cb, A.device)
        split_k, num_warps, num_stages, BLOCK_M = config
    else:
        num_warps, num_stages, BLOCK_M = 2, 2, 16

    if out is not None:
        C = out
    else:
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    if split_k > 1:
        # Split-K path: partial GEMM → fused reduce + Had
        tiles_per_split = triton.cdiv(tiles_k, split_k)
        partial_buf = _get_splitk_buf(split_k, M, N, A.device)
        C_partial = partial_buf[:split_k, :M, :N]

        grid_sk = (triton.cdiv(M, BLOCK_M), tiles_n128, split_k)
        _exl3_gemm_splitk_had128_kernel[grid_sk](
            A, B_i32, C_partial,
            word_idx_t, next_word_idx_t, shift_t,
            M, N, K,
            A.stride(0), A.stride(1),
            C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
            tiles_n, tiles_per_split,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            SPLIT_K=split_k,
            num_warps=num_warps, num_stages=num_stages,
        )

        grid_red = (triton.cdiv(M, BLOCK_M), tiles_n128)
        _reduce_had128_kernel[grid_red](
            C_partial, C,
            H16, svh,
            M, N,
            C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
            C.stride(0), C.stride(1),
            H16.stride(0), H16.stride(1),
            BLOCK_M=BLOCK_M, SPLIT_K=split_k, HAS_SVH=HAS_SVH,
            num_warps=num_warps, num_stages=num_stages,
        )

    else:
        # Non-split-K path: single fused kernel
        grid = (triton.cdiv(M, BLOCK_M), tiles_n128)
        _exl3_gemm_had128_kernel[grid](
            A, B_i32, C,
            word_idx_t, next_word_idx_t, shift_t,
            H16, svh,
            M, N, K,
            A.stride(0), A.stride(1),
            C.stride(0), C.stride(1),
            H16.stride(0), H16.stride(1),
            tiles_n,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            HAS_SVH=HAS_SVH,
            num_warps=num_warps, num_stages=num_stages,
        )

    return C


# =============================================================================
# Batched multi-GEMM kernels (for fused K+V and gate+up projections)
# =============================================================================

@triton.jit
def _exl3_mgemm_kernel(
    A_ptr, B_ptr, C_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    M, N, K,
    stride_a_batch, stride_am, stride_ak,
    stride_b_batch,
    stride_c_batch, stride_cm, stride_cn,
    tiles_n,
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
):
    """
    Batched EXL3 dequant + GEMM (non-split-K).

    Grid: (cdiv(M, BLOCK_M), N // 16, num_outputs)
    Each program computes a (BLOCK_M, 16) output tile for one batch element.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Offset pointers by batch
    A_batch = A_ptr + pid_batch * stride_a_batch
    B_batch = B_ptr + pid_batch * stride_b_batch
    C_batch = C_ptr + pid_batch * stride_c_batch

    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local

    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M

    num_k_tiles = K // 16
    for tk in range(num_k_tiles):
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE
        lo = tl.load(B_batch + b_base + word_idx)
        hi = tl.load(B_batch + b_base + next_word_idx)
        w = _dequant_tile(lo, hi, shift, shift_hi, CB)
        acc += tl.dot(a_tile, w)

    offs_n = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = C_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m)


@triton.jit
def _exl3_mgemm_splitk_kernel(
    A_ptr, B_ptr, C_partial_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    M, N, K,
    stride_a_batch, stride_am, stride_ak,
    stride_b_batch,
    stride_cp_batch, stride_cp_split, stride_cp_m, stride_cp_n,
    tiles_n,
    tiles_per_split,
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Batched split-K EXL3 dequant + GEMM.

    Grid: (cdiv(M, BLOCK_M), N // 16, num_outputs * split_k)
    pid(2) encodes batch and split-K indices.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_2 = tl.program_id(2)
    pid_batch = pid_2 // SPLIT_K
    pid_k = pid_2 % SPLIT_K

    A_batch = A_ptr + pid_batch * stride_a_batch
    B_batch = B_ptr + pid_batch * stride_b_batch
    C_batch = C_partial_ptr + pid_batch * stride_cp_batch

    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local

    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < M

    tk_start = pid_k * tiles_per_split
    tk_end = tl.minimum(tk_start + tiles_per_split, K // 16)

    for tk in range(tk_start, tk_end):
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE
        lo = tl.load(B_batch + b_base + word_idx)
        hi = tl.load(B_batch + b_base + next_word_idx)
        w = _dequant_tile(lo, hi, shift, shift_hi, CB)
        acc += tl.dot(a_tile, w)

    offs_n = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = (C_batch
              + pid_k * stride_cp_split
              + offs_m[:, None] * stride_cp_m
              + offs_n[None, :] * stride_cp_n)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m)


# =============================================================================
# Multi-GEMM partial buffer cache
# =============================================================================

def _get_mgemm_splitk_buf(num_outputs, split_k, M, N, device):
    """Allocate per-call partial buffer for batched multi-GEMM split-K reduction.

    Same race-condition fix as _get_splitk_buf: previously this was a cached
    buffer keyed by (device, num_outputs, split_k), but concurrent async CUDA
    kernels would corrupt each other's partials, producing garbled output.
    torch's caching allocator makes per-call allocation essentially free.
    """
    needed_shape = (num_outputs, split_k, max(M, 1), max(N, 1))
    return torch.empty(needed_shape, dtype=torch.float16, device=device)


# =============================================================================
# Multi-GEMM Python wrapper
# =============================================================================

def exl3_multi_gemm(
    A_batched: torch.Tensor,
    B_stacked: torch.Tensor,
    num_outputs: int,
    bits: int = 4,
    cb: int = 0,
    out: torch.Tensor = None,
    split_k: int = 0,
) -> torch.Tensor:
    """
    Batched EXL3 fused dequant + GEMM for multiple outputs.

    Args:
        A_batched: Input activations, float16, shape (num_outputs, M, K).
        B_stacked: Stacked packed weights, int32, shape (num_outputs, tiles_k, tiles_n, wpt).
        num_outputs: Number of output matrices.
        bits: Bits per weight (1-8).
        cb: Codebook variant (0=default, 1=MCG, 2=MUL1).
        out: Optional pre-allocated output, float16, shape (num_outputs, M, N).
        split_k: Split-K factor. 0 = auto-select.

    Returns:
        C: Output, float16, shape (num_outputs, M, N)
    """
    assert A_batched.dtype == torch.float16
    assert A_batched.ndim == 3 and A_batched.shape[0] == num_outputs
    assert B_stacked.ndim == 4 and B_stacked.shape[0] == num_outputs

    M, K = A_batched.shape[1], A_batched.shape[2]
    tiles_k, tiles_n = B_stacked.shape[1], B_stacked.shape[2]
    N = tiles_n * 16

    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, A_batched.device)
    WORDS_PER_TILE = 8 * bits

    # Auto-select kernel config (reuse existing cache — same per-output shape)
    if split_k == 0:
        if _AUTOTUNE_DISABLED:
            config = _heuristic_config(M, tiles_k, tiles_n)
        else:
            device_name = torch.cuda.get_device_name(A_batched.device)
            cache_key = (M, K, N, bits, cb, device_name)
            cached = _tune_cache.get(cache_key)
            if cached is not None:
                config = cached
            else:
                config = autotune_splitk(M, K, N, bits, cb, A_batched.device)
        split_k, num_warps, num_stages, BLOCK_M = config
    else:
        num_warps, num_stages, BLOCK_M = 2, 2, 16

    if split_k > 1:
        tiles_per_split = triton.cdiv(tiles_k, split_k)
        partial_buf = _get_mgemm_splitk_buf(num_outputs, split_k, M, N, A_batched.device)
        C_partial = partial_buf[:num_outputs, :split_k, :M, :N]

        grid = (triton.cdiv(M, BLOCK_M), tiles_n, num_outputs * split_k)
        _exl3_mgemm_splitk_kernel[grid](
            A_batched, B_stacked, C_partial,
            word_idx_t, next_word_idx_t, shift_t,
            M, N, K,
            A_batched.stride(0), A_batched.stride(1), A_batched.stride(2),
            B_stacked.stride(0),
            C_partial.stride(0), C_partial.stride(1), C_partial.stride(2), C_partial.stride(3),
            tiles_n, tiles_per_split,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            SPLIT_K=split_k,
            num_warps=num_warps, num_stages=num_stages,
        )

        if out is not None:
            torch.sum(C_partial, dim=1, out=out)
            return out
        else:
            return C_partial.sum(dim=1)
    else:
        if out is not None:
            C = out
        else:
            C = torch.empty((num_outputs, M, N), dtype=torch.float16, device=A_batched.device)

        grid = (triton.cdiv(M, BLOCK_M), tiles_n, num_outputs)
        _exl3_mgemm_kernel[grid](
            A_batched, B_stacked, C,
            word_idx_t, next_word_idx_t, shift_t,
            M, N, K,
            A_batched.stride(0), A_batched.stride(1), A_batched.stride(2),
            B_stacked.stride(0),
            C.stride(0), C.stride(1), C.stride(2),
            tiles_n,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            num_warps=num_warps, num_stages=num_stages,
        )
        return C


# =============================================================================
# Fused Multi-Expert MoE GEMM Kernel
#
# Processes ALL experts in ONE kernel launch using token indirection and
# expert-indexed weight lookup. Reduces MoE GEMM launches from
# ~150 (3 projections × ~50 active experts) to 3 (gate + up + down).
#
# Based on _exl3_gemm_kernel with two additions:
#   1. Expert indexing: expert_ids[pid_m] offsets into B[E, tiles_k, tiles_n, wpt]
#   2. Token indirection: sorted_token_ids[pid_m * BLOCK_M + ...] for A indexing
# =============================================================================

@triton.jit
def _exl3_fused_moe_gemm_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    sorted_token_ids_ptr,    # (EM,) int32 — sorted token indices into A
    expert_ids_ptr,          # (num_m_blocks,) int32 — expert ID per M-block
    num_tokens_post_padded_ptr,  # (1,) int32 — actual valid EM (tensor for graph capture)
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    # Dimensions
    M,                       # original number of tokens (for A bounds check)
    N, K,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    stride_be,               # B expert stride: B_ptr + eid * stride_be
    # Tile layout
    tiles_n,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
):
    """
    Fused multi-expert EXL3 dequant + GEMM.

    Grid: (num_m_blocks, tiles_n)
    Each program computes a (BLOCK_M, 16) output tile for one expert's tokens.

    B layout: (E, tiles_k, tiles_n, WORDS_PER_TILE) int32 — stacked expert weights.
    A layout: (total_tokens, K) float16 — flat token buffer.
    C layout: (EM, N) float16 — output indexed by sorted position.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Skip padding blocks beyond valid tokens (loaded from tensor for graph capture)
    num_valid_m = tl.load(num_tokens_post_padded_ptr)
    num_valid_m_blocks = (num_valid_m + BLOCK_M - 1) // BLOCK_M
    if pid_m >= num_valid_m_blocks:
        return

    # Load expert ID for this M-block
    off_expert = tl.load(expert_ids_ptr + pid_m)

    # Skip invalid expert blocks (EP: expert not on this rank)
    if off_expert < 0:
        return

    # Accumulator: (BLOCK_M, 16) in float32
    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    # Load precomputed bit extraction tables
    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local

    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    # Load sorted token IDs for this M-block (token indirection)
    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_block)

    # Expert-offset B pointer
    B_expert = B_ptr + off_expert.to(tl.int64) * stride_be

    # K-dimension loop
    num_k_tiles = K // 16
    for tk in range(num_k_tiles):
        # Load A tile via token indirection: A[token_ids, tk*16:(tk+1)*16]
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + token_ids[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
        mask_m = token_ids[:, None] < M
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        # Load and dequant B tile for this expert
        b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE
        lo = tl.load(B_expert + b_base + word_idx)
        hi = tl.load(B_expert + b_base + next_word_idx)
        w = _dequant_tile(lo, hi, shift, shift_hi, CB)

        acc += tl.dot(a_tile, w)

    # Store C tile: C[pid_m * BLOCK_M + ..., pid_n * 16 + ...]
    offs_n = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = C_ptr + offs_block[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
    mask_store = token_ids[:, None] < M
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_store)


@triton.jit
def _exl3_fused_moe_gemm_splitk_kernel(
    # Pointers
    A_ptr, B_ptr, C_partial_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,  # (1,) int32 — actual valid EM (tensor for graph capture)
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_cp_split, stride_cp_m, stride_cp_n,
    stride_be,
    # Tile layout
    tiles_n,
    tiles_per_split,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Split-K variant of fused multi-expert MoE GEMM.

    Grid: (num_m_blocks, tiles_n, split_k)
    Writes partial results to C_partial[split, EM, N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Skip padding blocks beyond valid tokens (loaded from tensor for graph capture)
    num_valid_m = tl.load(num_tokens_post_padded_ptr)
    num_valid_m_blocks = (num_valid_m + BLOCK_M - 1) // BLOCK_M
    if pid_m >= num_valid_m_blocks:
        return

    off_expert = tl.load(expert_ids_ptr + pid_m)

    # Skip invalid expert blocks (EP: expert not on this rank)
    if off_expert < 0:
        return

    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local

    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_block)

    B_expert = B_ptr + off_expert.to(tl.int64) * stride_be

    tk_start = pid_k * tiles_per_split
    tk_end = tl.minimum(tk_start + tiles_per_split, K // 16)

    for tk in range(tk_start, tk_end):
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + token_ids[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
        mask_m = token_ids[:, None] < M
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE
        lo = tl.load(B_expert + b_base + word_idx)
        hi = tl.load(B_expert + b_base + next_word_idx)
        w = _dequant_tile(lo, hi, shift, shift_hi, CB)

        acc += tl.dot(a_tile, w)

    offs_n = pid_n * 16 + tl.arange(0, 16)
    c_ptrs = (C_partial_ptr
              + pid_k * stride_cp_split
              + offs_block[:, None].to(tl.int64) * stride_cp_m
              + offs_n[None, :] * stride_cp_n)
    mask_store = token_ids[:, None] < M
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_store)


# =============================================================================
# Fused MoE GEMM + Hadamard-128 epilogue (non-split-K)
#
# Combines MoE routing (token indirection, expert B stride, EP skip) with
# the 8-accumulator Had-128 epilogue (H_16 matmul + H_8 butterfly + SVH).
# SVH is per-expert indexed: svh_ptr + off_expert * stride_svh_e + col_offset.
#
# Saves one kernel launch per projection vs separate GEMM + batched_had_r_128.
# =============================================================================

@triton.jit
def _exl3_fused_moe_gemm_had128_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    H16_ptr,           # (16, 16) fp16 normalized Hadamard
    svh_ptr,           # (E, N) fp16 per-expert sign-flips
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_cm, stride_cn,
    stride_be,          # B expert stride
    stride_svh_e,       # SVH expert stride (N elements)
    stride_h16_r, stride_h16_c,
    # Tile layout
    tiles_n,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
    HAS_SVH: tl.constexpr,
):
    """
    Fused multi-expert EXL3 dequant + GEMM + Hadamard-128 epilogue.

    Grid: (num_m_blocks, N // 128)
    Each program computes a (BLOCK_M, 128) output tile with fused H_128.
    """
    pid_m = tl.program_id(0)
    pid_n128 = tl.program_id(1)

    # Skip padding blocks beyond valid tokens
    num_valid_m = tl.load(num_tokens_post_padded_ptr)
    num_valid_m_blocks = (num_valid_m + BLOCK_M - 1) // BLOCK_M
    if pid_m >= num_valid_m_blocks:
        return

    # Load expert ID for this M-block
    off_expert = tl.load(expert_ids_ptr + pid_m)

    # Skip invalid expert blocks (EP: expert not on this rank)
    if off_expert < 0:
        return

    # Load bit extraction tables
    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local
    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    # Load sorted token IDs for this M-block
    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_block)

    # Expert-offset B pointer
    B_expert = B_ptr + off_expert.to(tl.int64) * stride_be

    # 8 accumulators, one per 16-col group within the 128-col block
    acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    # Base N-tile index for this 128-col block
    base_n_tile = pid_n128 * 8

    # K-loop
    num_k_tiles = K // 16
    for tk in range(num_k_tiles):
        # Load A tile via token indirection
        offs_k = tk * 16 + tl.arange(0, 16)
        a_ptrs = A_ptr + token_ids[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
        mask_m = token_ids[:, None] < M
        a_tile = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float16)

        # Dequant and accumulate for each of the 8 B tile columns
        for g in tl.static_range(8):
            b_col = base_n_tile + g
            b_base = (tk * tiles_n + b_col) * WORDS_PER_TILE
            lo = tl.load(B_expert + b_base + word_idx)
            hi = tl.load(B_expert + b_base + next_word_idx)
            w = _dequant_tile(lo, hi, shift, shift_hi, CB)

            if g == 0:
                acc0 += tl.dot(a_tile, w)
            elif g == 1:
                acc1 += tl.dot(a_tile, w)
            elif g == 2:
                acc2 += tl.dot(a_tile, w)
            elif g == 3:
                acc3 += tl.dot(a_tile, w)
            elif g == 4:
                acc4 += tl.dot(a_tile, w)
            elif g == 5:
                acc5 += tl.dot(a_tile, w)
            elif g == 6:
                acc6 += tl.dot(a_tile, w)
            elif g == 7:
                acc7 += tl.dot(a_tile, w)

    # --- Epilogue: H_128 = H_8 ⊗ H_16 ---

    # Load H_16 matrix (16x16, fp16)
    h16_r = tl.arange(0, 16)[:, None]
    h16_c = tl.arange(0, 16)[None, :]
    H16 = tl.load(H16_ptr + h16_r * stride_h16_r + h16_c * stride_h16_c)

    # Step 1: H_16 within each group
    acc0 = tl.dot(acc0.to(tl.float16), H16).to(tl.float32)
    acc1 = tl.dot(acc1.to(tl.float16), H16).to(tl.float32)
    acc2 = tl.dot(acc2.to(tl.float16), H16).to(tl.float32)
    acc3 = tl.dot(acc3.to(tl.float16), H16).to(tl.float32)
    acc4 = tl.dot(acc4.to(tl.float16), H16).to(tl.float32)
    acc5 = tl.dot(acc5.to(tl.float16), H16).to(tl.float32)
    acc6 = tl.dot(acc6.to(tl.float16), H16).to(tl.float32)
    acc7 = tl.dot(acc7.to(tl.float16), H16).to(tl.float32)

    # Step 2: H_8 butterfly across 8 groups (3 rounds, in fp32)
    t0 = acc0 + acc1
    t1 = acc0 - acc1
    t2 = acc2 + acc3
    t3 = acc2 - acc3
    t4 = acc4 + acc5
    t5 = acc4 - acc5
    t6 = acc6 + acc7
    t7 = acc6 - acc7

    s0 = t0 + t2
    s1 = t1 + t3
    s2 = t0 - t2
    s3 = t1 - t3
    s4 = t4 + t6
    s5 = t5 + t7
    s6 = t4 - t6
    s7 = t5 - t7

    acc0 = s0 + s4
    acc1 = s1 + s5
    acc2 = s2 + s6
    acc3 = s3 + s7
    acc4 = s0 - s4
    acc5 = s1 - s5
    acc6 = s2 - s6
    acc7 = s3 - s7

    # Scale: 1/sqrt(8) for H_8 (H_16 already has 1/sqrt(16) baked in)
    inv_sqrt8: tl.constexpr = 0.35355339059327373
    acc0 = acc0 * inv_sqrt8
    acc1 = acc1 * inv_sqrt8
    acc2 = acc2 * inv_sqrt8
    acc3 = acc3 * inv_sqrt8
    acc4 = acc4 * inv_sqrt8
    acc5 = acc5 * inv_sqrt8
    acc6 = acc6 * inv_sqrt8
    acc7 = acc7 * inv_sqrt8

    # Step 3: per-expert SVH sign-flip
    if HAS_SVH:
        svh_base = svh_ptr + off_expert.to(tl.int64) * stride_svh_e + pid_n128 * 128
        svh_offs = tl.arange(0, 16)[None, :]
        svh0 = tl.load(svh_base + 0 * 16 + svh_offs).to(tl.float32)
        svh1 = tl.load(svh_base + 1 * 16 + svh_offs).to(tl.float32)
        svh2 = tl.load(svh_base + 2 * 16 + svh_offs).to(tl.float32)
        svh3 = tl.load(svh_base + 3 * 16 + svh_offs).to(tl.float32)
        svh4 = tl.load(svh_base + 4 * 16 + svh_offs).to(tl.float32)
        svh5 = tl.load(svh_base + 5 * 16 + svh_offs).to(tl.float32)
        svh6 = tl.load(svh_base + 6 * 16 + svh_offs).to(tl.float32)
        svh7 = tl.load(svh_base + 7 * 16 + svh_offs).to(tl.float32)
        acc0 = acc0 * svh0
        acc1 = acc1 * svh1
        acc2 = acc2 * svh2
        acc3 = acc3 * svh3
        acc4 = acc4 * svh4
        acc5 = acc5 * svh5
        acc6 = acc6 * svh6
        acc7 = acc7 * svh7

    # Store all 8 groups: C[offs_block, ...] (sorted position)
    mask_store = token_ids[:, None] < M
    for g in tl.static_range(8):
        offs_n = pid_n128 * 128 + g * 16 + tl.arange(0, 16)
        c_ptrs = C_ptr + offs_block[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
        if g == 0:
            tl.store(c_ptrs, acc0.to(tl.float16), mask=mask_store)
        elif g == 1:
            tl.store(c_ptrs, acc1.to(tl.float16), mask=mask_store)
        elif g == 2:
            tl.store(c_ptrs, acc2.to(tl.float16), mask=mask_store)
        elif g == 3:
            tl.store(c_ptrs, acc3.to(tl.float16), mask=mask_store)
        elif g == 4:
            tl.store(c_ptrs, acc4.to(tl.float16), mask=mask_store)
        elif g == 5:
            tl.store(c_ptrs, acc5.to(tl.float16), mask=mask_store)
        elif g == 6:
            tl.store(c_ptrs, acc6.to(tl.float16), mask=mask_store)
        elif g == 7:
            tl.store(c_ptrs, acc7.to(tl.float16), mask=mask_store)


# =============================================================================
# Fused MoE split-K reduce + Hadamard-128 kernel
#
# Sums SPLIT_K partial results from _exl3_fused_moe_gemm_splitk_kernel,
# then applies H_128 epilogue + per-expert SVH sign-flip.
# =============================================================================

@triton.jit
def _moe_reduce_had128_kernel(
    C_partial_ptr, C_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    H16_ptr, svh_ptr,
    # Dimensions
    M, N, EM_max,
    # Strides
    stride_cp_split, stride_cp_m, stride_cp_n,
    stride_cm, stride_cn,
    stride_svh_e,
    stride_h16_r, stride_h16_c,
    # Compile-time constants
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_SVH: tl.constexpr,
):
    """
    Fused reduce + H_128 + per-expert SVH for MoE split-K partial results.

    Grid: (num_m_blocks, N // 128)
    """
    pid_m = tl.program_id(0)
    pid_n128 = tl.program_id(1)

    # Skip padding blocks beyond valid tokens
    num_valid_m = tl.load(num_tokens_post_padded_ptr)
    num_valid_m_blocks = (num_valid_m + BLOCK_M - 1) // BLOCK_M
    if pid_m >= num_valid_m_blocks:
        return

    off_expert = tl.load(expert_ids_ptr + pid_m)
    if off_expert < 0:
        return

    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_token_ids_ptr + offs_block)

    # Sum partials for each of the 8 groups
    acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    for sk in range(SPLIT_K):
        for g in tl.static_range(8):
            offs_n = pid_n128 * 128 + g * 16 + tl.arange(0, 16)
            cp_ptrs = (C_partial_ptr
                       + sk * stride_cp_split
                       + offs_block[:, None].to(tl.int64) * stride_cp_m
                       + offs_n[None, :] * stride_cp_n)
            mask_load = token_ids[:, None] < M
            partial = tl.load(cp_ptrs, mask=mask_load, other=0.0).to(tl.float32)

            if g == 0:
                acc0 += partial
            elif g == 1:
                acc1 += partial
            elif g == 2:
                acc2 += partial
            elif g == 3:
                acc3 += partial
            elif g == 4:
                acc4 += partial
            elif g == 5:
                acc5 += partial
            elif g == 6:
                acc6 += partial
            elif g == 7:
                acc7 += partial

    # --- H_128 epilogue ---

    h16_r = tl.arange(0, 16)[:, None]
    h16_c = tl.arange(0, 16)[None, :]
    H16 = tl.load(H16_ptr + h16_r * stride_h16_r + h16_c * stride_h16_c)

    # Step 1: H_16 per group
    acc0 = tl.dot(acc0.to(tl.float16), H16).to(tl.float32)
    acc1 = tl.dot(acc1.to(tl.float16), H16).to(tl.float32)
    acc2 = tl.dot(acc2.to(tl.float16), H16).to(tl.float32)
    acc3 = tl.dot(acc3.to(tl.float16), H16).to(tl.float32)
    acc4 = tl.dot(acc4.to(tl.float16), H16).to(tl.float32)
    acc5 = tl.dot(acc5.to(tl.float16), H16).to(tl.float32)
    acc6 = tl.dot(acc6.to(tl.float16), H16).to(tl.float32)
    acc7 = tl.dot(acc7.to(tl.float16), H16).to(tl.float32)

    # Step 2: H_8 butterfly
    t0 = acc0 + acc1
    t1 = acc0 - acc1
    t2 = acc2 + acc3
    t3 = acc2 - acc3
    t4 = acc4 + acc5
    t5 = acc4 - acc5
    t6 = acc6 + acc7
    t7 = acc6 - acc7

    s0 = t0 + t2
    s1 = t1 + t3
    s2 = t0 - t2
    s3 = t1 - t3
    s4 = t4 + t6
    s5 = t5 + t7
    s6 = t4 - t6
    s7 = t5 - t7

    acc0 = s0 + s4
    acc1 = s1 + s5
    acc2 = s2 + s6
    acc3 = s3 + s7
    acc4 = s0 - s4
    acc5 = s1 - s5
    acc6 = s2 - s6
    acc7 = s3 - s7

    inv_sqrt8: tl.constexpr = 0.35355339059327373
    acc0 = acc0 * inv_sqrt8
    acc1 = acc1 * inv_sqrt8
    acc2 = acc2 * inv_sqrt8
    acc3 = acc3 * inv_sqrt8
    acc4 = acc4 * inv_sqrt8
    acc5 = acc5 * inv_sqrt8
    acc6 = acc6 * inv_sqrt8
    acc7 = acc7 * inv_sqrt8

    # Step 3: per-expert SVH sign-flip
    if HAS_SVH:
        svh_base = svh_ptr + off_expert.to(tl.int64) * stride_svh_e + pid_n128 * 128
        svh_offs = tl.arange(0, 16)[None, :]
        svh0 = tl.load(svh_base + 0 * 16 + svh_offs).to(tl.float32)
        svh1 = tl.load(svh_base + 1 * 16 + svh_offs).to(tl.float32)
        svh2 = tl.load(svh_base + 2 * 16 + svh_offs).to(tl.float32)
        svh3 = tl.load(svh_base + 3 * 16 + svh_offs).to(tl.float32)
        svh4 = tl.load(svh_base + 4 * 16 + svh_offs).to(tl.float32)
        svh5 = tl.load(svh_base + 5 * 16 + svh_offs).to(tl.float32)
        svh6 = tl.load(svh_base + 6 * 16 + svh_offs).to(tl.float32)
        svh7 = tl.load(svh_base + 7 * 16 + svh_offs).to(tl.float32)
        acc0 = acc0 * svh0
        acc1 = acc1 * svh1
        acc2 = acc2 * svh2
        acc3 = acc3 * svh3
        acc4 = acc4 * svh4
        acc5 = acc5 * svh5
        acc6 = acc6 * svh6
        acc7 = acc7 * svh7

    # Store
    mask_store = token_ids[:, None] < M
    for g in tl.static_range(8):
        offs_n = pid_n128 * 128 + g * 16 + tl.arange(0, 16)
        c_ptrs = C_ptr + offs_block[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
        if g == 0:
            tl.store(c_ptrs, acc0.to(tl.float16), mask=mask_store)
        elif g == 1:
            tl.store(c_ptrs, acc1.to(tl.float16), mask=mask_store)
        elif g == 2:
            tl.store(c_ptrs, acc2.to(tl.float16), mask=mask_store)
        elif g == 3:
            tl.store(c_ptrs, acc3.to(tl.float16), mask=mask_store)
        elif g == 4:
            tl.store(c_ptrs, acc4.to(tl.float16), mask=mask_store)
        elif g == 5:
            tl.store(c_ptrs, acc5.to(tl.float16), mask=mask_store)
        elif g == 6:
            tl.store(c_ptrs, acc6.to(tl.float16), mask=mask_store)
        elif g == 7:
            tl.store(c_ptrs, acc7.to(tl.float16), mask=mask_store)


# =============================================================================
# Trellis → FP16 bulk dequant kernel (no GEMM, just decode packed bits)
#
# For prefill: dequant expert weights to FP16 scratch buffer, then use
# bandwidth-bound FP16 GEMM (rocBLAS / HIP) instead of ALU-bound trellis.
# One kernel launch per expert, processes all tiles in parallel.
# =============================================================================

@triton.jit
def _exl3_trellis_to_fp16_kernel(
    B_ptr, Out_ptr,
    word_idx_ptr, next_word_idx_ptr, shift_ptr,
    tiles_k, tiles_n,
    stride_bk, stride_bn, stride_bw,
    stride_ok, stride_on,
    WORDS_PER_TILE: tl.constexpr,
    CB: tl.constexpr,
):
    """Dequant one expert's trellis to (K, N) fp16.

    Grid: (tiles_k, tiles_n)
    Each program decodes one 16×16 B tile and writes to Out.
    """
    tk = tl.program_id(0)
    tn = tl.program_id(1)

    # Load bit extraction tables
    k_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 16)[None, :]
    mat_pos = k_local * 16 + n_local

    word_idx = tl.load(word_idx_ptr + mat_pos)
    next_word_idx = tl.load(next_word_idx_ptr + mat_pos)
    shift = tl.load(shift_ptr + mat_pos)
    shift_hi = (32 - shift) & 31

    # Load packed B words for this tile
    b_base = B_ptr + tk * stride_bk + tn * stride_bn
    lo = tl.load(b_base + word_idx * stride_bw)
    hi = tl.load(b_base + next_word_idx * stride_bw)

    # Dequant
    w = _dequant_tile(lo, hi, shift, shift_hi, CB)

    # Store to output (K, N) layout
    offs_k = tk * 16 + k_local
    offs_n = tn * 16 + n_local
    out_ptrs = Out_ptr + offs_k * stride_ok + offs_n * stride_on
    tl.store(out_ptrs, w)


def exl3_trellis_to_fp16(
    B_stacked: torch.Tensor,
    bits: int,
    cb: int = 0,
    B_i32: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bulk dequant stacked expert trellis weights to (E, K, N) FP16.

    Fast parallel decode: one kernel launch per expert, all tiles in parallel.
    Output is raw dequanted weights in Hadamard-rotated space (no Had baked in).

    Args:
        B_stacked: (E, tiles_k, tiles_n, wpt) int16 — packed trellis weights.
        bits: Quantization bits (1-8).
        cb: Codebook variant (0, 1, 2).
        B_i32: Optional pre-computed int32 view of B_stacked.

    Returns:
        (E, K, N) float16 — raw dequanted weights per expert.
    """
    if B_i32 is None:
        B_i32 = B_stacked.view(torch.int32)

    E, tiles_k, tiles_n = B_stacked.shape[0], B_stacked.shape[1], B_stacked.shape[2]
    K = tiles_k * 16
    N = tiles_n * 16
    WORDS_PER_TILE = 8 * bits

    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, B_stacked.device)

    out = torch.empty(E, K, N, dtype=torch.float16, device=B_stacked.device)

    for e in range(E):
        B_expert = B_i32[e]  # (tiles_k, tiles_n, wpt//2)
        grid = (tiles_k, tiles_n)
        _exl3_trellis_to_fp16_kernel[grid](
            B_expert, out[e],
            word_idx_t, next_word_idx_t, shift_t,
            tiles_k, tiles_n,
            B_expert.stride(0), B_expert.stride(1), B_expert.stride(2),
            out[e].stride(0), out[e].stride(1),
            WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            num_warps=1, num_stages=1,
        )

    return out


# =============================================================================
# Fused MoE GEMM Python wrapper
# =============================================================================

# Per-device cache for fused MoE split-K buffers
_moe_splitk_buf_cache = {}


def _get_moe_splitk_buf(split_k, EM, N, device):
    """Get or allocate cached partial buffer for fused MoE split-K."""
    key = (device, split_k)
    if key not in _moe_splitk_buf_cache or \
       _moe_splitk_buf_cache[key].shape[1] < EM or \
       _moe_splitk_buf_cache[key].shape[2] < N:
        _moe_splitk_buf_cache[key] = torch.empty(
            (split_k, max(EM, 1), max(N, 1)),
            dtype=torch.float16, device=device)
    return _moe_splitk_buf_cache[key]


def exl3_fused_moe_gemm(
    A: torch.Tensor,
    B_stacked: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: "int | torch.Tensor",
    EM_max: int = 0,
    bits: int = 4,
    cb: int = 0,
    num_valid_tokens: int = None,
    split_k: int = 0,
    B_i32: torch.Tensor = None,
    block_m: int = 16,
) -> torch.Tensor:
    """
    Fused multi-expert EXL3 dequant + GEMM for MoE layers.

    Processes all experts in ONE kernel launch using sorted token dispatch.

    Args:
        A: Input activations, float16, shape (M, K). Pre-Hadamard-transformed.
        B_stacked: Stacked expert weights, int16, shape (E, tiles_k, tiles_n, wpt).
        sorted_token_ids: (EM,) int32 — token indices sorted by expert.
            Padded to align to BLOCK_M. Padding entries have id >= M.
        expert_ids: (num_m_blocks,) int32 — expert ID for each M-block.
        num_tokens_post_padded: Total entries in sorted_token_ids (EM).
            Can be int (eager) or 1-element tensor (graph capture safe).
        EM_max: Fixed upper bound for output tensor rows and grid dims.
            When 0 (default), derived from num_tokens_post_padded (int path).
            When >0, used for all sizing (graph capture safe — no .item()).
        bits: Bits per weight (1-8).
        cb: Codebook variant (0, 1, 2).
        num_valid_tokens: Number of real tokens (for bounds). Defaults to M.
        split_k: Split-K factor. 0 = auto-select.
        B_i32: Optional pre-computed int32 view of B_stacked.
        block_m: M-dimension tile size (16 for decode, 64 for prefill).
            Larger values amortize trellis dequant across more rows.

    Returns:
        C: Output, float16, shape (EM_max, N) or (EM, N).
    """
    assert A.dtype == torch.float16
    assert A.is_contiguous()
    assert 1 <= bits <= 8
    assert cb in (0, 1, 2)

    M, K = A.shape
    E = B_stacked.shape[0]
    tiles_k, tiles_n = B_stacked.shape[1], B_stacked.shape[2]
    N = tiles_n * 16
    BLOCK_M = block_m
    WORDS_PER_TILE = 8 * bits

    if num_valid_tokens is None:
        num_valid_tokens = M

    # Support both int and tensor for num_tokens_post_padded.
    # When EM_max is provided, use it for all Python-level sizing (graph safe).
    # The tensor is passed to the kernel for runtime bounds checking.
    if isinstance(num_tokens_post_padded, torch.Tensor):
        num_post_pad_tensor = num_tokens_post_padded.to(torch.int32)
        if EM_max <= 0:
            # Fallback: extract int (breaks graph capture, but backward compat)
            EM_max = num_tokens_post_padded.item()
    else:
        # Scalar path: create 1-element tensor for kernel
        if EM_max <= 0:
            EM_max = num_tokens_post_padded
        num_post_pad_tensor = torch.tensor(
            [num_tokens_post_padded], dtype=torch.int32, device=A.device)

    EM = EM_max
    num_m_blocks = triton.cdiv(EM, BLOCK_M)

    if B_i32 is None:
        B_i32 = B_stacked.view(torch.int32)

    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, A.device)

    # Compute stride_be: number of int32 elements per expert in B
    stride_be = B_i32.stride(0)

    # Auto-select split-K config
    if split_k == 0:
        if _AUTOTUNE_DISABLED:
            config = _heuristic_config(1, tiles_k, tiles_n)
        else:
            device_name = torch.cuda.get_device_name(A.device)
            cache_key = (1, K, N, bits, cb, device_name)
            cached = _tune_cache.get(cache_key)
            if cached is not None:
                config = cached
            else:
                config = autotune_splitk(1, K, N, bits, cb, A.device)
        split_k, num_warps, num_stages, _ = config
    else:
        num_warps, num_stages = 2, 2

    C = torch.zeros(EM, N, dtype=torch.float16, device=A.device)

    if split_k > 1:
        tiles_per_split = triton.cdiv(tiles_k, split_k)
        partial_buf = _get_moe_splitk_buf(split_k, EM, N, A.device)
        C_partial = partial_buf[:split_k, :EM, :N]
        C_partial.zero_()  # Must zero: kernel skips remote expert blocks

        grid = (num_m_blocks, tiles_n, split_k)
        _exl3_fused_moe_gemm_splitk_kernel[grid](
            A, B_i32, C_partial,
            sorted_token_ids, expert_ids,
            num_post_pad_tensor,
            word_idx_t, next_word_idx_t, shift_t,
            num_valid_tokens, N, K,
            A.stride(0), A.stride(1),
            C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
            stride_be,
            tiles_n, tiles_per_split,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            SPLIT_K=split_k,
            num_warps=num_warps, num_stages=num_stages,
        )

        torch.sum(C_partial, dim=0, out=C)

    else:
        grid = (num_m_blocks, tiles_n)
        _exl3_fused_moe_gemm_kernel[grid](
            A, B_i32, C,
            sorted_token_ids, expert_ids,
            num_post_pad_tensor,
            word_idx_t, next_word_idx_t, shift_t,
            num_valid_tokens, N, K,
            A.stride(0), A.stride(1),
            C.stride(0), C.stride(1),
            stride_be,
            tiles_n,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            num_warps=num_warps, num_stages=num_stages,
        )

    return C


# =============================================================================
# Fused MoE GEMM + Hadamard-128 Python wrapper
# =============================================================================

# Auto-tune cache for fused MoE Had kernels
_moe_had_tune_cache = {}

# Disk cache path
_MOE_HAD_TUNE_CACHE_FILE = os.environ.get(
    "EXL3_MOE_HAD_CACHE",
    os.path.join(str(Path.home()), ".cache", "exl3_triton", "moe_had_cache.json"),
)


def _load_moe_had_disk_cache():
    """Load fused MoE Had tune cache from disk."""
    try:
        with open(_MOE_HAD_TUNE_CACHE_FILE, "r") as f:
            disk_data = json.load(f)
        for key_str, val in disk_data.items():
            parts = key_str.split("|")
            if len(parts) == 6:
                device_name, M, K, N, bits, cb = parts
                cache_key = (int(M), int(K), int(N), int(bits), int(cb), device_name)
                _moe_had_tune_cache[cache_key] = _normalize_config(val)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass


def _save_moe_had_to_disk_cache(cache_key, config):
    """Save a single entry to the fused MoE Had disk cache."""
    M, K, N, bits, cb, device_name = cache_key
    key_str = f"{device_name}|{M}|{K}|{N}|{bits}|{cb}"

    disk_data = {}
    try:
        with open(_MOE_HAD_TUNE_CACHE_FILE, "r") as f:
            disk_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    disk_data[key_str] = list(config)

    os.makedirs(os.path.dirname(_MOE_HAD_TUNE_CACHE_FILE), exist_ok=True)
    with open(_MOE_HAD_TUNE_CACHE_FILE, "w") as f:
        json.dump(disk_data, f, indent=2)


# Load eagerly at import time
_load_moe_had_disk_cache()


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
    num_valid_tokens: int = None,
    split_k: int = 0,
    B_i32: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused multi-expert EXL3 dequant + GEMM + output Hadamard-128 + per-expert SVH.

    Equivalent to: batched_had_r_128(exl3_fused_moe_gemm(...), svh, eid, pre=False)
    but fused into fewer kernel launches (saves 1 launch per projection).

    Args:
        A: Input activations, float16, shape (M, K). Pre-Hadamard-transformed.
        B_stacked: Stacked expert weights, int16, shape (E, tiles_k, tiles_n, wpt).
        sorted_token_ids: (EM,) int32 — token indices sorted by expert.
        expert_ids: (num_m_blocks,) int32 — expert ID per M-block.
        num_tokens_post_padded: Total entries in sorted_token_ids (EM).
        svh_stacked: (E, N) fp16 — per-expert SVH sign-flip vectors.
        EM_max: Fixed upper bound for output tensor rows and grid dims.
        bits: Bits per weight (1-8).
        cb: Codebook variant (0, 1, 2).
        num_valid_tokens: Number of real tokens (for bounds). Defaults to M.
        split_k: Split-K factor. 0 = auto-select.
        B_i32: Optional pre-computed int32 view of B_stacked.

    Returns:
        C: Output, float16, shape (EM_max, N) or (EM, N).
    """
    assert A.dtype == torch.float16
    assert A.is_contiguous()
    assert 1 <= bits <= 8
    assert cb in (0, 1, 2)

    M, K = A.shape
    E = B_stacked.shape[0]
    tiles_k, tiles_n = B_stacked.shape[1], B_stacked.shape[2]
    N = tiles_n * 16
    BLOCK_M = 16
    WORDS_PER_TILE = 8 * bits

    if num_valid_tokens is None:
        num_valid_tokens = M

    # Fallback: if N not divisible by 128, use separate GEMM + Had
    if N % 128 != 0:
        C = exl3_fused_moe_gemm(
            A, B_stacked, sorted_token_ids, expert_ids,
            num_tokens_post_padded, EM_max=EM_max, bits=bits, cb=cb,
            num_valid_tokens=num_valid_tokens, split_k=split_k, B_i32=B_i32,
        )
        # Apply per-expert SVH manually (fallback for non-128-aligned dims)
        if svh_stacked is not None:
            # Need to gather per-token SVH from expert IDs
            num_m_blocks = C.shape[0] // BLOCK_M
            eid_per_token = expert_ids[:num_m_blocks].repeat_interleave(
                BLOCK_M).clamp(min=0)
            svh_per_token = svh_stacked[eid_per_token]
            from vllm.model_executor.layers.quantization.exl3_kernels.hadamard import (
                _get_had128,
            )
            H = _get_had128(C.device)
            result = torch.mm(C.reshape(-1, 128), H.T)
            result = result.reshape_as(C)
            C = result * svh_per_token
        return C

    # Handle num_tokens_post_padded as int or tensor
    if isinstance(num_tokens_post_padded, torch.Tensor):
        num_post_pad_tensor = num_tokens_post_padded.to(torch.int32)
        if EM_max <= 0:
            EM_max = num_tokens_post_padded.item()
    else:
        if EM_max <= 0:
            EM_max = num_tokens_post_padded
        num_post_pad_tensor = torch.tensor(
            [num_tokens_post_padded], dtype=torch.int32, device=A.device)

    EM = EM_max
    num_m_blocks = triton.cdiv(EM, BLOCK_M)
    tiles_n128 = N // 128

    if B_i32 is None:
        B_i32 = B_stacked.view(torch.int32)

    word_idx_t, next_word_idx_t, shift_t = get_bit_tables(bits, A.device)
    H16 = _get_h16(A.device)

    stride_be = B_i32.stride(0)
    stride_svh_e = svh_stacked.stride(0)
    HAS_SVH = svh_stacked is not None

    # Auto-select split-K config
    if split_k == 0:
        if _AUTOTUNE_DISABLED:
            config = _heuristic_config(1, tiles_k, tiles_n)
        else:
            device_name = torch.cuda.get_device_name(A.device)
            cache_key = (1, K, N, bits, cb, device_name)
            cached = _moe_had_tune_cache.get(cache_key)
            if cached is not None:
                config = cached
            else:
                # Use the standard MoE tune cache as starting point
                cached = _tune_cache.get(cache_key)
                if cached is not None:
                    config = cached
                else:
                    config = autotune_splitk(1, K, N, bits, cb, A.device)
                _moe_had_tune_cache[cache_key] = config
                _save_moe_had_to_disk_cache(cache_key, config)
        split_k, num_warps, num_stages, _ = config
    else:
        num_warps, num_stages = 2, 2

    C = torch.zeros(EM, N, dtype=torch.float16, device=A.device)

    if split_k > 1:
        # Split-K path: reuse existing splitk GEMM kernel + new reduce+Had kernel
        tiles_per_split = triton.cdiv(tiles_k, split_k)
        partial_buf = _get_moe_splitk_buf(split_k, EM, N, A.device)
        C_partial = partial_buf[:split_k, :EM, :N]
        C_partial.zero_()  # Must zero: kernel skips remote expert blocks

        # Phase 1: partial GEMM sums (reuse existing split-K kernel)
        grid_sk = (num_m_blocks, tiles_n, split_k)
        _exl3_fused_moe_gemm_splitk_kernel[grid_sk](
            A, B_i32, C_partial,
            sorted_token_ids, expert_ids,
            num_post_pad_tensor,
            word_idx_t, next_word_idx_t, shift_t,
            num_valid_tokens, N, K,
            A.stride(0), A.stride(1),
            C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
            stride_be,
            tiles_n, tiles_per_split,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            SPLIT_K=split_k,
            num_warps=num_warps, num_stages=num_stages,
        )

        # Phase 2: reduce + Had + SVH
        grid_red = (num_m_blocks, tiles_n128)
        _moe_reduce_had128_kernel[grid_red](
            C_partial, C,
            sorted_token_ids, expert_ids,
            num_post_pad_tensor,
            H16, svh_stacked,
            num_valid_tokens, N, EM,
            C_partial.stride(0), C_partial.stride(1), C_partial.stride(2),
            C.stride(0), C.stride(1),
            stride_svh_e,
            H16.stride(0), H16.stride(1),
            BLOCK_M=BLOCK_M, SPLIT_K=split_k, HAS_SVH=HAS_SVH,
            num_warps=num_warps, num_stages=num_stages,
        )

    else:
        # Non-split-K path: single fused kernel
        grid = (num_m_blocks, tiles_n128)
        _exl3_fused_moe_gemm_had128_kernel[grid](
            A, B_i32, C,
            sorted_token_ids, expert_ids,
            num_post_pad_tensor,
            word_idx_t, next_word_idx_t, shift_t,
            H16, svh_stacked,
            num_valid_tokens, N, K,
            A.stride(0), A.stride(1),
            C.stride(0), C.stride(1),
            stride_be,
            stride_svh_e,
            H16.stride(0), H16.stride(1),
            tiles_n,
            BLOCK_M=BLOCK_M, WORDS_PER_TILE=WORDS_PER_TILE, CB=cb,
            HAS_SVH=HAS_SVH,
            num_warps=num_warps, num_stages=num_stages,
        )

    return C
