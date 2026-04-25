// rocWMMA requires __half conversions/operators that torch's build system disables
// via -D flags. We must undef these BEFORE any hip headers are included so that
// __half is defined with its full set of constructors (including float->half).
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

// Re-define torch's macros before including torch headers to avoid ambiguous
// overload issues between torch's half type and hip's __half operators.
#define __HIP_NO_HALF_OPERATORS__ 1
#define __HIP_NO_HALF_CONVERSIONS__ 1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// =====================================================================
// EXL3 dequant + GEMM kernel using rocWMMA for RDNA3 (gfx1100)
//
// Phase 1: CB=0, 4-bit only. Correctness first, optimization later.
//
// Algorithm per K-tile (16x16 weight tile):
//   1. Load A tile: (BLOCK_M, 16) fp16
//   2. Load packed B words, extract 16-bit indices via funnel shift
//   3. CB=0 decode: x = index * 89226354 + 64248484, LOP3, fp16 add
//   4. WMMA: acc += A_tile @ weight via rocwmma::mma_sync()
//   5. Store (BLOCK_M, 16) fp16 output
//
// Grid: (cdiv(M, 16), tiles_n)
// Block: 32 threads (1 wave)
// =====================================================================

#define WARP_SIZE 32
#define TILE_DIM  16

// Use _Float16 (rocwmma::float16_t) for LDS buffers that rocWMMA reads/writes.
// __half and _Float16 have the same binary layout but are different C++ types.
using f16_t = rocwmma::float16_t;  // = _Float16

// rocWMMA fragment types for 16x16x16 fp16 -> fp32
using FragA   = rocwmma::fragment<rocwmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM,
                                  f16_t, rocwmma::row_major>;
using FragB   = rocwmma::fragment<rocwmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM,
                                  f16_t, rocwmma::row_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM,
                                  float>;

// -----------------------------------------------------------------
// LDS layout
// -----------------------------------------------------------------
// Bit extraction tables: 3 * 256 int32 = 3072 bytes
// A tile buffer: 16 * 16 * 2 = 512 bytes (f16_t)
// B tile buffer: 16 * 16 * 2 = 512 bytes (f16_t)
// Acc buffer: 16 * 16 * 4 = 1024 bytes (float)
// Total: ~5120 bytes per block (64KB available per CU)
// -----------------------------------------------------------------

struct SharedMem {
    int32_t  s_word_idx[256];
    int32_t  s_next_word_idx[256];
    int32_t  s_shift[256];
    f16_t    s_A[TILE_DIM * TILE_DIM];   // (16, 16) row-major
    f16_t    s_B[TILE_DIM * TILE_DIM];   // (16, 16) row-major
    float    s_acc[TILE_DIM * TILE_DIM];  // (16, 16) accumulator output
};

// -----------------------------------------------------------------
// CB=0 decode: single 16-bit index -> fp16 weight
// Returns _Float16 for direct storage into rocWMMA-compatible buffer.
//
// Optimized for RDNA3 ISA (gfx1100) based on Triton disassembly:
//   1. Packed DWORD AND+XOR: processes both fp16 halves in 2 ops (was 4+)
//   2. v_mad_u64_u32 fused multiply-add: 1 op instead of separate mul+add
// -----------------------------------------------------------------

// Fused multiply-add: returns (a * b + c) as uint32_t
// Maps to v_mad_u64_u32 on RDNA3 (1 instruction vs 2)
__device__ __forceinline__ uint32_t cb0_mad_u32(uint32_t a, uint32_t b, uint32_t c)
{
    uint64_t result;
    uint64_t c64 = c;  // v_mad_u64_u32 src2 is 64-bit
    asm("v_mad_u64_u32 %0, null, %1, %2, %3"
        : "=v"(result) : "v"(a), "v"(b), "v"(c64));
    return static_cast<uint32_t>(result);
}

// V5: CB=0 multiply-add using VOP2 instructions for VOPD eligibility.
// v_mul_lo_u32 (VOP2, 4 cycles) + v_add_u32 (VOP2, 1 cycle)
// vs v_mad_u64_u32 (VOP3, 4 cycles, NOT VOPD-eligible).
// Both compute identical low-32-bit result, but VOP2 can dual-issue.
__device__ __forceinline__ uint32_t cb0_mul_add_v5(uint32_t index)
{
    uint32_t result;
    asm("v_mul_lo_u32 %0, %1, 0x05517C72\n"   // index * 89226354 (MUL)
        "v_add_u32 %0, %0, 0x03D45AA4"         // + 64248484 (ADD)
        : "=v"(result) : "v"(index));
    return result;
}

// V5: CB=1 (MCG) multiply-only using VOP2 instruction.
// MUL=0xCBAC1FED, ADD=0 — no add needed.
__device__ __forceinline__ uint32_t cb1_mul_v5(uint32_t index)
{
    uint32_t result;
    asm("v_mul_lo_u32 %0, %1, 0xCBAC1FED"      // index * 0xCBAC1FED
        : "=v"(result) : "v"(index));
    return result;
}

__device__ __forceinline__ f16_t cb0_decode(uint32_t index)
{
    // V5: VOP2 multiply-add (VOPD-eligible)
    uint32_t x = cb0_mul_add_v5(index);

    // Packed DWORD AND+XOR: both fp16 halves simultaneously (2 ops)
    // Triton pattern: v_and_b32 + v_xor_b32 on full 32-bit word
    uint32_t xored = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;

    // Extract halves and add (v_cvt_f16 + v_add_f16)
    f16_t lo_f16, hi_f16;
    uint16_t lo_bits = static_cast<uint16_t>(xored);
    uint16_t hi_bits = static_cast<uint16_t>(xored >> 16);
    __builtin_memcpy(&lo_f16, &lo_bits, 2);
    __builtin_memcpy(&hi_f16, &hi_bits, 2);
    return lo_f16 + hi_f16;
}

// -----------------------------------------------------------------
// Branchless funnel shift: extract 16-bit index from packed B words
// Maps to v_cndmask_b32 on RDNA3 (no branch divergence)
// -----------------------------------------------------------------
__device__ __forceinline__ uint32_t funnel_shift_16(
    uint32_t lo_word, uint32_t hi_word, int shift)
{
    uint32_t shift_hi = (32u - shift) & 31u;
    uint32_t funnel = (lo_word >> shift) | (hi_word << shift_hi);
    return (shift != 0) ? (funnel & 0xFFFFu) : (lo_word & 0xFFFFu);
}

// -----------------------------------------------------------------
// V4 ISA-optimized dequant primitives for RDNA3 (gfx1100)
//
// Optimizations:
//   1. v_alignbit_b32: single-instruction funnel shift (was 5 ops)
//   2. cb0_decode_2: process 2 indices simultaneously for ILP/VOPD
//   3. dequant_tile_v4: fused loop using both optimizations
// -----------------------------------------------------------------

// v_alignbit_b32: extract 32 bits from 64-bit {hi,lo} >> shift
// Maps to single RDNA3 instruction (was 5 ops: sub, and, shr, shl, or)
__device__ __forceinline__ uint32_t alignbit_b32(
    uint32_t hi, uint32_t lo, uint32_t shift)
{
    uint32_t result;
    asm volatile(
        "v_alignbit_b32 %0, %1, %2, %3"
        : "=v"(result) : "v"(hi), "v"(lo), "v"(shift));
    return result;
}

// Optimized funnel shift using v_alignbit_b32
// Extracts 16-bit index from packed B words.
// 2 ops (alignbit + AND) vs 5 ops in funnel_shift_16
__device__ __forceinline__ uint32_t funnel_shift_16_v4(
    uint32_t lo_word, uint32_t hi_word, int shift)
{
    if (shift == 0)
        return lo_word & 0xFFFFu;
    // v_alignbit_b32: extracts bits from {hi, lo} >> shift
    // hi is the "higher order" word (appears in upper 32 bits)
    return alignbit_b32(hi_word, lo_word, static_cast<uint32_t>(shift)) & 0xFFFFu;
}

// Process 2 indices simultaneously for better ILP and VOPD pairing.
// Two independent MAD+AND+XOR+extract chains can overlap on the
// RDNA3 VALU since they have no data dependencies.
__device__ __forceinline__ void cb0_decode_2(
    uint32_t idx0, uint32_t idx1,
    f16_t& out0, f16_t& out1)
{
    // Stage 1: Independent MUL+ADD using VOP2 (VOPD-eligible)
    uint32_t x0 = cb0_mul_add_v5(idx0);
    uint32_t x1 = cb0_mul_add_v5(idx1);

    // Stage 2: Independent AND+XOR pairs
    // VOPD opportunity: v_and(x0,...) :: v_xor(x1_prev,...)
    uint32_t a0 = x0 & 0x8FFF8FFFu;
    uint32_t a1 = x1 & 0x8FFF8FFFu;
    uint32_t e0 = a0 ^ 0x3B603B60u;
    uint32_t e1 = a1 ^ 0x3B603B60u;

    // Stage 3: Extract halves and add for each
    f16_t lo0, hi0, lo1, hi1;
    uint16_t lo0_bits = static_cast<uint16_t>(e0);
    uint16_t hi0_bits = static_cast<uint16_t>(e0 >> 16);
    uint16_t lo1_bits = static_cast<uint16_t>(e1);
    uint16_t hi1_bits = static_cast<uint16_t>(e1 >> 16);
    __builtin_memcpy(&lo0, &lo0_bits, 2);
    __builtin_memcpy(&hi0, &hi0_bits, 2);
    __builtin_memcpy(&lo1, &lo1_bits, 2);
    __builtin_memcpy(&hi1, &hi1_bits, 2);
    out0 = lo0 + hi0;
    out1 = lo1 + hi1;
}

// V4 optimized dequant for the inner B tile loop.
// Uses v_alignbit_b32 funnel shift + dual cb0_decode for ILP.
// Processes 8 elements per thread (same as v2), but with fewer ops.
//
// For 1-wave (32-thread) kernels: call with the thread's lane index.
// For 2-wave (64-thread) kernels: call with lane within wave.
__device__ __forceinline__ void dequant_tile_v4(
    const int32_t* __restrict__ B_base_ptr,  // B + b_base
    const int* reg_word_idx,                 // preloaded from LDS [8]
    const int* reg_next_word_idx,            // preloaded from LDS [8]
    const int* reg_shift,                    // preloaded from LDS [8]
    int lane,
    f16_t* lds_out)                          // per-wave B LDS buffer
{
    // Process pairs of indices for ILP
    #pragma unroll
    for (int j = 0; j < 8; j += 2)
    {
        // Load B words for both indices
        uint32_t lo0 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j]]);
        uint32_t hi0 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j]]);
        uint32_t lo1 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j+1]]);
        uint32_t hi1 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j+1]]);

        // Funnel shift to extract 16-bit indices
        uint32_t index0 = funnel_shift_16_v4(lo0, hi0, reg_shift[j]);
        uint32_t index1 = funnel_shift_16_v4(lo1, hi1, reg_shift[j+1]);

        // Dual CB0 decode
        cb0_decode_2(index0, index1,
                     lds_out[lane + j * WARP_SIZE],
                     lds_out[lane + (j+1) * WARP_SIZE]);
    }
}

// V4 optimized dequant with direct fragment fill (no LDS for B).
// Same optimizations as dequant_tile_v4 but writes to frag_b.x[] directly.
// Uses temporaries to avoid non-const reference to vector element.
__device__ __forceinline__ void dequant_frag_v4(
    const int32_t* __restrict__ B_base_ptr,
    const int* reg_word_idx,
    const int* reg_next_word_idx,
    const int* reg_shift,
    FragB& frag_b)
{
    #pragma unroll
    for (int j = 0; j < 8; j += 2)
    {
        uint32_t lo0 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j]]);
        uint32_t hi0 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j]]);
        uint32_t lo1 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j+1]]);
        uint32_t hi1 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j+1]]);

        uint32_t index0 = funnel_shift_16_v4(lo0, hi0, reg_shift[j]);
        uint32_t index1 = funnel_shift_16_v4(lo1, hi1, reg_shift[j+1]);

        f16_t out0, out1;
        cb0_decode_2(index0, index1, out0, out1);
        frag_b.x[j]   = out0;
        frag_b.x[j+1] = out1;
    }
}


// -----------------------------------------------------------------
// V5 ISA-optimized dequant for RDNA3 (gfx1100)
//
// Optimizations over V4:
//   1. Interleaved independent ops: AND/XOR of two elements are adjacent,
//      enabling the compiler's VOPD auto-pairing at -O3.
//   2. Streamlined horizontal add: v_lshrrev_b32 + v_add_f16 (2 ops)
//      vs V4's memcpy-based extract (compiler may generate extra moves).
//   3. Fused funnel_shift pair: both indices extracted before decode,
//      maximizing distance between load and use for latency hiding.
//
// Note: SDWA was removed in GFX11, and explicit v_dual_* asm requires
// even/odd VGPR bank alignment that inline asm can't guarantee. Instead
// we rely on hipcc -O3 VOPD auto-pairing of adjacent independent VOP2 ops.
// -----------------------------------------------------------------

// Streamlined CB0 decode for 2 independent indices.
// Interleaves operations to maximize VOPD auto-pairing opportunities.
// The compiler can pair: and0 :: and1, xor0 :: xor1, lshr0 :: lshr1,
// add0 :: add1 — up to 4 VOPD pairs per decode of 2 elements.
__device__ __forceinline__ void cb0_decode_2_v5(
    uint32_t idx0, uint32_t idx1,
    f16_t& out0, f16_t& out1)
{
    // Stage 1: Independent MUL+ADD using VOP2 (VOPD-eligible)
    // v_mul_lo_u32 pair can dual-issue, v_add_u32 pair can dual-issue
    uint32_t x0 = cb0_mul_add_v5(idx0);
    uint32_t x1 = cb0_mul_add_v5(idx1);

    // Stage 2: AND pair (VOPD candidate: v_and_b32 :: v_and_b32)
    uint32_t a0 = x0 & 0x8FFF8FFFu;
    uint32_t a1 = x1 & 0x8FFF8FFFu;

    // Stage 3: XOR pair (VOPD candidate: v_xor_b32 :: v_xor_b32)
    uint32_t e0 = a0 ^ 0x3B603B60u;
    uint32_t e1 = a1 ^ 0x3B603B60u;

    // Stage 4: Horizontal add — extract lo16 + hi16 as fp16
    // v_lshrrev_b32 (shift hi16 down) + v_add_f16 = 2 ops each
    // Interleaved for VOPD pairing of the shifts and adds
    uint32_t hi0_u32 = e0 >> 16;
    uint32_t hi1_u32 = e1 >> 16;
    f16_t lo0, hi0, lo1, hi1;
    __builtin_memcpy(&lo0, &e0, 2);
    __builtin_memcpy(&hi0, &hi0_u32, 2);
    __builtin_memcpy(&lo1, &e1, 2);
    __builtin_memcpy(&hi1, &hi1_u32, 2);
    out0 = lo0 + hi0;
    out1 = lo1 + hi1;
}

// -----------------------------------------------------------------
// CB=1 (MCG) decode: multiply only, no additive constant
// MCG uses MUL=0xCBAC1FED, ADD=0 (vs CB=0: MUL=89226354, ADD=64248484)
// One fewer instruction than CB=0 (no additive).
// -----------------------------------------------------------------

__device__ __forceinline__ void cb1_decode_2_v5(
    uint32_t idx0, uint32_t idx1,
    f16_t& out0, f16_t& out1)
{
    // Stage 1: MUL only using VOP2 (VOPD-eligible, no additive constant)
    uint32_t x0 = cb1_mul_v5(idx0);
    uint32_t x1 = cb1_mul_v5(idx1);

    // Stage 2: AND pair (same masks as cb0)
    uint32_t a0 = x0 & 0x8FFF8FFFu;
    uint32_t a1 = x1 & 0x8FFF8FFFu;

    // Stage 3: XOR pair (same masks as cb0)
    uint32_t e0 = a0 ^ 0x3B603B60u;
    uint32_t e1 = a1 ^ 0x3B603B60u;

    // Stage 4: Horizontal add — extract lo16 + hi16 as fp16
    uint32_t hi0_u32 = e0 >> 16;
    uint32_t hi1_u32 = e1 >> 16;
    f16_t lo0, hi0, lo1, hi1;
    __builtin_memcpy(&lo0, &e0, 2);
    __builtin_memcpy(&hi0, &hi0_u32, 2);
    __builtin_memcpy(&lo1, &e1, 2);
    __builtin_memcpy(&hi1, &hi1_u32, 2);
    out0 = lo0 + hi0;
    out1 = lo1 + hi1;
}

// Single-element CB=1 decode (for v3 kernels / benchmarks)
__device__ __forceinline__ f16_t cb1_decode(uint32_t index)
{
    uint32_t x = cb1_mul_v5(index);
    uint32_t xored = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    f16_t lo_f16, hi_f16;
    uint16_t lo_bits = static_cast<uint16_t>(xored);
    uint16_t hi_bits = static_cast<uint16_t>(xored >> 16);
    __builtin_memcpy(&lo_f16, &lo_bits, 2);
    __builtin_memcpy(&hi_f16, &hi_bits, 2);
    return lo_f16 + hi_f16;
}

// CB-dispatching decode pair: cb==0 uses cb0, cb==1 uses cb1
__device__ __forceinline__ void cb_decode_2_v5(
    uint32_t idx0, uint32_t idx1,
    f16_t& out0, f16_t& out1, int cb)
{
    if (cb == 0)
        cb0_decode_2_v5(idx0, idx1, out0, out1);
    else
        cb1_decode_2_v5(idx0, idx1, out0, out1);
}

// CB-dispatching single decode
__device__ __forceinline__ f16_t cb_decode(uint32_t index, int cb)
{
    return (cb == 0) ? cb0_decode(index) : cb1_decode(index);
}

// V5 funnel shift pair: extract 2 indices with interleaved AND masks.
__device__ __forceinline__ void funnel_shift_16_v5_pair(
    uint32_t lo0, uint32_t hi0, int shift0,
    uint32_t lo1, uint32_t hi1, int shift1,
    uint32_t& idx0, uint32_t& idx1)
{
    // v_alignbit_b32 is VOP3 — cannot VOPD, must be sequential
    uint32_t raw0, raw1;
    if (shift0 == 0)
        raw0 = lo0;
    else
        raw0 = alignbit_b32(hi0, lo0, static_cast<uint32_t>(shift0));

    if (shift1 == 0)
        raw1 = lo1;
    else
        raw1 = alignbit_b32(hi1, lo1, static_cast<uint32_t>(shift1));

    // AND pair (VOPD candidate: v_and_b32 :: v_and_b32)
    idx0 = raw0 & 0xFFFFu;
    idx1 = raw1 & 0xFFFFu;
}

// V5 optimized dequant tile: VOPD + SDWA throughout.
__device__ __forceinline__ void dequant_tile_v5(
    const int32_t* __restrict__ B_base_ptr,
    const int* reg_word_idx,
    const int* reg_next_word_idx,
    const int* reg_shift,
    int lane,
    f16_t* lds_out)
{
    #pragma unroll
    for (int j = 0; j < 8; j += 2)
    {
        uint32_t lo0 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j]]);
        uint32_t hi0 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j]]);
        uint32_t lo1 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j+1]]);
        uint32_t hi1 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j+1]]);

        uint32_t index0, index1;
        funnel_shift_16_v5_pair(
            lo0, hi0, reg_shift[j],
            lo1, hi1, reg_shift[j+1],
            index0, index1);

        cb0_decode_2_v5(index0, index1,
                        lds_out[lane + j * WARP_SIZE],
                        lds_out[lane + (j+1) * WARP_SIZE]);
    }
}

// V5 optimized dequant with direct fragment fill.
__device__ __forceinline__ void dequant_frag_v5(
    const int32_t* __restrict__ B_base_ptr,
    const int* reg_word_idx,
    const int* reg_next_word_idx,
    const int* reg_shift,
    FragB& frag_b)
{
    #pragma unroll
    for (int j = 0; j < 8; j += 2)
    {
        uint32_t lo0 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j]]);
        uint32_t hi0 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j]]);
        uint32_t lo1 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j+1]]);
        uint32_t hi1 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j+1]]);

        uint32_t index0, index1;
        funnel_shift_16_v5_pair(
            lo0, hi0, reg_shift[j],
            lo1, hi1, reg_shift[j+1],
            index0, index1);

        f16_t out0, out1;
        cb0_decode_2_v5(index0, index1, out0, out1);
        frag_b.x[j]   = out0;
        frag_b.x[j+1] = out1;
    }
}

// CB-dispatching dequant tile: routes to cb0 or cb1 decode
__device__ __forceinline__ void dequant_tile_v5_cb(
    const int32_t* __restrict__ B_base_ptr,
    const int* reg_word_idx,
    const int* reg_next_word_idx,
    const int* reg_shift,
    int lane,
    f16_t* lds_out,
    int cb)
{
    #pragma unroll
    for (int j = 0; j < 8; j += 2)
    {
        uint32_t lo0 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j]]);
        uint32_t hi0 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j]]);
        uint32_t lo1 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j+1]]);
        uint32_t hi1 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j+1]]);

        uint32_t index0, index1;
        funnel_shift_16_v5_pair(
            lo0, hi0, reg_shift[j],
            lo1, hi1, reg_shift[j+1],
            index0, index1);

        cb_decode_2_v5(index0, index1,
                       lds_out[lane + j * WARP_SIZE],
                       lds_out[lane + (j+1) * WARP_SIZE], cb);
    }
}

// CB-dispatching dequant with direct fragment fill
__device__ __forceinline__ void dequant_frag_v5_cb(
    const int32_t* __restrict__ B_base_ptr,
    const int* reg_word_idx,
    const int* reg_next_word_idx,
    const int* reg_shift,
    FragB& frag_b,
    int cb)
{
    #pragma unroll
    for (int j = 0; j < 8; j += 2)
    {
        uint32_t lo0 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j]]);
        uint32_t hi0 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j]]);
        uint32_t lo1 = static_cast<uint32_t>(B_base_ptr[reg_word_idx[j+1]]);
        uint32_t hi1 = static_cast<uint32_t>(B_base_ptr[reg_next_word_idx[j+1]]);

        uint32_t index0, index1;
        funnel_shift_16_v5_pair(
            lo0, hi0, reg_shift[j],
            lo1, hi1, reg_shift[j+1],
            index0, index1);

        f16_t out0, out1;
        cb_decode_2_v5(index0, index1, out0, out1, cb);
        frag_b.x[j]   = out0;
        frag_b.x[j+1] = out1;
    }
}


// -----------------------------------------------------------------
// Main kernel
// -----------------------------------------------------------------
__global__ __launch_bounds__(WARP_SIZE)
void exl3_gemm_hip_kernel(
    const f16_t*    __restrict__ A,           // (M, K) row-major
    const int32_t*  __restrict__ B,           // flat (K/16, N/16, WORDS_PER_TILE) int32
    f16_t*          __restrict__ C,           // (M, N) row-major
    const int32_t*  __restrict__ word_idx,    // (256,) bit extraction table
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int M, int N, int K,
    int tiles_n,
    int WORDS_PER_TILE,
    int cb)
{
    // Grid: (cdiv(M, 16), tiles_n)
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int lane  = threadIdx.x;

    __shared__ SharedMem smem;

    // 1. Cooperatively load bit extraction tables into LDS
    //    256 entries, 32 threads -> 8 entries each
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    // Preload bit tables into registers (avoids LDS reads in K-loop)
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[i];
        reg_next_word_idx[j] = smem.s_next_word_idx[i];
        reg_shift[j]         = smem.s_shift[i];
    }

    // 2. Initialize accumulator
    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    int num_k_tiles = K / TILE_DIM;
    int row_base = pid_m * TILE_DIM;

    for (int tk = 0; tk < num_k_tiles; tk++)
    {
        // --- Load A tile into LDS: (16, 16) fp16 ---
        // 256 elements, 32 threads -> 8 elements each
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_row = row_base + r;
            int global_col = tk * TILE_DIM + c;
            if (global_row < M)
                smem.s_A[i] = A[global_row * K + global_col];
            else
                smem.s_A[i] = static_cast<f16_t>(0.0f);
        }

        // --- V4 optimized dequant B tile into LDS ---
        int b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE;
        dequant_tile_v5_cb(
            B + b_base,
            reg_word_idx, reg_next_word_idx, reg_shift,
            lane, smem.s_B, cb);

        __syncthreads();

        // --- WMMA: acc += A_tile @ B_tile ---
        FragA frag_a;
        FragB frag_b;

        // Load from LDS (row-major, ldm=16)
        rocwmma::load_matrix_sync(frag_a, smem.s_A, TILE_DIM);
        rocwmma::load_matrix_sync(frag_b, smem.s_B, TILE_DIM);

        // MMA
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);

        __syncthreads();
    }

    // 3. Store accumulator (fp32) to LDS, then convert to fp16 and write to C
    rocwmma::store_matrix_sync(smem.s_acc, acc, TILE_DIM, rocwmma::mem_row_major);
    __syncthreads();

    // Convert fp32 -> fp16 and write to global C
    for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
    {
        int r = i / TILE_DIM;
        int c = i % TILE_DIM;
        int global_row = row_base + r;
        int global_col = pid_n * TILE_DIM + c;
        if (global_row < M && global_col < N)
        {
            float val = smem.s_acc[i];
            // Clamp to fp16 range
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            C[global_row * N + global_col] = static_cast<f16_t>(val);
        }
    }
}


// =====================================================================
// Phase 2: LDS layout without A tile buffer (direct global load)
// =====================================================================

struct SharedMemV2 {
    int32_t  s_word_idx[256];
    int32_t  s_next_word_idx[256];
    int32_t  s_shift[256];
    f16_t    s_A[TILE_DIM * TILE_DIM];   // (16, 16) — only for boundary tiles
    f16_t    s_B[TILE_DIM * TILE_DIM];   // (16, 16) row-major
    float    s_acc[TILE_DIM * TILE_DIM];  // (16, 16) accumulator output
};

// =====================================================================
// Phase 2: Split-K kernel with direct A load from global memory
//
// Grid: (cdiv(M, 16), tiles_n, split_k)
// Each block processes a K-range [k_start, k_end) and writes fp16
// partial sums to C_partial[pid_k * M * N + row * N + col].
// =====================================================================

__global__ __launch_bounds__(WARP_SIZE)
void exl3_gemm_hip_splitk_kernel(
    const f16_t*    __restrict__ A,           // (M, K) row-major
    const int32_t*  __restrict__ B,           // flat (K/16, N/16, WORDS_PER_TILE) int32
    f16_t*          __restrict__ C_partial,   // (split_k, M, N) row-major partials
    const int32_t*  __restrict__ word_idx,    // (256,) bit extraction table
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int M, int N, int K,
    int tiles_n,
    int WORDS_PER_TILE,
    int num_k_tiles,                          // total K tiles
    int cb)
{
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int pid_k = blockIdx.z;
    int split_k = gridDim.z;
    int lane  = threadIdx.x;

    __shared__ SharedMemV2 smem;

    // Load bit extraction tables into LDS
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    // Preload bit tables into registers (avoids LDS reads in K-loop)
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[i];
        reg_next_word_idx[j] = smem.s_next_word_idx[i];
        reg_shift[j]         = smem.s_shift[i];
    }

    // Compute K-range for this split
    int tiles_per_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_start = pid_k * tiles_per_split;
    int tk_end   = min(tk_start + tiles_per_split, num_k_tiles);

    // Initialize accumulator
    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    int row_base = pid_m * TILE_DIM;
    bool boundary_m = (row_base + TILE_DIM > M);  // need masking for A rows

    for (int tk = tk_start; tk < tk_end; tk++)
    {
        // --- Load A tile ---
        FragA frag_a;
        if (boundary_m)
        {
            // Boundary: load via LDS with zero-padding for OOB rows
            for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
            {
                int r = i / TILE_DIM;
                int c = i % TILE_DIM;
                int global_row = row_base + r;
                int global_col = tk * TILE_DIM + c;
                if (global_row < M)
                    smem.s_A[i] = A[global_row * K + global_col];
                else
                    smem.s_A[i] = static_cast<f16_t>(0.0f);
            }
            __syncthreads();
            rocwmma::load_matrix_sync(frag_a, smem.s_A, TILE_DIM);
        }
        else
        {
            // Full tile: load directly from global memory (row-major, stride=K)
            rocwmma::load_matrix_sync(frag_a, A + row_base * K + tk * TILE_DIM, K);
        }

        // --- V4 optimized dequant B tile into LDS ---
        int b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE;
        dequant_tile_v5_cb(
            B + b_base,
            reg_word_idx, reg_next_word_idx, reg_shift,
            lane, smem.s_B, cb);

        __syncthreads();

        // --- WMMA: acc += A_tile @ B_tile ---
        FragB frag_b;
        rocwmma::load_matrix_sync(frag_b, smem.s_B, TILE_DIM);
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);

        __syncthreads();
    }

    // Store partial result to C_partial[pid_k * M * N + ...]
    rocwmma::store_matrix_sync(smem.s_acc, acc, TILE_DIM, rocwmma::mem_row_major);
    __syncthreads();

    f16_t* out = C_partial + pid_k * M * N;
    for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
    {
        int r = i / TILE_DIM;
        int c = i % TILE_DIM;
        int global_row = row_base + r;
        int global_col = pid_n * TILE_DIM + c;
        if (global_row < M && global_col < N)
        {
            float val = smem.s_acc[i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            out[global_row * N + global_col] = static_cast<f16_t>(val);
        }
    }
}

// =====================================================================
// Phase 2: Reduction kernel — sum split-K partials
//
// Grid: (cdiv(M*N, 256))
// Block: 256 threads
// Each thread sums split_k partials for one output element
// =====================================================================

__global__ void exl3_gemm_reduce_kernel(
    const f16_t* __restrict__ C_partial,  // (split_k, M, N) contiguous
    f16_t*       __restrict__ C,          // (M, N) output
    int MN,                               // M * N
    int split_k)
{
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= MN) return;

    float sum = 0.0f;
    for (int s = 0; s < split_k; s++)
    {
        f16_t val = C_partial[s * MN + idx];
        sum += static_cast<float>(val);
    }

    // Clamp and store
    sum = fminf(fmaxf(sum, -65504.0f), 65504.0f);
    C[idx] = static_cast<f16_t>(sum);
}


// =====================================================================
// Host launchers
// =====================================================================

void hip_exl3_gemm(
    at::Tensor A,           // (M, K) float16
    at::Tensor B_i32,       // flat int32 packed trellis data
    at::Tensor C,           // (M, N) float16 output
    at::Tensor word_idx,    // (256,) int32
    at::Tensor next_word_idx, // (256,) int32
    at::Tensor shift_tbl,   // (256,) int32
    int bits,
    int cb)
{
    TORCH_CHECK(A.is_contiguous(), "exl3_gemm: A must be contiguous");
    TORCH_CHECK(B_i32.is_contiguous(), "exl3_gemm: B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "exl3_gemm: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "exl3_gemm: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "exl3_gemm: C must be float16");
    TORCH_CHECK(B_i32.dtype() == at::kInt, "exl3_gemm: B must be int32");
    TORCH_CHECK(word_idx.dtype() == at::kInt, "exl3_gemm: word_idx must be int32");
    TORCH_CHECK(next_word_idx.dtype() == at::kInt, "exl3_gemm: next_word_idx must be int32");
    TORCH_CHECK(shift_tbl.dtype() == at::kInt, "exl3_gemm: shift_tbl must be int32");

    // Phase 1: CB=0 only
    TORCH_CHECK(cb == 0 || cb == 1, "exl3_gemm HIP: only cb=0,1 supported, got cb=", cb);

    int M = A.size(0);
    int K = A.size(1);
    int N = C.size(1);

    TORCH_CHECK(K % 16 == 0, "exl3_gemm: K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 16 == 0, "exl3_gemm: N must be divisible by 16, got ", N);

    int tiles_n = N / 16;
    int WORDS_PER_TILE = 256 * bits / 32;  // = 8 * bits (for 4-bit: 32)

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Grid: (cdiv(M, 16), tiles_n)
    int grid_m = (M + 15) / 16;
    dim3 grid(grid_m, tiles_n);
    dim3 block(WARP_SIZE);

    // Cast torch's half* to _Float16* — same binary layout
    exl3_gemm_hip_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const f16_t*>(A.data_ptr()),
        reinterpret_cast<const int32_t*>(B_i32.data_ptr()),
        reinterpret_cast<f16_t*>(C.data_ptr()),
        reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
        reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
        reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
        M, N, K,
        tiles_n,
        WORDS_PER_TILE,
        cb
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "exl3_gemm kernel launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// Phase 2: Optimized launcher with split-K + direct A load
// =====================================================================

void hip_exl3_gemm_v2(
    at::Tensor A,             // (M, K) float16
    at::Tensor B_i32,         // flat int32 packed trellis data
    at::Tensor C,             // (M, N) float16 output
    at::Tensor word_idx,      // (256,) int32
    at::Tensor next_word_idx, // (256,) int32
    at::Tensor shift_tbl,     // (256,) int32
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial)     // (split_k, M, N) float16 — only used when split_k > 1
{
    TORCH_CHECK(A.is_contiguous(), "exl3_gemm_v2: A must be contiguous");
    TORCH_CHECK(B_i32.is_contiguous(), "exl3_gemm_v2: B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "exl3_gemm_v2: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "exl3_gemm_v2: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "exl3_gemm_v2: C must be float16");
    TORCH_CHECK(B_i32.dtype() == at::kInt, "exl3_gemm_v2: B must be int32");
    TORCH_CHECK(cb == 0 || cb == 1, "exl3_gemm_v2 HIP: only cb=0,1 supported, got cb=", cb);
    TORCH_CHECK(split_k >= 1, "exl3_gemm_v2: split_k must be >= 1, got ", split_k);

    int M = A.size(0);
    int K = A.size(1);
    int N = C.size(1);

    TORCH_CHECK(K % 16 == 0, "exl3_gemm_v2: K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 16 == 0, "exl3_gemm_v2: N must be divisible by 16, got ", N);

    int tiles_n = N / 16;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;

    // Clamp split_k to not exceed k tiles
    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int grid_m = (M + 15) / 16;

    if (split_k == 1)
    {
        // No split-K: single kernel writes directly to C
        dim3 grid(grid_m, tiles_n, 1);
        dim3 block(WARP_SIZE);

        exl3_gemm_hip_splitk_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            M, N, K, tiles_n, WORDS_PER_TILE, num_k_tiles, cb
        );
    }
    else
    {
        // Split-K: write partials, then reduce
        TORCH_CHECK(C_partial.is_contiguous(), "exl3_gemm_v2: C_partial must be contiguous");
        TORCH_CHECK(C_partial.dtype() == at::kHalf, "exl3_gemm_v2: C_partial must be float16");
        TORCH_CHECK(C_partial.size(0) >= split_k && C_partial.size(1) >= M && C_partial.size(2) >= N,
                    "exl3_gemm_v2: C_partial too small");

        dim3 grid(grid_m, tiles_n, split_k);
        dim3 block(WARP_SIZE);

        exl3_gemm_hip_splitk_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            M, N, K, tiles_n, WORDS_PER_TILE, num_k_tiles, cb
        );

        // Reduction kernel
        int MN = M * N;
        int reduce_threads = 256;
        int reduce_blocks = (MN + reduce_threads - 1) / reduce_threads;

        exl3_gemm_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            reinterpret_cast<const f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            MN, split_k
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "exl3_gemm_v2 kernel launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// Phase 3: Fused MoE GEMM kernel using rocWMMA
//
// One kernel launch processes ALL experts for a single projection.
// vLLM pre-gathers tokens into contiguous x_sorted with identity_ids,
// so A is always contiguous and all tiles are full (EM_max % 16 == 0).
//
// Grid: (num_m_blocks, tiles_n, split_k)
// Block: 32 threads (1 wave)
//
// Each M-block maps to one expert via expert_ids[pid_m].
// B_stacked is (E, tiles_k, tiles_n, WPT) — expert offset via stride_be.
// =====================================================================

struct SharedMemMoE {
    int32_t  s_word_idx[256];      // 1024 bytes
    int32_t  s_next_word_idx[256]; // 1024 bytes
    int32_t  s_shift[256];         // 1024 bytes
    f16_t    s_B[TILE_DIM * TILE_DIM];   // 512 bytes
    float    s_acc[TILE_DIM * TILE_DIM];  // 1024 bytes
};  // Total: ~4608 bytes


__global__ __launch_bounds__(WARP_SIZE)
void exl3_fused_moe_gemm_hip_kernel(
    const f16_t*    __restrict__ A,           // (EM_max, K) contiguous
    const int32_t*  __restrict__ B_stacked,   // (E, tiles_k, tiles_n, WPT) int32
    f16_t*          __restrict__ C,           // output (or C_partial for split-K)
    const int32_t*  __restrict__ expert_ids,  // (num_m_blocks,) int32
    const int32_t*  __restrict__ num_tokens_post_padded, // (1,) tensor
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int EM_max, int N, int K,
    int stride_be,              // expert stride in B_stacked (int32 elements)
    int tiles_n,
    int WORDS_PER_TILE,
    int num_k_tiles,
    int split_k,
    int stride_cp_split,        // stride between split-K slices = EM_max * N
    int cb)
{
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int pid_k = blockIdx.z;
    int lane  = threadIdx.x;

    // Early return: check if this M-block is beyond valid tokens
    int num_valid = num_tokens_post_padded[0];
    int num_valid_m_blocks = (num_valid + TILE_DIM - 1) / TILE_DIM;
    if (pid_m >= num_valid_m_blocks)
        return;

    // Load expert ID for this M-block; skip remote experts (id < 0)
    int off_expert = expert_ids[pid_m];
    if (off_expert < 0)
        return;

    __shared__ SharedMemMoE smem;

    // Load bit extraction tables into LDS
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    // Preload bit tables into registers (avoids LDS reads in K-loop)
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[i];
        reg_next_word_idx[j] = smem.s_next_word_idx[i];
        reg_shift[j]         = smem.s_shift[i];
    }

    // Compute K-range for this split
    int tiles_per_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_start = pid_k * tiles_per_split;
    int tk_end   = min(tk_start + tiles_per_split, num_k_tiles);

    // Initialize accumulator
    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    int row_base = pid_m * TILE_DIM;

    // B offset for this expert
    const int32_t* B_expert = B_stacked + off_expert * stride_be;

    for (int tk = tk_start; tk < tk_end; tk++)
    {
        // --- Load A tile directly from global memory ---
        // A is contiguous (EM_max, K), all M-tiles are full (EM_max % 16 == 0)
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A + row_base * K + tk * TILE_DIM, K);

        // --- V4 optimized dequant B tile into LDS ---
        int b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE;
        dequant_tile_v5_cb(
            B_expert + b_base,
            reg_word_idx, reg_next_word_idx, reg_shift,
            lane, smem.s_B, cb);

        __syncthreads();

        // --- WMMA: acc += A_tile @ B_tile ---
        FragB frag_b;
        rocwmma::load_matrix_sync(frag_b, smem.s_B, TILE_DIM);
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);

        __syncthreads();
    }

    // Store result
    rocwmma::store_matrix_sync(smem.s_acc, acc, TILE_DIM, rocwmma::mem_row_major);
    __syncthreads();

    // Compute output pointer: split_k==1 writes to C directly,
    // split_k>1 writes to C[pid_k * stride_cp_split + ...]
    f16_t* out = C + pid_k * stride_cp_split;

    for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
    {
        int r = i / TILE_DIM;
        int c = i % TILE_DIM;
        int global_row = row_base + r;
        int global_col = pid_n * TILE_DIM + c;
        if (global_row < EM_max && global_col < N)
        {
            float val = smem.s_acc[i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            out[global_row * N + global_col] = static_cast<f16_t>(val);
        }
    }
}


// =====================================================================
// Phase 3b: 2-wave fused MoE GEMM kernel
//
// 2 waves (64 threads) per block. Each wave processes half the K-tiles
// independently, then accumulators are merged via LDS.
//
// Benefits:
//   - K-loop iterations halved per wave → dequant work halved
//   - No per-K-tile __syncthreads() between waves (own B buffer)
//   - Shared bit tables loaded once by 64 threads (2x faster)
//   - RDNA3 dual-SIMD: both SIMDs in a CU execute in parallel
//
// Grid: (num_m_blocks, tiles_n, split_k)
// Block: 64 threads (2 waves)
// =====================================================================

#define NUM_WAVES_MOE 2
#define BLOCK_SIZE_MOE (WARP_SIZE * NUM_WAVES_MOE)  // 64

struct SharedMemMoE2W {
    int32_t  s_word_idx[256];                          // 1024B shared
    int32_t  s_next_word_idx[256];                     // 1024B shared
    int32_t  s_shift[256];                             // 1024B shared
    f16_t    s_B[NUM_WAVES_MOE][TILE_DIM * TILE_DIM];  // 1024B (512 per wave)
    float    s_acc[NUM_WAVES_MOE][TILE_DIM * TILE_DIM]; // 2048B (1024 per wave)
};  // Total: ~6144 bytes


// K-split 2-wave kernel: each wave processes half the K-tiles independently,
// then accumulators are merged via LDS.
// Grid: (num_m_blocks, tiles_n, split_k)
// Block: 64 threads (2 waves)
__global__ __launch_bounds__(BLOCK_SIZE_MOE)
void exl3_fused_moe_gemm_hip_2w_kernel(
    const f16_t*    __restrict__ A,
    const int32_t*  __restrict__ B_stacked,
    f16_t*          __restrict__ C,
    const int32_t*  __restrict__ expert_ids,
    const int32_t*  __restrict__ num_tokens_post_padded,
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int EM_max, int N, int K,
    int stride_be,
    int tiles_n,
    int WORDS_PER_TILE,
    int num_k_tiles,
    int split_k,
    int stride_cp_split,
    int cb)
{
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int pid_k = blockIdx.z;
    int tid   = threadIdx.x;
    int wave_id = tid / WARP_SIZE;   // 0 or 1
    int lane    = tid % WARP_SIZE;   // 0..31

    // Early return: write zeros to prevent stale data in cached split-K buffers
    int num_valid = num_tokens_post_padded[0];
    int num_valid_m_blocks = (num_valid + TILE_DIM - 1) / TILE_DIM;
    if (pid_m >= num_valid_m_blocks) return;

    int off_expert = expert_ids[pid_m];
    if (off_expert < 0) return;

    __shared__ SharedMemMoE2W smem;

    // Load bit tables cooperatively: 64 threads → 4 entries each
    for (int i = tid; i < 256; i += BLOCK_SIZE_MOE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();  // Only cross-wave sync: bit tables ready

    // Compute K-range for this grid split
    int tiles_per_grid_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_grid_start = pid_k * tiles_per_grid_split;
    int tk_grid_end   = min(tk_grid_start + tiles_per_grid_split, num_k_tiles);
    int grid_tiles    = tk_grid_end - tk_grid_start;

    // Split K-range between 2 waves within the block
    int tiles_per_wave = (grid_tiles + NUM_WAVES_MOE - 1) / NUM_WAVES_MOE;
    int tk_start = tk_grid_start + wave_id * tiles_per_wave;
    int tk_end   = min(tk_start + tiles_per_wave, tk_grid_end);

    // Preload bit tables into registers: 8 entries per lane (avoids LDS reads in K-loop)
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[i];
        reg_next_word_idx[j] = smem.s_next_word_idx[i];
        reg_shift[j]         = smem.s_shift[i];
    }

    // Initialize accumulator
    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    int row_base = pid_m * TILE_DIM;
    const int32_t* B_expert = B_stacked + off_expert * stride_be;

    // K-loop: each wave processes its own K-range independently
    // No cross-wave sync needed — each wave uses s_B[wave_id]
    for (int tk = tk_start; tk < tk_end; tk++)
    {
        // Load A tile from global
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A + row_base * K + tk * TILE_DIM, K);

        // V4 optimized dequant: v_alignbit_b32 + dual cb0_decode
        int b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE;
        dequant_tile_v5_cb(
            B_expert + b_base,
            reg_word_idx, reg_next_word_idx, reg_shift,
            lane, smem.s_B[wave_id], cb);

        // WMMA: load from per-wave LDS buffer
        FragB frag_b;
        rocwmma::load_matrix_sync(frag_b, smem.s_B[wave_id], TILE_DIM);
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);
    }

    // Store per-wave accumulator to LDS
    rocwmma::store_matrix_sync(smem.s_acc[wave_id], acc, TILE_DIM,
                               rocwmma::mem_row_major);
    __syncthreads();  // Both waves must finish before merge

    // Merge accumulators: all 64 threads cooperatively add s_acc[0] + s_acc[1]
    // and write to output
    f16_t* out = C + pid_k * stride_cp_split;

    for (int i = tid; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE_MOE)
    {
        int r = i / TILE_DIM;
        int c = i % TILE_DIM;
        int global_row = row_base + r;
        int global_col = pid_n * TILE_DIM + c;
        if (global_row < EM_max && global_col < N)
        {
            float val = smem.s_acc[0][i] + smem.s_acc[1][i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            out[global_row * N + global_col] = static_cast<f16_t>(val);
        }
    }
}


// =====================================================================
// Phase 3d: BLOCK_M=64 prefill kernel (4× M-tile reuse of dequanted B)
//
// For prefill (large M), each block processes 4 consecutive M-tiles
// (64 rows), dequanting each B tile once and reusing it for 4 A loads.
// This gives 4× less dequant work at the cost of 4 accumulators.
//
// At MoE expert boundaries within a block, B is re-dequanted for the
// new expert. Since tokens are sorted by expert, most blocks have all
// 4 sub-tiles from the same expert.
//
// Grid: (cdiv(num_m_blocks, 4), tiles_n, split_k)
//   where num_m_blocks = cdiv(EM_max, 16) — original 16-row blocks
// Block: 64 threads (2 waves)
// =====================================================================

#define M_FACTOR 2
#define BLOCK_M_PREFILL (TILE_DIM * M_FACTOR)  // 32

struct SharedMemMoE2W_M64 {
    int32_t  s_word_idx[256];                          // 1024B
    int32_t  s_next_word_idx[256];                     // 1024B
    int32_t  s_shift[256];                             // 1024B
    f16_t    s_B[NUM_WAVES_MOE][TILE_DIM * TILE_DIM];  // 1024B (shared B buffer, reused)
    float    s_acc[NUM_WAVES_MOE][TILE_DIM * TILE_DIM]; // 2048B (reused per M-sub merge)
};  // Total: ~6144 bytes (same as original!)


__global__ __launch_bounds__(BLOCK_SIZE_MOE)
void exl3_fused_moe_gemm_hip_2w_m64_kernel(
    const f16_t*    __restrict__ A,
    const int32_t*  __restrict__ B_stacked,
    f16_t*          __restrict__ C,
    const int32_t*  __restrict__ expert_ids,     // (num_m_blocks_16,) — per 16-row block
    const int32_t*  __restrict__ num_tokens_post_padded,
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int EM_max, int N, int K,
    int stride_be,
    int tiles_n,
    int WORDS_PER_TILE,
    int num_k_tiles,
    int split_k,
    int stride_cp_split,
    int num_m_blocks_16,        // total 16-row M-blocks (for bounds check)
    int cb)
{
    int pid_m64 = blockIdx.x;   // Index into 64-row super-blocks
    int pid_n   = blockIdx.y;
    int pid_k   = blockIdx.z;
    int tid     = threadIdx.x;
    int wave_id = tid / WARP_SIZE;   // 0 or 1
    int lane    = tid % WARP_SIZE;   // 0..31

    // Early return: check if ANY of our sub-tiles are valid
    int first_m_block = pid_m64 * M_FACTOR;
    if (first_m_block >= num_m_blocks_16) return;

    int num_valid = num_tokens_post_padded[0];
    int num_valid_m_blocks = (num_valid + TILE_DIM - 1) / TILE_DIM;

    __shared__ SharedMemMoE2W_M64 smem;

    // Load bit tables cooperatively
    for (int i = tid; i < 256; i += BLOCK_SIZE_MOE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    // Compute K-range for this grid split
    int tiles_per_grid_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_grid_start = pid_k * tiles_per_grid_split;
    int tk_grid_end   = min(tk_grid_start + tiles_per_grid_split, num_k_tiles);
    int grid_tiles    = tk_grid_end - tk_grid_start;

    // Split K-range between 2 waves within the block
    int tiles_per_wave = (grid_tiles + NUM_WAVES_MOE - 1) / NUM_WAVES_MOE;
    int tk_start = tk_grid_start + wave_id * tiles_per_wave;
    int tk_end   = min(tk_start + tiles_per_wave, tk_grid_end);

    // Preload bit tables into registers
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[i];
        reg_next_word_idx[j] = smem.s_next_word_idx[i];
        reg_shift[j]         = smem.s_shift[i];
    }

    // Precompute expert IDs and row bases for each sub-tile
    int sub_expert[M_FACTOR];
    int sub_row_base[M_FACTOR];
    int num_active_subs = 0;
    #pragma unroll
    for (int m = 0; m < M_FACTOR; m++)
    {
        int m_block = first_m_block + m;
        if (m_block < num_valid_m_blocks)
        {
            sub_expert[m] = expert_ids[m_block];
            sub_row_base[m] = m_block * TILE_DIM;
            if (sub_expert[m] >= 0)
                num_active_subs = m + 1;  // Track last active sub-tile
            else
                sub_expert[m] = -1;
        }
        else
        {
            sub_expert[m] = -1;
            sub_row_base[m] = 0;
        }
    }

    if (num_active_subs == 0) return;

    // Initialize accumulators (one per M-sub-tile)
    FragAcc acc[M_FACTOR];
    #pragma unroll
    for (int m = 0; m < M_FACTOR; m++)
        rocwmma::fill_fragment(acc[m], 0.0f);

    // K-loop: dequant B once, reuse for up to 4 A tiles
    for (int tk = tk_start; tk < tk_end; tk++)
    {
        int b_tile_offset = (tk * tiles_n + pid_n) * WORDS_PER_TILE;
        int last_expert = -1;

        FragB frag_b;  // Reused across M-subs with same expert

        #pragma unroll
        for (int m = 0; m < M_FACTOR; m++)
        {
            if (sub_expert[m] < 0) continue;

            // Dequant B only when expert changes
            if (sub_expert[m] != last_expert)
            {
                const int32_t* B_expert = B_stacked + sub_expert[m] * stride_be;
                dequant_tile_v5_cb(
                    B_expert + b_tile_offset,
                    reg_word_idx, reg_next_word_idx, reg_shift,
                    lane, smem.s_B[wave_id], cb);

                rocwmma::load_matrix_sync(frag_b, smem.s_B[wave_id], TILE_DIM);
                last_expert = sub_expert[m];
            }

            // Load A tile for this M-sub
            FragA frag_a;
            rocwmma::load_matrix_sync(frag_a,
                A + sub_row_base[m] * K + tk * TILE_DIM, K);

            // MMA: acc[m] += A @ B
            rocwmma::mma_sync(acc[m], frag_a, frag_b, acc[m]);
        }
    }

    // Merge and output: process each M-sub-tile sequentially
    // (reuse the same s_acc buffer to avoid 4× smem increase)
    f16_t* out = C + pid_k * stride_cp_split;

    for (int m = 0; m < M_FACTOR; m++)
    {
        if (sub_expert[m] < 0) continue;

        rocwmma::store_matrix_sync(smem.s_acc[wave_id], acc[m], TILE_DIM,
                                   rocwmma::mem_row_major);
        __syncthreads();

        for (int i = tid; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE_MOE)
        {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_row = sub_row_base[m] + r;
            int global_col = pid_n * TILE_DIM + c;
            if (global_row < EM_max && global_col < N)
            {
                float val = smem.s_acc[0][i] + smem.s_acc[1][i];
                val = fminf(fmaxf(val, -65504.0f), 65504.0f);
                out[global_row * N + global_col] = static_cast<f16_t>(val);
            }
        }
        __syncthreads();  // Ensure merge is done before next sub-tile overwrites s_acc
    }
}


// =====================================================================
// Phase 3c: Direct fragment fill kernel (no LDS for B)
//
// Same as 2-wave K-split kernel but bypasses LDS for B tile:
// dequant result goes directly into frag_b.x[] registers.
// Requires that the rocWMMA register layout matches our lane→element mapping:
//   lane L, iteration j → frag_b.x[j] = B[2*j + L/16][L%16]
//
// LDS only used for: bit tables (3KB) + acc merge (2KB) = 5KB
// Grid: (num_m_blocks, tiles_n, split_k)
// Block: 64 threads (2 waves)
// =====================================================================

struct SharedMemMoEDirect {
    int32_t  s_word_idx[256];                          // 1024B
    int32_t  s_next_word_idx[256];                     // 1024B
    int32_t  s_shift[256];                             // 1024B
    float    s_acc[NUM_WAVES_MOE][TILE_DIM * TILE_DIM]; // 2048B
};  // Total: ~5120 bytes (no B buffer!)


__global__ __launch_bounds__(BLOCK_SIZE_MOE)
void exl3_fused_moe_gemm_hip_direct_kernel(
    const f16_t*    __restrict__ A,
    const int32_t*  __restrict__ B_stacked,
    f16_t*          __restrict__ C,
    const int32_t*  __restrict__ expert_ids,
    const int32_t*  __restrict__ num_tokens_post_padded,
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int EM_max, int N, int K,
    int stride_be,
    int tiles_n,
    int WORDS_PER_TILE,
    int num_k_tiles,
    int split_k,
    int stride_cp_split,
    int cb)
{
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int pid_k = blockIdx.z;
    int tid   = threadIdx.x;
    int wave_id = tid / WARP_SIZE;
    int lane    = tid % WARP_SIZE;

    int num_valid = num_tokens_post_padded[0];
    int num_valid_m_blocks = (num_valid + TILE_DIM - 1) / TILE_DIM;
    if (pid_m >= num_valid_m_blocks) return;

    int off_expert = expert_ids[pid_m];
    if (off_expert < 0) return;

    __shared__ SharedMemMoEDirect smem;

    // Load bit tables cooperatively
    for (int i = tid; i < 256; i += BLOCK_SIZE_MOE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    // K-range: grid split then intra-block 2-wave split
    int tiles_per_grid_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_grid_start = pid_k * tiles_per_grid_split;
    int tk_grid_end   = min(tk_grid_start + tiles_per_grid_split, num_k_tiles);
    int grid_tiles    = tk_grid_end - tk_grid_start;

    int tiles_per_wave = (grid_tiles + NUM_WAVES_MOE - 1) / NUM_WAVES_MOE;
    int tk_start = tk_grid_start + wave_id * tiles_per_wave;
    int tk_end   = min(tk_start + tiles_per_wave, tk_grid_end);

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    int row_base = pid_m * TILE_DIM;
    const int32_t* B_expert = B_stacked + off_expert * stride_be;

    // Preload bit tables into registers (8 per lane)
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[i];
        reg_next_word_idx[j] = smem.s_next_word_idx[i];
        reg_shift[j]         = smem.s_shift[i];
    }

    for (int tk = tk_start; tk < tk_end; tk++)
    {
        // Load A tile from global
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A + row_base * K + tk * TILE_DIM, K);

        // V4 optimized dequant directly into fragment registers — no LDS!
        FragB frag_b;
        int b_base = (tk * tiles_n + pid_n) * WORDS_PER_TILE;
        dequant_frag_v5_cb(
            B_expert + b_base,
            reg_word_idx, reg_next_word_idx, reg_shift,
            frag_b, cb);

        rocwmma::mma_sync(acc, frag_a, frag_b, acc);
    }

    // Store + merge via LDS (same as K-split 2-wave)
    rocwmma::store_matrix_sync(smem.s_acc[wave_id], acc, TILE_DIM,
                               rocwmma::mem_row_major);
    __syncthreads();

    f16_t* out = C + pid_k * stride_cp_split;

    for (int i = tid; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE_MOE)
    {
        int r = i / TILE_DIM;
        int c = i % TILE_DIM;
        int global_row = row_base + r;
        int global_col = pid_n * TILE_DIM + c;
        if (global_row < EM_max && global_col < N)
        {
            float val = smem.s_acc[0][i] + smem.s_acc[1][i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            out[global_row * N + global_col] = static_cast<f16_t>(val);
        }
    }
}


// =====================================================================
// Phase 4: FP16 fused MoE GEMM kernel (pre-dequanted weights)
//
// Same structure as 2-wave dequant kernel but B is already FP16:
//   B_stacked_fp16: (E, K, N) row-major fp16
// No bit tables, no funnel shift, no cb0_decode — just load+MMA.
//
// Grid: (num_m_blocks, tiles_n, split_k)
// Block: 64 threads (2 waves)
// =====================================================================

struct SharedMemMoeFP16 {
    float    s_acc[NUM_WAVES_MOE][TILE_DIM * TILE_DIM]; // 2048B
};  // Total: ~2048 bytes (no bit tables, no B buffer!)


__global__ __launch_bounds__(BLOCK_SIZE_MOE)
void exl3_fused_moe_gemm_fp16_kernel(
    const f16_t*    __restrict__ A,            // (EM_max, K) contiguous
    const f16_t*    __restrict__ B_stacked,    // (E, K, N) row-major fp16
    f16_t*          __restrict__ C,            // output (or C_partial for split-K)
    const int32_t*  __restrict__ expert_ids,   // (num_m_blocks,) int32
    const int32_t*  __restrict__ num_tokens_post_padded, // (1,) tensor
    int EM_max, int N, int K,
    int stride_be,              // expert stride in B_stacked (fp16 elements) = K * N
    int tiles_n,
    int num_k_tiles,
    int split_k,
    int stride_cp_split)        // stride between split-K slices = EM_max * N
{
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int pid_k = blockIdx.z;
    int tid   = threadIdx.x;
    int wave_id = tid / WARP_SIZE;   // 0 or 1
    int lane    = tid % WARP_SIZE;   // 0..31

    // Early return: write zeros to prevent stale data in cached split-K buffers
    int num_valid = num_tokens_post_padded[0];
    int num_valid_m_blocks = (num_valid + TILE_DIM - 1) / TILE_DIM;
    if (pid_m >= num_valid_m_blocks) return;

    int off_expert = expert_ids[pid_m];
    if (off_expert < 0) return;

    __shared__ SharedMemMoeFP16 smem;

    // Compute K-range for this grid split
    int tiles_per_grid_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_grid_start = pid_k * tiles_per_grid_split;
    int tk_grid_end   = min(tk_grid_start + tiles_per_grid_split, num_k_tiles);
    int grid_tiles    = tk_grid_end - tk_grid_start;

    // Split K-range between 2 waves within the block
    int tiles_per_wave = (grid_tiles + NUM_WAVES_MOE - 1) / NUM_WAVES_MOE;
    int tk_start = tk_grid_start + wave_id * tiles_per_wave;
    int tk_end   = min(tk_start + tiles_per_wave, tk_grid_end);

    // Initialize accumulator
    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    int row_base = pid_m * TILE_DIM;

    // B offset for this expert: (E, K, N) row-major
    const f16_t* B_expert = B_stacked + off_expert * stride_be;

    // K-loop: each wave processes its own K-range independently
    for (int tk = tk_start; tk < tk_end; tk++)
    {
        // Load A tile from global
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A + row_base * K + tk * TILE_DIM, K);

        // Load B tile directly from global memory — already FP16!
        // B_expert is (K, N) row-major: tile at (tk*16, pid_n*16) with stride N
        FragB frag_b;
        rocwmma::load_matrix_sync(frag_b, B_expert + tk * TILE_DIM * N + pid_n * TILE_DIM, N);

        // MMA
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);
    }

    // Store + merge via LDS (same as dequant 2-wave kernel)
    rocwmma::store_matrix_sync(smem.s_acc[wave_id], acc, TILE_DIM,
                               rocwmma::mem_row_major);
    __syncthreads();  // Both waves must finish before merge

    f16_t* out = C + pid_k * stride_cp_split;

    for (int i = tid; i < TILE_DIM * TILE_DIM; i += BLOCK_SIZE_MOE)
    {
        int r = i / TILE_DIM;
        int c = i % TILE_DIM;
        int global_row = row_base + r;
        int global_col = pid_n * TILE_DIM + c;
        if (global_row < EM_max && global_col < N)
        {
            float val = smem.s_acc[0][i] + smem.s_acc[1][i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            out[global_row * N + global_col] = static_cast<f16_t>(val);
        }
    }
}


// =====================================================================
// Phase 3: Fused MoE reduction kernel
//
// Identical to dense reduce but uses MN = EM_max * N.
// Grid: (cdiv(EM_max * N, 256))
// Block: 256 threads
// =====================================================================

__global__ void exl3_fused_moe_reduce_kernel(
    const f16_t* __restrict__ C_partial,  // (split_k, EM_max, N) contiguous
    f16_t*       __restrict__ C,          // (EM_max, N) output
    int MN,                               // EM_max * N
    int split_k)
{
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= MN) return;

    float sum = 0.0f;
    for (int s = 0; s < split_k; s++)
    {
        f16_t val = C_partial[s * MN + idx];
        sum += static_cast<float>(val);
    }

    sum = fminf(fmaxf(sum, -65504.0f), 65504.0f);
    C[idx] = static_cast<f16_t>(sum);
}


// =====================================================================
// Phase 3: Host launcher for fused MoE GEMM
// =====================================================================

void hip_exl3_fused_moe_gemm(
    at::Tensor A,               // (EM_max, K) fp16
    at::Tensor B_stacked_i32,   // (E, tiles_k, tiles_n, WPT) int32
    at::Tensor C,               // (EM_max, N) fp16 — pre-zeroed by caller
    at::Tensor expert_ids,      // (num_m_blocks,) int32
    at::Tensor num_tokens_post_padded, // (1,) int32
    at::Tensor word_idx,        // (256,) int32
    at::Tensor next_word_idx,   // (256,) int32
    at::Tensor shift_tbl,       // (256,) int32
    int EM_max,
    int bits,
    int cb,
    int split_k,                // 0=auto, 1=no split, >1=explicit
    at::Tensor C_partial)       // (split_k, EM_max, N) fp16 — only used when split_k > 1
{
    TORCH_CHECK(A.is_contiguous(), "fused_moe_gemm: A must be contiguous");
    TORCH_CHECK(B_stacked_i32.is_contiguous(), "fused_moe_gemm: B_stacked must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "fused_moe_gemm: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "fused_moe_gemm: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "fused_moe_gemm: C must be float16");
    TORCH_CHECK(B_stacked_i32.dtype() == at::kInt, "fused_moe_gemm: B_stacked must be int32");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "fused_moe_gemm: expert_ids must be int32");
    TORCH_CHECK(num_tokens_post_padded.dtype() == at::kInt, "fused_moe_gemm: num_tokens_post_padded must be int32");
    TORCH_CHECK(cb == 0 || cb == 1, "fused_moe_gemm HIP: only cb=0,1 supported, got cb=", cb);

    int K = A.size(1);
    int N = C.size(1);
    int num_m_blocks = expert_ids.size(0);

    TORCH_CHECK(K % 16 == 0, "fused_moe_gemm: K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 16 == 0, "fused_moe_gemm: N must be divisible by 16, got ", N);

    int tiles_n = N / 16;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;

    // Expert stride in B_stacked: elements per expert (int32)
    int stride_be = B_stacked_i32.stride(0);

    // Auto split-K selection
    if (split_k == 0)
    {
        if (num_m_blocks <= 16 && num_k_tiles >= 16)
            split_k = min(8, num_k_tiles);
        else
            split_k = 1;
    }

    // Clamp split_k to not exceed k tiles
    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (split_k == 1)
    {
        // No split-K: 2-wave kernel writes directly to C
        dim3 grid(num_m_blocks, tiles_n, 1);
        dim3 block(BLOCK_SIZE_MOE);

        exl3_fused_moe_gemm_hip_2w_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            EM_max, N, K,
            stride_be, tiles_n, WORDS_PER_TILE,
            num_k_tiles,
            1,    // split_k
            0,    // stride_cp_split (unused when split_k==1)
            cb
        );
    }
    else
    {
        // Split-K: 2-wave kernel writes partials, then reduce
        TORCH_CHECK(C_partial.is_contiguous(), "fused_moe_gemm: C_partial must be contiguous");
        TORCH_CHECK(C_partial.dtype() == at::kHalf, "fused_moe_gemm: C_partial must be float16");

        int stride_cp_split = EM_max * N;

        dim3 grid(num_m_blocks, tiles_n, split_k);
        dim3 block(BLOCK_SIZE_MOE);

        exl3_fused_moe_gemm_hip_2w_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            EM_max, N, K,
            stride_be, tiles_n, WORDS_PER_TILE,
            num_k_tiles,
            split_k,
            stride_cp_split,
            cb
        );

        // Reduction kernel
        int MN = EM_max * N;
        int reduce_threads = 256;
        int reduce_blocks = (MN + reduce_threads - 1) / reduce_threads;

        exl3_fused_moe_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            reinterpret_cast<const f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            MN, split_k
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_moe_gemm kernel launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// Phase 3d: Host launcher for BLOCK_M=64 prefill fused MoE GEMM
// =====================================================================

void hip_exl3_fused_moe_gemm_m64(
    at::Tensor A,               // (EM_max, K) fp16
    at::Tensor B_stacked_i32,   // (E, tiles_k, tiles_n, WPT) int32
    at::Tensor C,               // (EM_max, N) fp16 — pre-zeroed by caller
    at::Tensor expert_ids,      // (num_m_blocks_16,) int32 — per 16-row block
    at::Tensor num_tokens_post_padded, // (1,) int32
    at::Tensor word_idx,        // (256,) int32
    at::Tensor next_word_idx,   // (256,) int32
    at::Tensor shift_tbl,       // (256,) int32
    int EM_max,
    int bits,
    int cb,
    int split_k,                // 0=auto, 1=no split, >1=explicit
    at::Tensor C_partial)       // (split_k, EM_max, N) fp16
{
    TORCH_CHECK(A.is_contiguous(), "fused_moe_gemm_m64: A must be contiguous");
    TORCH_CHECK(B_stacked_i32.is_contiguous(), "fused_moe_gemm_m64: B_stacked must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "fused_moe_gemm_m64: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "fused_moe_gemm_m64: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "fused_moe_gemm_m64: C must be float16");
    TORCH_CHECK(B_stacked_i32.dtype() == at::kInt, "fused_moe_gemm_m64: B_stacked must be int32");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "fused_moe_gemm_m64: expert_ids must be int32");
    TORCH_CHECK(cb == 0 || cb == 1, "fused_moe_gemm_m64 HIP: only cb=0,1 supported, got cb=", cb);

    int K = A.size(1);
    int N = C.size(1);
    int num_m_blocks_16 = expert_ids.size(0);  // Original 16-row blocks

    TORCH_CHECK(K % 16 == 0, "fused_moe_gemm_m64: K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 16 == 0, "fused_moe_gemm_m64: N must be divisible by 16, got ", N);

    int tiles_n = N / 16;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;
    int stride_be = B_stacked_i32.stride(0);

    // Grid M-dimension: 64-row super-blocks
    int num_m_blocks_64 = (num_m_blocks_16 + M_FACTOR - 1) / M_FACTOR;

    // Auto split-K: for prefill (large M), usually split_k=1 is best
    if (split_k == 0)
    {
        if (num_m_blocks_64 <= 4 && num_k_tiles >= 16)
            split_k = min(4, num_k_tiles);
        else
            split_k = 1;
    }
    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (split_k == 1)
    {
        dim3 grid(num_m_blocks_64, tiles_n, 1);
        dim3 block(BLOCK_SIZE_MOE);

        exl3_fused_moe_gemm_hip_2w_m64_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            EM_max, N, K,
            stride_be, tiles_n, WORDS_PER_TILE,
            num_k_tiles,
            1, 0,  // split_k=1, stride_cp_split unused
            num_m_blocks_16,
            cb
        );
    }
    else
    {
        TORCH_CHECK(C_partial.is_contiguous(), "fused_moe_gemm_m64: C_partial must be contiguous");
        TORCH_CHECK(C_partial.dtype() == at::kHalf, "fused_moe_gemm_m64: C_partial must be float16");

        int stride_cp_split = EM_max * N;

        dim3 grid(num_m_blocks_64, tiles_n, split_k);
        dim3 block(BLOCK_SIZE_MOE);

        exl3_fused_moe_gemm_hip_2w_m64_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            EM_max, N, K,
            stride_be, tiles_n, WORDS_PER_TILE,
            num_k_tiles,
            split_k, stride_cp_split,
            num_m_blocks_16,
            cb
        );

        // Reduction kernel (same as original)
        int MN = EM_max * N;
        int reduce_threads = 256;
        int reduce_blocks = (MN + reduce_threads - 1) / reduce_threads;

        exl3_fused_moe_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            reinterpret_cast<const f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            MN, split_k
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_moe_gemm_m64 kernel launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// Phase 4: Host launcher for FP16 fused MoE GEMM
// =====================================================================

void hip_exl3_fused_moe_gemm_fp16(
    at::Tensor A,               // (EM_max, K) fp16
    at::Tensor B_stacked_fp16,  // (E, K, N) fp16 row-major
    at::Tensor C,               // (EM_max, N) fp16 — pre-zeroed by caller
    at::Tensor expert_ids,      // (num_m_blocks,) int32
    at::Tensor num_tokens_post_padded, // (1,) int32
    int EM_max,
    int split_k,                // 0=auto, 1=no split, >1=explicit
    at::Tensor C_partial)       // (split_k, EM_max, N) fp16 — only used when split_k > 1
{
    TORCH_CHECK(A.is_contiguous(), "fused_moe_gemm_fp16: A must be contiguous");
    TORCH_CHECK(B_stacked_fp16.is_contiguous(), "fused_moe_gemm_fp16: B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "fused_moe_gemm_fp16: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "fused_moe_gemm_fp16: A must be float16");
    TORCH_CHECK(B_stacked_fp16.dtype() == at::kHalf, "fused_moe_gemm_fp16: B must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "fused_moe_gemm_fp16: C must be float16");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "fused_moe_gemm_fp16: expert_ids must be int32");
    TORCH_CHECK(num_tokens_post_padded.dtype() == at::kInt, "fused_moe_gemm_fp16: num_tokens_post_padded must be int32");

    int K = A.size(1);
    int N = C.size(1);
    int num_m_blocks = expert_ids.size(0);

    TORCH_CHECK(K % 16 == 0, "fused_moe_gemm_fp16: K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 16 == 0, "fused_moe_gemm_fp16: N must be divisible by 16, got ", N);

    int tiles_n = N / 16;
    int num_k_tiles = K / 16;

    // Expert stride in B_stacked: elements per expert (fp16) = K * N
    int stride_be = B_stacked_fp16.stride(0);

    // Auto split-K selection
    if (split_k == 0)
    {
        if (num_m_blocks <= 16 && num_k_tiles >= 16)
            split_k = min(8, num_k_tiles);
        else
            split_k = 1;
    }

    // Clamp split_k to not exceed k tiles
    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (split_k == 1)
    {
        dim3 grid(num_m_blocks, tiles_n, 1);
        dim3 block(BLOCK_SIZE_MOE);

        exl3_fused_moe_gemm_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const f16_t*>(B_stacked_fp16.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            EM_max, N, K,
            stride_be, tiles_n, num_k_tiles,
            1,    // split_k
            0     // stride_cp_split (unused when split_k==1)
        );
    }
    else
    {
        TORCH_CHECK(C_partial.is_contiguous(), "fused_moe_gemm_fp16: C_partial must be contiguous");
        TORCH_CHECK(C_partial.dtype() == at::kHalf, "fused_moe_gemm_fp16: C_partial must be float16");

        int stride_cp_split = EM_max * N;

        dim3 grid(num_m_blocks, tiles_n, split_k);
        dim3 block(BLOCK_SIZE_MOE);

        exl3_fused_moe_gemm_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const f16_t*>(B_stacked_fp16.data_ptr()),
            reinterpret_cast<f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            EM_max, N, K,
            stride_be, tiles_n, num_k_tiles,
            split_k,
            stride_cp_split
        );

        // Reduction kernel (reuse existing fused MoE reduce)
        int MN = EM_max * N;
        int reduce_threads = 256;
        int reduce_blocks = (MN + reduce_threads - 1) / reduce_threads;

        exl3_fused_moe_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            reinterpret_cast<const f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            MN, split_k
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_moe_gemm_fp16 kernel launch failed: ", cudaGetErrorString(err));
}



// =====================================================================
// Phase 6: Fused MoE GEMM + Hadamard-128 kernel
//
// Each block computes (BLOCK_M=16, 128) output tile (was 16×16).
// Grid: (num_m_blocks, N/128, split_k)
// Block: 32 threads (1 wave)
//
// The K-loop dequants 8 B tiles (each 16×16) per K iteration,
// accumulating into 8 FragAcc accumulators. After the K-loop,
// a Had-128 epilogue applies H_128 = H_8 ⊗ H_16:
//   Step 1: H_16 via WMMA (acc[g] × H_16 for each of 8 groups)
//   Step 2: H_8 butterfly across the 8 groups via LDS
//   Step 3: SVH scale + global store
//
// This eliminates the intermediate VRAM round-trip between GEMM and Had.
// =====================================================================

// Number of N-tile groups per block (Had-128 = 8 groups of 16)
#define HAD_GROUPS 8
#define HAD_N_PER_BLOCK (HAD_GROUPS * TILE_DIM)  // 128

struct SharedMemMoEHad {
    int32_t  s_word_idx[256];      // 1024 bytes
    int32_t  s_next_word_idx[256]; // 1024 bytes
    int32_t  s_shift[256];         // 1024 bytes
    f16_t    s_B[TILE_DIM * TILE_DIM];   // 512 bytes — dequant B tile buffer
    float    s_had[HAD_GROUPS][TILE_DIM * TILE_DIM]; // 8192 bytes — Had epilogue
    f16_t    s_h16[TILE_DIM * TILE_DIM]; // 512 bytes — H_16 matrix
    f16_t    s_tmp[TILE_DIM * TILE_DIM]; // 512 bytes — fp32→fp16 conversion
};  // Total: ~12.8KB


__global__ __launch_bounds__(WARP_SIZE)
void exl3_fused_moe_gemm_had_hip_kernel(
    const f16_t*    __restrict__ A,           // (EM_max, K) contiguous
    const int32_t*  __restrict__ B_stacked,   // (E, tiles_k, tiles_n, WPT) int32
    f16_t*          __restrict__ C,           // output (EM_max, N) or partial
    const int32_t*  __restrict__ expert_ids,  // (num_m_blocks,) int32
    const int32_t*  __restrict__ num_tokens_post_padded, // (1,) tensor
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    const f16_t*    __restrict__ H16,         // (16, 16) fp16 Hadamard matrix
    const f16_t*    __restrict__ svh,         // (E, N) fp16 per-expert scale
    int EM_max, int N, int K,
    int stride_be,              // expert stride in B_stacked (int32 elements)
    int tiles_n,                // N / 16
    int WORDS_PER_TILE,
    int num_k_tiles,
    int has_svh,                // 1 if svh is provided, 0 otherwise
    int stride_cp_split,        // stride between split-K slices = EM_max * N
    int cb)
{
    int pid_m = blockIdx.x;
    int pid_n128 = blockIdx.y;  // N/128 block index
    int pid_k = blockIdx.z;
    int lane  = threadIdx.x;

    // Early return: write zeros to prevent stale data in cached split-K buffers
    int num_valid = num_tokens_post_padded[0];
    int num_valid_m_blocks = (num_valid + TILE_DIM - 1) / TILE_DIM;
    if (pid_m >= num_valid_m_blocks) return;

    // Load expert ID for this M-block; skip remote experts (id < 0)
    int off_expert = expert_ids[pid_m];
    if (off_expert < 0) return;

    __shared__ SharedMemMoEHad smem;

    // Load bit extraction tables into LDS
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }

    // Load H_16 matrix into LDS (256 fp16 elements)
    for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
    {
        smem.s_h16[i] = H16[i];
    }
    __syncthreads();

    // Preload bit tables into registers (avoids LDS reads in K-loop)
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[i];
        reg_next_word_idx[j] = smem.s_next_word_idx[i];
        reg_shift[j]         = smem.s_shift[i];
    }

    // Compute K-range for split-K (gridDim.z == 1 for non-split)
    int split_k = gridDim.z;
    int tiles_per_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_start = pid_k * tiles_per_split;
    int tk_end   = min(tk_start + tiles_per_split, num_k_tiles);

    // Initialize 8 accumulators (one per 16-column group)
    FragAcc acc[HAD_GROUPS];
    #pragma unroll
    for (int g = 0; g < HAD_GROUPS; g++)
        rocwmma::fill_fragment(acc[g], 0.0f);

    int row_base = pid_m * TILE_DIM;
    int base_n_tile = pid_n128 * HAD_GROUPS;  // first N-tile index for this block

    // B offset for this expert
    const int32_t* B_expert = B_stacked + off_expert * stride_be;

    // =================================================================
    // K-loop: 8 sequential B-tile dequants per K iteration
    // =================================================================
    for (int tk = tk_start; tk < tk_end; tk++)
    {
        // Load A tile from global memory (same for all 8 B groups)
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A + row_base * K + tk * TILE_DIM, K);

        // Process 8 B tile groups
        #pragma unroll
        for (int g = 0; g < HAD_GROUPS; g++)
        {
            // Dequant B tile for group g into LDS
            int b_base = (tk * tiles_n + base_n_tile + g) * WORDS_PER_TILE;
            dequant_tile_v5_cb(
                B_expert + b_base,
                reg_word_idx, reg_next_word_idx, reg_shift,
                lane, smem.s_B, cb);

            __syncthreads();

            // WMMA: acc[g] += A_tile @ B_tile
            FragB frag_b;
            rocwmma::load_matrix_sync(frag_b, smem.s_B, TILE_DIM);
            rocwmma::mma_sync(acc[g], frag_a, frag_b, acc[g]);

            __syncthreads();
        }
    }

    // =================================================================
    // Had-128 epilogue (only when split_k == 1 or last split)
    // For split_k > 1: handled by separate reduce+Had kernel
    // =================================================================
    if (split_k > 1)
    {
        // Split-K partial: store raw accumulators without Had
        f16_t* out = C + pid_k * stride_cp_split;
        for (int g = 0; g < HAD_GROUPS; g++)
        {
            rocwmma::store_matrix_sync(smem.s_had[0], acc[g], TILE_DIM,
                                       rocwmma::mem_row_major);
            __syncthreads();

            for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
            {
                int r = i / TILE_DIM;
                int c = i % TILE_DIM;
                int global_row = row_base + r;
                int global_col = (base_n_tile + g) * TILE_DIM + c;
                if (global_row < EM_max && global_col < N)
                {
                    float val = smem.s_had[0][i];
                    val = fminf(fmaxf(val, -65504.0f), 65504.0f);
                    out[global_row * N + global_col] = static_cast<f16_t>(val);
                }
            }
            __syncthreads();
        }
        return;
    }

    // -----------------------------------------------------------------
    // Step 1: H_16 via WMMA — acc[g] = acc[g] @ H_16 for each group
    // -----------------------------------------------------------------
    // Load H_16 as FragB (stays the same for all groups)
    FragB frag_h16;
    rocwmma::load_matrix_sync(frag_h16, smem.s_h16, TILE_DIM);

    #pragma unroll
    for (int g = 0; g < HAD_GROUPS; g++)
    {
        // Store acc[g] (fp32) to LDS, convert to fp16
        rocwmma::store_matrix_sync(smem.s_had[0], acc[g], TILE_DIM,
                                   rocwmma::mem_row_major);
        __syncthreads();

        // Convert fp32 → fp16 into s_tmp
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            float val = smem.s_had[0][i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            smem.s_tmp[i] = static_cast<f16_t>(val);
        }
        __syncthreads();

        // Load as FragA and multiply by H_16
        FragA frag_acc_f16;
        rocwmma::load_matrix_sync(frag_acc_f16, smem.s_tmp, TILE_DIM);

        // Reset accumulator and compute acc[g] = frag_acc_f16 @ H_16
        rocwmma::fill_fragment(acc[g], 0.0f);
        rocwmma::mma_sync(acc[g], frag_acc_f16, frag_h16, acc[g]);
    }

    // -----------------------------------------------------------------
    // Step 2: Store all 8 post-H_16 accumulators to LDS for H_8 butterfly
    // -----------------------------------------------------------------
    #pragma unroll
    for (int g = 0; g < HAD_GROUPS; g++)
    {
        rocwmma::store_matrix_sync(smem.s_had[g], acc[g], TILE_DIM,
                                   rocwmma::mem_row_major);
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // Step 3: H_8 butterfly across 8 groups (through LDS)
    // Each thread reads same position from all 8 groups, butterflies,
    // writes back.
    // -----------------------------------------------------------------
    // 256 elements per group, 32 threads → 8 elements per thread
    for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
    {
        float v[HAD_GROUPS];
        #pragma unroll
        for (int g = 0; g < HAD_GROUPS; g++)
            v[g] = smem.s_had[g][i];

        // 3-round H_8 butterfly (in registers, no cross-lane needed)
        // Round 1: pairs (0,1), (2,3), (4,5), (6,7)
        float t0 = v[0] + v[1], t1 = v[0] - v[1];
        float t2 = v[2] + v[3], t3 = v[2] - v[3];
        float t4 = v[4] + v[5], t5 = v[4] - v[5];
        float t6 = v[6] + v[7], t7 = v[6] - v[7];

        // Round 2: pairs (0,2), (1,3), (4,6), (5,7)
        float s0 = t0 + t2, s1 = t1 + t3;
        float s2 = t0 - t2, s3 = t1 - t3;
        float s4 = t4 + t6, s5 = t5 + t7;
        float s6 = t4 - t6, s7 = t5 - t7;

        // Round 3: pairs (0,4), (1,5), (2,6), (3,7)
        v[0] = s0 + s4; v[1] = s1 + s5;
        v[2] = s2 + s6; v[3] = s3 + s7;
        v[4] = s0 - s4; v[5] = s1 - s5;
        v[6] = s2 - s6; v[7] = s3 - s7;

        // Scale: H_16 already has 1/√16 baked in, so we only need 1/√8
        // 1/√8 = 0.35355339f
        #pragma unroll
        for (int g = 0; g < HAD_GROUPS; g++)
            smem.s_had[g][i] = v[g] * 0.35355339f;
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // Step 4: SVH scale + store to global memory
    // -----------------------------------------------------------------
    f16_t* out = C;  // split_k==1 writes directly to C

    for (int g = 0; g < HAD_GROUPS; g++)
    {
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_row = row_base + r;
            int global_col = pid_n128 * HAD_N_PER_BLOCK + g * TILE_DIM + c;

            if (global_row < EM_max && global_col < N)
            {
                float val = smem.s_had[g][i];
                if (has_svh)
                    val *= static_cast<float>(svh[off_expert * N + global_col]);
                val = fminf(fmaxf(val, -65504.0f), 65504.0f);
                out[global_row * N + global_col] = static_cast<f16_t>(val);
            }
        }
    }
}


// =====================================================================
// Phase 6: Reduce + Hadamard-128 kernel for split-K partial sums
//
// Used when split_k > 1: sums partial results, then applies Had-128 + SVH.
// Grid: (num_m_blocks, N/128)
// Block: 32 threads (1 wave)
// =====================================================================

struct SharedMemReduceHad {
    float    s_had[HAD_GROUPS][TILE_DIM * TILE_DIM]; // 8192 bytes
    f16_t    s_h16[TILE_DIM * TILE_DIM];             // 512 bytes
    f16_t    s_tmp[TILE_DIM * TILE_DIM];             // 512 bytes
};

__global__ __launch_bounds__(WARP_SIZE)
void exl3_fused_moe_reduce_had_kernel(
    const f16_t*    __restrict__ C_partial,   // (split_k, EM_max, N)
    f16_t*          __restrict__ C,           // (EM_max, N)
    const int32_t*  __restrict__ expert_ids,  // (num_m_blocks,) for SVH lookup
    const f16_t*    __restrict__ H16,         // (16, 16) fp16
    const f16_t*    __restrict__ svh,         // (E, N) fp16
    int EM_max, int N,
    int split_k,
    int has_svh)
{
    int pid_m = blockIdx.x;
    int pid_n128 = blockIdx.y;
    int lane = threadIdx.x;

    int row_base = pid_m * TILE_DIM;
    int base_n128 = pid_n128 * HAD_N_PER_BLOCK;

    // Look up expert ID for this M-block; write zeros for inactive blocks
    int off_expert = expert_ids[pid_m];
    if (off_expert < 0) return;

    __shared__ SharedMemReduceHad smem;

    // Load H_16 matrix into LDS
    for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        smem.s_h16[i] = H16[i];
    __syncthreads();

    // Sum split-K partials for each group into s_had
    for (int g = 0; g < HAD_GROUPS; g++)
    {
        // Initialize from first split
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_row = row_base + r;
            int global_col = base_n128 + g * TILE_DIM + c;

            float sum = 0.0f;
            if (global_row < EM_max && global_col < N)
            {
                for (int s = 0; s < split_k; s++)
                    sum += static_cast<float>(
                        C_partial[s * EM_max * N + global_row * N + global_col]);
            }
            smem.s_had[g][i] = sum;
        }
    }
    __syncthreads();

    // H_16 via WMMA for each group
    FragB frag_h16;
    rocwmma::load_matrix_sync(frag_h16, smem.s_h16, TILE_DIM);

    for (int g = 0; g < HAD_GROUPS; g++)
    {
        // Convert fp32 → fp16
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            float val = smem.s_had[g][i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            smem.s_tmp[i] = static_cast<f16_t>(val);
        }
        __syncthreads();

        FragA frag_acc_f16;
        rocwmma::load_matrix_sync(frag_acc_f16, smem.s_tmp, TILE_DIM);

        FragAcc acc_g;
        rocwmma::fill_fragment(acc_g, 0.0f);
        rocwmma::mma_sync(acc_g, frag_acc_f16, frag_h16, acc_g);

        rocwmma::store_matrix_sync(smem.s_had[g], acc_g, TILE_DIM,
                                   rocwmma::mem_row_major);
        __syncthreads();
    }

    // H_8 butterfly across 8 groups
    for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
    {
        float v[HAD_GROUPS];
        #pragma unroll
        for (int g = 0; g < HAD_GROUPS; g++)
            v[g] = smem.s_had[g][i];

        float t0 = v[0] + v[1], t1 = v[0] - v[1];
        float t2 = v[2] + v[3], t3 = v[2] - v[3];
        float t4 = v[4] + v[5], t5 = v[4] - v[5];
        float t6 = v[6] + v[7], t7 = v[6] - v[7];

        float s0 = t0 + t2, s1 = t1 + t3;
        float s2 = t0 - t2, s3 = t1 - t3;
        float s4 = t4 + t6, s5 = t5 + t7;
        float s6 = t4 - t6, s7 = t5 - t7;

        v[0] = s0 + s4; v[1] = s1 + s5;
        v[2] = s2 + s6; v[3] = s3 + s7;
        v[4] = s0 - s4; v[5] = s1 - s5;
        v[6] = s2 - s6; v[7] = s3 - s7;

        #pragma unroll
        for (int g = 0; g < HAD_GROUPS; g++)
            smem.s_had[g][i] = v[g] * 0.35355339f;
    }
    __syncthreads();

    // SVH scale + store to global
    for (int g = 0; g < HAD_GROUPS; g++)
    {
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM;
            int c = i % TILE_DIM;
            int global_row = row_base + r;
            int global_col = base_n128 + g * TILE_DIM + c;

            if (global_row < EM_max && global_col < N)
            {
                float val = smem.s_had[g][i];
                if (has_svh)
                    val *= static_cast<float>(svh[off_expert * N + global_col]);
                val = fminf(fmaxf(val, -65504.0f), 65504.0f);
                C[global_row * N + global_col] = static_cast<f16_t>(val);
            }
        }
    }
}


// =====================================================================
// Phase 6: Host launcher for fused MoE GEMM + Hadamard-128
// =====================================================================

void hip_exl3_fused_moe_gemm_had(
    at::Tensor A,               // (EM_max, K) fp16
    at::Tensor B_stacked_i32,   // (E, tiles_k, tiles_n, WPT) int32
    at::Tensor C,               // (EM_max, N) fp16 — pre-zeroed by caller
    at::Tensor expert_ids,      // (num_m_blocks,) int32
    at::Tensor num_tokens_post_padded, // (1,) int32
    at::Tensor word_idx,        // (256,) int32
    at::Tensor next_word_idx,   // (256,) int32
    at::Tensor shift_tbl,       // (256,) int32
    at::Tensor H16,             // (16, 16) fp16 — precomputed normalized Hadamard
    at::Tensor svh,             // (E, N) fp16 per-expert sign-flip scale
    int EM_max,
    int bits,
    int cb,
    int split_k,                // 0=auto, 1=no split, >1=explicit
    int has_svh,
    at::Tensor C_partial)       // (split_k, EM_max, N) fp16 — only used when split_k > 1
{
    TORCH_CHECK(A.is_contiguous(), "fused_moe_gemm_had: A must be contiguous");
    TORCH_CHECK(B_stacked_i32.is_contiguous(), "fused_moe_gemm_had: B_stacked must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "fused_moe_gemm_had: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "fused_moe_gemm_had: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "fused_moe_gemm_had: C must be float16");
    TORCH_CHECK(B_stacked_i32.dtype() == at::kInt, "fused_moe_gemm_had: B_stacked must be int32");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "fused_moe_gemm_had: expert_ids must be int32");
    TORCH_CHECK(H16.dtype() == at::kHalf, "fused_moe_gemm_had: H16 must be float16");
    TORCH_CHECK(cb == 0 || cb == 1, "fused_moe_gemm_had HIP: only cb=0,1 supported, got cb=", cb);

    int K = A.size(1);
    int N = C.size(1);
    int num_m_blocks = expert_ids.size(0);

    TORCH_CHECK(K % 16 == 0, "fused_moe_gemm_had: K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 128 == 0, "fused_moe_gemm_had: N must be divisible by 128, got ", N);

    int tiles_n = N / 16;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;
    int stride_be = B_stacked_i32.stride(0);

    // Auto split-K selection
    if (split_k == 0)
    {
        if (num_m_blocks <= 16 && num_k_tiles >= 16)
            split_k = min(8, num_k_tiles);
        else
            split_k = 1;
    }
    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int n128_blocks = N / 128;

    if (split_k == 1)
    {
        dim3 grid(num_m_blocks, n128_blocks, 1);
        dim3 block(WARP_SIZE);

        exl3_fused_moe_gemm_had_hip_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            reinterpret_cast<const f16_t*>(H16.data_ptr()),
            has_svh ? reinterpret_cast<const f16_t*>(svh.data_ptr()) : nullptr,
            EM_max, N, K,
            stride_be, tiles_n, WORDS_PER_TILE,
            num_k_tiles,
            has_svh,
            0,  // stride_cp_split (unused when split_k==1)
            cb
        );
    }
    else
    {
        // Split-K: GEMM writes partials (no Had), then reduce+Had kernel
        TORCH_CHECK(C_partial.is_contiguous(), "fused_moe_gemm_had: C_partial must be contiguous");
        TORCH_CHECK(C_partial.dtype() == at::kHalf, "fused_moe_gemm_had: C_partial must be float16");

        int stride_cp_split = EM_max * N;

        dim3 grid(num_m_blocks, n128_blocks, split_k);
        dim3 block(WARP_SIZE);

        exl3_fused_moe_gemm_had_hip_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            reinterpret_cast<const f16_t*>(H16.data_ptr()),
            has_svh ? reinterpret_cast<const f16_t*>(svh.data_ptr()) : nullptr,
            EM_max, N, K,
            stride_be, tiles_n, WORDS_PER_TILE,
            num_k_tiles,
            has_svh,
            stride_cp_split,
            cb
        );

        // Reduce + Had-128 kernel
        dim3 reduce_grid(num_m_blocks, n128_blocks);
        dim3 reduce_block(WARP_SIZE);

        exl3_fused_moe_reduce_had_kernel<<<reduce_grid, reduce_block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
            reinterpret_cast<const f16_t*>(H16.data_ptr()),
            has_svh ? reinterpret_cast<const f16_t*>(svh.data_ptr()) : nullptr,
            EM_max, N,
            split_k,
            has_svh
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_moe_gemm_had kernel launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// rocWMMA Microbenchmark Kernels
//
// Each kernel isolates a single operation, run N_ITER times in a loop.
// rocprofv3 kernel trace gives per-kernel GPU time → divide by N_ITER
// to get per-operation cost.
//
// All kernels: 1 block, 32 threads (1 wave) — matches our GEMM kernel.
// LDS layout matches our production kernel.
// =====================================================================

#define BENCH_ITERS 4096

// Shared memory for benchmarks
struct SharedMemBench {
    f16_t    s_A[TILE_DIM * TILE_DIM];     // 512B
    f16_t    s_B[TILE_DIM * TILE_DIM];     // 512B
    float    s_acc[TILE_DIM * TILE_DIM];   // 1024B
    int32_t  s_word_idx[256];              // 1024B
    int32_t  s_next_word_idx[256];         // 1024B
    int32_t  s_shift[256];                 // 1024B
};

// --- 1. mma_sync only (WMMA instruction) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_mma_sync(f16_t* __restrict__ buf_f16, float* __restrict__ buf_f32)
{
    __shared__ SharedMemBench smem;
    int lane = threadIdx.x;

    // Init LDS with something to load
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        smem.s_A[i] = static_cast<f16_t>(0.001f * i);
        smem.s_B[i] = static_cast<f16_t>(0.001f * i);
    }
    __syncthreads();

    FragA frag_a;
    FragB frag_b;
    FragAcc acc;
    rocwmma::load_matrix_sync(frag_a, smem.s_A, TILE_DIM);
    rocwmma::load_matrix_sync(frag_b, smem.s_B, TILE_DIM);
    rocwmma::fill_fragment(acc, 0.0f);

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);
    }

    // Prevent dead-code elimination
    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}

// --- 2. load_matrix_sync from global memory (A-tile pattern) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_load_global(const f16_t* __restrict__ A, float* __restrict__ buf_f32, int K)
{
    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A, K);
        // Accumulate to prevent DCE
        for (int i = 0; i < frag_a.num_elements; i++)
            acc.x[i] += static_cast<float>(frag_a.x[i]);
    }

    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}

// --- 3. load_matrix_sync from LDS ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_load_lds(f16_t* __restrict__ buf_f16, float* __restrict__ buf_f32)
{
    __shared__ f16_t s_tile[TILE_DIM * TILE_DIM];
    int lane = threadIdx.x;

    for (int i = lane; i < 256; i += WARP_SIZE)
        s_tile[i] = static_cast<f16_t>(0.001f * i);
    __syncthreads();

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        FragB frag_b;
        rocwmma::load_matrix_sync(frag_b, s_tile, TILE_DIM);
        for (int i = 0; i < frag_b.num_elements; i++)
            acc.x[i] += static_cast<float>(frag_b.x[i]);
    }

    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}

// --- 4. store_matrix_sync to LDS (acc → LDS float) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_store_lds(f16_t* __restrict__ buf_f16, float* __restrict__ buf_f32)
{
    __shared__ float s_acc[TILE_DIM * TILE_DIM];

    FragAcc acc;
    rocwmma::fill_fragment(acc, 1.0f);

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        rocwmma::store_matrix_sync(s_acc, acc, TILE_DIM, rocwmma::mem_row_major);
        // Touch LDS to prevent DCE
        acc.x[0] += s_acc[threadIdx.x] * 0.0001f;
    }

    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}

// --- 5. store_matrix_sync to global (acc → global f16) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_store_global(f16_t* __restrict__ buf_f16, float* __restrict__ buf_f32)
{
    // We can't store float acc directly to f16.
    // This benchmarks: store_matrix_sync(LDS) + manual convert + global write
    // which is our actual output path.
    __shared__ float s_acc[TILE_DIM * TILE_DIM];
    int lane = threadIdx.x;

    FragAcc acc;
    rocwmma::fill_fragment(acc, 1.0f);

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        rocwmma::store_matrix_sync(s_acc, acc, TILE_DIM, rocwmma::mem_row_major);
        for (int i = lane; i < 256; i += WARP_SIZE)
        {
            float val = s_acc[i];
            val = fminf(fmaxf(val, -65504.0f), 65504.0f);
            buf_f16[i] = static_cast<f16_t>(val);
        }
        // Touch output to prevent DCE
        acc.x[0] += static_cast<float>(buf_f16[lane]) * 0.0001f;
    }

    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}

// --- 6. fill_fragment (zero init) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_fill_fragment(float* __restrict__ buf_f32)
{
    FragAcc acc;

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        rocwmma::fill_fragment(acc, 0.0f);
        // Touch to prevent DCE
        acc.x[0] += 0.0001f;
    }

    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}

// --- 7. Dequant loop only (8 iters: funnel shift + cb0_decode → LDS) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_dequant_to_lds(
    const int32_t* __restrict__ B,
    const int32_t* __restrict__ word_idx,
    const int32_t* __restrict__ next_word_idx,
    const int32_t* __restrict__ shift_tbl,
    f16_t* __restrict__ buf_f16,
    int WORDS_PER_TILE)
{
    __shared__ SharedMemBench smem;
    int lane = threadIdx.x;

    // Load bit tables
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    float sink = 0.0f;

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        for (int i = lane; i < 256; i += WARP_SIZE)
        {
            int lo_idx  = smem.s_word_idx[i];
            int hi_idx  = smem.s_next_word_idx[i];
            int shift   = smem.s_shift[i];

            uint32_t lo_word = static_cast<uint32_t>(B[lo_idx]);
            uint32_t hi_word = static_cast<uint32_t>(B[hi_idx]);

            uint32_t index;
            if (shift > 0)
            {
                int shift_hi = (32 - shift) & 31;
                uint32_t lo_part = (lo_word >> shift) & (((1u << shift_hi) - 1u) & 0xFFFFu);
                uint32_t hi_part = (hi_word << shift_hi) & 0xFFFFu;
                index = lo_part | hi_part;
            }
            else
            {
                index = lo_word & 0xFFFFu;
            }

            smem.s_B[i] = cb0_decode(index);
        }
        __syncthreads();
        sink += static_cast<float>(smem.s_B[lane]);
    }

    buf_f16[lane] = static_cast<f16_t>(sink);
}

// --- 8. Dequant with register preload (no LDS reads for tables in loop) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_dequant_regpreload(
    const int32_t* __restrict__ B,
    const int32_t* __restrict__ word_idx,
    const int32_t* __restrict__ next_word_idx,
    const int32_t* __restrict__ shift_tbl,
    f16_t* __restrict__ buf_f16,
    int WORDS_PER_TILE)
{
    __shared__ f16_t s_B[TILE_DIM * TILE_DIM];
    int lane = threadIdx.x;

    // Preload bit tables into registers
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = word_idx[i];
        reg_next_word_idx[j] = next_word_idx[i];
        reg_shift[j]         = shift_tbl[i];
    }

    float sink = 0.0f;

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            int lo_idx = reg_word_idx[j];
            int hi_idx = reg_next_word_idx[j];
            int shift  = reg_shift[j];

            uint32_t lo_word = static_cast<uint32_t>(B[lo_idx]);
            uint32_t hi_word = static_cast<uint32_t>(B[hi_idx]);

            uint32_t index;
            if (shift > 0)
            {
                int shift_hi = (32 - shift) & 31;
                uint32_t lo_part = (lo_word >> shift) & (((1u << shift_hi) - 1u) & 0xFFFFu);
                uint32_t hi_part = (hi_word << shift_hi) & 0xFFFFu;
                index = lo_part | hi_part;
            }
            else
            {
                index = lo_word & 0xFFFFu;
            }

            s_B[lane + j * WARP_SIZE] = cb0_decode(index);
        }
        __syncthreads();
        sink += static_cast<float>(s_B[lane]);
    }

    buf_f16[lane] = static_cast<f16_t>(sink);
}

// --- 9. __syncthreads cost (bare sync in a loop) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_syncthreads(float* __restrict__ buf_f32)
{
    __shared__ float s_val[WARP_SIZE];
    s_val[threadIdx.x] = 1.0f;

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        __syncthreads();
        s_val[threadIdx.x] += 0.0001f;
    }

    buf_f32[threadIdx.x] = s_val[threadIdx.x];
}

// --- 10. Full K-tile pipeline: dequant→LDS→load_matrix_sync→mma_sync+sync ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_full_k_tile(
    const f16_t*   __restrict__ A,
    const int32_t* __restrict__ B,
    const int32_t* __restrict__ word_idx,
    const int32_t* __restrict__ next_word_idx,
    const int32_t* __restrict__ shift_tbl,
    float* __restrict__ buf_f32,
    int K, int WORDS_PER_TILE)
{
    __shared__ SharedMemBench smem;
    int lane = threadIdx.x;

    // Load bit tables
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        // Load A from global
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A, K);

        // Dequant B to LDS
        for (int i = lane; i < 256; i += WARP_SIZE)
        {
            int lo_idx  = smem.s_word_idx[i];
            int hi_idx  = smem.s_next_word_idx[i];
            int shift   = smem.s_shift[i];

            uint32_t lo_word = static_cast<uint32_t>(B[lo_idx]);
            uint32_t hi_word = static_cast<uint32_t>(B[hi_idx]);

            uint32_t index;
            if (shift > 0)
            {
                int shift_hi = (32 - shift) & 31;
                uint32_t lo_part = (lo_word >> shift) & (((1u << shift_hi) - 1u) & 0xFFFFu);
                uint32_t hi_part = (hi_word << shift_hi) & 0xFFFFu;
                index = lo_part | hi_part;
            }
            else
            {
                index = lo_word & 0xFFFFu;
            }

            smem.s_B[i] = cb0_decode(index);
        }
        __syncthreads();

        // WMMA
        FragB frag_b;
        rocwmma::load_matrix_sync(frag_b, smem.s_B, TILE_DIM);
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);

        __syncthreads();
    }

    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}

// --- 11. Full K-tile with register preload (optimized pipeline) ---
__global__ __launch_bounds__(WARP_SIZE)
void bench_full_k_tile_regpreload(
    const f16_t*   __restrict__ A,
    const int32_t* __restrict__ B,
    const int32_t* __restrict__ word_idx,
    const int32_t* __restrict__ next_word_idx,
    const int32_t* __restrict__ shift_tbl,
    float* __restrict__ buf_f32,
    int K, int WORDS_PER_TILE)
{
    __shared__ f16_t s_B[TILE_DIM * TILE_DIM];
    int lane = threadIdx.x;

    // Preload bit tables into registers
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int i = lane + j * WARP_SIZE;
        reg_word_idx[j]      = word_idx[i];
        reg_next_word_idx[j] = next_word_idx[i];
        reg_shift[j]         = shift_tbl[i];
    }

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    for (int it = 0; it < BENCH_ITERS; it++)
    {
        // Load A from global
        FragA frag_a;
        rocwmma::load_matrix_sync(frag_a, A, K);

        // Dequant B with register preload
        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            int lo_idx = reg_word_idx[j];
            int hi_idx = reg_next_word_idx[j];
            int shift  = reg_shift[j];

            uint32_t lo_word = static_cast<uint32_t>(B[lo_idx]);
            uint32_t hi_word = static_cast<uint32_t>(B[hi_idx]);

            uint32_t index;
            if (shift > 0)
            {
                int shift_hi = (32 - shift) & 31;
                uint32_t lo_part = (lo_word >> shift) & (((1u << shift_hi) - 1u) & 0xFFFFu);
                uint32_t hi_part = (hi_word << shift_hi) & 0xFFFFu;
                index = lo_part | hi_part;
            }
            else
            {
                index = lo_word & 0xFFFFu;
            }

            s_B[lane + j * WARP_SIZE] = cb0_decode(index);
        }

        // WMMA (no syncthreads needed — single wave, lockstep)
        FragB frag_b;
        rocwmma::load_matrix_sync(frag_b, s_B, TILE_DIM);
        rocwmma::mma_sync(acc, frag_a, frag_b, acc);
    }

    rocwmma::store_matrix_sync(buf_f32, acc, TILE_DIM, rocwmma::mem_row_major);
}


// =====================================================================
// Host launcher: runs each benchmark kernel once
// =====================================================================

void hip_rocwmma_bench(
    at::Tensor A,           // (16, K) fp16 — for load_global/full_k_tile
    at::Tensor B_i32,       // (WPT,) int32 — packed B tile
    at::Tensor word_idx,    // (256,) int32
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int bits)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int K = A.size(1);
    int WORDS_PER_TILE = 256 * bits / 32;

    // Scratch buffers
    auto buf_f16 = at::zeros({256}, at::TensorOptions().dtype(at::kHalf).device(A.device()));
    auto buf_f32 = at::zeros({256}, at::TensorOptions().dtype(at::kFloat).device(A.device()));

    f16_t*   p_f16  = reinterpret_cast<f16_t*>(buf_f16.data_ptr());
    float*   p_f32  = reinterpret_cast<float*>(buf_f32.data_ptr());
    const f16_t*   p_A    = reinterpret_cast<const f16_t*>(A.data_ptr());
    const int32_t* p_B    = reinterpret_cast<const int32_t*>(B_i32.data_ptr());
    const int32_t* p_wi   = reinterpret_cast<const int32_t*>(word_idx.data_ptr());
    const int32_t* p_nwi  = reinterpret_cast<const int32_t*>(next_word_idx.data_ptr());
    const int32_t* p_sh   = reinterpret_cast<const int32_t*>(shift_tbl.data_ptr());

    dim3 grid1(1), block32(WARP_SIZE);

    // Launch each benchmark as a separate named kernel
    bench_mma_sync<<<grid1, block32, 0, stream>>>(p_f16, p_f32);
    bench_load_global<<<grid1, block32, 0, stream>>>(p_A, p_f32, K);
    bench_load_lds<<<grid1, block32, 0, stream>>>(p_f16, p_f32);
    bench_store_lds<<<grid1, block32, 0, stream>>>(p_f16, p_f32);
    bench_store_global<<<grid1, block32, 0, stream>>>(p_f16, p_f32);
    bench_fill_fragment<<<grid1, block32, 0, stream>>>(p_f32);
    bench_dequant_to_lds<<<grid1, block32, 0, stream>>>(p_B, p_wi, p_nwi, p_sh, p_f16, WORDS_PER_TILE);
    bench_dequant_regpreload<<<grid1, block32, 0, stream>>>(p_B, p_wi, p_nwi, p_sh, p_f16, WORDS_PER_TILE);
    bench_syncthreads<<<grid1, block32, 0, stream>>>(p_f32);
    bench_full_k_tile<<<grid1, block32, 0, stream>>>(p_A, p_B, p_wi, p_nwi, p_sh, p_f32, K, WORDS_PER_TILE);
    bench_full_k_tile_regpreload<<<grid1, block32, 0, stream>>>(p_A, p_B, p_wi, p_nwi, p_sh, p_f32, K, WORDS_PER_TILE);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rocwmma_bench launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// Phase 5: Pipelined 4-wave GEMM kernel with 3-stage fragment pipeline
//          and lock-based split-K reduction
//
// Architecture (modeled after turboderp's CUDA FSTAGE pipeline):
//   Block: 128 threads = 4 waves (wave32)
//   Grid:  (num_m_blocks, cdiv(N, 64), split_k)
//
//   Each wave handles one 16x16 N-tile (4 waves = 64 N-columns per block).
//   3-stage pipeline: while MMA runs on frag[m], we load A[s]→LDS and
//   dequant B[s]→frag_b (direct fill). Next iteration advances slots.
//
//   Lock-based split-K: pid_k=0 writes, pid_k>0 reads+adds+writes.
//   No separate reduce kernel needed.
//
// Register budget per thread (~90 VGPRs):
//   frag_a[3] = 24, frag_b[3] = 24, acc = 8
//   reg_tables[3*8] = 24, temps ~10 → 100% occupancy on RDNA3
// =====================================================================

#define V3_NUM_WAVES 4
#define V3_BLOCK_SIZE (WARP_SIZE * V3_NUM_WAVES)  // 128
#define V3_FRAG_STAGES 3
#define V3_N_PER_BLOCK 64

// -----------------------------------------------------------------
// Lock-based split-K barriers (agent-scope for cross-CU visibility)
// -----------------------------------------------------------------
__device__ __forceinline__ void barrier_acquire_v3(int* lock, int target)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        while (__hip_atomic_load(lock, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != target) {}
    }
    __threadfence();  // ensure all prior global writes from releasing block are visible
    __syncthreads();  // broadcast to all threads
}

__device__ __forceinline__ void barrier_release_v3(int* lock, int val)
{
    __syncthreads();   // ensure all threads finished their writes
    __threadfence();   // flush global memory writes to L2
    if (threadIdx.x == 0)
    {
        __hip_atomic_store(lock, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// -----------------------------------------------------------------
// LDS layout for v3: bit tables + triple-buffered A + output acc
// -----------------------------------------------------------------
struct SharedMemV3 {
    int32_t  s_word_idx[256];                            // 1024B
    int32_t  s_next_word_idx[256];                       // 1024B
    int32_t  s_shift[256];                               // 1024B
    f16_t    s_A[V3_FRAG_STAGES][TILE_DIM * TILE_DIM];  // 3 × 512B = 1536B
    f16_t    s_B[V3_NUM_WAVES][TILE_DIM * TILE_DIM];    // 4 × 512B = 2048B (per-wave B buffer)
    float    s_acc[V3_NUM_WAVES][TILE_DIM * TILE_DIM];   // 4 × 1024B = 4096B
};  // Total: ~10.5KB — well within 64KB per CU


// =====================================================================
// Dense v3 kernel
// =====================================================================
__global__ __launch_bounds__(V3_BLOCK_SIZE)
void exl3_gemm_hip_v3_kernel(
    const f16_t*    __restrict__ A,           // (M, K) row-major
    const int32_t*  __restrict__ B,           // (tiles_k, tiles_n, WPT) int32
    f16_t*          __restrict__ C,           // (M, N) row-major
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int32_t*        locks,                    // (grid_m * tiles_n64) int32
    int M, int N, int K,
    int tiles_n,        // N/16
    int tiles_n64,      // N/64
    int WORDS_PER_TILE,
    int num_k_tiles,
    int split_k,
    int cb)
{
    int pid_m   = blockIdx.x;
    int pid_n64 = blockIdx.y;
    int pid_k   = blockIdx.z;
    int tid     = threadIdx.x;
    int wave_id = tid / WARP_SIZE;   // 0..3
    int lane    = tid % WARP_SIZE;   // 0..31

    __shared__ SharedMemV3 smem;

    // 1. Load bit tables cooperatively: 128 threads, 256 entries = 2 each
    for (int i = tid; i < 256; i += V3_BLOCK_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    // 2. Compute K-range for this split
    int tiles_per_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_start = pid_k * tiles_per_split;
    int tk_end   = min(tk_start + tiles_per_split, num_k_tiles);
    int num_tiles = tk_end - tk_start;

    // 3. Setup
    int row_base = pid_m * TILE_DIM;
    int pid_n    = pid_n64 * 4 + wave_id;   // per-wave 16-wide N-tile
    bool boundary_m = (row_base + TILE_DIM > M);

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    FragA frag_a[V3_FRAG_STAGES];
    FragB frag_b[V3_FRAG_STAGES];

    // ----- Helper: fill one pipeline slot (A + B) -----
    // B dequant → per-wave LDS → load_matrix_sync (packed f16x2 on RDNA3)
    // Preload bit tables into registers for V4 optimized dequant
    int v3_reg_word_idx[8], v3_reg_next_word_idx[8], v3_reg_shift[8];
    #pragma unroll
    for (int _j = 0; _j < 8; _j++)
    {
        int _i = lane + _j * WARP_SIZE;
        v3_reg_word_idx[_j]      = smem.s_word_idx[_i];
        v3_reg_next_word_idx[_j] = smem.s_next_word_idx[_i];
        v3_reg_shift[_j]         = smem.s_shift[_i];
    }

    #define V3_FILL_SLOT_DENSE(tk_val, slot)                                           \
    {                                                                                  \
        /* Cooperative A load */                                                       \
        if (boundary_m)                                                                \
        {                                                                              \
            for (int _i = tid; _i < TILE_DIM * TILE_DIM; _i += V3_BLOCK_SIZE)         \
            {                                                                          \
                int _r = _i / TILE_DIM, _c = _i % TILE_DIM;                           \
                int _gr = row_base + _r, _gc = (tk_val) * TILE_DIM + _c;              \
                smem.s_A[(slot)][_i] = (_gr < M) ? A[_gr * K + _gc]                   \
                                                 : static_cast<f16_t>(0.0f);          \
            }                                                                          \
        }                                                                              \
        else                                                                           \
        {                                                                              \
            for (int _i = tid; _i < TILE_DIM * TILE_DIM; _i += V3_BLOCK_SIZE)         \
            {                                                                          \
                int _r = _i / TILE_DIM, _c = _i % TILE_DIM;                           \
                smem.s_A[(slot)][_i] = A[(row_base + _r) * K + (tk_val) * TILE_DIM + _c]; \
            }                                                                          \
        }                                                                              \
        /* V4 optimized per-wave B dequant → LDS */                                    \
        {                                                                              \
            int _b_base = ((tk_val) * tiles_n + pid_n) * WORDS_PER_TILE;               \
            dequant_tile_v5_cb(                                                        \
                B + _b_base,                                                           \
                v3_reg_word_idx, v3_reg_next_word_idx, v3_reg_shift,                   \
                lane, smem.s_B[wave_id], cb);                                          \
        }                                                                              \
        __syncthreads();                                                               \
        /* Load A and B fragments from LDS */                                          \
        rocwmma::load_matrix_sync(frag_a[(slot)], smem.s_A[(slot)], TILE_DIM);         \
        rocwmma::load_matrix_sync(frag_b[(slot)], smem.s_B[wave_id], TILE_DIM);        \
    }

    // 4. Prologue: fill pipeline (use tk%3 for slot to stay in sync with main loop)
    if (num_tiles >= 1)
        V3_FILL_SLOT_DENSE(tk_start, tk_start % V3_FRAG_STAGES);
    if (num_tiles >= 2)
        V3_FILL_SLOT_DENSE(tk_start + 1, (tk_start + 1) % V3_FRAG_STAGES);

    // 5. Main loop: 3-stage pipeline
    for (int tk = tk_start + 2; tk < tk_end; tk++)
    {
        int s = tk % V3_FRAG_STAGES;
        int m = (tk - 2) % V3_FRAG_STAGES;

        rocwmma::mma_sync(acc, frag_a[m], frag_b[m], acc);
        V3_FILL_SLOT_DENSE(tk, s);
    }

    // 6. Drain
    if (num_tiles >= 2)
        rocwmma::mma_sync(acc, frag_a[(tk_end - 2) % V3_FRAG_STAGES],
                           frag_b[(tk_end - 2) % V3_FRAG_STAGES], acc);
    if (num_tiles >= 1)
        rocwmma::mma_sync(acc, frag_a[(tk_end - 1) % V3_FRAG_STAGES],
                           frag_b[(tk_end - 1) % V3_FRAG_STAGES], acc);

    #undef V3_FILL_SLOT_DENSE

    // 8. Store accumulator to LDS
    rocwmma::store_matrix_sync(smem.s_acc[wave_id], acc, TILE_DIM,
                               rocwmma::mem_row_major);

    // 9. Lock-based split-K output
    int n_col_base = pid_n64 * V3_N_PER_BLOCK + wave_id * TILE_DIM;
    int lock_idx = pid_m * tiles_n64 + pid_n64;

    if (split_k == 1)
    {
        // No sync needed — each wave reads only its own s_acc[wave_id]
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM, c = i % TILE_DIM;
            int gr = row_base + r, gc = n_col_base + c;
            if (gr < M && gc < N)
            {
                float val = smem.s_acc[wave_id][i];
                C[gr * N + gc] = static_cast<f16_t>(fminf(fmaxf(val, -65504.f), 65504.f));
            }
        }
    }
    else if (pid_k == 0)
    {
        // First split: write partial to C, signal next
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM, c = i % TILE_DIM;
            int gr = row_base + r, gc = n_col_base + c;
            if (gr < M && gc < N)
            {
                float val = smem.s_acc[wave_id][i];
                C[gr * N + gc] = static_cast<f16_t>(fminf(fmaxf(val, -65504.f), 65504.f));
            }
        }
        barrier_release_v3(&locks[lock_idx], 1);
    }
    else
    {
        // Wait for previous split, read+add+write
        barrier_acquire_v3(&locks[lock_idx], pid_k);
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM, c = i % TILE_DIM;
            int gr = row_base + r, gc = n_col_base + c;
            if (gr < M && gc < N)
            {
                float val = smem.s_acc[wave_id][i] + static_cast<float>(C[gr * N + gc]);
                C[gr * N + gc] = static_cast<f16_t>(fminf(fmaxf(val, -65504.f), 65504.f));
            }
        }
        barrier_release_v3(&locks[lock_idx], (pid_k == split_k - 1) ? 0 : pid_k + 1);
    }
}


// =====================================================================
// Fused MoE v3 kernel — same pipeline architecture, expert-routed
// =====================================================================
__global__ __launch_bounds__(V3_BLOCK_SIZE)
void exl3_fused_moe_gemm_hip_v3_kernel(
    const f16_t*    __restrict__ A,             // (EM_max, K) contiguous
    const int32_t*  __restrict__ B_stacked,     // (E, tiles_k, tiles_n, WPT) int32
    f16_t*          __restrict__ C,             // (EM_max, N) row-major
    const int32_t*  __restrict__ expert_ids,    // (num_m_blocks,) int32
    const int32_t*  __restrict__ num_tokens_post_padded,
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int32_t*        locks,
    int EM_max, int N, int K,
    int stride_be,          // expert stride in B_stacked (int32 elements)
    int tiles_n,            // N/16
    int tiles_n64,          // N/64
    int WORDS_PER_TILE,
    int num_k_tiles,
    int split_k,
    int cb)
{
    int pid_m   = blockIdx.x;
    int pid_n64 = blockIdx.y;
    int pid_k   = blockIdx.z;
    int tid     = threadIdx.x;
    int wave_id = tid / WARP_SIZE;
    int lane    = tid % WARP_SIZE;

    // Early exit check (uniform across block — safe for __syncthreads)
    int num_valid = num_tokens_post_padded[0];
    int num_valid_m_blocks = (num_valid + TILE_DIM - 1) / TILE_DIM;
    int off_expert = (pid_m < num_valid_m_blocks) ? expert_ids[pid_m] : -1;

    bool skip = (pid_m >= num_valid_m_blocks || off_expert < 0);

    if (skip)
    {
        // Pass through lock chain so subsequent pid_k blocks don't hang
        if (split_k > 1)
        {
            int lock_idx = pid_m * tiles_n64 + pid_n64;
            if (pid_k == 0)
                barrier_release_v3(&locks[lock_idx], 1);
            else
            {
                barrier_acquire_v3(&locks[lock_idx], pid_k);
                barrier_release_v3(&locks[lock_idx],
                                   (pid_k == split_k - 1) ? 0 : pid_k + 1);
            }
        }
        return;
    }

    __shared__ SharedMemV3 smem;

    // 1. Load bit tables cooperatively
    for (int i = tid; i < 256; i += V3_BLOCK_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    // 2. K-range for this split
    int tiles_per_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_start = pid_k * tiles_per_split;
    int tk_end   = min(tk_start + tiles_per_split, num_k_tiles);
    int num_tiles = tk_end - tk_start;

    // 4. Setup
    int row_base = pid_m * TILE_DIM;
    int pid_n    = pid_n64 * 4 + wave_id;
    const int32_t* B_expert = B_stacked + off_expert * stride_be;

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    FragA frag_a[V3_FRAG_STAGES];
    FragB frag_b[V3_FRAG_STAGES];

    // Preload bit tables into registers for V4 optimized dequant
    int v3m_reg_word_idx[8], v3m_reg_next_word_idx[8], v3m_reg_shift[8];
    #pragma unroll
    for (int _j = 0; _j < 8; _j++)
    {
        int _i = lane + _j * WARP_SIZE;
        v3m_reg_word_idx[_j]      = smem.s_word_idx[_i];
        v3m_reg_next_word_idx[_j] = smem.s_next_word_idx[_i];
        v3m_reg_shift[_j]         = smem.s_shift[_i];
    }

    // ----- Helper: fill one pipeline slot -----
    // V4 optimized dequant B → per-wave LDS → load_matrix_sync
    #define V3_FILL_SLOT_MOE(tk_val, slot)                                             \
    {                                                                                  \
        /* Cooperative A load (EM_max always % 16 == 0, no boundary check) */          \
        for (int _i = tid; _i < TILE_DIM * TILE_DIM; _i += V3_BLOCK_SIZE)             \
        {                                                                              \
            int _r = _i / TILE_DIM;                                                    \
            int _c = _i % TILE_DIM;                                                    \
            smem.s_A[(slot)][_i] = A[(row_base + _r) * K + (tk_val) * TILE_DIM + _c]; \
        }                                                                              \
        /* V4 optimized per-wave B dequant → LDS */                                    \
        {                                                                              \
            int _b_base = ((tk_val) * tiles_n + pid_n) * WORDS_PER_TILE;               \
            dequant_tile_v5_cb(                                                        \
                B_expert + _b_base,                                                    \
                v3m_reg_word_idx, v3m_reg_next_word_idx, v3m_reg_shift,                \
                lane, smem.s_B[wave_id], cb);                                          \
        }                                                                              \
        __syncthreads();                                                               \
        /* Load A and B fragments from LDS */                                          \
        rocwmma::load_matrix_sync(frag_a[(slot)], smem.s_A[(slot)], TILE_DIM);         \
        rocwmma::load_matrix_sync(frag_b[(slot)], smem.s_B[wave_id], TILE_DIM);        \
    }

    // 5. Prologue: fill pipeline (use tk%3 for slot to stay in sync with main loop)
    if (num_tiles >= 1)
        V3_FILL_SLOT_MOE(tk_start, tk_start % V3_FRAG_STAGES);
    if (num_tiles >= 2)
        V3_FILL_SLOT_MOE(tk_start + 1, (tk_start + 1) % V3_FRAG_STAGES);

    // 6. Main loop: 3-stage pipeline
    for (int tk = tk_start + 2; tk < tk_end; tk++)
    {
        int s = tk % V3_FRAG_STAGES;
        int m = (tk - 2) % V3_FRAG_STAGES;

        rocwmma::mma_sync(acc, frag_a[m], frag_b[m], acc);
        V3_FILL_SLOT_MOE(tk, s);
    }

    // 7. Drain
    if (num_tiles >= 2)
        rocwmma::mma_sync(acc, frag_a[(tk_end - 2) % V3_FRAG_STAGES],
                           frag_b[(tk_end - 2) % V3_FRAG_STAGES], acc);
    if (num_tiles >= 1)
        rocwmma::mma_sync(acc, frag_a[(tk_end - 1) % V3_FRAG_STAGES],
                           frag_b[(tk_end - 1) % V3_FRAG_STAGES], acc);

    #undef V3_FILL_SLOT_MOE

    // 8. Store accumulator to LDS
    rocwmma::store_matrix_sync(smem.s_acc[wave_id], acc, TILE_DIM,
                               rocwmma::mem_row_major);

    // 9. Lock-based split-K output
    int n_col_base = pid_n64 * V3_N_PER_BLOCK + wave_id * TILE_DIM;
    int lock_idx = pid_m * tiles_n64 + pid_n64;

    if (split_k == 1)
    {
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM, c = i % TILE_DIM;
            int gr = row_base + r, gc = n_col_base + c;
            if (gr < EM_max && gc < N)
            {
                float val = smem.s_acc[wave_id][i];
                C[gr * N + gc] = static_cast<f16_t>(fminf(fmaxf(val, -65504.f), 65504.f));
            }
        }
    }
    else if (pid_k == 0)
    {
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM, c = i % TILE_DIM;
            int gr = row_base + r, gc = n_col_base + c;
            if (gr < EM_max && gc < N)
            {
                float val = smem.s_acc[wave_id][i];
                C[gr * N + gc] = static_cast<f16_t>(fminf(fmaxf(val, -65504.f), 65504.f));
            }
        }
        barrier_release_v3(&locks[lock_idx], 1);
    }
    else
    {
        barrier_acquire_v3(&locks[lock_idx], pid_k);
        for (int i = lane; i < TILE_DIM * TILE_DIM; i += WARP_SIZE)
        {
            int r = i / TILE_DIM, c = i % TILE_DIM;
            int gr = row_base + r, gc = n_col_base + c;
            if (gr < EM_max && gc < N)
            {
                float val = smem.s_acc[wave_id][i] + static_cast<float>(C[gr * N + gc]);
                C[gr * N + gc] = static_cast<f16_t>(fminf(fmaxf(val, -65504.f), 65504.f));
            }
        }
        barrier_release_v3(&locks[lock_idx], (pid_k == split_k - 1) ? 0 : pid_k + 1);
    }
}


// =====================================================================
// Phase 5: Host launchers for v3 kernels
// =====================================================================

void hip_exl3_gemm_v3(
    at::Tensor A,             // (M, K) float16
    at::Tensor B_i32,         // flat int32 packed trellis data
    at::Tensor C,             // (M, N) float16 output
    at::Tensor word_idx,      // (256,) int32
    at::Tensor next_word_idx, // (256,) int32
    at::Tensor shift_tbl,     // (256,) int32
    at::Tensor locks,         // lock buffer (pre-allocated)
    int bits,
    int cb,
    int split_k)
{
    TORCH_CHECK(A.is_contiguous(), "exl3_gemm_v3: A must be contiguous");
    TORCH_CHECK(B_i32.is_contiguous(), "exl3_gemm_v3: B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "exl3_gemm_v3: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "exl3_gemm_v3: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "exl3_gemm_v3: C must be float16");
    TORCH_CHECK(B_i32.dtype() == at::kInt, "exl3_gemm_v3: B must be int32");
    TORCH_CHECK(cb == 0 || cb == 1, "exl3_gemm_v3 HIP: only cb=0,1 supported, got cb=", cb);
    TORCH_CHECK(split_k >= 1, "exl3_gemm_v3: split_k must be >= 1");

    int M = A.size(0);
    int K = A.size(1);
    int N = C.size(1);

    TORCH_CHECK(K % 16 == 0, "exl3_gemm_v3: K must be divisible by 16");
    TORCH_CHECK(N % 64 == 0, "exl3_gemm_v3: N must be divisible by 64 (4-wave), got ", N);

    int tiles_n = N / 16;
    int tiles_n64 = N / 64;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;

    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int grid_m = (M + 15) / 16;

    // Zero lock buffer for split-K
    if (split_k > 1)
    {
        int num_locks = grid_m * tiles_n64;
        TORCH_CHECK(locks.numel() >= num_locks, "exl3_gemm_v3: locks buffer too small");
        cudaMemsetAsync(locks.data_ptr(), 0, num_locks * sizeof(int32_t), stream);
    }

    dim3 grid(grid_m, tiles_n64, split_k);
    dim3 block(V3_BLOCK_SIZE);

    exl3_gemm_hip_v3_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const f16_t*>(A.data_ptr()),
        reinterpret_cast<const int32_t*>(B_i32.data_ptr()),
        reinterpret_cast<f16_t*>(C.data_ptr()),
        reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
        reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
        reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
        reinterpret_cast<int32_t*>(locks.data_ptr()),
        M, N, K,
        tiles_n, tiles_n64, WORDS_PER_TILE,
        num_k_tiles, split_k, cb
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "exl3_gemm_v3 kernel launch failed: ", cudaGetErrorString(err));
}


void hip_exl3_fused_moe_gemm_v3(
    at::Tensor A,               // (EM_max, K) fp16
    at::Tensor B_stacked_i32,   // (E, tiles_k, tiles_n, WPT) int32
    at::Tensor C,               // (EM_max, N) fp16 — pre-zeroed by caller
    at::Tensor expert_ids,      // (num_m_blocks,) int32
    at::Tensor num_tokens_post_padded,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    at::Tensor locks,
    int EM_max,
    int bits,
    int cb,
    int split_k)
{
    TORCH_CHECK(A.is_contiguous(), "fused_moe_gemm_v3: A must be contiguous");
    TORCH_CHECK(B_stacked_i32.is_contiguous(), "fused_moe_gemm_v3: B_stacked must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "fused_moe_gemm_v3: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "fused_moe_gemm_v3: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "fused_moe_gemm_v3: C must be float16");
    TORCH_CHECK(B_stacked_i32.dtype() == at::kInt, "fused_moe_gemm_v3: B must be int32");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "fused_moe_gemm_v3: expert_ids must be int32");
    TORCH_CHECK(cb == 0 || cb == 1, "fused_moe_gemm_v3 HIP: only cb=0,1 supported, got cb=", cb);
    TORCH_CHECK(split_k >= 1, "fused_moe_gemm_v3: split_k must be >= 1");

    int K = A.size(1);
    int N = C.size(1);
    int num_m_blocks = expert_ids.size(0);

    TORCH_CHECK(K % 16 == 0, "fused_moe_gemm_v3: K must be divisible by 16");
    TORCH_CHECK(N % 64 == 0, "fused_moe_gemm_v3: N must be divisible by 64, got ", N);

    int tiles_n = N / 16;
    int tiles_n64 = N / 64;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;

    int stride_be = B_stacked_i32.stride(0);

    // Auto split-K
    if (split_k == 0)
    {
        if (num_m_blocks <= 16 && num_k_tiles >= 16)
            split_k = min(8, num_k_tiles);
        else
            split_k = 1;
    }
    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Zero lock buffer
    if (split_k > 1)
    {
        int num_locks = num_m_blocks * tiles_n64;
        TORCH_CHECK(locks.numel() >= num_locks,
                    "fused_moe_gemm_v3: locks buffer too small (",
                    locks.numel(), " < ", num_locks, ")");
        cudaMemsetAsync(locks.data_ptr(), 0, num_locks * sizeof(int32_t), stream);
    }

    dim3 grid(num_m_blocks, tiles_n64, split_k);
    dim3 block(V3_BLOCK_SIZE);

    exl3_fused_moe_gemm_hip_v3_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const f16_t*>(A.data_ptr()),
        reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
        reinterpret_cast<f16_t*>(C.data_ptr()),
        reinterpret_cast<const int32_t*>(expert_ids.data_ptr()),
        reinterpret_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
        reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
        reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
        reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
        reinterpret_cast<int32_t*>(locks.data_ptr()),
        EM_max, N, K,
        stride_be, tiles_n, tiles_n64, WORDS_PER_TILE,
        num_k_tiles, split_k, cb
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_moe_gemm_v3 kernel launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// V4: Register-only VALU kernel for M=1 decode
//
// Eliminates WMMA overhead for M=1 by using scalar FMA:
// - No WMMA: dequant -> register -> multiply by A -> accumulate (float)
// - No LDS for B: dequant directly to registers
// - No LDS for A: load directly from global (L1 cached)
// - 4 warps/block: each warp handles 16 N-columns independently
// - Cross-lane reduce: __shfl_xor(acc, 16) combines even/odd rows
//
// Bit table mapping for 32-thread warp:
//   flat_index = lane + j * 32  (lane in [0,31], j in [0,7])
//   B_col = flat_index % 16 = lane % 16
//   B_row = flat_index / 16 = (lane / 16) + j * 2
//   Lanes 0-15:  even rows (0,2,4,...,14) of columns 0-15
//   Lanes 16-31: odd rows  (1,3,5,...,15) of columns 0-15
//
// Grid: (cdiv(N, 64), 1, split_k)
// Block: 128 threads = 4 warps
// =====================================================================

#define V4_BLOCK_SIZE 128
#define V4_WARPS      4

struct SharedMemV4 {
    int32_t s_word_idx[256];       // 1024 bytes
    int32_t s_next_word_idx[256];  // 1024 bytes
    int32_t s_shift[256];          // 1024 bytes
};  // 3072 bytes total

__global__ __launch_bounds__(V4_BLOCK_SIZE)
void exl3_gemm_v4_kernel(
    const f16_t*    __restrict__ A,            // (1, K) row-major
    const int32_t*  __restrict__ B,            // (K/16, N/16, WPT) int32
    f16_t*          __restrict__ C_out,        // C or C_partial slice
    const int32_t*  __restrict__ word_idx,     // (256,) bit table
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int N, int K,
    int tiles_n,
    int WORDS_PER_TILE,
    int num_k_tiles,
    int cb)
{
    int warp_id = threadIdx.x / WARP_SIZE;   // 0-3
    int lane    = threadIdx.x % WARP_SIZE;   // 0-31
    int col     = lane % 16;                 // output column within N-tile
    int row_parity = lane / 16;              // 0=even rows, 1=odd rows

    // Global N-tile index for this warp
    int n_tile = blockIdx.x * V4_WARPS + warp_id;
    bool active = (n_tile < tiles_n);

    int col_global = n_tile * 16 + col;

    __shared__ SharedMemV4 smem;

    // Cooperatively load bit tables to LDS (128 threads, 256 entries)
    // ALL threads must participate before __syncthreads — no early return!
    for (int i = threadIdx.x; i < 256; i += V4_BLOCK_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    if (!active) return;

    // Preload bit tables into registers (avoids LDS reads in K-loop)
    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int idx = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[idx];
        reg_next_word_idx[j] = smem.s_next_word_idx[idx];
        reg_shift[j]         = smem.s_shift[idx];
    }

    // Split-K range
    int split_k = gridDim.z;
    int pid_k   = blockIdx.z;
    int tiles_per_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_start = pid_k * tiles_per_split;
    int tk_end   = min(tk_start + tiles_per_split, num_k_tiles);

    float acc = 0.0f;

    for (int tk = tk_start; tk < tk_end; tk++)
    {
        int b_base = (tk * tiles_n + n_tile) * WORDS_PER_TILE;
        int a_base = tk * 16;

        // Dequant + FMA: 8 elements per thread, paired for ILP
        #pragma unroll
        for (int j = 0; j < 8; j += 2)
        {
            // Load packed B words
            uint32_t lo0 = static_cast<uint32_t>(B[b_base + reg_word_idx[j]]);
            uint32_t hi0 = static_cast<uint32_t>(B[b_base + reg_next_word_idx[j]]);
            uint32_t lo1 = static_cast<uint32_t>(B[b_base + reg_word_idx[j+1]]);
            uint32_t hi1 = static_cast<uint32_t>(B[b_base + reg_next_word_idx[j+1]]);

            // Funnel shift to extract 16-bit indices
            uint32_t idx0, idx1;
            funnel_shift_16_v5_pair(lo0, hi0, reg_shift[j],
                                    lo1, hi1, reg_shift[j+1],
                                    idx0, idx1);

            // Codebook decode to fp16 weights
            f16_t w0, w1;
            cb_decode_2_v5(idx0, idx1, w0, w1, cb);

            // FMA: weight * A[row] -> accumulate
            int row0 = row_parity + j * 2;
            int row1 = row_parity + (j + 1) * 2;
            acc += static_cast<float>(w0) * static_cast<float>(A[a_base + row0]);
            acc += static_cast<float>(w1) * static_cast<float>(A[a_base + row1]);
        }
    }

    // Cross-lane reduce: combine even/odd row partials
    acc += __shfl_xor(acc, 16);

    // Lanes 0-15 write output (complete dot product for one column)
    if (lane < 16 && col_global < N)
    {
        float clamped = fminf(fmaxf(acc, -65504.0f), 65504.0f);
        C_out[pid_k * N + col_global] = static_cast<f16_t>(clamped);
    }
}


// =====================================================================
// V4 host launcher
// =====================================================================

void hip_exl3_gemm_v4(
    at::Tensor A,             // (1, K) float16
    at::Tensor B_i32,         // flat int32 packed trellis data
    at::Tensor C,             // (1, N) float16 output
    at::Tensor word_idx,      // (256,) int32
    at::Tensor next_word_idx, // (256,) int32
    at::Tensor shift_tbl,     // (256,) int32
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial)     // (split_k, 1, N) float16 — only used when split_k > 1
{
    TORCH_CHECK(A.is_contiguous(), "exl3_gemm_v4: A must be contiguous");
    TORCH_CHECK(B_i32.is_contiguous(), "exl3_gemm_v4: B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "exl3_gemm_v4: C must be contiguous");
    TORCH_CHECK(A.dtype() == at::kHalf, "exl3_gemm_v4: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "exl3_gemm_v4: C must be float16");
    TORCH_CHECK(B_i32.dtype() == at::kInt, "exl3_gemm_v4: B must be int32");
    TORCH_CHECK(cb == 0 || cb == 1, "exl3_gemm_v4: only cb=0,1 supported, got cb=", cb);
    TORCH_CHECK(split_k >= 1, "exl3_gemm_v4: split_k must be >= 1");

    int M = A.size(0);
    int K = A.size(1);
    int N = C.size(1);

    TORCH_CHECK(M == 1, "exl3_gemm_v4: M must be 1 (decode-only kernel), got ", M);
    TORCH_CHECK(K % 16 == 0, "exl3_gemm_v4: K must be divisible by 16, got ", K);
    TORCH_CHECK(N % 16 == 0, "exl3_gemm_v4: N must be divisible by 16, got ", N);

    int tiles_n = N / 16;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;

    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int grid_n = (tiles_n + V4_WARPS - 1) / V4_WARPS;

    if (split_k == 1)
    {
        dim3 grid(grid_n, 1, 1);
        dim3 block(V4_BLOCK_SIZE);

        exl3_gemm_v4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            N, K, tiles_n, WORDS_PER_TILE, num_k_tiles, cb
        );
    }
    else
    {
        TORCH_CHECK(C_partial.is_contiguous(), "exl3_gemm_v4: C_partial must be contiguous");
        TORCH_CHECK(C_partial.dtype() == at::kHalf, "exl3_gemm_v4: C_partial must be float16");
        TORCH_CHECK(C_partial.size(0) >= split_k && C_partial.size(1) >= M && C_partial.size(2) >= N,
                    "exl3_gemm_v4: C_partial too small");

        dim3 grid(grid_n, 1, split_k);
        dim3 block(V4_BLOCK_SIZE);

        exl3_gemm_v4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A.data_ptr()),
            reinterpret_cast<const int32_t*>(B_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            N, K, tiles_n, WORDS_PER_TILE, num_k_tiles, cb
        );

        // Reduce split-K partials
        int MN = N;  // M=1
        int reduce_threads = 256;
        int reduce_blocks = (MN + reduce_threads - 1) / reduce_threads;

        exl3_gemm_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            reinterpret_cast<const f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            MN, split_k
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "exl3_gemm_v4 launch failed: ", cudaGetErrorString(err));
}


// =====================================================================
// Batched V4: Single-launch multi-GEMM for merged projections (M=1)
//
// Grid: (cdiv(N, 64), num_outputs, split_k)
//   blockIdx.x = N-tile group
//   blockIdx.y = output index (sub-projection)
//   blockIdx.z = split-K partition
// =====================================================================

__global__ __launch_bounds__(V4_BLOCK_SIZE)
void exl3_batched_gemm_v4_kernel(
    const f16_t*    __restrict__ A,
    const int32_t*  __restrict__ B,
    f16_t*          __restrict__ C_out,
    const int32_t*  __restrict__ word_idx,
    const int32_t*  __restrict__ next_word_idx,
    const int32_t*  __restrict__ shift_tbl,
    int N, int K,
    int tiles_n,
    int WORDS_PER_TILE,
    int num_k_tiles,
    int num_outputs,
    int cb)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int col     = lane % 16;
    int row_parity = lane / 16;

    int n_tile = blockIdx.x * V4_WARPS + warp_id;
    bool active = (n_tile < tiles_n);

    int out_idx = blockIdx.y;
    int col_global = n_tile * 16 + col;

    __shared__ SharedMemV4 smem;

    // ALL threads must participate before __syncthreads — no early return!
    for (int i = threadIdx.x; i < 256; i += V4_BLOCK_SIZE)
    {
        smem.s_word_idx[i]      = word_idx[i];
        smem.s_next_word_idx[i] = next_word_idx[i];
        smem.s_shift[i]         = shift_tbl[i];
    }
    __syncthreads();

    if (!active) return;

    int reg_word_idx[8], reg_next_word_idx[8], reg_shift[8];
    #pragma unroll
    for (int j = 0; j < 8; j++)
    {
        int idx = lane + j * WARP_SIZE;
        reg_word_idx[j]      = smem.s_word_idx[idx];
        reg_next_word_idx[j] = smem.s_next_word_idx[idx];
        reg_shift[j]         = smem.s_shift[idx];
    }

    const f16_t*   A_out = A + out_idx * K;
    int b_stride_per_output = num_k_tiles * tiles_n * WORDS_PER_TILE;
    const int32_t* B_out = B + out_idx * b_stride_per_output;

    int split_k = gridDim.z;
    int pid_k   = blockIdx.z;
    int tiles_per_split = (num_k_tiles + split_k - 1) / split_k;
    int tk_start = pid_k * tiles_per_split;
    int tk_end   = min(tk_start + tiles_per_split, num_k_tiles);

    float acc = 0.0f;

    for (int tk = tk_start; tk < tk_end; tk++)
    {
        int b_base = (tk * tiles_n + n_tile) * WORDS_PER_TILE;
        int a_base = tk * 16;

        #pragma unroll
        for (int j = 0; j < 8; j += 2)
        {
            uint32_t lo0 = static_cast<uint32_t>(B_out[b_base + reg_word_idx[j]]);
            uint32_t hi0 = static_cast<uint32_t>(B_out[b_base + reg_next_word_idx[j]]);
            uint32_t lo1 = static_cast<uint32_t>(B_out[b_base + reg_word_idx[j+1]]);
            uint32_t hi1 = static_cast<uint32_t>(B_out[b_base + reg_next_word_idx[j+1]]);

            uint32_t idx0, idx1;
            funnel_shift_16_v5_pair(lo0, hi0, reg_shift[j],
                                    lo1, hi1, reg_shift[j+1],
                                    idx0, idx1);

            f16_t w0, w1;
            cb_decode_2_v5(idx0, idx1, w0, w1, cb);

            int row0 = row_parity + j * 2;
            int row1 = row_parity + (j + 1) * 2;
            acc += static_cast<float>(w0) * static_cast<float>(A_out[a_base + row0]);
            acc += static_cast<float>(w1) * static_cast<float>(A_out[a_base + row1]);
        }
    }

    acc += __shfl_xor(acc, 16);

    if (lane < 16 && col_global < N)
    {
        float clamped = fminf(fmaxf(acc, -65504.0f), 65504.0f);
        C_out[pid_k * (num_outputs * N) + out_idx * N + col_global] = static_cast<f16_t>(clamped);
    }
}


void hip_exl3_batched_gemm_v4(
    at::Tensor A_batched,
    at::Tensor B_stacked_i32,
    at::Tensor C,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int num_outputs,
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial)
{
    TORCH_CHECK(A_batched.is_contiguous(), "batched_v4: A must be contiguous");
    TORCH_CHECK(B_stacked_i32.is_contiguous(), "batched_v4: B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "batched_v4: C must be contiguous");
    TORCH_CHECK(A_batched.dtype() == at::kHalf, "batched_v4: A must be float16");
    TORCH_CHECK(C.dtype() == at::kHalf, "batched_v4: C must be float16");
    TORCH_CHECK(B_stacked_i32.dtype() == at::kInt, "batched_v4: B must be int32");
    TORCH_CHECK(cb == 0 || cb == 1, "batched_v4: only cb=0,1 supported");
    TORCH_CHECK(split_k >= 1, "batched_v4: split_k must be >= 1");

    int K = A_batched.size(2);
    int N = C.size(2);

    TORCH_CHECK(A_batched.size(0) == num_outputs, "batched_v4: A dim0 != num_outputs");
    TORCH_CHECK(A_batched.size(1) == 1, "batched_v4: M must be 1");
    TORCH_CHECK(K % 16 == 0, "batched_v4: K must be divisible by 16");
    TORCH_CHECK(N % 16 == 0, "batched_v4: N must be divisible by 16");

    int tiles_n = N / 16;
    int num_k_tiles = K / 16;
    int WORDS_PER_TILE = 256 * bits / 32;

    if (split_k > num_k_tiles)
        split_k = num_k_tiles;

    const at::cuda::OptionalCUDAGuard device_guard(A_batched.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int grid_n = (tiles_n + V4_WARPS - 1) / V4_WARPS;

    if (split_k == 1)
    {
        dim3 grid(grid_n, num_outputs, 1);
        dim3 block(V4_BLOCK_SIZE);

        exl3_batched_gemm_v4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A_batched.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            N, K, tiles_n, WORDS_PER_TILE, num_k_tiles, num_outputs, cb
        );
    }
    else
    {
        TORCH_CHECK(C_partial.is_contiguous(), "batched_v4: C_partial must be contiguous");
        TORCH_CHECK(C_partial.dtype() == at::kHalf, "batched_v4: C_partial must be float16");

        dim3 grid(grid_n, num_outputs, split_k);
        dim3 block(V4_BLOCK_SIZE);

        exl3_batched_gemm_v4_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const f16_t*>(A_batched.data_ptr()),
            reinterpret_cast<const int32_t*>(B_stacked_i32.data_ptr()),
            reinterpret_cast<f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<const int32_t*>(word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(next_word_idx.data_ptr()),
            reinterpret_cast<const int32_t*>(shift_tbl.data_ptr()),
            N, K, tiles_n, WORDS_PER_TILE, num_k_tiles, num_outputs, cb
        );

        int MN = num_outputs * N;
        int reduce_threads = 256;
        int reduce_blocks = (MN + reduce_threads - 1) / reduce_threads;

        exl3_gemm_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            reinterpret_cast<const f16_t*>(C_partial.data_ptr()),
            reinterpret_cast<f16_t*>(C.data_ptr()),
            MN, split_k
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "batched_v4 launch failed: ", cudaGetErrorString(err));
}
