#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// =====================================================================
// DPP-fused butterfly Hadamard-128 kernel for RDNA3 (gfx1100)
//
// V4 Phase 2: Exploits RDNA3-unique DPP (Data Parallel Primitives)
// and ds_swizzle for cross-lane operations. DPP fuses the shuffle+VALU
// into a single instruction (1 cycle), vs __shfl_xor which compiles to
// ds_bpermute_b32 (LDS unit, separate from VALU).
//
// Algorithm: 1 warp (32 threads) handles one 128-element block.
// Each thread holds 4 float values. Steps:
//   1. Load 4 fp16 -> float
//   2. Optional pre-scale (sign flips before Hadamard)
//   3. Local 4-element Hadamard in registers (had4_local)
//   4. 5 rounds of DPP/ds_swizzle butterfly (Hadamard-32)
//   5. Scale by r_scale / sqrt(128)
//   6. Optional post-scale (sign flips after Hadamard)
//   7. Store as fp16
//
// Cross-lane performance on RDNA3 (measured, 4 shuffles per iter):
//   DPP quad_perm:  ~3.6 ns  (register-path, fused, fastest)
//   __shfl_xor:    ~14.9 ns  (ds_bpermute_b32, compiler-managed waitcnt)
//   ds_swizzle:    ~17.4 ns  (single-op latency; requires +v wait at -O3)
//   ds_swizzle sustained: 0.88x of __shfl_xor (12% faster in pure shuffle)
//   ds_swizzle + DPP mix:  1.10x of __shfl_xor (10% slower when interleaved)
//
// Rounds 1-2: DPP quad_perm (register path, fused with VALU, 1 cycle)
// Rounds 3-5: __shfl_xor (faster than ds_swizzle when mixed with DPP)
//
// FLOPs: 896 per 128-element block (vs 16,384 for matmul)
// No shared memory needed — pure registers + cross-lane ops.
// =====================================================================

#define WARP_SIZE 32
#define HAD_BLOCK 128
#define VALS_PER_THREAD 4

// Normalized scale: 1/sqrt(128)
static constexpr float RSQRT_128 = 0.08838834764831845f;

// -----------------------------------------------------------------
// DPP / ds_swizzle inline ASM primitives for RDNA3 (gfx1100)
//
// DPP (Data Parallel Primitives):
//   - quad_perm: 4-lane permute within each quad (lanes 0-3, 4-7, etc.)
//   - Fused with VALU: v_mov_b32 + DPP modifier = 1 instruction
//   - Register-path (not LDS), lower latency than ds_bpermute
//
// NOTE: ds_swizzle_b32 works on RDNA3 but has two pitfalls:
//   1. Offset encoding: bits 14:10=xor, 9:5=or, 4:0=and (easy to swap and/or)
//   2. At -O3, the compiler schedules VALU reads of result VGPRs BEFORE
//      s_waitcnt lgkmcnt(0), causing stale reads. Fix: use +v constraints.
//
// Scaling test (1-192 blocks): ds_swizzle is 12% FASTER than __shfl_xor in
// pure shuffle workloads (0.88x ratio), but 10% SLOWER when mixed with DPP
// (1.10x ratio). The ratio is flat across all occupancy levels — no scaling
// advantage for either. Since the Hadamard butterfly mixes DPP (rounds 1-2)
// with shuffle (rounds 3-5), __shfl_xor is the right choice here.
// For pure-shuffle kernels (reductions, allreduce), prefer ds_swizzle.
// See tests/test_rdna3_cross_lane.hip for the primitive validation suite.
// -----------------------------------------------------------------

// DPP quad permute: swap pairs within each 4-lane quad (XOR-1)
// quad_perm:[1,0,3,2] means: lane 0→1, 1→0, 2→3, 3→2
__device__ __forceinline__ float dpp_xor1(float v)
{
    float r;
    asm volatile(
        "v_mov_b32 %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1"
        : "=v"(r) : "v"(v));
    return r;
}

// DPP quad permute: swap halves within each 4-lane quad (XOR-2)
// quad_perm:[2,3,0,1] means: lane 0→2, 1→3, 2→0, 3→1
__device__ __forceinline__ float dpp_xor2(float v)
{
    float r;
    asm volatile(
        "v_mov_b32 %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1"
        : "=v"(r) : "v"(v));
    return r;
}

// __shfl_xor wrapper for cross-lane XOR with arbitrary stride
// Compiles to ds_bpermute_b32 on RDNA3 — correct for all strides.
__device__ __forceinline__ float shfl_xor(float v, int mask)
{
    return __shfl_xor(v, mask);
}


// -----------------------------------------------------------------
// Butterfly Hadamard: 4 local values + 5 DPP/ds_swizzle rounds
// -----------------------------------------------------------------

// Local 4-element Hadamard transform (in registers)
__device__ __forceinline__ void had4_local(float& v0, float& v1, float& v2, float& v3)
{
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v0 = s0 + s1;
    v1 = d0 + d1;
    v2 = s0 - s1;
    v3 = d0 - d1;
}

// Templated butterfly round using DPP (strides 1-2) or __shfl_xor (strides 4-16)
// For each round with stride S, lanes where (lane & S) != 0 compute
// partner - val instead of val + partner.
template<int STRIDE>
__device__ __forceinline__ void butterfly_dpp(
    float& v0, float& v1, float& v2, float& v3, int lane)
{
    float p0, p1, p2, p3;

    if constexpr (STRIDE == 1)
    {
        // DPP quad_perm: fused register-path shuffle (1 instruction, 1 cycle)
        p0 = dpp_xor1(v0); p1 = dpp_xor1(v1);
        p2 = dpp_xor1(v2); p3 = dpp_xor1(v3);
    }
    else if constexpr (STRIDE == 2)
    {
        // DPP quad_perm: fused register-path shuffle (1 instruction, 1 cycle)
        p0 = dpp_xor2(v0); p1 = dpp_xor2(v1);
        p2 = dpp_xor2(v2); p3 = dpp_xor2(v3);
    }
    else
    {
        // __shfl_xor for strides 4, 8, 16 (faster than ds_swizzle on RDNA3)
        p0 = shfl_xor(v0, STRIDE);
        p1 = shfl_xor(v1, STRIDE);
        p2 = shfl_xor(v2, STRIDE);
        p3 = shfl_xor(v3, STRIDE);
    }

    // Sign-flip: lanes where (lane & STRIDE) != 0 compute partner - val
    // Compiler maps this to v_cndmask_b32 (branchless)
    bool flip = (lane & STRIDE) != 0;
    v0 = flip ? (p0 - v0) : (v0 + p0);
    v1 = flip ? (p1 - v1) : (v1 + p1);
    v2 = flip ? (p2 - v2) : (v2 + p2);
    v3 = flip ? (p3 - v3) : (v3 + p3);
}

// 5 rounds of DPP + __shfl_xor butterfly across 32 lanes.
// Rounds 1-2: DPP quad_perm (register-path, fused)
// Rounds 3-5: __shfl_xor (ds_bpermute_b32, LDS-path)
// Computes Hadamard-32 for each of the 4 values per thread.
__device__ __forceinline__ void dpp_had32(
    float& v0, float& v1, float& v2, float& v3,
    int lane)
{
    butterfly_dpp<1>(v0, v1, v2, v3, lane);
    butterfly_dpp<2>(v0, v1, v2, v3, lane);
    butterfly_dpp<4>(v0, v1, v2, v3, lane);
    butterfly_dpp<8>(v0, v1, v2, v3, lane);
    butterfly_dpp<16>(v0, v1, v2, v3, lane);
}

// Clamp to fp16 range
__device__ __forceinline__ __half clamp_f16(float v)
{
    return __float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f));
}


// -----------------------------------------------------------------
// Dense Had-128 kernel (DPP butterfly version)
// -----------------------------------------------------------------
// Grid: (num_rows, dim / 128)
// Block: 32 threads (one warp)
// -----------------------------------------------------------------

__global__ __launch_bounds__(WARP_SIZE)
void had_r_128_kernel(
    const __half* __restrict__ input,
    __half*       __restrict__ output,
    const __half* __restrict__ pre_scale,   // may be nullptr
    const __half* __restrict__ post_scale,  // may be nullptr
    float                      combined_scale,  // r_scale * RSQRT_128
    int                        dim,
    int                        num_rows)
{
    int row = blockIdx.x;
    int blk = blockIdx.y;   // which 128-element block within the row
    int lane = threadIdx.x; // 0..31

    if (row >= num_rows) { return; }

    int block_offset = row * dim + blk * HAD_BLOCK;
    int elem_offset = block_offset + lane * VALS_PER_THREAD;

    // 1. Load 4 fp16 -> float
    float v0 = __half2float(input[elem_offset + 0]);
    float v1 = __half2float(input[elem_offset + 1]);
    float v2 = __half2float(input[elem_offset + 2]);
    float v3 = __half2float(input[elem_offset + 3]);

    // 2. Pre-scale (sign flips before Hadamard)
    if (pre_scale != nullptr)
    {
        int scale_offset = blk * HAD_BLOCK + lane * VALS_PER_THREAD;
        v0 *= __half2float(pre_scale[scale_offset + 0]);
        v1 *= __half2float(pre_scale[scale_offset + 1]);
        v2 *= __half2float(pre_scale[scale_offset + 2]);
        v3 *= __half2float(pre_scale[scale_offset + 3]);
    }

    // 3. Local 4-element Hadamard in registers
    had4_local(v0, v1, v2, v3);

    // 4. 5 rounds of DPP/ds_swizzle butterfly (Hadamard-32)
    dpp_had32(v0, v1, v2, v3, lane);

    // 5. Scale by r_scale / sqrt(128)
    v0 *= combined_scale;
    v1 *= combined_scale;
    v2 *= combined_scale;
    v3 *= combined_scale;

    // 6. Post-scale (sign flips after Hadamard)
    if (post_scale != nullptr)
    {
        int scale_offset = blk * HAD_BLOCK + lane * VALS_PER_THREAD;
        v0 *= __half2float(post_scale[scale_offset + 0]);
        v1 *= __half2float(post_scale[scale_offset + 1]);
        v2 *= __half2float(post_scale[scale_offset + 2]);
        v3 *= __half2float(post_scale[scale_offset + 3]);
    }

    // 7. Store as fp16 (clamped to prevent inf)
    output[elem_offset + 0] = clamp_f16(v0);
    output[elem_offset + 1] = clamp_f16(v1);
    output[elem_offset + 2] = clamp_f16(v2);
    output[elem_offset + 3] = clamp_f16(v3);
}


// =====================================================================
// Batched Had-128 kernel for fused MoE (DPP butterfly, multi-row)
//
// Applies Hadamard-128 transform to sorted MoE tokens with
// per-expert scale vectors. One launch processes all experts.
//
// Multi-row: each block processes ROWS_PER_BLOCK consecutive rows,
// reducing block count by 8× (e.g., 248K → 31K for M=15504, dim=2048).
// This eliminates the block-scheduling overhead that dominates at
// large M (prefill). For decode (M=1), loop runs once — no penalty.
//
// Grid: (ceildiv(M, ROWS_PER_BLOCK), dim / 128)
// Block: 32 threads (one warp)
//
// Supports both pre-scale (suh) and post-scale (svh) modes.
// Pre-scale: scale → Hadamard → rsqrt(128)
// Post-scale: Hadamard → rsqrt(128) → scale
// =====================================================================

#define ROWS_PER_BLOCK 64

template<int RPB>
__global__ __launch_bounds__(WARP_SIZE)
void batched_had_r_128_dpp_kernel(
    const __half* __restrict__ x_sorted,       // (EM, dim) fp16
    const __half* __restrict__ scale_stacked,  // (E, dim) fp16
    __half*       __restrict__ output,         // (EM, dim) fp16
    const int*    __restrict__ expert_ids,     // (EM,) expert ID per row
    int                        dim,
    int                        M,
    int                        PRE)            // 1=pre-scale, 0=post-scale
{
    int row_base = blockIdx.x * RPB;
    int blk = blockIdx.y;  // which 128-element block
    int lane = threadIdx.x;
    int blk_off = blk * HAD_BLOCK + lane * VALS_PER_THREAD;

    for (int r = 0; r < RPB; r++)
    {
        int row = row_base + r;
        if (row >= M) return;

        // Clamp expert ID (padding rows have -1)
        int eid = expert_ids[row];
        eid = (eid < 0) ? 0 : eid;

        int base = row * dim + blk_off;
        int s_base = eid * dim + blk_off;

        // Load 4 fp16 → float
        float v0 = __half2float(x_sorted[base + 0]);
        float v1 = __half2float(x_sorted[base + 1]);
        float v2 = __half2float(x_sorted[base + 2]);
        float v3 = __half2float(x_sorted[base + 3]);

        // Load scale
        float s0 = __half2float(scale_stacked[s_base + 0]);
        float s1 = __half2float(scale_stacked[s_base + 1]);
        float s2 = __half2float(scale_stacked[s_base + 2]);
        float s3 = __half2float(scale_stacked[s_base + 3]);

        // Pre-scale
        if (PRE) { v0 *= s0; v1 *= s1; v2 *= s2; v3 *= s3; }

        // Butterfly Had-128 = Had-4 (local) × Had-32 (cross-lane DPP)
        had4_local(v0, v1, v2, v3);
        dpp_had32(v0, v1, v2, v3, lane);

        // Normalize
        v0 *= RSQRT_128; v1 *= RSQRT_128; v2 *= RSQRT_128; v3 *= RSQRT_128;

        // Post-scale
        if (!PRE) { v0 *= s0; v1 *= s1; v2 *= s2; v3 *= s3; }

        // Store
        output[base + 0] = clamp_f16(v0);
        output[base + 1] = clamp_f16(v1);
        output[base + 2] = clamp_f16(v2);
        output[base + 3] = clamp_f16(v3);
    }
}


// =====================================================================
// Host launchers
// =====================================================================

void hip_had_r_128(
    at::Tensor x,
    at::Tensor out,
    c10::optional<at::Tensor> pre_scale,
    c10::optional<at::Tensor> post_scale,
    float r_scale)
{
    TORCH_CHECK(x.is_contiguous(), "had_r_128: x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "had_r_128: out must be contiguous");
    TORCH_CHECK(x.dtype() == at::kHalf, "had_r_128: x must be float16");
    TORCH_CHECK(out.dtype() == at::kHalf, "had_r_128: out must be float16");

    int ndim = x.dim();
    TORCH_CHECK(ndim >= 1, "had_r_128: x must be at least 1D");
    int dim = x.size(ndim - 1);
    TORCH_CHECK(dim % HAD_BLOCK == 0,
                "had_r_128: last dimension must be divisible by 128, got ", dim);

    int64_t numel = x.numel();
    int num_rows = numel / dim;
    int num_blocks = dim / HAD_BLOCK;

    TORCH_CHECK(out.numel() == numel,
                "had_r_128: output must have same number of elements as input");

    const __half* pre_ptr = nullptr;
    if (pre_scale.has_value() && pre_scale->defined())
    {
        TORCH_CHECK(pre_scale->is_contiguous(), "had_r_128: pre_scale must be contiguous");
        TORCH_CHECK(pre_scale->dtype() == at::kHalf, "had_r_128: pre_scale must be float16");
        TORCH_CHECK(pre_scale->numel() == dim,
                    "had_r_128: pre_scale must have ", dim, " elements, got ", pre_scale->numel());
        pre_ptr = reinterpret_cast<const __half*>(pre_scale->data_ptr());
    }

    const __half* post_ptr = nullptr;
    if (post_scale.has_value() && post_scale->defined())
    {
        TORCH_CHECK(post_scale->is_contiguous(), "had_r_128: post_scale must be contiguous");
        TORCH_CHECK(post_scale->dtype() == at::kHalf, "had_r_128: post_scale must be float16");
        TORCH_CHECK(post_scale->numel() == dim,
                    "had_r_128: post_scale must have ", dim, " elements, got ", post_scale->numel());
        post_ptr = reinterpret_cast<const __half*>(post_scale->data_ptr());
    }

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    float combined_scale = r_scale * RSQRT_128;

    dim3 grid(num_rows, num_blocks);
    dim3 block(WARP_SIZE);

    had_r_128_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        pre_ptr,
        post_ptr,
        combined_scale,
        dim,
        num_rows
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "had_r_128 kernel launch failed: ", cudaGetErrorString(err));
}


void hip_batched_had_r_128(
    at::Tensor x_sorted,       // (EM, dim) fp16
    at::Tensor scale_stacked,  // (E, dim) fp16
    at::Tensor output,         // (EM, dim) fp16
    at::Tensor expert_ids,     // (EM,) int32
    int pre)                   // 1=pre-scale, 0=post-scale
{
    TORCH_CHECK(x_sorted.is_contiguous(), "batched_had_r_128: x_sorted must be contiguous");
    TORCH_CHECK(scale_stacked.is_contiguous(), "batched_had_r_128: scale_stacked must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "batched_had_r_128: output must be contiguous");
    TORCH_CHECK(x_sorted.dtype() == at::kHalf, "batched_had_r_128: x_sorted must be float16");
    TORCH_CHECK(scale_stacked.dtype() == at::kHalf, "batched_had_r_128: scale_stacked must be float16");
    TORCH_CHECK(output.dtype() == at::kHalf, "batched_had_r_128: output must be float16");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "batched_had_r_128: expert_ids must be int32");

    int M = x_sorted.size(0);
    int dim = x_sorted.size(1);
    TORCH_CHECK(dim % HAD_BLOCK == 0,
                "batched_had_r_128: dim must be divisible by 128, got ", dim);
    TORCH_CHECK(output.size(0) == M && output.size(1) == dim,
                "batched_had_r_128: output shape must match x_sorted");
    TORCH_CHECK(expert_ids.size(0) == M,
                "batched_had_r_128: expert_ids must have M elements");

    if (M == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(x_sorted.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int num_blocks = dim / HAD_BLOCK;
    int row_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    dim3 grid(row_blocks, num_blocks);
    dim3 block_dim(WARP_SIZE);

    batched_had_r_128_dpp_kernel<ROWS_PER_BLOCK><<<grid, block_dim, 0, stream>>>(
        reinterpret_cast<const __half*>(x_sorted.data_ptr()),
        reinterpret_cast<const __half*>(scale_stacked.data_ptr()),
        reinterpret_cast<__half*>(output.data_ptr()),
        reinterpret_cast<const int*>(expert_ids.data_ptr()),
        dim,
        M,
        pre
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "batched_had_r_128 kernel launch failed: ", cudaGetErrorString(err));
}
