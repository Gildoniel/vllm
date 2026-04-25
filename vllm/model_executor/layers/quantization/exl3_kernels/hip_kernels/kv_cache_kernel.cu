#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// =====================================================================
// Fused paged KV cache quantize/dequantize kernel for RDNA3 (gfx1100)
//
// Replaces the pure-PyTorch quant_cache_paged / dequant_cache_paged
// with a single-launch HIP kernel using warp shuffles.
//
// Per 32-element block:
//   Quant:   load fp16 → Hadamard-32 (butterfly) → absmax → quantize → ballot pack
//   Dequant: unpack (shfl broadcast) → scale → Hadamard-32 (butterfly) → store fp16
//
// 1 warp (32 threads) processes one 32-element block.
// No shared memory needed — pure registers + warp shuffles.
// =====================================================================

#define WARP_SIZE 32
#define CQ_PAGE_SIZE 256

// 1/sqrt(32)
static constexpr float RSQRT32 = 0.17677669529663688f;

// =====================================================================
// Device helpers
// =====================================================================

// Butterfly Hadamard-32 via warp shuffles (5 rounds)
// Each thread holds 1 float value. After 5 XOR-shuffle rounds,
// the 32 values across the warp are Hadamard-transformed.
__device__ __forceinline__ float shuffle_had_32(float v, int lane)
{
    #pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1)
    {
        float p = __shfl_xor(v, stride);
        bool flip = (lane & stride) != 0;
        // v = flip ? (p - v) : (v + p)
        unsigned int u = __float_as_uint(v);
        if (flip) { u ^= 0x80000000u; }
        v = __uint_as_float(u) + p;
    }
    return v;
}

// Warp-reduce max (absolute value) across 32 lanes
__device__ __forceinline__ float shuffle_absmax_32(float v)
{
    float a = fabsf(v);
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1)
    {
        float p = __shfl_xor(a, offset);
        a = fmaxf(a, p);
    }
    // Now all lanes have the max
    return a;
}

// =====================================================================
// Dequant kernel — one warp per 32-element block
// =====================================================================
//
// Template parameter BITS selects the bit width (2-8).
//
// Layout:
//   q_in:  (num_pages, page_size, num_blocks_per_token * bits) int32
//   s_in:  (num_pages, page_size, num_blocks_per_token) fp16
//   fp_out: (num_pages, page_size, token_dim) fp16
//   block_table: (batch, pages_per_seq) int32
//   cache_seqlens: (batch,) int32
//
// Grid:  (total_warps_needed, 1, 2)   — z=0 for K, z=1 for V
// Block: (WARP_SIZE * WARPS_PER_CTA)
//
// Each warp processes one (batch_elem, token, block_within_token).

template <int K_BITS, int V_BITS>
__global__ void dequant_cache_paged_kernel(
    const int*    __restrict__ qk,
    const __half* __restrict__ sk,
    __half*       __restrict__ k_out,
    const int*    __restrict__ qv,
    const __half* __restrict__ sv,
    __half*       __restrict__ v_out,
    const int*    __restrict__ cache_seqlens,
    const int*    __restrict__ block_table,
    int                        pages_per_seq,
    int                        num_blocks_per_token,
    int                        token_dim,
    int                        page_size,
    int                        q_stride_k,     // qk dim2: num_blocks_per_token * k_bits
    int                        q_stride_v,     // qv dim2: num_blocks_per_token * v_bits
    int                        batch_size,
    const int*   __restrict__  batch_offsets,   // prefix-sum of seqlens (batch+1,)
    int                        total_tokens)    // sum of all seqlens
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Total work items = total_tokens * num_blocks_per_token
    int total_work = total_tokens * num_blocks_per_token;
    if (global_warp_id >= total_work) return;

    // Decompose into token index and block-within-token
    int flat_token = global_warp_id / num_blocks_per_token;
    int blk_idx    = global_warp_id % num_blocks_per_token;

    // Find which batch element this token belongs to via binary search on batch_offsets
    int b = 0;
    {
        int lo = 0, hi = batch_size;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (batch_offsets[mid + 1] <= flat_token)
                lo = mid + 1;
            else
                hi = mid;
        }
        b = lo;
    }

    int token_within_seq = flat_token - batch_offsets[b];
    // Resolve physical page
    int logical_page = token_within_seq / page_size;
    int page_off     = token_within_seq % page_size;
    int phys_page    = block_table[b * pages_per_seq + logical_page];

    // Base index into the (num_pages, page_size, ...) tensor
    int page_token_idx = phys_page * page_size + page_off;

    // Select K or V based on blockIdx.z
    const int* q_in;
    const __half* s_in;
    __half* fp_out;
    int bits;
    int q_stride;

    if (blockIdx.z == 0)
    {
        q_in    = qk;
        s_in    = sk;
        fp_out  = k_out;
        bits    = K_BITS;
        q_stride = q_stride_k;
    }
    else
    {
        q_in    = qv;
        s_in    = sv;
        fp_out  = v_out;
        bits    = V_BITS;
        q_stride = q_stride_v;
    }

    // Load scale for this block
    float scale = __half2float(s_in[page_token_idx * num_blocks_per_token + blk_idx]);

    // Load bitplane words: q_in[page_token_idx, blk_idx * bits + i]
    // Each lane < bits loads one bitplane word
    int q_base = page_token_idx * q_stride + blk_idx * bits;
    int my_word = 0;
    if (lane < bits)
    {
        my_word = q_in[q_base + lane];
    }

    // Unpack via shfl broadcast: for each bit plane i, broadcast word from lane i
    // then extract this lane's bit
    float center = (float)(1 << (bits - 1));
    float v = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < bits; i++)
    {
        // Only unroll up to bits, but we unroll max 8 and break
        if (i >= bits) break;
        int word = __shfl(my_word, i);
        int bit_val = (word >> lane) & 1;
        v += (float)bit_val * (float)(1 << i);
    }

    // Dequantize: v = (q - center) * (rsqrt32 / center) * scale
    v = (v - center) * (RSQRT32 / center) * scale;

    // Inverse Hadamard-32 via butterfly
    v = shuffle_had_32(v, lane);

    // Store fp16 (clamped)
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    fp_out[page_token_idx * token_dim + blk_idx * WARP_SIZE + lane] = __float2half_rn(v);
}


// =====================================================================
// Quant kernel — one warp per 32-element block
// =====================================================================

template <int K_BITS, int V_BITS>
__global__ void quant_cache_paged_kernel(
    const __half* __restrict__ k_in,
    int*          __restrict__ qk,
    __half*       __restrict__ sk,
    const __half* __restrict__ v_in,
    int*          __restrict__ qv,
    __half*       __restrict__ sv,
    const int*    __restrict__ cache_seqlens,
    const int*    __restrict__ block_table,
    int                        pages_per_seq,
    int                        num_blocks_per_token,
    int                        token_dim,
    int                        page_size,
    int                        q_stride_k,
    int                        q_stride_v,
    int                        batch_size,
    int                        length)        // tokens to quant per batch element
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Total work = batch_size * length * num_blocks_per_token
    int total_work = batch_size * length * num_blocks_per_token;
    if (global_warp_id >= total_work) return;

    // Decompose
    int flat_token = global_warp_id / num_blocks_per_token;
    int blk_idx    = global_warp_id % num_blocks_per_token;
    int b          = flat_token / length;
    int tok_offset = flat_token % length;

    // Token position = cache_seqlens[b] + tok_offset
    int token_pos = cache_seqlens[b] + tok_offset;

    // Resolve physical page
    int logical_page = token_pos / page_size;
    int page_off     = token_pos % page_size;
    int phys_page    = block_table[b * pages_per_seq + logical_page];
    int page_token_idx = phys_page * page_size + page_off;

    // Select K or V
    const __half* fp_in;
    int* q_out;
    __half* s_out;
    int bits;
    int q_stride;

    if (blockIdx.z == 0)
    {
        fp_in   = k_in;
        q_out   = qk;
        s_out   = sk;
        bits    = K_BITS;
        q_stride = q_stride_k;
    }
    else
    {
        fp_in   = v_in;
        q_out   = qv;
        s_out   = sv;
        bits    = V_BITS;
        q_stride = q_stride_v;
    }

    // Load one fp16 value
    float v = __half2float(fp_in[page_token_idx * token_dim + blk_idx * WARP_SIZE + lane]);

    // Forward Hadamard-32
    v = shuffle_had_32(v, lane);

    // Scale by 1/sqrt(32)
    v *= RSQRT32;

    // Warp-reduce absmax
    float amax = shuffle_absmax_32(v);
    amax = fmaxf(amax, 1e-10f);  // avoid div-by-zero

    // Store scale (only lane 0 writes, but all lanes have same amax)
    if (lane == 0)
    {
        s_out[page_token_idx * num_blocks_per_token + blk_idx] =
            __float2half_rn(fminf(amax, 65504.0f));
    }

    // Normalize to [-1, 1] and quantize
    float norm = v / amax;
    int center = 1 << (bits - 1);
    int q = __float2int_rn(norm * (float)center) + center;
    int max_val = (1 << bits) - 1;
    q = max(0, min(q, max_val));

    // Bitplane pack via __ballot
    int q_base = page_token_idx * q_stride + blk_idx * bits;
    for (int i = 0; i < bits; i++)
    {
        int bit_set = (q >> i) & 1;
        // __ballot returns a 64-bit value on HIP, cast to int for 32-lane usage
        unsigned long long ballot_result = __ballot(bit_set);
        int word = (int)(ballot_result & 0xFFFFFFFFu);
        if (lane == 0)
        {
            q_out[q_base + i] = word;
        }
    }
}


// =====================================================================
// Template dispatch macro
// =====================================================================

#define DISPATCH_BITS(K_BITS, V_BITS, KERNEL, ...) \
    if (k_bits == K_BITS && v_bits == V_BITS) { \
        KERNEL<K_BITS, V_BITS><<<grid, block, 0, stream>>>(__VA_ARGS__); \
    }

#define DISPATCH_ALL_V(KB, KERNEL, ...) \
    DISPATCH_BITS(KB, 2, KERNEL, __VA_ARGS__) else \
    DISPATCH_BITS(KB, 3, KERNEL, __VA_ARGS__) else \
    DISPATCH_BITS(KB, 4, KERNEL, __VA_ARGS__) else \
    DISPATCH_BITS(KB, 5, KERNEL, __VA_ARGS__) else \
    DISPATCH_BITS(KB, 6, KERNEL, __VA_ARGS__) else \
    DISPATCH_BITS(KB, 7, KERNEL, __VA_ARGS__) else \
    DISPATCH_BITS(KB, 8, KERNEL, __VA_ARGS__)

#define DISPATCH_KV_BITS(KERNEL, ...) \
    DISPATCH_ALL_V(2, KERNEL, __VA_ARGS__) else \
    DISPATCH_ALL_V(3, KERNEL, __VA_ARGS__) else \
    DISPATCH_ALL_V(4, KERNEL, __VA_ARGS__) else \
    DISPATCH_ALL_V(5, KERNEL, __VA_ARGS__) else \
    DISPATCH_ALL_V(6, KERNEL, __VA_ARGS__) else \
    DISPATCH_ALL_V(7, KERNEL, __VA_ARGS__) else \
    DISPATCH_ALL_V(8, KERNEL, __VA_ARGS__) else \
    { TORCH_CHECK(false, "Unsupported k_bits=", k_bits, " v_bits=", v_bits); }


// =====================================================================
// Host launchers
// =====================================================================

void hip_dequant_cache_paged(
    at::Tensor qk,
    at::Tensor sk,
    at::Tensor k_out,
    at::Tensor qv,
    at::Tensor sv,
    at::Tensor v_out,
    at::Tensor cache_seqlens,
    at::Tensor block_table,
    int page_size)
{
    TORCH_CHECK(qk.is_contiguous(), "qk must be contiguous");
    TORCH_CHECK(sk.is_contiguous(), "sk must be contiguous");
    TORCH_CHECK(k_out.is_contiguous(), "k_out must be contiguous");
    TORCH_CHECK(qv.is_contiguous(), "qv must be contiguous");
    TORCH_CHECK(sv.is_contiguous(), "sv must be contiguous");
    TORCH_CHECK(v_out.is_contiguous(), "v_out must be contiguous");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");

    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "page_size must be ", CQ_PAGE_SIZE);

    int batch_size = cache_seqlens.size(0);
    int pages_per_seq = block_table.size(1);

    // Determine token_dim from k_out shape
    int token_dim;
    if (k_out.dim() == 4)
        token_dim = k_out.size(2) * k_out.size(3);
    else
        token_dim = k_out.size(2);

    int num_blocks_per_token = token_dim / WARP_SIZE;
    int k_bits = qk.size(2) / num_blocks_per_token;
    int v_bits = qv.size(2) / num_blocks_per_token;

    TORCH_CHECK(k_bits >= 2 && k_bits <= 8, "k_bits must be 2-8, got ", k_bits);
    TORCH_CHECK(v_bits >= 2 && v_bits <= 8, "v_bits must be 2-8, got ", v_bits);

    // Compute batch_offsets (prefix sum of seqlens) on CPU
    auto seqlens_cpu = cache_seqlens.to(at::kCPU, at::kInt);
    int* sl = seqlens_cpu.data_ptr<int>();

    int total_tokens = 0;
    std::vector<int> offsets(batch_size + 1);
    offsets[0] = 0;
    for (int b = 0; b < batch_size; b++)
    {
        total_tokens += sl[b];
        offsets[b + 1] = total_tokens;
    }

    if (total_tokens == 0) return;

    // Upload batch_offsets to GPU
    auto batch_offsets = torch::from_blob(offsets.data(), {batch_size + 1},
                                          torch::TensorOptions().dtype(torch::kInt32))
                            .to(qk.device());

    const at::cuda::OptionalCUDAGuard device_guard(qk.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int total_work = total_tokens * num_blocks_per_token;
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * WARP_SIZE;
    int num_blocks_grid = (total_work + warps_per_block - 1) / warps_per_block;

    // z=2: one for K, one for V
    dim3 grid(num_blocks_grid, 1, 2);
    dim3 block(threads_per_block);

    int q_stride_k = qk.size(2);
    int q_stride_v = qv.size(2);

    DISPATCH_KV_BITS(dequant_cache_paged_kernel,
        qk.data_ptr<int>(),
        reinterpret_cast<const __half*>(sk.data_ptr()),
        reinterpret_cast<__half*>(k_out.data_ptr()),
        qv.data_ptr<int>(),
        reinterpret_cast<const __half*>(sv.data_ptr()),
        reinterpret_cast<__half*>(v_out.data_ptr()),
        cache_seqlens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        pages_per_seq,
        num_blocks_per_token,
        token_dim,
        page_size,
        q_stride_k,
        q_stride_v,
        batch_size,
        batch_offsets.data_ptr<int>(),
        total_tokens
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "dequant_cache_paged kernel launch failed: ", cudaGetErrorString(err));
}


void hip_quant_cache_paged(
    at::Tensor k_in,
    at::Tensor qk,
    at::Tensor sk,
    at::Tensor v_in,
    at::Tensor qv,
    at::Tensor sv,
    at::Tensor cache_seqlens,
    at::Tensor block_table,
    int page_size,
    int length)
{
    TORCH_CHECK(k_in.is_contiguous(), "k_in must be contiguous");
    TORCH_CHECK(qk.is_contiguous(), "qk must be contiguous");
    TORCH_CHECK(sk.is_contiguous(), "sk must be contiguous");
    TORCH_CHECK(v_in.is_contiguous(), "v_in must be contiguous");
    TORCH_CHECK(qv.is_contiguous(), "qv must be contiguous");
    TORCH_CHECK(sv.is_contiguous(), "sv must be contiguous");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");

    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "page_size must be ", CQ_PAGE_SIZE);

    int batch_size = cache_seqlens.size(0);
    int pages_per_seq = block_table.size(1);

    int token_dim;
    if (k_in.dim() == 4)
        token_dim = k_in.size(2) * k_in.size(3);
    else
        token_dim = k_in.size(2);

    int num_blocks_per_token = token_dim / WARP_SIZE;
    int k_bits = qk.size(2) / num_blocks_per_token;
    int v_bits = qv.size(2) / num_blocks_per_token;

    TORCH_CHECK(k_bits >= 2 && k_bits <= 8, "k_bits must be 2-8, got ", k_bits);
    TORCH_CHECK(v_bits >= 2 && v_bits <= 8, "v_bits must be 2-8, got ", v_bits);

    if (length == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int total_work = batch_size * length * num_blocks_per_token;
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * WARP_SIZE;
    int num_blocks_grid = (total_work + warps_per_block - 1) / warps_per_block;

    dim3 grid(num_blocks_grid, 1, 2);
    dim3 block(threads_per_block);

    int q_stride_k = qk.size(2);
    int q_stride_v = qv.size(2);

    DISPATCH_KV_BITS(quant_cache_paged_kernel,
        reinterpret_cast<const __half*>(k_in.data_ptr()),
        qk.data_ptr<int>(),
        reinterpret_cast<__half*>(sk.data_ptr()),
        reinterpret_cast<const __half*>(v_in.data_ptr()),
        qv.data_ptr<int>(),
        reinterpret_cast<__half*>(sv.data_ptr()),
        cache_seqlens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        pages_per_seq,
        num_blocks_per_token,
        token_dim,
        page_size,
        q_stride_k,
        q_stride_v,
        batch_size,
        length
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "quant_cache_paged kernel launch failed: ", cudaGetErrorString(err));
}
