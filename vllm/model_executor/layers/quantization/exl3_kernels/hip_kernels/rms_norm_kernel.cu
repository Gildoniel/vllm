#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// =====================================================================
// Fused RMSNorm kernel for RDNA3 (gfx1100)
// Ported from exllamav3/exllamav3_ext/norm.cu
//
// One block per row. Two passes:
//   1) Sum of squares with warp-shuffle + shared-mem reduction
//   2) Normalize * weight, write output
// =====================================================================

// -- Vectorized read/write helpers for fp16 (via uint2 = 4 halfs) -----

__device__ __forceinline__ void read_half4(float4& f4, const __half* addr)
{
    // Load 4 halfs (8 bytes) as uint2
    uint2 raw = *reinterpret_cast<const uint2*>(addr);
    __half2 h2_lo = *reinterpret_cast<const __half2*>(&raw.x);
    __half2 h2_hi = *reinterpret_cast<const __half2*>(&raw.y);
    f4.x = __half2float(__low2half(h2_lo));
    f4.y = __half2float(__high2half(h2_lo));
    f4.z = __half2float(__low2half(h2_hi));
    f4.w = __half2float(__high2half(h2_hi));
}

__device__ __forceinline__ void write_half4(const float4& f4, __half* addr)
{
    __half2 h2_lo = __halves2half2(__float2half_rn(f4.x), __float2half_rn(f4.y));
    __half2 h2_hi = __halves2half2(__float2half_rn(f4.z), __float2half_rn(f4.w));
    uint2 raw;
    raw.x = *reinterpret_cast<const unsigned int*>(&h2_lo);
    raw.y = *reinterpret_cast<const unsigned int*>(&h2_hi);
    *reinterpret_cast<uint2*>(addr) = raw;
}

// -- Vectorized read/write for fp32 -----------------------------------

__device__ __forceinline__ void read_float4(float4& f4, const float* addr)
{
    f4 = *reinterpret_cast<const float4*>(addr);
}

__device__ __forceinline__ void write_float4(const float4& f4, float* addr)
{
    *reinterpret_cast<float4*>(addr) = f4;
}

// -- Math helpers -----------------------------------------------------

__device__ __forceinline__ float sum_sq4(float acc, const float4& f4)
{
    acc = fma(f4.x, f4.x, acc);
    acc = fma(f4.y, f4.y, acc);
    acc = fma(f4.z, f4.z, acc);
    acc = fma(f4.w, f4.w, acc);
    return acc;
}

// -- Warp + block reduction -------------------------------------------

template <int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum(float val)
{
    const int warp_id = threadIdx.x / warpSize;  // warpSize=32 on RDNA3
    const int lane_id = threadIdx.x % warpSize;

    // Warp-level reduction via butterfly shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset);

    if constexpr (NUM_THREADS <= 32)
        return val;

    // Cross-warp reduction via shared memory
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    __shared__ float shared[NUM_WARPS];
    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();

    val = (lane_id < NUM_WARPS) ? shared[lane_id] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset);

    return val;
}

// -- Main kernel ------------------------------------------------------

template <typename input_t, typename output_t, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void rms_norm_kernel(
    const input_t* __restrict__ x,
    const __half*  __restrict__ w,   // weight (may be nullptr)
    output_t*      __restrict__ y,
    const float    epsilon,
    const int      rows,
    const int      dim,
    const float    constant_bias)
{
    constexpr bool in_fp16  = std::is_same_v<input_t,  __half>;
    constexpr bool out_fp16 = std::is_same_v<output_t, __half>;

    const int t   = threadIdx.x;
    const int row = blockIdx.x;
    const int columns = dim / 4;  // number of float4 chunks

    const input_t* x_row = x + (size_t)row * dim;

    // ---- Pass 1: sum of squares ------------------------------------
    float sum = 0.0f;
    for (int col = t; col < columns; col += NUM_THREADS)
    {
        float4 v;
        if constexpr (in_fp16)
            read_half4(v, reinterpret_cast<const __half*>(x_row) + col * 4);
        else
            read_float4(v, reinterpret_cast<const float*>(x_row) + col * 4);
        sum = sum_sq4(sum, v);
    }
    sum = block_reduce_sum<NUM_THREADS>(sum);

    const float rmf = rsqrtf(sum / (float)dim + epsilon);

    // ---- Pass 2: normalize (& scale by weight) and write -----------
    output_t* y_row = y + (size_t)row * dim;

    for (int col = t; col < columns; col += NUM_THREADS)
    {
        float4 v;
        if constexpr (in_fp16)
            read_half4(v, reinterpret_cast<const __half*>(x_row) + col * 4);
        else
            read_float4(v, reinterpret_cast<const float*>(x_row) + col * 4);

        if (w)
        {
            float4 wv;
            read_half4(wv, w + col * 4);
            if (constant_bias != 0.0f)
            {
                wv.x += constant_bias;
                wv.y += constant_bias;
                wv.z += constant_bias;
                wv.w += constant_bias;
            }
            v.x = v.x * wv.x * rmf;
            v.y = v.y * wv.y * rmf;
            v.z = v.z * wv.z * rmf;
            v.w = v.w * wv.w * rmf;
        }
        else
        {
            v.x *= rmf;
            v.y *= rmf;
            v.z *= rmf;
            v.w *= rmf;
        }

        if constexpr (out_fp16)
            write_half4(v, reinterpret_cast<__half*>(y_row) + col * 4);
        else
            write_float4(v, reinterpret_cast<float*>(y_row) + col * 4);
    }
}

// =====================================================================
// Host launcher
// =====================================================================

void hip_rms_norm(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    bool span_heads)
{
    if (span_heads)
    {
        x = x.flatten(-2);
        y = y.flatten(-2);
    }

    TORCH_CHECK(x.dim() >= 1, "rms_norm: x must have at least 1 dimension");
    TORCH_CHECK(x.size(-1) % 4 == 0, "rms_norm: last dim must be divisible by 4");
    TORCH_CHECK(x.is_contiguous(), "rms_norm: x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "rms_norm: y must be contiguous");

    const __half* w_ptr = nullptr;
    if (w.has_value() && w->defined())
    {
        TORCH_CHECK(w->dtype() == at::kHalf, "rms_norm: weight must be float16");
        TORCH_CHECK(w->size(0) == x.size(-1), "rms_norm: weight size must match last dim of x");
        w_ptr = reinterpret_cast<const __half*>(w->data_ptr());
    }

    bool input_fp16  = (x.dtype() == at::kHalf);
    bool output_fp16 = (y.dtype() == at::kHalf);
    TORCH_CHECK(input_fp16 || x.dtype() == at::kFloat,
                "rms_norm: input must be float16 or float32");
    TORCH_CHECK(output_fp16 || y.dtype() == at::kFloat,
                "rms_norm: output must be float16 or float32");

    int rows = 1;
    for (int i = 0; i < x.dim() - 1; ++i)
        rows *= x.size(i);
    int dim = x.size(-1);

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // 256 threads = 8 warps, good occupancy on RDNA3
    constexpr int NUM_THREADS = 256;
    dim3 grid(rows);
    dim3 block(NUM_THREADS);

    #define LAUNCH(IN_T, OUT_T) \
        rms_norm_kernel<IN_T, OUT_T, NUM_THREADS><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const IN_T*>(x.data_ptr()), \
            w_ptr, \
            reinterpret_cast<OUT_T*>(y.data_ptr()), \
            epsilon, rows, dim, constant_bias)

    if (input_fp16 && output_fp16)
        LAUNCH(__half, __half);
    else if (input_fp16 && !output_fp16)
        LAUNCH(__half, float);
    else if (!input_fp16 && output_fp16)
        LAUNCH(float, __half);
    else
        LAUNCH(float, float);

    #undef LAUNCH

    // Check for launch errors (cudaError_t is aliased to hipError_t on ROCm)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "rms_norm kernel launch failed: ", cudaGetErrorString(err));
}
