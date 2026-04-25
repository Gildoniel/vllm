#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// =====================================================================
// Fused activation * mul kernels for RDNA3 (gfx1100)
// Ported from exllamav3/exllamav3_ext/activation_kernels.cuh
//
//   z = act(g) * u
//
// Supports SiLU, GELU (tanh approx), ReLU² via template.
// Half and float inputs, half output. Vectorized via half2/float2.
// =====================================================================

#define NUM_THREADS 256
#define ACT_SILU  0
#define ACT_GELU  1
#define ACT_RELU2 2

// -- Activation functions ---------------------------------------------

__device__ __forceinline__ float _silu_f(float x)
{
    float e = __expf(-x);
    return x / (1.0f + e);
}

__device__ __forceinline__ float _gelu_f(float x)
{
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float tanh_arg = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

__device__ __forceinline__ float _relu2_f(float x)
{
    x = fmaxf(0.0f, x);
    return x * x;
}

// -- FP16 clamp (prevent inf in half) ---------------------------------

__device__ __forceinline__ __half clamp_half(float v)
{
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    return __float2half_rn(v);
}

// -- Kernel: half input -----------------------------------------------

template <int ACT_TYPE>
__global__ __launch_bounds__(NUM_THREADS)
void act_mul_kernel_h(
    const __half* __restrict__ g,
    const __half* __restrict__ u,
    __half*       __restrict__ z,
    const size_t numel)
{
    size_t idx = blockIdx.x * (size_t)NUM_THREADS + threadIdx.x;
    if (idx >= numel / 2) return;

    // Load 2 halfs at a time
    __half2 g2 = reinterpret_cast<const __half2*>(g)[idx];
    __half2 u2 = reinterpret_cast<const __half2*>(u)[idx];

    float gx = __half2float(__low2half(g2));
    float gy = __half2float(__high2half(g2));
    float ux = __half2float(__low2half(u2));
    float uy = __half2float(__high2half(u2));

    if constexpr (ACT_TYPE == ACT_SILU)  { gx = _silu_f(gx);  gy = _silu_f(gy);  }
    if constexpr (ACT_TYPE == ACT_GELU)  { gx = _gelu_f(gx);  gy = _gelu_f(gy);  }
    if constexpr (ACT_TYPE == ACT_RELU2) { gx = _relu2_f(gx);  gy = _relu2_f(gy); }

    __half2 r = __halves2half2(clamp_half(gx * ux), clamp_half(gy * uy));
    reinterpret_cast<__half2*>(z)[idx] = r;
}

// -- Kernel: float input, half output ---------------------------------

template <int ACT_TYPE>
__global__ __launch_bounds__(NUM_THREADS)
void act_mul_kernel_f(
    const float* __restrict__ g,
    const float* __restrict__ u,
    __half*      __restrict__ z,
    const size_t numel)
{
    size_t idx = blockIdx.x * (size_t)NUM_THREADS + threadIdx.x;
    if (idx >= numel / 2) return;

    float2 g2 = reinterpret_cast<const float2*>(g)[idx];
    float2 u2 = reinterpret_cast<const float2*>(u)[idx];

    if constexpr (ACT_TYPE == ACT_SILU)  { g2.x = _silu_f(g2.x);  g2.y = _silu_f(g2.y);  }
    if constexpr (ACT_TYPE == ACT_GELU)  { g2.x = _gelu_f(g2.x);  g2.y = _gelu_f(g2.y);  }
    if constexpr (ACT_TYPE == ACT_RELU2) { g2.x = _relu2_f(g2.x);  g2.y = _relu2_f(g2.y); }

    g2.x *= u2.x;
    g2.y *= u2.y;

    __half2 r = __halves2half2(clamp_half(g2.x), clamp_half(g2.y));
    reinterpret_cast<__half2*>(z)[idx] = r;
}

// =====================================================================
// Host launchers
// =====================================================================

static void _act_mul_dispatch(
    int act_type,
    at::Tensor g,
    at::Tensor u,
    at::Tensor z)
{
    TORCH_CHECK(g.is_contiguous(), "act_mul: g must be contiguous");
    TORCH_CHECK(u.is_contiguous(), "act_mul: u must be contiguous");
    TORCH_CHECK(z.is_contiguous(), "act_mul: z must be contiguous");
    TORCH_CHECK(z.dtype() == at::kHalf, "act_mul: output z must be float16");
    TORCH_CHECK(g.numel() == u.numel() && g.numel() == z.numel(),
                "act_mul: g, u, z must have same number of elements");
    TORCH_CHECK(g.numel() % 2 == 0, "act_mul: numel must be even");

    bool float_input = (g.dtype() == at::kFloat);
    if (float_input) {
        TORCH_CHECK(u.dtype() == at::kFloat, "act_mul: if g is float32, u must also be float32");
    } else {
        TORCH_CHECK(g.dtype() == at::kHalf, "act_mul: g must be float16 or float32");
        TORCH_CHECK(u.dtype() == at::kHalf, "act_mul: if g is float16, u must also be float16");
    }

    const at::cuda::OptionalCUDAGuard device_guard(g.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    size_t numel = g.numel();
    size_t blocks = (numel / 2 + NUM_THREADS - 1) / NUM_THREADS;

    const __half* g_h = reinterpret_cast<const __half*>(g.data_ptr());
    const __half* u_h = reinterpret_cast<const __half*>(u.data_ptr());
    const float*  g_f = reinterpret_cast<const float*>(g.data_ptr());
    const float*  u_f = reinterpret_cast<const float*>(u.data_ptr());
    __half*       z_h = reinterpret_cast<__half*>(z.data_ptr());

    if (float_input)
    {
        if      (act_type == ACT_SILU)  act_mul_kernel_f<ACT_SILU> <<<blocks, NUM_THREADS, 0, stream>>>(g_f, u_f, z_h, numel);
        else if (act_type == ACT_GELU)  act_mul_kernel_f<ACT_GELU> <<<blocks, NUM_THREADS, 0, stream>>>(g_f, u_f, z_h, numel);
        else if (act_type == ACT_RELU2) act_mul_kernel_f<ACT_RELU2><<<blocks, NUM_THREADS, 0, stream>>>(g_f, u_f, z_h, numel);
        else TORCH_CHECK(false, "act_mul: unknown activation type");
    }
    else
    {
        if      (act_type == ACT_SILU)  act_mul_kernel_h<ACT_SILU> <<<blocks, NUM_THREADS, 0, stream>>>(g_h, u_h, z_h, numel);
        else if (act_type == ACT_GELU)  act_mul_kernel_h<ACT_GELU> <<<blocks, NUM_THREADS, 0, stream>>>(g_h, u_h, z_h, numel);
        else if (act_type == ACT_RELU2) act_mul_kernel_h<ACT_RELU2><<<blocks, NUM_THREADS, 0, stream>>>(g_h, u_h, z_h, numel);
        else TORCH_CHECK(false, "act_mul: unknown activation type");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "act_mul kernel launch failed: ", cudaGetErrorString(err));
}

void hip_silu_mul(at::Tensor g, at::Tensor u, at::Tensor z)
{
    _act_mul_dispatch(ACT_SILU, g, u, z);
}

void hip_gelu_mul(at::Tensor g, at::Tensor u, at::Tensor z)
{
    _act_mul_dispatch(ACT_GELU, g, u, z);
}

void hip_relu2_mul(at::Tensor g, at::Tensor u, at::Tensor z)
{
    _act_mul_dispatch(ACT_RELU2, g, u, z);
}
