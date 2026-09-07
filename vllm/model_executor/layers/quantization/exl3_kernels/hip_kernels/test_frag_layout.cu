// Test: determine exact FragB REGISTER layout for mma_sync on RDNA3
// Strategy: fill LDS with known values, load_matrix_sync, read back x[j]
// This tells us what mma_sync actually sees per (lane, j)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <stdio.h>

#define TILE_DIM 16
#define WARP_SIZE 32

using f16_t = rocwmma::float16_t;
using FragB = rocwmma::fragment<rocwmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, f16_t, rocwmma::col_major>;

// Test 1: Load col-major indices 0-255 via load_matrix_sync, read x[j]
// This shows: frag_b.x[j] for lane L = which col-major position?
__global__ __launch_bounds__(32)
void test_register_layout(float* out_map)
{
    int lane = threadIdx.x;

    __shared__ f16_t s_B[256];

    // Fill s_B[i] = i (col-major, stride=16)
    for (int i = lane; i < 256; i += WARP_SIZE)
        s_B[i] = static_cast<f16_t>(static_cast<float>(i));
    __syncthreads();

    FragB frag_b;
    rocwmma::load_matrix_sync(frag_b, s_B, TILE_DIM);

    // Read frag_b.x[j] and write to output indexed by (lane, j)
    for (int j = 0; j < 8; j++)
        out_map[lane * 8 + j] = static_cast<float>(frag_b.x[j]);
}

// Test 2: Direct fill frag_b.x[j] = value, use mma_sync with identity-like A
// Then check if C matches expected A × B
__global__ __launch_bounds__(32)
void test_mma_with_direct_fill(float* out_c)
{
    using FragA = rocwmma::fragment<rocwmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, f16_t, rocwmma::row_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>;

    int lane = threadIdx.x;
    __shared__ f16_t s_A[256], s_B[256];
    __shared__ float s_C[256];

    // A = identity (16×16)
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        int r = i / TILE_DIM, c = i % TILE_DIM;
        s_A[i] = static_cast<f16_t>((r == c) ? 1.0f : 0.0f);
    }
    // B = values 0-255 (col-major)
    for (int i = lane; i < 256; i += WARP_SIZE)
        s_B[i] = static_cast<f16_t>(static_cast<float>(i));
    __syncthreads();

    // Load A normally
    FragA frag_a;
    rocwmma::load_matrix_sync(frag_a, s_A, TILE_DIM);

    // Method A: load B normally
    FragB frag_b_load;
    rocwmma::load_matrix_sync(frag_b_load, s_B, TILE_DIM);

    FragAcc acc_load;
    rocwmma::fill_fragment(acc_load, 0.0f);
    rocwmma::mma_sync(acc_load, frag_a, frag_b_load, acc_load);
    rocwmma::store_matrix_sync(s_C, acc_load, TILE_DIM, rocwmma::mem_row_major);
    __syncthreads();

    // Output: C should be I × B = B (in row-major output)
    for (int i = lane; i < 256; i += WARP_SIZE)
        out_c[i] = s_C[i];
}

// Test 3: Direct fill with the "register layout" mapping and verify via mma
__global__ __launch_bounds__(32)
void test_mma_direct_fill_mapped(float* out_c, const float* reg_map)
{
    using FragA = rocwmma::fragment<rocwmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, f16_t, rocwmma::row_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>;

    int lane = threadIdx.x;
    __shared__ f16_t s_A[256];
    __shared__ float s_C[256];

    // A = identity
    for (int i = lane; i < 256; i += WARP_SIZE)
    {
        int r = i / TILE_DIM, c = i % TILE_DIM;
        s_A[i] = static_cast<f16_t>((r == c) ? 1.0f : 0.0f);
    }
    __syncthreads();

    FragA frag_a;
    rocwmma::load_matrix_sync(frag_a, s_A, TILE_DIM);

    // Direct fill using register layout mapping from Test 1
    FragB frag_b;
    for (int j = 0; j < 8; j++)
    {
        // reg_map[lane*8+j] tells us which col-major position goes into x[j]
        int col_major_pos = (int)reg_map[lane * 8 + j];
        frag_b.x[j] = static_cast<f16_t>(static_cast<float>(col_major_pos));
    }

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);
    rocwmma::mma_sync(acc, frag_a, frag_b, acc);
    rocwmma::store_matrix_sync(s_C, acc, TILE_DIM, rocwmma::mem_row_major);
    __syncthreads();

    for (int i = lane; i < 256; i += WARP_SIZE)
        out_c[i] = s_C[i];
}

int main()
{
    float *d_map, *d_c;
    float h_map[256], h_c[256], h_c2[256];
    hipMalloc(&d_map, 256 * sizeof(float));
    hipMalloc(&d_c, 256 * sizeof(float));

    // Test 1: Determine register layout
    printf("=== Test 1: load_matrix_sync register layout ===\n");
    printf("frag_b.x[j] for lane L after loading col-major indices 0-255:\n\n");
    test_register_layout<<<1, 32>>>(d_map);
    hipDeviceSynchronize();
    hipMemcpy(h_map, d_map, 256 * sizeof(float), hipMemcpyDeviceToHost);

    printf("Lane | x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7]\n");
    printf("-----|------------------------------------------------\n");
    for (int l = 0; l < 32; l++)
    {
        printf("%4d | ", l);
        for (int j = 0; j < 8; j++)
            printf("%4d ", (int)h_map[l * 8 + j]);
        printf("\n");
    }

    // Check: is the mapping lane + j*32?
    printf("\n--- Checking if mapping is lane + j*32: ---\n");
    bool match_flat = true;
    for (int l = 0; l < 32; l++)
        for (int j = 0; j < 8; j++)
            if ((int)h_map[l * 8 + j] != l + j * 32) { match_flat = false; break; }
    printf("lane + j*32: %s\n", match_flat ? "YES" : "NO");

    // Check: is the mapping j + 8*(l/16) + (l%16)*16?
    bool match_frag = true;
    for (int l = 0; l < 32; l++)
        for (int j = 0; j < 8; j++)
            if ((int)h_map[l * 8 + j] != j + 8*(l/16) + (l%16)*16) { match_frag = false; break; }
    printf("j + 8*(l/16) + (l%16)*16: %s\n", match_frag ? "YES" : "NO");

    // Print the actual formula
    printf("\n--- Reverse mapping (col-major pos -> lane, j): ---\n");
    for (int p = 0; p < 16; p++)
    {
        // Find which (lane, j) holds position p
        for (int l = 0; l < 32; l++)
            for (int j = 0; j < 8; j++)
                if ((int)h_map[l * 8 + j] == p)
                    printf("pos %3d -> lane %2d, j=%d\n", p, l, j);
    }

    // Test 2: I × B via normal load
    printf("\n=== Test 2: I × B via load_matrix_sync (reference) ===\n");
    test_mma_with_direct_fill<<<1, 32>>>(d_c);
    hipDeviceSynchronize();
    hipMemcpy(h_c, d_c, 256 * sizeof(float), hipMemcpyDeviceToHost);

    printf("C[0][0..15] = ");
    for (int i = 0; i < 16; i++) printf("%.0f ", h_c[i]);
    printf("\n");
    printf("C[1][0..15] = ");
    for (int i = 16; i < 32; i++) printf("%.0f ", h_c[i]);
    printf("\n");

    // Test 3: I × B via direct fill using register layout from Test 1
    printf("\n=== Test 3: I × B via direct fill (register-mapped) ===\n");
    // Upload the register map to device
    hipMemcpy(d_map, h_map, 256 * sizeof(float), hipMemcpyHostToDevice);
    test_mma_direct_fill_mapped<<<1, 32>>>(d_c, d_map);
    hipDeviceSynchronize();
    hipMemcpy(h_c2, d_c, 256 * sizeof(float), hipMemcpyDeviceToHost);

    printf("C[0][0..15] = ");
    for (int i = 0; i < 16; i++) printf("%.0f ", h_c2[i]);
    printf("\n");
    printf("C[1][0..15] = ");
    for (int i = 16; i < 32; i++) printf("%.0f ", h_c2[i]);
    printf("\n");

    bool match = true;
    for (int i = 0; i < 256; i++)
        if ((int)h_c[i] != (int)h_c2[i]) { match = false; break; }
    printf("\nDirect fill matches load_matrix_sync: %s\n", match ? "YES" : "NO");
    if (!match) {
        printf("First mismatches:\n");
        int cnt = 0;
        for (int i = 0; i < 256 && cnt < 10; i++)
            if ((int)h_c[i] != (int)h_c2[i]) {
                printf("  C[%d][%d]: load=%.0f direct=%.0f\n", i/16, i%16, h_c[i], h_c2[i]);
                cnt++;
            }
    }

    hipFree(d_map);
    hipFree(d_c);
    return 0;
}
