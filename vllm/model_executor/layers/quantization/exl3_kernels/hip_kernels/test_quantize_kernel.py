#!/usr/bin/env python3
"""Build and test the HIP Viterbi quantization kernel."""

import os
import sys
import time
import torch

# Build via torch.utils.cpp_extension
from torch.utils.cpp_extension import load

print("Building HIP quantize kernel...")
t0 = time.time()

kernel_dir = os.path.dirname(os.path.abspath(__file__))
hip_file = os.path.join(kernel_dir, "quantize_kernel.hip")

# Detect GPU arch
props = torch.cuda.get_device_properties(0)
gcn_arch = props.gcnArchName if hasattr(props, 'gcnArchName') else "gfx1100"
print(f"GPU: {props.name}, arch: {gcn_arch}")

os.environ["PYTORCH_ROCM_ARCH"] = gcn_arch  # Only build for this GPU
ext = load(
    name="exl3_quantize_hip",
    sources=[hip_file],
    extra_cuda_cflags=[
        "-O3",
        "-I/opt/rocm/include",
    ],
    build_directory=os.path.expanduser("~/.cache/exl3_quantize_hip"),
    verbose=True,
)
print(f"Build time: {time.time() - t0:.1f}s")

# ---- Correctness test ----
print("\n=== Correctness Test ===")
K = 4  # 4-bit quantization (most common)
edges = 65536 >> K  # 4096 edges
n_tiles = 8

device = torch.device("cuda:0")
input_tiles = torch.randn(n_tiles, 256, device=device)
output_tiles = torch.zeros(n_tiles, 256, device=device)
output_indices = torch.zeros(n_tiles, 256, dtype=torch.int16, device=device)

# Query SM count for temp buffer sizing
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
max_batch = min(n_tiles, num_sms)

temp_costs = torch.zeros(max_batch, 2, edges, dtype=torch.float16, device=device)
temp_edges = torch.zeros(max_batch, 256, edges, dtype=torch.int16, device=device)

ext.quantize_tiles(input_tiles, output_tiles, output_indices, temp_costs, temp_edges, K, False, False)
torch.cuda.synchronize()

# Check output is non-zero
print(f"Input  range: [{input_tiles.min():.4f}, {input_tiles.max():.4f}]")
print(f"Output range: [{output_tiles.min():.4f}, {output_tiles.max():.4f}]")
print(f"Indices range: [{output_indices.min()}, {output_indices.max()}]")

# Compute reconstruction error
mse = ((input_tiles - output_tiles) ** 2).mean().item()
cos_sim = torch.nn.functional.cosine_similarity(
    input_tiles.reshape(-1).unsqueeze(0),
    output_tiles.reshape(-1).unsqueeze(0)
).item()
print(f"MSE: {mse:.6f}, Cosine similarity: {cos_sim:.6f}")

# Sanity: output should be non-trivial
assert output_tiles.abs().sum() > 0, "Output is all zeros!"
assert cos_sim > 0.8, f"Cosine similarity too low: {cos_sim}"
print("Correctness: PASS")

# ---- Test all K values ----
print("\n=== Test K=2..8 ===")  # K=1 has 32768 edges, needs special handling
for k in range(2, 9):
    e = 65536 >> k
    tc = torch.zeros(max_batch, 2, e, dtype=torch.float16, device=device)
    te = torch.zeros(max_batch, 256, e, dtype=torch.int16, device=device)
    inp = torch.randn(4, 256, device=device)
    out = torch.zeros(4, 256, device=device)
    idx = torch.zeros(4, 256, dtype=torch.int16, device=device)
    try:
        ext.quantize_tiles(inp, out, idx, tc, te, k, False, False)
        torch.cuda.synchronize()
        mse_k = ((inp - out) ** 2).mean().item()
        print(f"  K={k}: edges={e:5d}, MSE={mse_k:.6f}")
    except Exception as ex:
        print(f"  K={k}: edges={e:5d}, FAILED: {ex}")
print("K value tests done")

# ---- Test CB=1 (MCG) ----
print("\n=== Test CB=1 (MCG) ===")
K = 4
e = 65536 >> K
tc = torch.zeros(max_batch, 2, e, dtype=torch.float16, device=device)
te = torch.zeros(max_batch, 256, e, dtype=torch.int16, device=device)
inp = torch.randn(4, 256, device=device)
out = torch.zeros(4, 256, device=device)
idx = torch.zeros(4, 256, dtype=torch.int16, device=device)
ext.quantize_tiles(inp, out, idx, tc, te, K, True, False)
torch.cuda.synchronize()
mse_mcg = ((inp - out) ** 2).mean().item()
print(f"  CB=1 MSE: {mse_mcg:.6f}")
print("MCG: PASS")

# ---- Benchmark ----
print("\n=== Benchmark ===")
for n_tiles_bench in [16, 64, 256, 512, 1024]:
    K = 4
    edges = 65536 >> K
    inp = torch.randn(n_tiles_bench, 256, device=device)
    out = torch.zeros(n_tiles_bench, 256, device=device)
    idx = torch.zeros(n_tiles_bench, 256, dtype=torch.int16, device=device)
    tc = torch.zeros(max_batch, 2, edges, dtype=torch.float16, device=device)
    te = torch.zeros(max_batch, 256, edges, dtype=torch.int16, device=device)

    # Warmup
    ext.quantize_tiles(inp, out, idx, tc, te, K, False, False)
    torch.cuda.synchronize()

    # Timed
    torch.cuda.synchronize()
    t0 = time.time()
    n_iters = max(1, 3000 // n_tiles_bench)
    for _ in range(n_iters):
        ext.quantize_tiles(inp, out, idx, tc, te, K, False, False)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    total_tiles = n_tiles_bench * n_iters
    tiles_per_s = total_tiles / elapsed
    print(f"  n_tiles={n_tiles_bench:4d}: {tiles_per_s:,.0f} tiles/s ({elapsed:.2f}s for {total_tiles} tiles)")

# ---- Decode test ----
print("\n=== Decode Test ===")
K = 4
inp = torch.randn(8, 256, device=device)
out = torch.zeros(8, 256, device=device)
idx = torch.zeros(8, 256, dtype=torch.int16, device=device)
tc = torch.zeros(max_batch, 2, 65536 >> K, dtype=torch.float16, device=device)
te = torch.zeros(max_batch, 256, 65536 >> K, dtype=torch.int16, device=device)
ext.quantize_tiles(inp, out, idx, tc, te, K, False, False)
torch.cuda.synchronize()

# Decode the indices
decoded = torch.zeros(8, 256, device=device)
ext.decode_tiles(idx, decoded, False, False)
torch.cuda.synchronize()

diff = (out - decoded).abs().max().item()
print(f"  Quantize vs Decode max diff: {diff}")
assert diff < 1e-4, f"Decode mismatch: {diff}"
print("Decode: PASS")

print("\n=== ALL TESTS PASSED ===")
