#!/usr/bin/env python3
"""Build and test the HIP recurrent GDN kernel."""
import os, sys, time, torch
from torch.utils.cpp_extension import load

kernel_dir = os.path.dirname(os.path.abspath(__file__))
hip_file = os.path.join(kernel_dir, "recurrent_gdn_kernel.hip")

props = torch.cuda.get_device_properties(0)
gcn_arch = props.gcnArchName if hasattr(props, 'gcnArchName') else "gfx1100"
print(f"GPU: {props.name}, arch: {gcn_arch}")

os.environ["PYTORCH_ROCM_ARCH"] = gcn_arch
cache_dir = os.path.expanduser("~/.cache/exl3_recurrent_gdn_hip")
os.makedirs(cache_dir, exist_ok=True)

print("Building HIP recurrent GDN kernel...")
t0 = time.time()
ext = load(
    name="exl3_recurrent_gdn_hip",
    sources=[hip_file],
    extra_cuda_cflags=["-O3", "-I/opt/rocm/include"],
    build_directory=cache_dir,
    verbose=True,
)
print(f"Build time: {time.time() - t0:.1f}s")

# Qwen3.5-9B GDN dimensions
bsz, seqlen, num_heads = 1, 2048, 16
k_dim, v_dim = 128, 128
device = "cuda:0"
scale = k_dim ** -0.5

# Create test data in float32
q = torch.randn(bsz, seqlen, num_heads, k_dim, device=device, dtype=torch.float32)
k = torch.randn(bsz, seqlen, num_heads, k_dim, device=device, dtype=torch.float32)
v = torch.randn(bsz, seqlen, num_heads, v_dim, device=device, dtype=torch.float32)
g = torch.randn(bsz, seqlen, num_heads, device=device, dtype=torch.float32) * 0.1  # small g for stability
beta = torch.randn(bsz, seqlen, num_heads, device=device, dtype=torch.float32).sigmoid()
state = torch.zeros(bsz, num_heads, k_dim, v_dim, device=device, dtype=torch.float32)
out = torch.zeros(bsz, seqlen, num_heads, v_dim, device=device, dtype=torch.float32)

# === Test HIP kernel ===
print("\n=== HIP Kernel Test ===")
state_hip = state.clone()
out_hip = out.clone()
ext.recurrent_gdn_forward(q, k, k, v, g, beta, state_hip, out_hip, scale, False)
torch.cuda.synchronize()
print(f"HIP output range: [{out_hip.min():.4f}, {out_hip.max():.4f}]")
print(f"HIP state range: [{state_hip.min():.4f}, {state_hip.max():.4f}]")
assert out_hip.abs().sum() > 0, "HIP output is all zeros!"
print("HIP kernel: PASS")

# === Test PyTorch reference ===
print("\n=== PyTorch Reference ===")
state_ref = state.clone()
out_ref = torch.zeros_like(out)
for t in range(seqlen):
    q_t = q[:, t, :]
    k_t = k[:, t, :]
    v_t = v[:, t, :]
    g_t = g[:, t, :].exp().unsqueeze(-1)
    beta_t = beta[:, t, :].unsqueeze(-1)
    kv_mem = (state_ref * k_t.unsqueeze(-1)).sum(dim=-2)
    v_adj = v_t - kv_mem * g_t
    upd = k_t.unsqueeze(-1) * v_adj.unsqueeze(-2) * beta_t.unsqueeze(-1)
    state_ref = state_ref * g_t.unsqueeze(-1) + upd
    out_ref[:, t, :] = (state_ref * q_t.unsqueeze(-1)).sum(dim=-2) * scale

# Compare
max_diff_out = (out_hip - out_ref).abs().max().item()
max_diff_state = (state_hip - state_ref).abs().max().item()
cos_sim = torch.nn.functional.cosine_similarity(
    out_hip.reshape(-1).unsqueeze(0), out_ref.reshape(-1).unsqueeze(0)
).item()
print(f"Max output diff: {max_diff_out:.6f}")
print(f"Max state diff: {max_diff_state:.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")

# Allow some numerical difference due to FP32 accumulation order
if cos_sim > 0.99:
    print("Correctness: PASS")
else:
    print(f"Correctness: FAIL (cos_sim={cos_sim})")

# === Benchmark ===
print("\n=== Benchmark ===")
# Warmup
state_bench = state.clone()
out_bench = out.clone()
ext.recurrent_gdn_forward(q, k, k, v, g, beta, state_bench, out_bench, scale, False)
torch.cuda.synchronize()

# HIP kernel
state_bench = state.clone()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    state_bench.zero_()
    ext.recurrent_gdn_forward(q, k, k, v, g, beta, state_bench, out_bench, scale, False)
torch.cuda.synchronize()
hip_time = (time.time() - t0) / 10
print(f"HIP kernel: {hip_time*1000:.1f}ms per call (seqlen={seqlen})")

# PyTorch reference
state_bench = state.clone()
torch.cuda.synchronize()
t0 = time.time()
for t in range(seqlen):
    q_t = q[:, t, :]
    k_t = k[:, t, :]
    v_t = v[:, t, :]
    g_t = g[:, t, :].exp().unsqueeze(-1)
    beta_t = beta[:, t, :].unsqueeze(-1)
    kv_mem = (state_bench * k_t.unsqueeze(-1)).sum(dim=-2)
    v_adj = v_t - kv_mem * g_t
    upd = k_t.unsqueeze(-1) * v_adj.unsqueeze(-2) * beta_t.unsqueeze(-1)
    state_bench = state_bench * g_t.unsqueeze(-1) + upd
torch.cuda.synchronize()
py_time = time.time() - t0
print(f"PyTorch loop: {py_time*1000:.1f}ms per call (seqlen={seqlen})")
print(f"Speedup: {py_time/hip_time:.1f}x")

# Estimate conversion time saving
cal_samples = 250
gdn_layers = 24  # Qwen3.5-9B has ~24 GDN layers
hip_total = hip_time * cal_samples * gdn_layers
py_total = py_time * cal_samples * gdn_layers
print(f"\nConversion estimate ({cal_samples} samples, {gdn_layers} GDN layers):")
print(f"  Python loop: {py_total/3600:.1f}h")
print(f"  HIP kernel:  {hip_total/3600:.2f}h")
print(f"  Time saved:  {(py_total - hip_total)/3600:.1f}h")
