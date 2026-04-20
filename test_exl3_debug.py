"""Debug EXL3: verify weights and single forward pass."""
import sys
sys.path.insert(0, '/home/yiyuanti/vllm')

import torch
from safetensors.torch import load_file

# Load weights directly to compare
sf_weights = load_file('/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw/model.safetensors')

# Test that our EXL3 kernel produces correct output standalone
sys.path.insert(0, '/home/yiyuanti/ai_inference/exl3_triton_kernel')
from hadamard import had_r_128
from triton_kernel import exl3_gemm

device = torch.device('cuda')

# Test full layer 0 q_proj: suh -> had -> gemm -> had -> svh
trellis = sf_weights['model.layers.0.self_attn.q_proj.trellis'].to(device)
suh = sf_weights['model.layers.0.self_attn.q_proj.suh'].to(device)
svh = sf_weights['model.layers.0.self_attn.q_proj.svh'].to(device)

x = torch.randn(1, 1024, dtype=torch.float16, device=device)

# Step 1: had(x, suh, None)
xh = torch.empty_like(x)
had_r_128(x, xh, suh, None, 1.0)
print(f"Step 1 - xh: range=[{xh.min():.4f}, {xh.max():.4f}], std={xh.std():.4f}")

# Step 2: gemm
out = exl3_gemm(xh, trellis, bits=4, cb=0)
print(f"Step 2 - gemm out: range=[{out.min():.4f}, {out.max():.4f}], std={out.std():.4f}")

# Step 3: had(out, None, svh)
out_h = torch.empty_like(out)
had_r_128(out, out_h, None, svh, 1.0)
print(f"Step 3 - final out: range=[{out_h.min():.4f}, {out_h.max():.4f}], std={out_h.std():.4f}")

# Now test via vLLM's loaded model
print("\n--- Loading vLLM model ---")
from vllm.model_executor.layers.quantization.exl3 import EXL3Config, EXL3LinearMethod
from vllm.model_executor.layers.quantization.exl3_kernels import exl3_gemm as vllm_gemm, had_r_128 as vllm_had

# Check if vLLM's kernels produce same output
xh2 = torch.empty_like(x)
vllm_had(x, xh2, suh, None, 1.0)
print(f"\nvLLM had match: {torch.allclose(xh, xh2, atol=1e-4)}")

out2 = vllm_gemm(xh2, trellis, bits=4, cb=0)
print(f"vLLM gemm match: {torch.allclose(out, out2, atol=1e-4)}")

out_h2 = torch.empty_like(out2)
vllm_had(out2, out_h2, None, svh, 1.0)
print(f"vLLM final match: {torch.allclose(out_h, out_h2, atol=1e-4)}")

# Check the trellis shape info
print(f"\nTrellis info: shape={trellis.shape}, nonzero={trellis.count_nonzero()}")
print(f"suh info: shape={suh.shape}, nonzero={suh.count_nonzero()}, range=[{suh.min():.4f}, {suh.max():.4f}]")
print(f"svh info: shape={svh.shape}, nonzero={svh.count_nonzero()}, range=[{svh.min():.4f}, {svh.max():.4f}]")

# Check QKV merged loading
# In vLLM, QKV are merged: output_sizes = [2048, 1024, 1024] for Qwen3-0.6B
# Total: 4096 output features = 256 tiles_n
# words_per_tile = 64 (4bpw)
# So the merged trellis should be (64, 256, 64)

print("\n--- Checking QKV merge ---")
q_trellis = sf_weights['model.layers.0.self_attn.q_proj.trellis']  # (64, 128, 64)
k_trellis = sf_weights['model.layers.0.self_attn.k_proj.trellis']  # (64, 64, 64)
v_trellis = sf_weights['model.layers.0.self_attn.v_proj.trellis']  # (64, 64, 64)
print(f"Q: {q_trellis.shape}, K: {k_trellis.shape}, V: {v_trellis.shape}")
print(f"Concatenated tiles_n: {q_trellis.shape[1] + k_trellis.shape[1] + v_trellis.shape[1]} (expected 256)")
