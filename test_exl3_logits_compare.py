"""Compare EXL3 logits: standalone kernel vs vLLM.

Runs a single forward pass through the model using:
1. debug_pure_pytorch.py style (standalone, known to work — produces "Paris" at 75%)
2. vLLM pipeline

Compares the logits to see where they diverge.
"""
import sys
sys.path.insert(0, '/home/yiyuanti/vllm')
sys.path.insert(0, '/home/yiyuanti/ai_inference/exl3_triton_kernel')

import torch
import json
from pathlib import Path
from safetensors.torch import load_file

MODEL_DIR = '/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw'

def load_model_weights():
    """Load all safetensor files."""
    sf = {}
    for p in Path(MODEL_DIR).glob('*.safetensors'):
        sf.update(load_file(str(p)))
    return sf

def standalone_forward(sf, input_ids):
    """Run forward pass using standalone kernel (known to work)."""
    from hadamard import had_r_128
    from triton_kernel import exl3_gemm

    device = torch.device('cuda')

    # Load config
    with open(f'{MODEL_DIR}/config.json') as f:
        config = json.load(f)

    num_layers = config['num_hidden_layers']  # 24
    hidden_size = config['hidden_size']  # 1024
    num_heads = config['num_attention_heads']  # 16
    num_kv_heads = config['num_key_value_heads']  # 8
    head_dim = hidden_size // num_heads  # 64
    intermediate_size = config['intermediate_size']  # 3072
    rms_norm_eps = config['rms_norm_eps']

    # Embedding
    embed = sf['model.embed_tokens.weight'].to(device)
    x = embed[input_ids].unsqueeze(0)  # (1, seq_len, hidden_size)

    def rms_norm(x, weight, eps):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + eps)
        return (x * weight.float()).half()

    def exl3_linear(x, prefix, bits=4):
        """Run a single EXL3 linear layer."""
        trellis = sf[f'{prefix}.trellis'].to(device)
        suh = sf[f'{prefix}.suh'].to(device)
        svh = sf[f'{prefix}.svh'].to(device)

        M = x.shape[0]
        K = x.shape[1]

        xh = torch.empty_like(x)
        had_r_128(x, xh, suh, None, 1.0)

        out = exl3_gemm(xh, trellis, bits=bits, cb=0)

        out_h = torch.empty_like(out)
        had_r_128(out, out_h, None, svh, 1.0)
        return out_h

    B, S, D = x.shape
    print(f"Input shape: {x.shape}, tokens: {input_ids}")

    for layer_idx in range(num_layers):
        prefix = f'model.layers.{layer_idx}'

        # RMSNorm
        ln_w = sf[f'{prefix}.input_layernorm.weight'].to(device)
        normed = rms_norm(x, ln_w, rms_norm_eps)

        # QKV
        q = exl3_linear(normed.view(-1, D), f'{prefix}.self_attn.q_proj')
        k = exl3_linear(normed.view(-1, D), f'{prefix}.self_attn.k_proj')
        v = exl3_linear(normed.view(-1, D), f'{prefix}.self_attn.v_proj')

        # Simple attention (no cache, no RoPE for comparison)
        q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

        # GQA: repeat KV
        if num_kv_heads < num_heads:
            rep = num_heads // num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Scaled dot-product attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # O projection
        o_out = exl3_linear(attn_out.view(-1, D), f'{prefix}.self_attn.o_proj')
        o_out = o_out.view(B, S, D)

        # Residual
        x = x + o_out

        # MLP
        ln2_w = sf[f'{prefix}.post_attention_layernorm.weight'].to(device)
        normed2 = rms_norm(x, ln2_w, rms_norm_eps)

        gate = exl3_linear(normed2.view(-1, D), f'{prefix}.mlp.gate_proj')
        up = exl3_linear(normed2.view(-1, D), f'{prefix}.mlp.up_proj')

        # SiLU activation
        hidden = torch.nn.functional.silu(gate.float()).half() * up
        down = exl3_linear(hidden, f'{prefix}.mlp.down_proj')
        down = down.view(B, S, D)

        # Residual
        x = x + down

        if layer_idx < 2 or layer_idx >= num_layers - 2:
            print(f"  Layer {layer_idx}: x range=[{x.min():.3f}, {x.max():.3f}], std={x.std():.4f}")

    # Final norm
    final_ln_w = sf['model.norm.weight'].to(device)
    x_normed = rms_norm(x, final_ln_w, rms_norm_eps)

    # lm_head (tied with embed_tokens for Qwen3-0.6B)
    logits = x_normed @ embed.T  # (B, S, vocab_size)

    # Get last token logits
    last_logits = logits[0, -1, :]
    return last_logits


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: {prompt!r}")
    print(f"Token IDs: {input_ids}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids]}")

    sf = load_model_weights()
    print(f"\nLoaded {len(sf)} tensors")

    # 1. Standalone forward
    print("\n=== Standalone Forward ===")
    standalone_logits = standalone_forward(sf, input_ids)
    top_k = 10
    top_vals, top_ids = standalone_logits.topk(top_k)
    print(f"\nStandalone top-{top_k} predictions:")
    for val, tid in zip(top_vals, top_ids):
        token = tokenizer.decode([tid.item()])
        print(f"  {token!r:20s} (id={tid.item():6d}) logit={val.item():.3f}")

    # 2. vLLM forward
    print("\n=== vLLM Forward ===")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_DIR,
        dtype='half',
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    params = SamplingParams(max_tokens=5, temperature=0.0, logprobs=10)
    outputs = llm.generate([prompt], sampling_params=params)

    out = outputs[0].outputs[0]
    print(f"\nvLLM output: {out.text!r}")
    print(f"Token IDs: {list(out.token_ids)}")
    if out.logprobs:
        print(f"\nvLLM top logprobs at step 0:")
        for token_id, logprob in sorted(out.logprobs[0].items(),
                                         key=lambda x: -x[1].logprob)[:10]:
            print(f"  {logprob.decoded_token!r:20s} (id={token_id:6d}) "
                  f"logprob={logprob.logprob:.3f}")


if __name__ == "__main__":
    main()
