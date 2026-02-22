"""Hook into vLLM model to capture intermediate tensors."""
import sys
sys.path.insert(0, '/home/yiyuanti/vllm')


def main():
    import torch
    from vllm import LLM, SamplingParams

    # Monkey-patch the EXL3 apply to capture the first real inference call
    from vllm.model_executor.layers.quantization import exl3

    original_apply = exl3.EXL3LinearMethod.apply
    captured = {"calls": [], "capture": False}

    def hooked_apply(self, layer, x, bias=None):
        result = original_apply(self, layer, x, bias)
        if captured["capture"] and len(captured["calls"]) < 500:
            captured["calls"].append({
                "x_shape": list(x.shape),
                "x_min": x.min().item(),
                "x_max": x.max().item(),
                "out_shape": list(result.shape),
                "out_min": result.min().item(),
                "out_max": result.max().item(),
                "trellis_shape": list(layer.trellis.shape),
                "bits": layer.exl3_bits,
            })
        return result

    exl3.EXL3LinearMethod.apply = hooked_apply

    # Also hook into the logits processor to capture final hidden states
    from vllm.model_executor.layers import logits_processor as lp_mod
    original_get_logits = lp_mod.LogitsProcessor._get_logits

    def hooked_get_logits(self, hidden_states, lm_head, embedding_bias):
        result = original_get_logits(self, hidden_states, lm_head, embedding_bias)
        if captured["capture"]:
            captured["hidden_states"] = {
                "shape": list(hidden_states.shape),
                "min": hidden_states.min().item(),
                "max": hidden_states.max().item(),
                "std": hidden_states.std().item(),
                "first_10": hidden_states[0, :10].tolist() if hidden_states.dim() >= 2 else hidden_states[:10].tolist(),
            }
            if result is not None:
                captured["logits"] = {
                    "shape": list(result.shape),
                    "min": result.min().item(),
                    "max": result.max().item(),
                    "std": result.std().item(),
                }
                # Top-10 tokens from first sequence
                if result.dim() >= 2:
                    last_logits = result[-1]  # last token in batch
                else:
                    last_logits = result
                top_vals, top_ids = last_logits.topk(10)
                captured["top_tokens"] = list(zip(top_ids.tolist(), top_vals.tolist()))
        return result

    lp_mod.LogitsProcessor._get_logits = hooked_get_logits

    llm = LLM(
        model='/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw',
        dtype='half',
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )

    # Enable capture for actual inference
    captured["capture"] = True

    params = SamplingParams(max_tokens=3, temperature=0.0)
    outputs = llm.generate(['The capital of France is'], sampling_params=params)

    out = outputs[0].outputs[0]
    print(f"\n=== vLLM Output ===")
    print(f"Text: {out.text!r}")
    print(f"Token IDs: {list(out.token_ids)}")

    print(f"\n=== Captured Data ===")
    if "hidden_states" in captured:
        hs = captured["hidden_states"]
        print(f"Hidden states before lm_head: shape={hs['shape']}, "
              f"range=[{hs['min']:.4f}, {hs['max']:.4f}], std={hs['std']:.4f}")
        print(f"  First 10 values: {hs['first_10']}")

    if "logits" in captured:
        lg = captured["logits"]
        print(f"Logits: shape={lg['shape']}, "
              f"range=[{lg['min']:.4f}, {lg['max']:.4f}], std={lg['std']:.4f}")

    if "top_tokens" in captured:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            '/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw')
        print(f"\nTop-10 logits predictions:")
        for tid, val in captured["top_tokens"]:
            token = tokenizer.decode([tid])
            print(f"  {token!r:20s} (id={tid:6d}) logit={val:.3f}")

    print(f"\n=== Layer call pattern (first 20) ===")
    for i, c in enumerate(captured["calls"][:20]):
        print(f"  #{i}: x={c['x_shape']} [{c['x_min']:.3f},{c['x_max']:.3f}] "
              f"-> out={c['out_shape']} [{c['out_min']:.3f},{c['out_max']:.3f}] "
              f"trellis={c['trellis_shape']} bits={c['bits']}")


if __name__ == "__main__":
    main()
