"""Verify EXL3 weights loaded correctly in vLLM model."""
import sys
sys.path.insert(0, '/home/yiyuanti/vllm')


def main():
    import torch
    from safetensors.torch import load_file
    from vllm import LLM

    # Load raw weights
    sf = load_file('/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw/model.safetensors')

    # Load vLLM model
    llm = LLM(
        model='/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw',
        dtype='half',
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )

    # Access model directly -- vLLM v1 uses engine_core process
    # We can't easily access the model this way, so let's check via the worker

    # Instead, let's check the weight loading by hooking into the apply method
    # Use a simple test by comparing weight checksums
    print("--- Weight checksums from safetensors ---")
    for name in sorted(sf.keys()):
        if 'layers.0.' in name:
            w = sf[name]
            print(f"  {name}: shape={list(w.shape)}, sum={w.float().sum():.4f}")

    print("\nModel loaded successfully, checking via generation test...")

    # Since we can't access weights in the subprocess, let's do a different test:
    # Run with ExLlamaV3 reference and compare
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=5, temperature=0.0, logprobs=5)
    outputs = llm.generate(['Paris'], sampling_params=params)

    out = outputs[0].outputs[0]
    print(f"\nPrompt: 'Paris'")
    print(f"Text: {repr(out.text)}")
    print(f"Token IDs: {list(out.token_ids)}")
    if out.logprobs:
        for i, lp in enumerate(out.logprobs[:3]):
            print(f"  Step {i}: top tokens = ", end='')
            for token_id, logprob in sorted(lp.items(), key=lambda x: -x[1].logprob)[:5]:
                print(f"{repr(logprob.decoded_token)}({logprob.logprob:.2f}) ", end='')
            print()


if __name__ == "__main__":
    main()
