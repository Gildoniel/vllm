"""Test EXL3 quantization backend with multiple models."""
import sys
import os


def test_model(model_dir, prompts, max_tokens=20):
    from vllm import LLM, SamplingParams

    model_name = os.path.basename(model_dir)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    try:
        llm = LLM(
            model=model_dir,
            dtype='half',
            max_model_len=256,
            gpu_memory_utilization=0.7,
            enforce_eager=True,
        )

        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outputs = llm.generate(prompts, sampling_params=params)

        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            print(f"  Prompt: {prompts[i]!r}")
            print(f"  Output: {text[:120]!r}")
        print(f"  [OK] {model_name}")

        del llm
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  [FAIL] {model_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    models = [
        '/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw',
        '/home/yiyuanti/ai_inference/models/Llama-3.2-1B-Instruct-exl3-4bpw',
        '/home/yiyuanti/ai_inference/models/SmolLM3-3B-exl3-4bpw',
        '/home/yiyuanti/ai_inference/models/Qwen3-8B-exl3-4bpw',
    ]

    prompts = [
        'The capital of France is',
        '1 + 1 =',
    ]

    for model_dir in models:
        if os.path.exists(model_dir):
            test_model(model_dir, prompts)
        else:
            print(f"\n[SKIP] {model_dir} not found")


if __name__ == "__main__":
    main()
