"""Test EXL3 quantization backend in vLLM."""

import sys


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model='/home/yiyuanti/ai_inference/models/Qwen3-0.6B-exl3-4bpw',
        dtype='half',
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )

    params = SamplingParams(max_tokens=30, temperature=0.0)
    prompts = [
        'The capital of France is',
        'Hello, how are you?',
        '1 + 1 =',
    ]
    outputs = llm.generate(prompts, sampling_params=params)

    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        print(f'--- Prompt {i}: {prompts[i]}')
        print(f'    Text: {repr(text[:200])}')
        print(f'    Token IDs ({len(token_ids)}): {list(token_ids[:20])}')
        sys.stdout.flush()


if __name__ == "__main__":
    main()
