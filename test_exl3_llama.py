"""Quick test: Llama-1B only."""
import sys
sys.path.insert(0, '/home/yiyuanti/vllm')

def main():
    from vllm import LLM, SamplingParams
    llm = LLM(
        model='/home/yiyuanti/ai_inference/models/Llama-3.2-1B-Instruct-exl3-4bpw',
        dtype='half',
        max_model_len=256,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    params = SamplingParams(max_tokens=20, temperature=0.0)
    outputs = llm.generate(['The capital of France is'], sampling_params=params)
    text = outputs[0].outputs[0].text
    print(f"Output: {text!r}")
    print(f"Token IDs: {list(outputs[0].outputs[0].token_ids[:10])}")

if __name__ == "__main__":
    main()
