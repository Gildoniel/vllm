"""Benchmark EXL3 vs AWQ on Qwen3-8B via vLLM.

Tests:
1. Decode latency (batch=1, input=64, output=128) — single-user chat
2. Prefill throughput (batch=1, input=512, output=1) — time-to-first-token
3. Batch throughput (batch=8, input=64, output=128) — concurrent users
"""
import sys
import os
import time
import json
import gc

sys.path.insert(0, '/home/yiyuanti/vllm')
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODELS = {
    "EXL3-4bpw": "/home/yiyuanti/ai_inference/models/Qwen3-8B-exl3-4bpw",
    "AWQ-4bit": "/home/yiyuanti/ai_inference/models/Qwen3-8B-AWQ",
}

SCENARIOS = [
    {"name": "decode_b1",     "batch": 1, "input_len": 64,  "output_len": 128, "desc": "Decode B=1"},
    {"name": "prefill_512",   "batch": 1, "input_len": 512, "output_len": 1,   "desc": "Prefill 512"},
    {"name": "batch_8",       "batch": 8, "input_len": 64,  "output_len": 128, "desc": "Batch B=8"},
]

NUM_WARMUP = 3
NUM_ITERS = 10


def make_prompts(batch_size, input_len, tokenizer):
    """Create prompts of approximately input_len tokens."""
    # Use a repeated pattern to get predictable token count
    base = "The quick brown fox jumps over the lazy dog. "
    tokens_per_base = len(tokenizer.encode(base))
    repeats = max(1, input_len // tokens_per_base)
    text = base * repeats
    # Trim to exact length
    ids = tokenizer.encode(text)[:input_len]
    prompt = tokenizer.decode(ids)
    return [prompt] * batch_size


def bench_model(model_name, model_path, scenarios):
    from vllm import LLM, SamplingParams

    print(f"\n{'='*70}")
    print(f"  {model_name}: {os.path.basename(model_path)}")
    print(f"{'='*70}")

    force_eager = os.environ.get("BENCH_EAGER", "0") == "1"
    mode = "eager" if force_eager else "compiled"
    print(f"  Mode: {mode}")

    llm = LLM(
        model=model_path,
        dtype='half',
        max_model_len=1024,
        gpu_memory_utilization=0.80,
        enforce_eager=force_eager,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()

    results = {}

    for scenario in scenarios:
        name = scenario["name"]
        batch = scenario["batch"]
        input_len = scenario["input_len"]
        output_len = scenario["output_len"]
        desc = scenario["desc"]

        prompts = make_prompts(batch, input_len, tokenizer)
        params = SamplingParams(max_tokens=output_len, temperature=0.0)

        # Verify prompt length
        actual_len = len(tokenizer.encode(prompts[0]))

        # Warmup
        for _ in range(NUM_WARMUP):
            llm.generate(prompts, sampling_params=params)

        # Timed iterations
        latencies = []
        total_output_tokens = 0
        total_input_tokens = 0
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=params)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            for out in outputs:
                total_output_tokens += len(out.outputs[0].token_ids)
                total_input_tokens += len(out.prompt_token_ids)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        avg_output_toks = total_output_tokens / NUM_ITERS
        avg_input_toks = total_input_tokens / NUM_ITERS
        output_tps = avg_output_toks / avg_latency
        total_tps = (avg_input_toks + avg_output_toks) / avg_latency

        results[name] = {
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min_latency * 1000,
            "output_tok_per_sec": output_tps,
            "total_tok_per_sec": total_tps,
            "avg_output_tokens": avg_output_toks,
            "avg_input_tokens": avg_input_toks,
        }

        print(f"\n  [{desc}] input≈{actual_len}, output={output_len}, batch={batch}")
        print(f"    Avg latency:   {avg_latency*1000:8.1f} ms  (min: {min_latency*1000:.1f} ms)")
        print(f"    Output tok/s:  {output_tps:8.1f}")
        print(f"    Total tok/s:   {total_tps:8.1f}")

    # Cleanup
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    return results


def main():
    all_results = {}

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"[SKIP] {model_path} not found")
            continue
        all_results[model_name] = bench_model(model_name, model_path, SCENARIOS)

    # Summary comparison
    if len(all_results) == 2:
        names = list(all_results.keys())
        print(f"\n{'='*70}")
        print(f"  COMPARISON: {names[0]} vs {names[1]}")
        print(f"{'='*70}")
        print(f"  {'Scenario':<20s} {'':>8s}  {'':>8s}  {'Ratio':>8s}")
        print(f"  {'':20s} {names[0]:>8s}  {names[1]:>8s}  {'EXL3/AWQ':>8s}")
        print(f"  {'-'*56}")

        for scenario in SCENARIOS:
            name = scenario["name"]
            desc = scenario["desc"]
            r0 = all_results[names[0]][name]
            r1 = all_results[names[1]][name]
            tps0 = r0["output_tok_per_sec"]
            tps1 = r1["output_tok_per_sec"]
            ratio = tps0 / tps1 if tps1 > 0 else 0
            print(f"  {desc:<20s} {tps0:7.1f}t/s  {tps1:7.1f}t/s  {ratio:7.2f}x")

        print()
        for scenario in SCENARIOS:
            name = scenario["name"]
            desc = scenario["desc"]
            r0 = all_results[names[0]][name]
            r1 = all_results[names[1]][name]
            lat0 = r0["avg_latency_ms"]
            lat1 = r1["avg_latency_ms"]
            ratio = lat0 / lat1 if lat1 > 0 else 0
            print(f"  {desc:<20s} {lat0:7.1f}ms  {lat1:7.1f}ms  {ratio:7.2f}x")

    # Save JSON
    out_path = "/home/yiyuanti/vllm/bench_exl3_vs_awq_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
