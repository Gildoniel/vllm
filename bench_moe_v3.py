#!/usr/bin/env python3
"""
Benchmark: Qwen3-Next-80B MoE with HIP v3 pipelined kernel vs baseline.

Usage:
    # v3 kernel (under test)
    EXL3_HIP_GEMM_V3=1 EXL3_HIP_MOE_GEMM=1 python bench_moe_v3.py

    # Baseline (Triton / HIP v2)
    EXL3_HIP_MOE_GEMM=1 python bench_moe_v3.py

    # AWQ reference (no EXL3 env vars)
    python bench_moe_v3.py --awq

    # Eager mode (no compile / CUDA graphs)
    BENCH_EAGER=1 EXL3_HIP_GEMM_V3=1 EXL3_HIP_MOE_GEMM=1 python bench_moe_v3.py

    # Concurrency test
    EXL3_HIP_GEMM_V3=1 EXL3_HIP_MOE_GEMM=1 python bench_moe_v3.py --parallel 1 2 4

Requires 8 GPUs (TP=8, EP=8 for EXL3; TP=8 for AWQ).
"""
import sys
import os
import time
import json
import gc
import argparse

sys.path.insert(0, '/home/yiyuanti/vllm')
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

EXL3_MODEL = "/opt/models/turboderp/Qwen3-Next-80B-A3B-Instruct-exl3"
AWQ_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"  # HF name if AWQ exists

NUM_WARMUP = 3
NUM_ITERS = 5

SCENARIOS = [
    {"name": "decode_b1",      "batch": 1, "input_len": 64,   "output_len": 128, "desc": "Decode B=1"},
    {"name": "decode_1k",      "batch": 1, "input_len": 1024, "output_len": 128, "desc": "Decode 1k ctx"},
    {"name": "batch_4",        "batch": 4, "input_len": 64,   "output_len": 128, "desc": "Batch B=4"},
]


def make_prompts(batch_size, input_len, tokenizer):
    """Create prompts of approximately input_len tokens."""
    base = "The quick brown fox jumps over the lazy dog. "
    tokens_per_base = len(tokenizer.encode(base))
    repeats = max(1, input_len // tokens_per_base)
    text = base * repeats
    ids = tokenizer.encode(text)[:input_len]
    prompt = tokenizer.decode(ids)
    return [prompt] * batch_size


def bench_model(model_path, label, tp_size=8, enforce_eager=False,
                quantization=None, enable_expert_parallel=False,
                scenarios=None, max_model_len=2048,
                parallel_levels=None):
    from vllm import LLM, SamplingParams

    force_eager = os.environ.get("BENCH_EAGER", "0") == "1" or enforce_eager
    mode = "eager" if force_eager else "compiled"

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  TP={tp_size}, EP={'ON' if enable_expert_parallel else 'OFF'}, mode={mode}")
    v3 = os.environ.get("EXL3_HIP_GEMM_V3", "0")
    hip_moe = os.environ.get("EXL3_HIP_MOE_GEMM", "0")
    print(f"  EXL3_HIP_GEMM_V3={v3}, EXL3_HIP_MOE_GEMM={hip_moe}")
    print(f"{'='*70}")

    import torch
    for i in range(min(tp_size, torch.cuda.device_count())):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {name} ({free/1024**3:.1f}/{total/1024**3:.1f} GB free)")

    llm_kwargs = dict(
        model=model_path,
        dtype='half',
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        enforce_eager=force_eager,
        disable_log_stats=True,
        tensor_parallel_size=tp_size,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    if enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True

    print(f"\n  Loading model...")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    tokenizer = llm.get_tokenizer()
    all_results = {}

    for scenario in scenarios:
        name = scenario["name"]
        batch = scenario["batch"]
        input_len = scenario["input_len"]
        output_len = scenario["output_len"]
        desc = scenario["desc"]

        prompts = make_prompts(batch, input_len, tokenizer)
        params = SamplingParams(max_tokens=output_len, temperature=0.0)

        actual_len = len(tokenizer.encode(prompts[0]))

        # Warmup
        print(f"\n  [{desc}] warming up ({NUM_WARMUP} iters)...")
        for _ in range(NUM_WARMUP):
            llm.generate(prompts, sampling_params=params)

        # Timed iterations
        latencies = []
        total_output_tokens = 0
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=params)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            for out in outputs:
                total_output_tokens += len(out.outputs[0].token_ids)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        avg_output_toks = total_output_tokens / NUM_ITERS
        output_tps = avg_output_toks / avg_latency

        # Print generated text from last run (sanity check)
        sample_text = outputs[0].outputs[0].text[:200]
        print(f"    Sample output: {sample_text!r}...")

        all_results[name] = {
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min_latency * 1000,
            "output_tok_per_sec": output_tps,
            "avg_output_tokens": avg_output_toks,
            "input_len": actual_len,
            "batch": batch,
        }

        print(f"    Input: ~{actual_len} tokens, batch={batch}")
        print(f"    Avg latency:   {avg_latency*1000:8.1f} ms  (min: {min_latency*1000:.1f} ms)")
        print(f"    Output tok/s:  {output_tps:8.1f}")

    # Concurrency sweep (parallel requests)
    if parallel_levels:
        print(f"\n  --- Concurrency scaling ---")
        conc_results = {}
        for par in parallel_levels:
            prompts_par = make_prompts(par, 1024, tokenizer)
            params_par = SamplingParams(max_tokens=128, temperature=0.0)

            # Warmup
            for _ in range(2):
                llm.generate(prompts_par, sampling_params=params_par)

            latencies_par = []
            total_toks = 0
            for _ in range(3):
                t0 = time.perf_counter()
                outs = llm.generate(prompts_par, sampling_params=params_par)
                t1 = time.perf_counter()
                latencies_par.append(t1 - t0)
                for out in outs:
                    total_toks += len(out.outputs[0].token_ids)

            avg_lat = sum(latencies_par) / len(latencies_par)
            tput = (total_toks / 3) / avg_lat
            per_req = tput / par

            conc_results[f"parallel_{par}"] = {
                "throughput_tps": tput,
                "per_request_tps": per_req,
                "avg_latency_ms": avg_lat * 1000,
            }
            print(f"    par={par}: {tput:.1f} tok/s total, {per_req:.1f}/req, "
                  f"{avg_lat*1000:.0f} ms")

        all_results["concurrency"] = conc_results

    # Cleanup
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Next-80B MoE benchmark")
    parser.add_argument("--awq", action="store_true",
                        help="Benchmark AWQ instead of EXL3")
    parser.add_argument("--tp", type=int, default=8,
                        help="Tensor parallel size")
    parser.add_argument("--eager", action="store_true",
                        help="Force eager mode (no torch.compile)")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--parallel", type=int, nargs="*", default=None,
                        help="Concurrency levels to test (e.g., --parallel 1 2 4 8)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model path")
    args = parser.parse_args()

    if args.awq:
        model = args.model or AWQ_MODEL
        label = f"AWQ Qwen3-Next-80B TP={args.tp}"
        results = bench_model(
            model, label, tp_size=args.tp,
            enforce_eager=args.eager,
            quantization="awq",
            enable_expert_parallel=False,
            scenarios=SCENARIOS,
            max_model_len=args.max_model_len,
            parallel_levels=args.parallel,
        )
    else:
        model = args.model or EXL3_MODEL
        label = f"EXL3 Qwen3-Next-80B TP={args.tp} EP={args.tp}"
        v3 = os.environ.get("EXL3_HIP_GEMM_V3", "0") == "1"
        if v3:
            label += " [v3 pipelined]"
        results = bench_model(
            model, label, tp_size=args.tp,
            enforce_eager=args.eager,
            enable_expert_parallel=True,
            scenarios=SCENARIOS,
            max_model_len=args.max_model_len,
            parallel_levels=args.parallel,
        )

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for name, data in results.items():
        if name == "concurrency":
            continue
        print(f"  {name:20s}: {data['output_tok_per_sec']:7.1f} tok/s "
              f"(lat {data['avg_latency_ms']:.0f}ms)")

    # Save
    tag = "v3" if os.environ.get("EXL3_HIP_GEMM_V3", "0") == "1" else "baseline"
    if args.awq:
        tag = "awq"
    mode = "eager" if (args.eager or os.environ.get("BENCH_EAGER", "0") == "1") else "compiled"
    out_path = f"/home/yiyuanti/vllm/bench_moe_{tag}_{mode}_tp{args.tp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
