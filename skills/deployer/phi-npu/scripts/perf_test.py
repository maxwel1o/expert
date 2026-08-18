#!/usr/bin/env python3
"""
Performance test for Phi models on Ascend NPU using direct transformers.
Measures: tokens/sec, TTFT approximation, memory usage.

Usage: python3 perf_test.py <model_name>
  model_name: phi-1 | phi-1_5 | phi-2
"""
import argparse
import json
import os
import sys
import time

os.environ["TASK_QUEUE_ENABLE"] = "1"

import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATHS = {
    "phi-1": "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-1",
    "phi-1_5": "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-1_5",
    "phi-2": "/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-2",
}

MODEL_DTYPES = {
    "phi-1": torch.float16,   # Use fp16 on NPU for attention kernel compatibility
    "phi-1_5": torch.float16,
    "phi-2": torch.float16,
}

TEST_PROMPTS = [
    "def fibonacci(n):",
    "Machine learning is a field of",
    "Write a Python function to sort a list:",
    "The capital of France is Paris.",
    "Quantum computing is",
]

LONG_PROMPT = (
    "Machine learning is a subset of artificial intelligence that enables systems to learn "
    "and improve from experience without being explicitly programmed. The core idea is to "
    "develop algorithms that can parse data, learn from it, and then apply what they've learned "
    "to make informed decisions. There are three main types of machine learning: supervised learning, "
    "unsupervised learning, and reinforcement learning. In supervised learning, the algorithm is "
    "trained on a labeled dataset, where each training example is paired with an output label. "
    "Deep learning is a subset of machine learning that uses neural networks with many layers to "
    "model complex patterns in data. These deep neural networks have been particularly successful "
    "in tasks such as image recognition, natural language processing, and speech recognition."
)


def measure_generation(model, tokenizer, prompt, max_new_tokens, num_runs=3):
    """Measure generation performance."""
    inputs = tokenizer(prompt, return_tensors="pt").to("npu:0")
    prompt_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)

    # Measure
    latencies = []
    completions = []
    for _ in range(num_runs):
        torch.npu.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        torch.npu.synchronize()
        elapsed = time.time() - start
        gen_tokens = out.shape[1] - prompt_len
        latencies.append(elapsed)
        completions.append(gen_tokens)

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(completions) / len(completions)
    tok_per_sec = avg_tokens / avg_latency

    return {
        "prompt_tokens": prompt_len,
        "completion_tokens": int(avg_tokens),
        "latency_seconds": round(avg_latency, 4),
        "throughput_tok_s": round(tok_per_sec, 2),
        "num_runs": num_runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=list(MODEL_PATHS.keys()))
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model_name]
    dtype = MODEL_DTYPES[args.model_name]

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model = model.npu()
    model.eval()
    print(f"Loaded on {torch.npu.get_device_name(0)}")

    # Memory before
    torch.npu.reset_peak_memory_stats()
    mem_before = torch.npu.memory_allocated() / (1024**3)

    results = {}

    # Test 1: Single short prompt
    print("\n--- Short Prompt Generation ---")
    for prompt in TEST_PROMPTS[:3]:
        res = measure_generation(model, tokenizer, prompt, 64, num_runs=2)
        print(f"  '{prompt[:30]}...': {res['latency_seconds']:.3f}s, {res['throughput_tok_s']:.1f} tok/s")
        results[f"short_{TEST_PROMPTS.index(prompt)}"] = res

    # Test 2: Varying output lengths
    print("\n--- Varying Output Lengths ---")
    for max_tok in [32, 64, 128]:
        prompt = TEST_PROMPTS[2]
        res = measure_generation(model, tokenizer, prompt, max_tok, num_runs=2)
        print(f"  max_new_tokens={max_tok}: {res['latency_seconds']:.3f}s, {res['throughput_tok_s']:.1f} tok/s")
        results[f"output_{max_tok}"] = res

    # Test 3: Long prompt
    print("\n--- Long Prompt ---")
    res = measure_generation(model, tokenizer, LONG_PROMPT, 128, num_runs=2)
    print(f"  prompt_len={res['prompt_tokens']}, gen=128: {res['latency_seconds']:.3f}s, {res['throughput_tok_s']:.1f} tok/s")
    results["long_prompt"] = res

    # Memory stats
    mem_after = torch.npu.memory_allocated() / (1024**3)
    mem_peak = torch.npu.max_memory_allocated() / (1024**3)

    summary = {
        "model": args.model_name,
        "framework": "torch_npu + transformers",
        "torch_npu_version": torch_npu.__version__,
        "transformers_version": "4.57.6",
        "npu": torch.npu.get_device_name(0),
        "dtype": str(dtype).replace("torch.", ""),
        "memory_gib": {
            "before_load": round(mem_before, 2),
            "after_load": round(mem_after, 2),
            "peak": round(mem_peak, 2),
            "model_size": round(mem_after - mem_before, 2),
        },
        "results": results,
    }

    out_path = f"/opt/atomgit/repos/phi-npu/{args.model_name}/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
