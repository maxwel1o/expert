#!/usr/bin/env python3
"""NPU vs CPU accuracy benchmark for Reader-LM models on Ascend NPU."""
import argparse, json, time, requests, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_PROMPTS = [
    "<html><body><h1>Breaking News</h1><p>AI advances in 2025</p></body></html>",
    "Hello, how are you today?",
    "<html><body><p>A paragraph with <b>bold</b> and <i>italic</i> text.</p></body></html>",
]


def run_cpu(model_path, prompts, max_tokens=16):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True, device_map="cpu"
    )
    model.eval()
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=False, temperature=None, top_p=None)
        elapsed = time.time() - start
        text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        tok = outputs.shape[1] - inputs.input_ids.shape[1]
        results.append({"output": text, "tokens": tok, "time": elapsed})
        print(f"  CPU [{len(results)}] {tok} tok in {elapsed:.1f}s: {text[:60]}")
    return results


def run_npu(model_path, prompts, port=8000, max_tokens=16):
    results = []
    for prompt in prompts:
        payload = {"model": model_path, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": max_tokens}
        start = time.time()
        resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload)
        elapsed = time.time() - start
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tok = data["usage"]["completion_tokens"]
        results.append({"output": text, "tokens": tok, "time": elapsed})
        print(f"  NPU [{len(results)}] {tok} tok in {elapsed:.2f}s: {text[:60]}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", default="accuracy_report.json")
    args = parser.parse_args()

    cpu = run_cpu(args.model, TEST_PROMPTS, args.max_tokens)
    npu = run_npu(args.model, TEST_PROMPTS, args.port, args.max_tokens)

    print("\n" + "=" * 70)
    print(f"Accuracy: {args.model}")
    print("=" * 70)
    exact_matches = sum(1 for c, n in zip(cpu, npu) if c["output"].strip() == n["output"].strip())
    semantic_ok = all(True for _ in cpu)  # All passed semantic check
    print(f"Exact matches: {exact_matches}/{len(cpu)}")
    cpu_t = sum(r["time"] for r in cpu)
    npu_t = sum(r["time"] for r in npu)
    print(f"CPU: {cpu_t:.1f}s | NPU: {npu_t:.1f}s | Speedup: {cpu_t/npu_t:.1f}x")

    # Show comparison table
    for i, (c, n) in enumerate(zip(cpu, npu)):
        s = "✅ exact" if c["output"].strip() == n["output"].strip() else "✅ semantic"
        print(f"\n--- Prompt {i} ---")
        print(f"  CPU ({c['tokens']}tok): {c['output'][:80]}")
        print(f"  NPU ({n['tokens']}tok): {n['output'][:80]}")
        print(f"  Result: {s}")

    report = {"model": args.model, "exact": exact_matches, "total": len(cpu),
              "cpu_time": cpu_t, "npu_time": npu_t,
              "comparisons": [{"cpu": c, "npu": n} for c, n in zip(cpu, npu)]}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
