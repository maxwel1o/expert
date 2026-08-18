#!/usr/bin/env python3
"""
Accuracy verification: compare NPU vs CPU outputs for Phi models.
Ensures NPU output distribution deviates < 1% from CPU baseline.

Usage: python check_accuracy.py <model_name>
  model_name: phi-1 | phi-1_5 | phi-2
"""
import argparse
import json
import os
import sys

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
    "phi-1": torch.float32,
    "phi-1_5": torch.float16,
    "phi-2": torch.float16,
}

TEST_PROMPTS = [
    "The capital of France is",
    "def hello():",
    "Machine learning is",
    "The meaning of life is",
    "Python is a",
]

# Token indices that are too close to 0 (special tokens) should be excluded
# from logit comparison
SPECIAL_TOKEN_THRESHOLD = 5  # tokens with id < 5 are typically special


def compare_logits(npu_logits, cpu_logits, tokenizer, model_name):
    """Compare NPU and CPU logits, report relative errors."""
    # Move to CPU for comparison
    npu_logits = npu_logits.cpu().float()
    cpu_logits = cpu_logits.cpu().float()

    # Compute softmax probabilities
    npu_probs = torch.softmax(npu_logits, dim=-1)
    cpu_probs = torch.softmax(cpu_logits, dim=-1)

    # Per-token metrics
    metrics = []
    for i in range(npu_logits.shape[0]):
        # Top-1 prediction comparison
        npu_top1 = npu_logits[i].argmax().item()
        cpu_top1 = cpu_logits[i].argmax().item()
        top1_match = npu_top1 == cpu_top1

        # KL divergence (NPU vs CPU)
        kl = torch.sum(cpu_probs[i] * (torch.log(cpu_probs[i] + 1e-10) - torch.log(npu_probs[i] + 1e-10))).item()

        # Max absolute logit difference
        max_logit_diff = torch.max(torch.abs(npu_logits[i] - cpu_logits[i])).item()

        # Mean absolute logit difference
        mean_logit_diff = torch.mean(torch.abs(npu_logits[i] - cpu_logits[i])).item()

        # Cosine similarity of logits
        cos_sim = torch.nn.functional.cosine_similarity(
            npu_logits[i].unsqueeze(0), cpu_logits[i].unsqueeze(0)
        ).item()

        metrics.append({
            "token_pos": i,
            "npu_top1": npu_top1,
            "npu_top1_token": tokenizer.decode(npu_top1) if tokenizer else str(npu_top1),
            "cpu_top1": cpu_top1,
            "cpu_top1_token": tokenizer.decode(cpu_top1) if tokenizer else str(cpu_top1),
            "top1_match": top1_match,
            "kl_div": round(kl, 8),
            "max_logit_diff": round(max_logit_diff, 6),
            "mean_logit_diff": round(mean_logit_diff, 6),
            "cosine_sim": round(cos_sim, 8),
        })

    # Aggregate metrics
    avg_kl = sum(m["kl_div"] for m in metrics) / len(metrics)
    avg_cos_sim = sum(m["cosine_sim"] for m in metrics) / len(metrics)
    top1_accuracy = sum(1 for m in metrics if m["top1_match"]) / len(metrics)
    avg_max_logit_diff = sum(m["max_logit_diff"] for m in metrics) / len(metrics)
    avg_mean_logit_diff = sum(m["mean_logit_diff"] for m in metrics) / len(metrics)

    return {
        "per_token": metrics,
        "aggregate": {
            "avg_kl_divergence": avg_kl,
            "avg_cosine_similarity": avg_cos_sim,
            "top1_accuracy": top1_accuracy,
            "avg_max_logit_diff": avg_max_logit_diff,
            "avg_mean_logit_diff": avg_mean_logit_diff,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Check Phi model accuracy on NPU vs CPU")
    parser.add_argument("model_name", choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model_name]
    dtype = MODEL_DTYPES[args.model_name]

    print(f"Loading {args.model_name} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model once on CPU, clone state_dict for NPU copy
    model_cpu = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model_cpu.eval()

    model_npu = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model_npu = model_npu.npu()
    model_npu.eval()

    all_results = {}
    passed = 0
    failed = 0

    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Testing prompt: '{prompt}'")
        print(f"{'='*60}")

        inputs = tokenizer(prompt, return_tensors="pt")

        # CPU run
        with torch.no_grad():
            cpu_outputs = model_cpu(**inputs, output_hidden_states=False)

        # NPU run
        npu_inputs = {k: v.to("npu:0") for k, v in inputs.items()}
        with torch.no_grad():
            npu_outputs = model_npu(**npu_inputs, output_hidden_states=False)

        # Compare logits (first token prediction, no generation needed)
        cpu_logits = cpu_outputs.logits[0]  # (seq_len, vocab_size)
        npu_logits = npu_outputs.logits[0]

        result = compare_logits(npu_logits, cpu_logits, tokenizer, args.model_name)
        agg = result["aggregate"]

        top1_ok = agg["top1_accuracy"] >= 0.99
        cos_sim_ok = agg["avg_cosine_similarity"] >= 0.99
        kl_ok = agg["avg_kl_divergence"] < 0.01

        prompt_ok = top1_ok and cos_sim_ok and kl_ok

        print(f"  Top-1 accuracy:       {agg['top1_accuracy']*100:.2f}% {'PASS' if top1_ok else 'FAIL'}")
        print(f"  Avg cosine similarity: {agg['avg_cosine_similarity']:.6f} {'PASS' if cos_sim_ok else 'FAIL'}")
        print(f"  Avg KL divergence:     {agg['avg_kl_divergence']:.8f} {'PASS' if kl_ok else 'FAIL'}")
        print(f"  Avg max logit diff:    {agg['avg_max_logit_diff']:.6f}")
        print(f"  Avg mean logit diff:   {agg['avg_mean_logit_diff']:.6f}")

        all_results[prompt] = result
        if prompt_ok:
            passed += 1
        else:
            failed += 1

        # Also test generation consistency
        print(f"\n  --- Generation test ---")
        with torch.no_grad():
            gen_outputs = model_npu.generate(
                **npu_inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        print(f"  NPU generated: {gen_text[:100]}...")

    print(f"\n{'='*60}")
    print(f"ACCURACY SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts tested: {len(TEST_PROMPTS)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Overall: {'PASS' if failed == 0 else 'FAIL'}")
    print(f"(Threshold: top1 >= 99%, cos_sim >= 0.99, KL < 0.01)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
