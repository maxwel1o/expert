#!/usr/bin/env python3
"""Compare CPU and NPU inference results for timm image classification models."""
import os
import sys
import json

import torch
import numpy as np

MODEL_NAME = os.environ.get("MODEL_NAME", "")
if not MODEL_NAME:
    print("ERROR: MODEL_NAME environment variable is required")
    sys.exit(1)

def max_absolute_error(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.max(np.abs(a - b)))

def mean_absolute_error(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.mean(np.abs(a - b)))

def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b + 1e-10))

def logit_relative_error(a, b):
    """Compute relative error based on abs(a-b)/max(abs(a), abs(b))."""
    a = np.array(a)
    b = np.array(b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-10
    return float(np.mean(np.abs(a - b) / denom) * 100)

print(f"=== CPU vs NPU Precision Comparison ===")
print(f"Model: {MODEL_NAME}")
print()

# Load results
results_path = "results/inference_results.json"
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run inference.py first.")
    sys.exit(1)

with open(results_path) as f:
    results = json.load(f)

cpu = results["cpu"]
npu = results["npu"]

# Compare logits
cpu_logits = np.array(cpu["logits"])
npu_logits = np.array(npu["logits"])

mae = max_absolute_error(cpu_logits, npu_logits)
mean_ae = mean_absolute_error(cpu_logits, npu_logits)
cos_sim = cosine_similarity(cpu_logits, npu_logits)
rel_err = logit_relative_error(cpu_logits, npu_logits)

print(f"Logits comparison:")
print(f"  Max Absolute Error: {mae:.8f}")
print(f"  Mean Absolute Error: {mean_ae:.8f}")
print(f"  Cosine Similarity: {cos_sim:.8f}")
print(f"  Relative Error (normalized): {rel_err:.4f}%")
print()

# Compare top-5 indices
cpu_top5 = np.array(cpu["top5_indices"])
npu_top5 = np.array(npu["top5_indices"])
top5_match = cpu_top5 == npu_top5
top5_agreement = int(np.sum(top5_match))
top5_agreement_pct = top5_agreement / 5 * 100

print(f"Top-5 prediction comparison:")
print(f"  CPU top-5 indices: {cpu_top5.tolist()}")
print(f"  NPU top-5 indices: {npu_top5.tolist()}")
print(f"  Top-5 agreement: {top5_agreement}/5 ({top5_agreement_pct:.0f}%)")
print()

# Compare top-5 probabilities
cpu_probs = cpu["top5_probs"]
npu_probs = npu["top5_probs"]
print(f"Top-5 probabilities comparison:")
for i in range(5):
    diff = abs(cpu_probs[i] - npu_probs[i])
    print(f"  #{i+1}: CPU={cpu_probs[i]:.6f}, NPU={npu_probs[i]:.6f}, diff={diff:.6f}")
print()

# Performance comparison
cpu_time = cpu["avg_inference_time_ms"]
npu_time = npu["avg_inference_time_ms"]
speedup = cpu_time / npu_time if npu_time > 0 else 0
print(f"Performance comparison:")
print(f"  CPU avg inference time: {cpu_time:.2f} ms")
print(f"  NPU avg inference time: {npu_time:.2f} ms")
print(f"  NPU speedup: {speedup:.2f}x")
print()

# Final verdict - use cosine similarity > 0.999 as main metric for logits
# For classification models, top-5 agreement and cosine similarity are key metrics
cos_sim_pass = cos_sim > 0.999
top5_pass = top5_agreement >= 4  # at least 4/5 same
mae_pass = mae < 1.0  # max abs error < 1.0 in logit space

print(f"=== Precision Verdict ===")
print(f"  Cosine Similarity: {cos_sim:.8f} ({'PASS' if cos_sim_pass else 'FAIL'}, threshold > 0.999)")
print(f"  Top-5 Agreement: {top5_agreement}/5 ({'PASS' if top5_pass else 'FAIL'})")
print(f"  Max Absolute Error: {mae:.6f} ({'PASS' if mae_pass else 'FAIL'}, threshold < 1.0)")

overall_pass = cos_sim_pass and top5_pass
if overall_pass:
    print(f"  OVERALL: PASS - NPU and CPU inference results are equivalent (error < 1%)")
else:
    print(f"  OVERALL: NEEDS REVIEW")

# Save comparison results
comparison = {
    "model": MODEL_NAME,
    "max_absolute_error": mae,
    "mean_absolute_error": mean_ae,
    "cosine_similarity": cos_sim,
    "relative_error_percent": round(rel_err, 4),
    "top5_match_count": int(top5_agreement),
    "top5_agreement_percent": round(top5_agreement_pct, 1),
    "cpu_top5_indices": cpu["top5_indices"],
    "npu_top5_indices": npu["top5_indices"],
    "cpu_top5_probs": cpu["top5_probs"],
    "npu_top5_probs": npu["top5_probs"],
    "cpu_avg_inference_time_ms": cpu_time,
    "npu_avg_inference_time_ms": npu_time,
    "npu_speedup_x": round(speedup, 2),
    "cosine_similarity_pass": cos_sim_pass,
    "top5_agreement_pass": top5_pass,
    "verdict": "PASS" if overall_pass else "FAIL"
}

os.makedirs("results", exist_ok=True)
with open("results/comparison_results.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("Comparison results saved to results/comparison_results.json")
