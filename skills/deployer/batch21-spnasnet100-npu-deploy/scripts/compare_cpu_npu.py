#!/usr/bin/env python3
"""CPU vs NPU precision comparison for SPNASNet_100."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


def compare_results(logits_path="results/logits.pt"):
    """Compare CPU and NPU inference results."""
    if not Path(logits_path).exists():
        print(f"Error: {logits_path} not found. Run inference.py first.")
        sys.exit(1)

    data = torch.load(logits_path, weights_only=True)
    cpu_logits = data["cpu"]
    npu_logits = data["npu"]

    print("=" * 70)
    print("CPU vs NPU Precision Comparison - SPNASNet_100")
    print("=" * 70)

    # 1. Logits comparison
    logits_diff = (cpu_logits - npu_logits).abs()
    max_abs_err = logits_diff.max().item()
    mean_abs_err = logits_diff.mean().item()
    mse = ((cpu_logits - npu_logits) ** 2).mean().item()

    # Relative error (avoid division by zero)
    cpu_mag = cpu_logits.abs().max().item()
    rel_err = max_abs_err / cpu_mag if cpu_mag > 0 else 0

    print(f"\n{'─'*70}")
    print("1. Logits Comparison")
    print(f"{'─'*70}")
    print(f"  Max absolute error:    {max_abs_err:.6e}")
    print(f"  Mean absolute error:   {mean_abs_err:.6e}")
    print(f"  MSE:                   {mse:.6e}")
    print(f"  Max relative error:    {rel_err*100:.6f}%")
    print(f"  CPU logits max:        {cpu_logits.abs().max().item():.4f}")
    print(f"  NPU logits max:        {npu_logits.abs().max().item():.4f}")

    # Ensure 2D for softmax
    if cpu_logits.dim() == 1:
        cpu_logits = cpu_logits.unsqueeze(0)
        npu_logits = npu_logits.unsqueeze(0)

    # 2. Probability comparison
    cpu_probs = F.softmax(cpu_logits, dim=1)
    npu_probs = F.softmax(npu_logits, dim=1)
    prob_diff = (cpu_probs - npu_probs).abs()
    max_prob_err = prob_diff.max().item() * 100  # in percentage
    mean_prob_err = prob_diff.mean().item() * 100
    prob_cosine = F.cosine_similarity(cpu_probs, npu_probs, dim=1).item()

    print(f"\n{'─'*70}")
    print("2. Probability Comparison")
    print(f"{'─'*70}")
    print(f"  Max probability diff:  {max_prob_err:.6f}%")
    print(f"  Mean probability diff: {mean_prob_err:.6f}%")
    print(f"  Cosine similarity:     {prob_cosine:.10f}")

    # 3. Top-1 / Top-5 consistency
    cpu_top1 = cpu_logits.argmax(dim=1).item()
    npu_top1 = npu_logits.argmax(dim=1).item()
    cpu_top5 = set(cpu_logits.topk(5).indices[0].tolist())
    npu_top5 = set(npu_logits.topk(5).indices[0].tolist())

    print(f"\n{'─'*70}")
    print("3. Classification Consistency")
    print(f"{'─'*70}")
    print(f"  CPU Top-1 class: {cpu_top1}")
    print(f"  NPU Top-1 class: {npu_top1}")
    print(f"  Top-1 match:     {'YES' if cpu_top1 == npu_top1 else 'NO'}")
    print(f"  Top-5 overlap:   {len(cpu_top5 & npu_top5)}/5 classes")
    print(f"  Top-5 match:     {'YES' if cpu_top5 == npu_top5 else 'NO'}")

    # 4. Detailed top-5
    try:
        from timm.data import IMAGENET_1k
        id2label = IMAGENET_1k
    except Exception:
        id2label = {i: f"class_{i}" for i in range(1000)}

    cpu_top5_vals, cpu_top5_idx = torch.topk(cpu_probs[0] * 100, k=5)
    npu_top5_vals, npu_top5_idx = torch.topk(npu_probs[0] * 100, k=5)

    print(f"\n{'─'*70}")
    print("4. Top-5 Predictions Detail")
    print(f"{'─'*70}")
    print(f"  {'Rank':<6} {'CPU Class':<60} {'CPU Prob':>8} {'NPU Class':<60} {'NPU Prob':>8}")
    print(f"  {'─'*5} {'─'*59} {'─'*8} {'─'*59} {'─'*8}")
    for i in range(5):
        cpu_lbl = id2label.get(int(cpu_top5_idx[i]), str(int(cpu_top5_idx[i])))
        npu_lbl = id2label.get(int(npu_top5_idx[i]), str(int(npu_top5_idx[i])))
        print(f"  {i+1:<6} {cpu_lbl:<59} {cpu_top5_vals[i]:>7.2f}% "
              f"{npu_lbl:<59} {npu_top5_vals[i]:>7.2f}%")

    # 5. Summary
    print(f"\n{'═'*70}")
    result = "PASS" if max_prob_err < 1.0 else "FAIL"
    print(f"VERDICT: {result}")
    print(f"  Max probability difference: {max_prob_err:.4f}%")
    if max_prob_err < 1.0:
        print(f"  ✓ NPU and CPU inference error < 1%")
    else:
        print(f"  ✗ NPU and CPU inference error >= 1%")
    print(f"{'═'*70}")

    # Save comparison results
    comparison = {
        "max_absolute_error": round(max_abs_err, 10),
        "mean_absolute_error": round(mean_abs_err, 10),
        "mse": round(mse, 10),
        "max_relative_error_pct": round(rel_err * 100, 6),
        "max_probability_diff_pct": round(max_prob_err, 6),
        "mean_probability_diff_pct": round(mean_prob_err, 6),
        "cosine_similarity": round(prob_cosine, 10),
        "top1_match": cpu_top1 == npu_top1,
        "top5_overlap": len(cpu_top5 & npu_top5),
        "verdict": result,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\nComparison results saved to results/comparison_results.json")

    return result == "PASS"


if __name__ == "__main__":
    success = compare_results()
    sys.exit(0 if success else 1)
