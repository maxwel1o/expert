#!/usr/bin/env python3
"""Compare CPU vs NPU inference results for timm MixNet models"""

import json
import torch
import numpy as np


def compare():
    print("=" * 60)
    print("CPU vs NPU Precision Comparison")
    print("=" * 60)

    # Load logits
    cpu_logits = torch.load("logits_cpu.pt")
    npu_logits = torch.load("logits_npu.pt")
    print(f"CPU logits shape: {cpu_logits.shape}")
    print(f"NPU logits shape: {npu_logits.shape}")

    # Numerical comparison
    diff = (cpu_logits - npu_logits).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    mse = ((cpu_logits - npu_logits) ** 2).mean().item()
    mae_ratio = mean_abs_err / cpu_logits.abs().mean().item() * 100

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_logits.view(1, -1), npu_logits.view(1, -1)
    ).item()

    print(f"\n--- Numerical Metrics ---")
    print(f"Max Absolute Error:   {max_abs_err:.8f}")
    print(f"Mean Absolute Error:  {mean_abs_err:.8f}")
    print(f"MSE:                  {mse:.8f}")
    print(f"MAE/Mean(|CPU|) Ratio:{mae_ratio:.6f}%")
    print(f"Cosine Similarity:    {cos_sim:.8f}")

    # Classification agreement
    cpu_top1 = cpu_logits.argmax(dim=1)
    npu_top1 = npu_logits.argmax(dim=1)
    top1_agreement = (cpu_top1 == npu_top1).item()

    cpu_top5 = cpu_logits.topk(5, dim=1).indices
    npu_top5 = npu_logits.topk(5, dim=1).indices
    top5_agreement = all(
        cpu_top5[0, i].item() in npu_top5[0].tolist()
        for i in range(5)
    )

    print(f"\n--- Classification Agreement ---")
    print(f"Top-1 Match:          {'YES' if top1_agreement else 'NO'}")
    print(f"Top-5 Overlap:        {'YES' if top5_agreement else 'NO'}")

    # Softmax probability comparison
    cpu_probs = torch.nn.functional.softmax(cpu_logits[0], dim=0)
    npu_probs = torch.nn.functional.softmax(npu_logits[0], dim=0)
    prob_diff = (cpu_probs - npu_probs).abs()
    print(f"\n--- Probability Metrics ---")
    print(f"Max Prob Difference:  {prob_diff.max().item():.8f}")
    print(f"Mean Prob Difference: {prob_diff.mean().item():.8f}")

    # Verdict
    print(f"\n--- Verdict ---")
    print(f"MAE/Mean(|CPU|) < 1%: {'PASS' if mae_ratio < 1.0 else 'FAIL'}")
    print(f"Max Abs Error < 0.01: {'PASS' if max_abs_err < 0.01 else 'FAIL'}")
    print(f"Cosine Similarity > 0.999: {'PASS' if cos_sim > 0.999 else 'WARN'}")

    overall = "PASS" if (mae_ratio < 1.0 and max_abs_err < 0.01) else "FAIL"
    print(f"Overall: {overall}")

    # Save comparison results
    results = {
        "max_absolute_error": round(max_abs_err, 8),
        "mean_absolute_error": round(mean_abs_err, 8),
        "mse": round(mse, 8),
        "mae_ratio_percent": round(mae_ratio, 6),
        "cosine_similarity": round(cos_sim, 8),
        "top1_match": bool(top1_agreement),
        "top5_overlap": bool(top5_agreement),
        "max_prob_difference": round(prob_diff.max().item(), 8),
        "mean_prob_difference": round(prob_diff.mean().item(), 8),
        "mae_ratio_under_1_percent": mae_ratio < 1.0,
        "overall_pass": overall == "PASS",
    }
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[INFO] Comparison results saved to comparison_results.json")

    return overall == "PASS"


if __name__ == "__main__":
    compare()
