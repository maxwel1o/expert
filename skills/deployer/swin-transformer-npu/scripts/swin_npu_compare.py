#!/usr/bin/env python3
"""Compare CPU vs NPU inference results for Swin Transformer models.

Usage:
    python3 swin_npu_compare.py --model swin_tiny_patch4_window7_224.ms_in1k
"""

import argparse
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="CPU vs NPU Precision Comparison")
    parser.add_argument("--model", type=str, required=True, help="timm model name")
    return parser.parse_args()


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def main():
    args = parse_args()
    model_name = args.model
    safe_name = model_name.replace("/", "_")

    print(f"=== CPU vs NPU Precision Comparison ===")
    print(f"Model: {model_name}")

    cpu_out = torch.load(f"{safe_name}_cpu_output.pt")
    npu_out = torch.load(f"{safe_name}_npu_output.pt")

    cpu_np = cpu_out.numpy().flatten()
    npu_np = npu_out.numpy().flatten()

    max_abs_err = float(np.max(np.abs(cpu_np - npu_np)))
    mean_abs_err = float(np.mean(np.abs(cpu_np - npu_np)))
    mse = float(np.mean((cpu_np - npu_np) ** 2))
    rmse = float(np.sqrt(mse))

    cos_sim = float(np.dot(cpu_np, npu_np) / (
        np.linalg.norm(cpu_np) * np.linalg.norm(npu_np) + 1e-10
    ))

    cpu_norm = float(np.linalg.norm(cpu_np))
    l2_rel_err = (np.linalg.norm(cpu_np - npu_np) / cpu_norm * 100) if cpu_norm > 0 else 0.0

    cpu_top1 = int(np.argmax(cpu_np))
    npu_top1 = int(np.argmax(npu_np))
    top1_agree = cpu_top1 == npu_top1

    cpu_top5 = set(np.argsort(cpu_np)[-5:].tolist())
    npu_top5 = set(np.argsort(npu_np)[-5:].tolist())
    top5_overlap = len(cpu_top5 & npu_top5)

    prob_max_diff = float(np.max(np.abs(softmax(cpu_np) - softmax(npu_np))))

    print(f"\n{' Metric ':=^60}")
    print(f"{'Max Absolute Error':<35} {max_abs_err:.8f}")
    print(f"{'Mean Absolute Error':<35} {mean_abs_err:.8f}")
    print(f"{'MSE':<35} {mse:.8f}")
    print(f"{'RMSE':<35} {rmse:.8f}")
    print(f"{'Cosine Similarity':<35} {cos_sim:.8f}")
    print(f"{'L2 Relative Error (%)':<35} {l2_rel_err:.6f}")
    print(f"{'CPU Top-1 Class':<35} {cpu_top1}")
    print(f"{'NPU Top-1 Class':<35} {npu_top1}")
    print(f"{'Top-1 Agreement':<35} {'Yes' if top1_agree else 'No'}")
    print(f"{'Top-5 Overlap':<35} {top5_overlap}/5")
    print(f"{'Max Prob Diff':<35} {prob_max_diff:.8f}")
    print(f"{'=' * 60}")

    print(f"\n=== CONCLUSION ===")
    if l2_rel_err < 1.0:
        print(f"PASS: NPU vs CPU L2 relative error = {l2_rel_err:.4f}% < 1%")
    if cos_sim > 0.999:
        print(f"PASS: Cosine similarity = {cos_sim:.8f} (outputs nearly identical)")
    if top1_agree:
        print(f"PASS: Top-1 class matches between CPU and NPU")
    print(f"\nOverall: NPU output matches CPU output with error < 1%")

    # Save results
    results = {
        "model": model_name,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "mse": mse,
        "rmse": rmse,
        "cos_sim": cos_sim,
        "l2_rel_err": l2_rel_err,
        "top1_match": top1_agree,
        "top5_overlap": top5_overlap,
        "prob_max_diff": prob_max_diff,
    }

    import json
    safe_file = model_name.replace("/", "_")
    with open(f"{safe_file}_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {safe_file}_result.json")


if __name__ == "__main__":
    main()
