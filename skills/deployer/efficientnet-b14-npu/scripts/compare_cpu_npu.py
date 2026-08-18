#!/usr/bin/env python3
"""Compare CPU vs NPU inference outputs for timm classification models.

Usage: python3 compare_cpu_npu.py <model_name>
"""
import argparse
import json
import os
import sys

import torch


def get_model_name(url_or_name: str) -> str:
    name = url_or_name.strip()
    if "modelscope.cn/models/timm/" in name:
        name = name.rsplit("timm/", 1)[-1]
    name = name.split("?")[0].strip("/")
    return name


def compare_outputs(model_name: str) -> dict:
    """Compare CPU and NPU inference outputs."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", model_name)

    cpu_path = os.path.join(output_dir, "output_cpu.pt")
    npu_path = os.path.join(output_dir, "output_npu.pt")

    if not os.path.exists(cpu_path):
        raise FileNotFoundError(f"CPU output not found: {cpu_path}. Run inference on CPU first.")
    if not os.path.exists(npu_path):
        raise FileNotFoundError(f"NPU output not found: {npu_path}. Run inference on NPU first.")

    cpu_out = torch.load(cpu_path, map_location="cpu", weights_only=True)
    npu_out = torch.load(npu_path, map_location="cpu", weights_only=True)

    print(f"\n{'='*60}")
    print(f"[COMPARE] Model: {model_name}")
    print(f"[COMPARE] CPU output shape: {list(cpu_out.shape)}")
    print(f"[COMPARE] NPU output shape: {list(npu_out.shape)}")
    print(f"{'='*60}")

    # Compute metrics
    mae = torch.abs(cpu_out - npu_out).mean().item()
    max_ae = torch.abs(cpu_out - npu_out).max().item()
    mse = ((cpu_out - npu_out) ** 2).mean().item()
    rmse = mse ** 0.5

    # Cosine similarity
    cpu_flat = cpu_out.flatten()
    npu_flat = npu_out.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(cpu_flat.unsqueeze(0), npu_flat.unsqueeze(0)).item()

    # Relative error
    cpu_norm = torch.norm(cpu_flat)
    npu_norm = torch.norm(npu_flat)
    if cpu_norm > 0:
        rel_error = torch.norm(cpu_flat - npu_flat).item() / cpu_norm.item()
    else:
        rel_error = float('inf')

    # Top-1 and Top-5 agreement
    cpu_top1 = cpu_out.argmax(dim=-1)
    npu_top1 = npu_out.argmax(dim=-1)
    top1_agree = (cpu_top1 == npu_top1).item()

    _, cpu_top5 = torch.topk(cpu_out, k=5, dim=-1)
    _, npu_top5 = torch.topk(npu_out, k=5, dim=-1)
    top5_agree = len(set(cpu_top5[0].tolist()) & set(npu_top5[0].tolist()))

    # Max probability difference
    cpu_probs = torch.nn.functional.softmax(cpu_out, dim=-1)
    npu_probs = torch.nn.functional.softmax(npu_out, dim=-1)
    max_prob_diff = torch.abs(cpu_probs - npu_probs).max().item()

    # Load timing info if available
    cpu_time = None
    npu_time = None
    timing_cpu_path = os.path.join(output_dir, "timing_cpu.txt")
    timing_npu_path = os.path.join(output_dir, "timing_npu.txt")
    if os.path.exists(timing_cpu_path):
        with open(timing_cpu_path) as f:
            for line in f:
                if "Average inference time" in line:
                    cpu_time = float(line.split()[-2].strip("s()"))
    if os.path.exists(timing_npu_path):
        with open(timing_npu_path) as f:
            for line in f:
                if "Average inference time" in line:
                    npu_time = float(line.split()[-2].strip("s()"))

    results = {
        "model_name": model_name,
        "mae": round(mae, 8),
        "max_ae": round(max_ae, 8),
        "mse": round(mse, 8),
        "rmse": round(rmse, 8),
        "cosine_similarity": round(cos_sim, 8),
        "relative_error": round(rel_error, 8),
        "top1_agreement": bool(top1_agree),
        "top5_agreement_count": top5_agree,
        "max_probability_difference": round(max_prob_diff, 8),
        "cpu_inference_time_s": cpu_time,
        "npu_inference_time_s": npu_time,
    }

    print(f"\n[RESULTS] Precision Comparison:")
    print(f"  Mean Absolute Error (MAE): {mae:.8f}")
    print(f"  Max Absolute Error: {max_ae:.8f}")
    print(f"  MSE: {mse:.8f}")
    print(f"  RMSE: {rmse:.8f}")
    print(f"  Cosine Similarity: {cos_sim:.8f}")
    print(f"  Relative Error: {rel_error:.8f}")
    print(f"  Top-1 Agreement: {top1_agree}")
    print(f"  Top-5 Agreements (out of 5): {top5_agree}")
    print(f"  Max Probability Difference: {max_prob_diff:.8f}")
    if cpu_time and npu_time:
        speedup = cpu_time / npu_time if npu_time > 0 else float('inf')
        print(f"  CPU Time: {cpu_time:.4f}s, NPU Time: {npu_time:.4f}s, Speedup: {speedup:.2f}x")

    # Conclusion
    error_pct = mae * 100
    if error_pct < 1.0 and cos_sim > 0.99:
        conclusion = "PASS"
        detail = f"误差<1% (MAE={mae:.6f}, cosine_sim={cos_sim:.6f})"
    else:
        conclusion = "FAIL"
        detail = f"误差>=1% (MAE={mae:.6f}, cosine_sim={cos_sim:.6f})"

    results["conclusion"] = conclusion
    results["detail"] = detail
    print(f"\n[VERDICT] {conclusion}: {detail}")

    # Save comparison results
    results_path = os.path.join(output_dir, "compare_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Comparison results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare CPU and NPU inference outputs")
    parser.add_argument("model_name", help="Model name")
    args = parser.parse_args()

    model_name = get_model_name(args.model_name)
    compare_outputs(model_name)


if __name__ == "__main__":
    main()
