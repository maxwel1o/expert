#!/usr/bin/env python3
"""Compare CPU vs NPU inference results for timm classification models."""
import json
import os
import sys

import torch
import numpy as np


def compare_results(model_name: str, base_dir: str = "."):
    output_dir = os.path.join(base_dir, "outputs", model_name)

    # Load CPU and NPU outputs
    cpu_output = torch.load(os.path.join(output_dir, "output_cpu.pt"), map_location="cpu")
    npu_output = torch.load(os.path.join(output_dir, "output_npu.pt"), map_location="cpu")

    with open(os.path.join(output_dir, "results_cpu.json")) as f:
        cpu_results = json.load(f)
    with open(os.path.join(output_dir, "results_npu.json")) as f:
        npu_results = json.load(f)

    print(f"\n{'='*60}")
    print(f"CPU vs NPU Accuracy Comparison: {model_name}")
    print(f"{'='*60}")

    cpu_np = cpu_output.numpy()
    npu_np = npu_output.numpy()

    # Metrics
    mae = np.mean(np.abs(cpu_np - npu_np))
    mse = np.mean((cpu_np - npu_np) ** 2)
    rmse = np.sqrt(mse)

    # Cosine similarity
    cpu_flat = cpu_np.flatten()
    npu_flat = npu_np.flatten()
    cos_sim = np.dot(cpu_flat, npu_flat) / (np.linalg.norm(cpu_flat) * np.linalg.norm(npu_flat) + 1e-10)

    # Probability comparison
    cpu_probs = torch.nn.functional.softmax(cpu_output[0], dim=0).numpy()
    npu_probs = torch.nn.functional.softmax(npu_output[0], dim=0).numpy()
    prob_mae = np.mean(np.abs(cpu_probs - npu_probs))
    max_prob_diff = np.max(np.abs(cpu_probs - npu_probs))

    # Top-1/Top-5 consistency
    cpu_top5 = set(torch.topk(torch.from_numpy(cpu_probs), 5).indices.tolist())
    npu_top5 = set(torch.topk(torch.from_numpy(npu_probs), 5).indices.tolist())
    top5_match = len(cpu_top5 & npu_top5)
    top1_match = torch.argmax(torch.from_numpy(cpu_probs)).item() == torch.argmax(torch.from_numpy(npu_probs)).item()

    # Conclusion
    error_pct = mae * 100
    conclusion = "PASS" if error_pct < 1.0 and cos_sim > 0.99 else "FAIL"

    print(f"  MAE (logits):          {mae:.6f}")
    print(f"  MSE:                   {mse:.6f}")
    print(f"  RMSE:                  {rmse:.6f}")
    print(f"  Cosine Similarity:     {cos_sim:.6f}")
    print(f"  Prob MAE:              {prob_mae:.6f}")
    print(f"  Max Prob Diff:         {max_prob_diff:.6f}")
    print(f"  Top-1 Match:           {top1_match}")
    print(f"  Top-5 Match:           {top5_match}/5")
    print(f"  Error Percentage:      {error_pct:.4f}%")
    print(f"  Conclusion:            {conclusion}")

    compare_results = {
        "model_name": model_name,
        "mae": round(float(mae), 6),
        "mse": round(float(mse), 6),
        "rmse": round(float(rmse), 6),
        "cosine_similarity": round(float(cos_sim), 6),
        "prob_mae": round(float(prob_mae), 6),
        "max_prob_diff": round(float(max_prob_diff), 6),
        "top1_match": bool(top1_match),
        "top5_match": int(top5_match),
        "error_percentage": round(float(error_pct), 4),
        "conclusion": conclusion,
        "cpu_inference_time_s": cpu_results.get("avg_inference_time_s", 0),
        "npu_inference_time_s": npu_results.get("avg_inference_time_s", 0),
    }

    with open(os.path.join(output_dir, "compare_results.json"), "w") as f:
        json.dump(compare_results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Comparison results saved to: {os.path.join(output_dir, 'compare_results.json')}")

    return compare_results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_cpu_npu.py <model_name>")
        sys.exit(1)
    model_name = sys.argv[1]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    compare_results(model_name, base_dir)


if __name__ == "__main__":
    main()
