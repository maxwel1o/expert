#!/usr/bin/env python3
"""CPU vs NPU accuracy comparison for ModelScope CV models.

Usage:
    # First run inference on both devices:
    python3 inference.py --device cpu
    python3 inference.py --device npu

    # Then compare:
    python3 compare_cpu_npu.py
"""

import json
import os
import sys

import numpy as np

MODEL_NAME = os.environ.get("MODEL_NAME", "cv_resnest101_general_recognition")


def load_results(device: str):
    path = f"/tmp/{MODEL_NAME}_{device}_results.json"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run inference.py on {device} first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def compare():
    print(f"{'='*60}")
    print(f"  CPU vs NPU Accuracy Comparison - {MODEL_NAME}")
    print(f"{'='*60}")

    cpu = load_results("cpu")
    npu = load_results("npu")

    cpu_logits = np.array(cpu["logits"])
    npu_logits = np.array(npu["logits"])
    cpu_probs = np.array(cpu["probabilities"])
    npu_probs = np.array(npu["probabilities"])

    # 1. Logits comparison
    logits_diff = np.abs(cpu_logits - npu_logits)
    logits_mae = float(np.mean(logits_diff))
    logits_max_ae = float(np.max(logits_diff))
    logits_rmse = float(np.sqrt(np.mean(logits_diff ** 2)))

    print(f"\n  --- Logits Comparison ---")
    print(f"  Mean Absolute Error (MAE):     {logits_mae:.8f}")
    print(f"  Max Absolute Error (MaxAE):    {logits_max_ae:.8f}")
    print(f"  Root Mean Square Error (RMSE): {logits_rmse:.8f}")

    # 2. Probabilities comparison
    probs_diff = np.abs(cpu_probs - npu_probs)
    probs_mae = float(np.mean(probs_diff))
    probs_max_ae = float(np.max(probs_diff))

    print(f"\n  --- Probabilities Comparison ---")
    print(f"  Mean Absolute Error (MAE):     {probs_mae:.8f}")
    print(f"  Max Absolute Error (MaxAE):    {probs_max_ae:.8f}")

    # 3. Top-1 and Top-5 agreement
    cpu_top1 = np.argmax(cpu_logits, axis=-1)
    npu_top1 = np.argmax(npu_logits, axis=-1)
    cpu_top5 = np.argsort(cpu_logits, axis=-1)[:, -5:]
    npu_top5 = np.argsort(npu_logits, axis=-1)[:, -5:]
    top1_match = bool((cpu_top1 == npu_top1).item())
    top5_match = bool(len(set(cpu_top5[0]) & set(npu_top5[0])) >= 5)

    print(f"\n  --- Prediction Agreement ---")
    print(f"  Top-1 match:  {'YES' if top1_match else 'NO'}")
    print(f"  Top-5 match:  {'YES' if top5_match else 'NO'}")

    # 4. Cosine similarity of logits
    cpu_norm = cpu_logits / (np.linalg.norm(cpu_logits) + 1e-12)
    npu_norm = npu_logits / (np.linalg.norm(npu_logits) + 1e-12)
    cos_sim = float(np.dot(cpu_norm[0], npu_norm[0]))

    print(f"\n  --- Cosine Similarity ---")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # 5. Logits relative error
    cpu_logits_max = float(np.max(np.abs(cpu_logits))) + 1e-12
    logits_rel_mae = (logits_mae / cpu_logits_max) * 100
    logits_rel_max = (logits_max_ae / cpu_logits_max) * 100

    print(f"\n  --- Logits Relative Error ---")
    print(f"  Relative MAE:    {logits_rel_mae:.6f}%")
    print(f"  Relative MaxAE:  {logits_rel_max:.6f}%")

    # 6. Probabilities Max Absolute Error is primary pass/fail metric
    probs_error_pct = probs_max_ae * 100
    passed = probs_error_pct < 1.0

    print(f"\n  {'='*50}")
    print(f"  OVERALL ASSESSMENT")
    print(f"  {'='*50}")
    print(f"  Probability MaxAE: {probs_error_pct:.6f}%")
    print(f"  Logits Relative MaxAE: {logits_rel_max:.6f}%")

    if passed:
        print(f"  ✓ PASS: NPU vs CPU probability error < 1% ({probs_error_pct:.6f}% < 1%)")
    else:
        print(f"  ✗ FAIL: NPU vs CPU probability error >= 1% ({probs_error_pct:.6f}% >= 1%)")
    print(f"  {'='*50}\n")

    results = {
        "model": MODEL_NAME,
        "logits_mae": logits_mae,
        "logits_max_ae": logits_max_ae,
        "logits_rmse": logits_rmse,
        "logits_rel_mae_pct": float("nan") if cpu_logits_max == 0 else logits_rel_mae,
        "logits_rel_max_pct": float("nan") if cpu_logits_max == 0 else logits_rel_max,
        "probs_mae": probs_mae,
        "probs_max_ae": probs_max_ae,
        "probs_error_pct": probs_error_pct,
        "cosine_similarity": cos_sim,
        "top1_match": top1_match,
        "top5_match": top5_match,
        "cpu_top1": int(cpu_top1.item()),
        "npu_top1": int(npu_top1.item()),
        "passed": passed,
    }
    output_path = f"/tmp/{MODEL_NAME}_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Comparison results saved to {output_path}")

    return results


if __name__ == "__main__":
    compare()
