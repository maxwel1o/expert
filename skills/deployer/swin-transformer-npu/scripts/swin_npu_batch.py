#!/usr/bin/env python3
"""Batch process all Swin Transformer models for NPU adaptation.

Runs CPU inference, NPU inference, and precision comparison serially
for each model. Releases NPU memory between models.

Usage:
    python3 swin_npu_batch.py --all
    python3 swin_npu_batch.py --models model1,model2
"""

import argparse
import gc
import json
import os
import sys
import time
import torch

from swin_npu_infer import run_inference
from swin_npu_compare import softmax

# All 19 Swin models with their input sizes
ALL_MODELS = [
    ("swin_tiny_patch4_window7_224.ms_in1k", 224),
    ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", 224),
    ("swin_tiny_patch4_window7_224.ms_in22k", 224),
    ("swin_small_patch4_window7_224.ms_in1k", 224),
    ("swin_small_patch4_window7_224.ms_in22k_ft_in1k", 224),
    ("swin_small_patch4_window7_224.ms_in22k", 224),
    ("swin_s3_tiny_224.ms_in1k", 224),
    ("swin_s3_small_224.ms_in1k", 224),
    ("swin_s3_base_224.ms_in1k", 224),
    ("swin_large_patch4_window7_224.ms_in22k_ft_in1k", 224),
    ("swin_large_patch4_window7_224.ms_in22k", 224),
    ("swin_large_patch4_window12_384.ms_in22k_ft_in1k", 384),
    ("swin_large_patch4_window12_384.ms_in22k", 384),
    ("swin_base_patch4_window7_224.ms_in22k", 224),
    ("swin_base_patch4_window7_224.ms_in22k_ft_in1k", 224),
    ("swin_base_patch4_window7_224.ms_in1k", 224),
    ("swin_base_patch4_window12_384.ms_in22k_ft_in1k", 384),
    ("swin_base_patch4_window12_384.ms_in22k", 384),
    ("swin_base_patch4_window12_384.ms_in1k", 384),
]

MODEL_MAP = {name: size for name, size in ALL_MODELS}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Swin Transformer NPU Adaptation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all 19 models")
    group.add_argument("--models", type=str, help="Comma-separated list of model names")
    return parser.parse_args()


def cleanup():
    """Release NPU memory."""
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()


def compute_precision(cpu_np, npu_np):
    """Compare CPU vs NPU outputs."""
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

    return {
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


def process_model(model_name, input_size):
    """Process a single model: CPU infer -> NPU infer -> compare."""
    import numpy as np

    safe_name = model_name.replace("/", "_")
    print(f"\n{'=' * 60}")
    print(f"Processing: {model_name} (input_size={input_size})")
    print(f"{'=' * 60}")

    # Step 1: CPU inference
    print("\n--- CPU Inference ---")
    cpu_result = run_inference(model_name, "cpu")
    cpu_np = cpu_result["output"].numpy().flatten()
    cpu_time = cpu_result["time_ms"]

    # Save CPU output
    torch.save(cpu_result["output"], f"{safe_name}_cpu_output.pt")
    print(f"CPU inference: {cpu_time:.1f}ms")

    cleanup()

    # Step 2: NPU inference
    print("\n--- NPU Inference ---")
    npu_result = run_inference(model_name, "npu")
    npu_np = npu_result["output"].numpy().flatten()
    npu_time = npu_result["time_ms"]

    # Save NPU output
    torch.save(npu_result["output"], f"{safe_name}_npu_output.pt")
    print(f"NPU inference: {npu_time:.1f}ms")

    # Step 3: Compare
    print("\n--- Precision Comparison ---")
    precision = compute_precision(cpu_np, npu_np)
    precision["cpu_time_ms"] = cpu_time
    precision["npu_time_ms"] = npu_time
    precision["speedup"] = cpu_time / npu_time if npu_time > 0 else 0
    precision["model"] = model_name
    precision["input_size"] = input_size

    # Save results
    with open(f"{safe_name}_result.json", "w") as f:
        json.dump(precision, f, indent=2)

    status = "PASS" if precision["l2_rel_err"] < 1.0 and precision["top1_match"] else "FAIL"
    print(f"\nStatus: {status}")
    print(f"CPU: {cpu_time:.1f}ms | NPU: {npu_time:.1f}ms | Speedup: {precision['speedup']:.1f}x")
    print(f"L2 Error: {precision['l2_rel_err']:.4f}% | Top-1 Match: {precision['top1_match']}")

    cleanup()
    return precision


def main():
    args = parse_args()

    if args.all:
        models = ALL_MODELS
    else:
        names = [n.strip() for n in args.models.split(",")]
        models = [(n, MODEL_MAP.get(n, 224)) for n in names]
        unknown = [n for n, _ in models if n not in MODEL_MAP]
        if unknown:
            print(f"Warning: unknown models: {unknown}")

    total = len(models)
    passed = 0
    failed = 0
    all_results = []

    print(f"\nBatch processing {total} models...\n")

    for i, (model_name, input_size) in enumerate(models, 1):
        print(f"\n[{i}/{total}] {model_name}")
        try:
            result = process_model(model_name, input_size)
            all_results.append(result)
            if result["l2_rel_err"] < 1.0 and result["top1_match"]:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAILED: {model_name} - {e}")
            failed += 1
            import traceback
            traceback.print_exc()
        time.sleep(1)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"BATCH SUMMARY: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'=' * 60}")
    print(f"\n{'Model':<50} {'CPU(ms)':<10} {'NPU(ms)':<10} {'Speedup':<10} {'L2 Err':<10} {'Status':<8}")
    print("-" * 98)
    for r in all_results:
        status = "PASS" if r["l2_rel_err"] < 1.0 and r["top1_match"] else "FAIL"
        print(f"{r['model']:<50} {r['cpu_time_ms']:<10.1f} {r['npu_time_ms']:<10.1f} {r['speedup']:<10.1f}x {r['l2_rel_err']:<10.4f}% {status:<8}")

    # Save batch summary
    with open("batch_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBatch summary saved to batch_summary.json")


if __name__ == "__main__":
    main()
