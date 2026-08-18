#!/usr/bin/env python3
"""Batch runner for SEMNASNet NPU deployment skill - run all models sequentially."""

import gc
import subprocess
import sys
import torch

MODELS = [
    "semnasnet_100.rmsp_in1k",
    "semnasnet_075.rmsp_in1k",
]


def run_model(model_name, device="npu"):
    """Run inference and comparison for a single model."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name} on {device}")
    print(f"{'='*60}\n")

    # Step 1: Inference
    print(f"[1/3] Running {device} inference...")
    result = subprocess.run(
        [sys.executable, "inference.py",
         "--model-name", model_name,
         "--device", device],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] Inference failed for {model_name}")
        return False

    # Step 2: Accuracy comparison (NPU only)
    if device == "npu":
        print(f"\n[2/3] Running CPU vs NPU accuracy comparison...")
        result = subprocess.run(
            [sys.executable, "compare_cpu_npu.py",
             "--model-name", model_name],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[ERROR] Comparison failed for {model_name}")
            return False

    # Step 3: Memory cleanup
    print(f"\n[3/3] Cleaning up resources...")
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()

    print(f"\n✓ Completed: {model_name}")
    return True


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "npu"

    for model_name in MODELS:
        success = run_model(model_name, device)
        if not success:
            print(f"[FAILED] {model_name} - continuing with next model")

    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
