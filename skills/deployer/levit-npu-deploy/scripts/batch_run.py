#!/usr/bin/env python3
"""Batch run all LeViT models sequentially to avoid NPU OOM.

Usage:
  python3 batch_run.py --models all --device npu
  python3 batch_run.py --models 128,192
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time


ALL_MODELS = ["levit-128", "levit-128S", "levit-192", "levit-256", "levit-384"]
SHORT_NAMES = {"128": "levit-128", "128s": "levit-128S", "192": "levit-192",
               "256": "levit-256", "384": "levit-384"}


def free_memory():
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except ImportError:
        pass


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Batch run LeViT models")
    parser.add_argument("--models", default="all", help="Models: all, or comma-separated (128,192,384)")
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--image", default="test.jpg", help="Test image path")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU inference")
    args = parser.parse_args()

    if args.models == "all":
        models = ALL_MODELS
    else:
        models = []
        for m in args.models.split(","):
            m = m.strip()
            models.append(SHORT_NAMES.get(m, m))

    print(f"\nBatch running {len(models)} models: {', '.join(models)}")
    print(f"Device: {args.device.upper()}")

    results = {}
    for model in models:
        print(f"\n{'#'*60}")
        print(f"#  Processing: {model}")
        print(f"{'#'*60}")

        model_dir = os.path.join(
            os.path.expanduser("~"), ".cache/modelscope/hub/models/facebook", model
        )

        # CPU inference
        if not args.skip_cpu:
            ret = run_cmd(
                f"MODEL_NAME={model} MODEL_DIR={model_dir} "
                f"python3 scripts/run_inference.py --model {model} "
                f"--device cpu --image {args.image}",
                f"CPU Inference: {model}"
            )
            if ret != 0:
                print(f"  CPU inference failed for {model}, continuing...")

        # NPU inference
        ret = run_cmd(
            f"MODEL_NAME={model} MODEL_DIR={model_dir} "
            f"python3 scripts/run_inference.py --model {model} "
            f"--device npu --image {args.image}",
            f"NPU Inference: {model}"
        )
        if ret != 0:
            print(f"  NPU inference failed for {model}, continuing...")

        # Compare
        if not args.skip_cpu:
            run_cmd(
                f"python3 scripts/compare.py --model {model}",
                f"Accuracy Comparison: {model}"
            )

        # Free memory
        print(f"\n  Freeing memory after {model}...")
        free_memory()
        time.sleep(1)

        # Read results
        try:
            cpu_t = json.load(open(f"/tmp/{model}_cpu_results.json"))["time_ms"]
            npu_t = json.load(open(f"/tmp/{model}_npu_results.json"))["time_ms"]
            results[model] = {
                "cpu_time_ms": cpu_t, "npu_time_ms": npu_t,
                "speedup": f"{cpu_t/npu_t:.1f}x"
            }
        except Exception as e:
            results[model] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch Run Summary")
    print(f"{'='*60}")
    for model, info in results.items():
        if "error" in info:
            print(f"  {model:15s} ERROR: {info['error']}")
        else:
            print(f"  {model:15s} CPU={info['cpu_time_ms']:.1f}ms "
                  f"NPU={info['npu_time_ms']:.1f}ms "
                  f"Speedup={info['speedup']}")


if __name__ == "__main__":
    main()
