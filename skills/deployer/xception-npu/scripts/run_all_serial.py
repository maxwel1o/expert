#!/usr/bin/env python3
"""Serial execution script for all Xception models - avoids NPU OOM."""
import os
import sys
import subprocess
import gc
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), "batch2_xception")

MODELS = [
    "xception71.tf_in1k",
    "xception65p.ra3_in1k",
    "xception65.tf_in1k",
    "xception65.ra3_in1k",
    "xception41p.ra3_in1k",
    "xception41.tf_in1k",
]

def release_resources():
    """Release GPU/NPU memory and CPU memory."""
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass

def run_model(model_name, script="compare_cpu_npu.py"):
    """Run a single model's inference and comparison."""
    model_dir = os.path.join(MODELS_DIR, model_name)
    script_path = os.path.join(model_dir, script)

    if not os.path.exists(script_path):
        print(f"[SKIP] {model_name}: {script} not found")
        return False

    print(f"\n{'='*60}")
    print(f"[RUN] {model_name} - {script}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=600,
        cwd=model_dir
    )

    # Filter output
    for line in result.stdout.split('\n'):
        if any(x in line for x in ['Warning', 'warnings.warn', 'Path manager',
                                    'does not match', 'cannot create',
                                    'path string is NULL']):
            continue
        if line.strip():
            print(line)

    if result.returncode != 0:
        print(f"[FAIL] {model_name} returned code {result.returncode}")
        return False

    print(f"[PASS] {model_name} completed successfully")
    return True

if __name__ == "__main__":
    results = {}
    for model in MODELS:
        success = run_model(model)
        results[model] = "PASS" if success else "FAIL"
        release_resources()
        time.sleep(2)

    print(f"\n{'='*60}")
    print("Final Results:")
    for model, status in results.items():
        print(f"  {model:<30} {status}")
