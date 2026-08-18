#!/usr/bin/env python3
"""Example: Run inference and comparison for a single cubeai model."""

import os
import sys
import json

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_inference import main as infer_main
from run_compare import compare

MODEL_NAME = "cv_level1_protected_animals_classification"
CACHE_DIR = os.path.expanduser("~/.cache/modelscope")

def main():
    model_path = os.path.join(CACHE_DIR, "cubeai", MODEL_NAME)

    if not os.path.isdir(model_path):
        print(f"Model not found. Download first with:")
        print(f"  python3 -c \"from modelscope import snapshot_download; snapshot_download('cubeai/{MODEL_NAME}')\"")
        return

    print(f"Model: {MODEL_NAME}")
    print(f"Path: {model_path}")
    print()

    # Run comparison
    results = compare(model_path)

    if results:
        print(f"\n{'='*50}")
        print(f"Precision Check: {'PASS' if results['passed'] else 'FAIL'}")
        print(f"CPU Time: {results['cpu_time_ms']} ms")
        print(f"NPU Time: {results['npu_time_ms']} ms")
        print(f"Speedup: {results['speedup']}x")
        print(f"Max Error: {results['max_error_pct']}%")

if __name__ == "__main__":
    main()
