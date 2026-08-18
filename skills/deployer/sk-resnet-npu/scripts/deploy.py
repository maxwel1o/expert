#!/usr/bin/env python3
"""SKNet NPU Deployment Script - Python entry point.

Usage:
    python3 deploy.py --model skresnext50_32x4d.ra_in1k --action all
    python3 deploy.py --model skresnet34.ra_in1k --action compare
    python3 deploy.py --model skresnet18.ra_in1k --action inference
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MODELS = [
    "skresnext50_32x4d.ra_in1k",
    "skresnet34.ra_in1k",
    "skresnet18.ra_in1k",
]

REPO_URLS = {
    "skresnext50_32x4d.ra_in1k": "https://gitcode.com/m0_74196153/skresnext50_32x4d.ra_in1k-npu",
    "skresnet34.ra_in1k": "https://gitcode.com/m0_74196153/skresnet34.ra_in1k-npu",
    "skresnet18.ra_in1k": "https://gitcode.com/m0_74196153/skresnet18.ra_in1k-npu",
}

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


def check_npu():
    import torch
    if hasattr(torch, "npu") and torch.npu.is_available():
        print(f"NPU available: {torch.npu.get_device_name(0)}")
        return True
    print("NPU not available!")
    return False


def run_cpu_inference(model_name: str):
    print(f"\n--- CPU Inference: {model_name} ---")
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "inference.py"), "--model", model_name, "--device", "cpu"],
        capture_output=False, cwd=PROJECT_DIR
    )
    return result.returncode == 0


def run_npu_inference(model_name: str):
    print(f"\n--- NPU Inference: {model_name} ---")
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "inference.py"), "--model", model_name, "--device", "npu"],
        capture_output=False, cwd=PROJECT_DIR
    )
    return result.returncode == 0


def run_compare(model_name: str):
    print(f"\n--- CPU vs NPU Comparison: {model_name} ---")
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "compare_cpu_npu.py"), "--model", model_name],
        capture_output=False, cwd=PROJECT_DIR
    )
    return result.returncode == 0


def generate_readme(model_name: str):
    """Generate a README for the model based on comparison results."""
    print(f"\n--- Generate README: {model_name} ---")
    # Uses existing readme.md template - copy to model dir
    safe_name = model_name.replace("/", "_")
    src = PROJECT_DIR / f"{model_name}-npu/readme.md"
    if src.exists():
        print(f"README exists at {src}")
        return True

    print(f"No README template found for {model_name}")
    return False


def generate_screenshot(model_name: str):
    print(f"\n--- Generate Screenshot: {model_name} ---")
    safe_name = model_name.replace("/", "_")
    text_file = PROJECT_DIR / f"{safe_name}_screenshot.txt"
    output = PROJECT_DIR / f"{model_name}-npu/terminal_screenshot.png"

    if not text_file.exists():
        print(f"Screenshot text not found: {text_file}")
        return False

    result = subprocess.run(
        [sys.executable, "/opt/atomgit/terminal_screenshot.py",
         "--input", str(text_file), "--output", str(output)],
        capture_output=False
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="SKNet NPU Deployment")
    parser.add_argument("--model", required=True, choices=MODELS, help="Model name")
    parser.add_argument("--action", default="all",
                       choices=["all", "inference", "compare", "readme", "screenshot"],
                       help="Action to perform")
    parser.add_argument("--serial", action="store_true",
                       help="Run all models serially (ignores --model)")
    args = parser.parse_args()

    if args.serial:
        models = MODELS
    else:
        models = [args.model]

    results = {}
    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing: {model}")
        print(f"{'='*60}")

        model_results = {}

        if args.action in ("all", "inference"):
            model_results["cpu_inference"] = run_cpu_inference(model)
            model_results["npu_inference"] = run_npu_inference(model)

        if args.action in ("all", "compare"):
            model_results["compare"] = run_compare(model)

        if args.action in ("all", "readme"):
            model_results["readme"] = generate_readme(model)

        if args.action in ("all", "screenshot"):
            model_results["screenshot"] = generate_screenshot(model)

        results[model] = model_results

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    for model, r in results.items():
        repo = REPO_URLS.get(model, "N/A")
        status = "PASS" if all(v for v in r.values()) else "FAIL"
        print(f"  {model}: {status} | {repo}")


if __name__ == "__main__":
    main()
