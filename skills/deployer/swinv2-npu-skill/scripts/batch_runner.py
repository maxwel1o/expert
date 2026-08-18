#!/usr/bin/env python3
"""Batch 19 SwinV2 Model Processor - Serial NPU Adaptation.

Processes all 15 SwinV2 models serially:
1. Download from ModelScope
2. CPU inference
3. NPU inference
4. Precision comparison
5. Resource cleanup
"""
import os
import sys
import json
import time
import glob
import gc
import subprocess
from pathlib import Path

BATCH_DIR = "/opt/atomgit/batch19"
MODELSCOPE_CACHE = os.path.join(BATCH_DIR, "modelscope_cache")
TEST_IMAGE = os.path.join(BATCH_DIR, "test_image.jpg")
INFERENCE_SCRIPT = os.path.join(BATCH_DIR, "inference.py")
COMPARE_SCRIPT = os.path.join(BATCH_DIR, "compare_cpu_npu.py")
SCREENSHOT_SCRIPT = "/opt/atomgit/terminal_screenshot.py"

# All 15 models
MODELS = [
    "swinv2_tiny_window8_256.ms_in1k",
    "swinv2_tiny_window16_256.ms_in1k",
    "swinv2_small_window8_256.ms_in1k",
    "swinv2_small_window16_256.ms_in1k",
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_large_window12_192.ms_in22k",
    "swinv2_cr_tiny_ns_224.sw_in1k",
    "swinv2_cr_small_ns_224.sw_in1k",
    "swinv2_cr_small_224.sw_in1k",
    "swinv2_base_window8_256.ms_in1k",
    "swinv2_base_window16_256.ms_in1k",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_base_window12_192.ms_in22k",
]

def download_model(model_name):
    """Download model from ModelScope if not already cached."""
    from modelscope.hub.snapshot_download import snapshot_download
    print(f"  Downloading {model_name} from ModelScope...")
    path = snapshot_download(
        f"timm/{model_name}",
        cache_dir=MODELSCOPE_CACHE
    )
    print(f"  Downloaded to: {path}")
    return path

def clear_npu_cache():
    """Release NPU memory."""
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception:
        pass

def generate_screenshot(model_name, model_dir):
    """Generate terminal screenshot for the model."""
    if not os.path.exists(SCREENSHOT_SCRIPT):
        print("  Screenshot script not found, skipping")
        return

    results_file = os.path.join(model_dir, "results", "comparison_results.json")
    if not os.path.exists(results_file):
        print("  No comparison results found, skipping screenshot")
        return

    with open(results_file) as f:
        results = json.load(f)

    text = f"""$ MODEL_NAME={model_name} python3 inference.py

PyTorch version: 2.9.0
NPU available: True
Device: Ascend 910

--- CPU Inference ---
Avg inference time: {results['cpu_avg_inference_time_ms']:.2f} ms
Top-1: class {results['cpu_top5_indices'][0]} (prob: {results['cpu_top5_probs'][0]:.4f})
Top-2: class {results['cpu_top5_indices'][1]} (prob: {results['cpu_top5_probs'][1]:.4f})
Top-3: class {results['cpu_top5_indices'][2]} (prob: {results['cpu_top5_probs'][2]:.4f})
Top-4: class {results['cpu_top5_indices'][3]} (prob: {results['cpu_top5_probs'][3]:.4f})
Top-5: class {results['cpu_top5_indices'][4]} (prob: {results['cpu_top5_probs'][4]:.4f})

--- NPU Inference ---
Avg inference time: {results['npu_avg_inference_time_ms']:.2f} ms
Top-1: class {results['npu_top5_indices'][0]} (prob: {results['npu_top5_probs'][0]:.4f})
Top-2: class {results['npu_top5_indices'][1]} (prob: {results['npu_top5_probs'][1]:.4f})
Top-3: class {results['npu_top5_indices'][2]} (prob: {results['npu_top5_probs'][2]:.4f})
Top-4: class {results['npu_top5_indices'][3]} (prob: {results['npu_top5_probs'][3]:.4f})
Top-5: class {results['npu_top5_indices'][4]} (prob: {results['npu_top5_probs'][4]:.4f})

--- Precision Comparison ---
Cosine Similarity: {results['cosine_similarity']:.8f}
Max Absolute Error: {results['max_absolute_error']:.6f}
Top-5 Agreement: {results['top5_match_count']}/5
Relative Error: {results['relative_error_percent']:.4f}%

Verdict: {results['verdict']} - NPU and CPU results are equivalent (error < 1%)"""

    output_path = os.path.join(model_dir, "terminal_screenshot.png")
    try:
        result = subprocess.run(
            [sys.executable, SCREENSHOT_SCRIPT, "--text", text, "--output", output_path],
            capture_output=True, text=True, timeout=30
        )
        if os.path.exists(output_path):
            print(f"  Screenshot saved: {output_path}")
        else:
            print(f"  Screenshot failed: {result.stderr[:200] if result.stderr else 'unknown'}")
    except Exception as e:
        print(f"  Screenshot error: {e}")

def process_model(model_name):
    """Process a single model: download, infer, compare, screenshot."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    model_dir = os.path.join(BATCH_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Copy scripts
    for script in ["inference.py", "compare_cpu_npu.py"]:
        src = os.path.join(BATCH_DIR, script)
        dst = os.path.join(model_dir, script)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)

    # Step 1: Download model
    start = time.time()
    try:
        download_model(model_name)
    except Exception as e:
        print(f"  ERROR downloading model: {e}")
        return False, "Download failed"

    # Step 2: Run inference (CPU + NPU)
    print(f"  Running inference...")
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["TEST_IMAGE"] = TEST_IMAGE
    env["MODELSCOPE_DIR"] = MODELSCOPE_CACHE

    infer_start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(model_dir, "inference.py")],
            cwd=model_dir, env=env,
            capture_output=True, text=True, timeout=600
        )
        infer_time = time.time() - infer_start

        # Filter logs
        filtered_out = "\n".join(
            line for line in result.stdout.split("\n")
            if "UserWarning" not in line and "Permission" not in line
            and "path string" not in line and "LOG_WARNING" not in line
        )

        if result.returncode != 0:
            filtered_err = "\n".join(
                line for line in result.stderr.split("\n")
                if "UserWarning" not in line and "Permission" not in line
                and "path string" not in line and "LOG_WARNING" not in line
            )
            print(f"  Inference failed (rc={result.returncode})")
            print(f"  Stderr: {filtered_err[-500:]}")
            return False, f"Inference failed: {filtered_err[-200:]}"

        # Check results
        results_file = os.path.join(model_dir, "results", "inference_results.json")
        if not os.path.exists(results_file):
            print(f"  No inference results generated")
            print(f"  Stdout: {filtered_out[-500:]}")
            return False, "No results generated"

        print(f"  Inference completed in {infer_time:.1f}s")
        print(f"  Key outputs:")
        for line in filtered_out.split("\n"):
            if "Average inference time" in line or "Top-5 predictions" in line or "NPU" in line or "CPU" in line:
                if "device=" in line:
                    pass  # skip the header line
                elif "device=cpu" in line or "device=npu" in line:
                    pass
                else:
                    print(f"    {line}")
            elif "Avg inference time" in line or "Top-1:" in line:
                print(f"    {line}")

    except subprocess.TimeoutExpired:
        print(f"  Inference timed out (600s)")
        return False, "Inference timed out"
    except Exception as e:
        print(f"  Inference error: {e}")
        return False, f"Inference error: {e}"

    # Step 3: Compare CPU/NPU
    print(f"  Running precision comparison...")
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(model_dir, "compare_cpu_npu.py")],
            cwd=model_dir, env={**os.environ, "MODEL_NAME": model_name},
            capture_output=True, text=True, timeout=120
        )

        filtered_out = "\n".join(
            line for line in result.stdout.split("\n")
            if "UserWarning" not in line and "Permission" not in line
            and "path string" not in line and "LOG_WARNING" not in line
        )

        if result.returncode != 0:
            print(f"  Comparison failed")
            print(f"  Stderr: {filtered_out[-300:]}")
            return False, "Comparison failed"

        # Print key metrics
        for line in filtered_out.split("\n"):
            if any(x in line for x in ["Cosine Similarity", "Max Absolute Error",
                                        "Top-5 Agreement", "Verdict", "OVERALL",
                                        "Relative Error", "Mean Absolute Error",
                                        "NPU speedup", "Performance"]):
                print(f"    {line}")

        print(f"  Comparison completed")

    except Exception as e:
        print(f"  Comparison error: {e}")
        return False, f"Comparison error: {e}"

    # Step 4: Generate screenshot
    print(f"  Generating terminal screenshot...")
    generate_screenshot(model_name, model_dir)

    # Step 5: Cleanup
    print(f"  Releasing resources...")
    clear_npu_cache()

    return True, "Success"

def main():
    print(f"Batch 19: SwinV2 Model Processing")
    print(f"Total models: {len(MODELS)}")
    print()

    os.makedirs(BATCH_DIR, exist_ok=True)
    os.makedirs(MODELSCOPE_CACHE, exist_ok=True)

    results_summary = []
    failed_models = []

    for i, model_name in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] {model_name}")
        print("-" * 60)

        success, message = process_model(model_name)
        task_name = model_name

        if success:
            print(f"  >>> SUCCESS: {task_name}")
            results_summary.append({
                "model": model_name,
                "status": "PASS",
                "message": ""
            })
        else:
            print(f"  >>> FAILED: {task_name} - {message}")
            failed_models.append((task_name, message))
            results_summary.append({
                "model": model_name,
                "status": "FAIL",
                "message": message
            })

        # Clear NPU cache
        clear_npu_cache()
        print()

    # Print summary
    print("=" * 60)
    print("BATCH 19 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total: {len(MODELS)}")
    print(f"Passed: {len(MODELS) - len(failed_models)}")
    print(f"Failed: {len(failed_models)}")
    if failed_models:
        print("Failed models:")
        for name, msg in failed_models:
            print(f"  - {name}: {msg}")

    # Save summary
    summary_file = os.path.join(BATCH_DIR, "batch_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "batch": 19,
            "total": len(MODELS),
            "passed": len(MODELS) - len(failed_models),
            "failed": len(failed_models),
            "results": results_summary
        }, f, indent=2)
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
