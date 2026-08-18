#!/usr/bin/env python3
"""Batch runner for all 19 Swin Transformer models - Phase 2 & 3.

Usage: python3 batch_runner.py
"""
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELS = [
    "swin_tiny_patch4_window7_224.ms_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k",
    "swin_small_patch4_window7_224.ms_in1k",
    "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_small_patch4_window7_224.ms_in22k",
    "swin_s3_tiny_224.ms_in1k",
    "swin_s3_small_224.ms_in1k",
    "swin_s3_base_224.ms_in1k",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_large_patch4_window7_224.ms_in22k",
    "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
    "swin_large_patch4_window12_384.ms_in22k",
    "swin_base_patch4_window7_224.ms_in22k",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_base_patch4_window7_224.ms_in1k",
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "swin_base_patch4_window12_384.ms_in22k",
    "swin_base_patch4_window12_384.ms_in1k",
]

BATCH_DIR = Path("/opt/atomgit/batch20")
RESULTS_FILE = BATCH_DIR / "batch_results.txt"
FAILED_MODELS = []


def run_cmd(cmd: list[str], desc: str, timeout: int = 300) -> tuple[bool, str]:
    print(f"\n  >>> {desc}")
    print(f"  Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONWARNINGS": "ignore"},
        )
        stdout = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        stderr = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
        print(stdout)
        if result.returncode != 0:
            print(f"  WARN: Exit code {result.returncode}")
            print(stderr)
            return False, stderr
        return True, stdout
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timed out after {timeout}s")
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, str(e)


def add_to_queue_file(model_name: str):
    """Remove model from queue file after completion."""
    # Batch entries are managed by batch number, not individual models
    pass


def generate_terminal_screenshot(log_text: str, output_path: str):
    """Generate terminal screenshot using the screenshot tool."""
    result = subprocess.run(
        ["python3", "/opt/atomgit/terminal_screenshot.py",
         "--text", log_text,
         "--output", output_path],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def main():
    n = len(MODELS)
    print(f"Batch 20: {n} Swin Transformer models")
    print("=" * 60)

    for i, model_name in enumerate(MODELS, 1):
        print(f"\n{'=' * 60}")
        print(f"Model [{i}/{n}]: {model_name}")
        print(f"{'=' * 60}")

        # Check HF cache
        hf_cache_dir = Path(f"/opt/atomgit/.cache/huggingface/hub/models--timm--{model_name.replace('/', '--')}")
        if not hf_cache_dir.exists():
            print(f"  WARNING: HF cache not found for {model_name}, skipping...")
            FAILED_MODELS.append((model_name, "HF cache not found"))
            continue

        model_results = {
            "model": model_name,
            "npu_ok": False,
            "error_pct": "N/A",
            "npu_time_ms": "N/A",
            "cpu_time_ms": "N/A",
        }

        # Step 1: Run compare_cpu_npu.py
        print(f"\n  --- Step 1/2: CPU vs NPU comparison ---")
        ok, output = run_cmd(
            ["python3", "-W", "ignore", str(BATCH_DIR / "compare_cpu_npu.py"),
             "--model", model_name],
            "Comparing CPU vs NPU",
            timeout=600,
        )

        if ok:
            model_results["npu_ok"] = True
            # Parse results from output
            for line in output.split('\n'):
                if "Max absolute error" in line:
                    try:
                        val = float(line.split(":")[-1].strip())
                        model_results["error_pct"] = f"{val * 100:.4f}%"
                    except: pass
                elif "NPU inference time" in line:
                    try:
                        val = line.split(":")[-1].strip().replace("ms", "")
                        model_results["npu_time_ms"] = val
                    except: pass
                elif "CPU inference time" in line:
                    try:
                        val = line.split(":")[-1].strip().replace("ms", "")
                        model_results["cpu_time_ms"] = val
                    except: pass

            # Save output for screenshot
            screenshot_text = f"$ python3 compare_cpu_npu.py --model {model_name}\n"
            for line in output.split('\n'):
                line = line.strip()
                if line and not line.startswith('/usr') and not line.startswith('path string'):
                    screenshot_text += line + "\n"

            screenshot_path = str(BATCH_DIR / f"screenshots" / f"{model_name.replace('.', '_')}.png")
            os.makedirs(str(BATCH_DIR / "screenshots"), exist_ok=True)
            generate_terminal_screenshot(screenshot_text, screenshot_path)
            print(f"  Screenshot saved to: {screenshot_path}")
        else:
            print(f"  FAILED: {model_name} comparison")
            FAILED_MODELS.append((model_name, "Comparison failed"))

        # Save model results
        result_line = (
            f"{model_results['model']} | "
            f"{'PASS' if model_results['npu_ok'] else 'FAIL'} | "
            f"{model_results['error_pct']} | "
            f"CPU: {model_results['cpu_time_ms']}ms | "
            f"NPU: {model_results['npu_time_ms']}ms"
        )
        with open(RESULTS_FILE, "a") as f:
            f.write(result_line + "\n")
        print(f"\n  Result: {result_line}")

        # Memory cleanup
        print("\n  Cleaning up...")
        gc.collect()
        try:
            import torch
            if hasattr(torch, "npu"):
                torch.npu.empty_cache()
        except Exception:
            pass

        print(f"  Completed: {model_name}")

    # Summary
    print(f"\n{'=' * 60}")
    print("BATCH 20 SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total: {n} models")
    print(f"Successful: {n - len(FAILED_MODELS)}")
    print(f"Failed: {len(FAILED_MODELS)}")
    for name, reason in FAILED_MODELS:
        print(f"  - {name}: {reason}")

    return 0 if not FAILED_MODELS else 1


if __name__ == "__main__":
    sys.exit(main())
