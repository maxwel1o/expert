#!/usr/bin/env python3
"""
vLLM-Ascend environment configuration check script.

Usage:
    python env_check.py
"""

import os
import subprocess


def check_env_var(name: str, recommended: str = None, required: bool = False,
                  condition: str = None):
    """Check an environment variable and print status."""
    value = os.environ.get(name)

    if value is None:
        if required:
            print(f"[FAIL] {name} not set (required)")
        elif recommended:
            print(f"[WARN] {name} not set (recommend: {recommended})")
        else:
            print(f"[INFO] {name} not set")
        return

    if recommended and value != recommended:
        print(f"[WARN] {name} = {value} (recommend: {recommended})")
    else:
        print(f"[OK]   {name} = {value}")


def check_npu():
    """Check NPU availability via npu-smi."""
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Count NPU entries
            count = sum(1 for line in lines if "910" in line or "310" in line)
            print(f"[OK]   npu-smi: {count} NPU(s) detected")
        else:
            print(f"[FAIL] npu-smi failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("[FAIL] npu-smi not found in PATH")
    except Exception as e:
        print(f"[FAIL] npu-smi check error: {e}")


def check_cann():
    """Check CANN environment."""
    cann_path = os.environ.get("ASCEND_HOME_PATH")
    if cann_path:
        print(f"[OK]   ASCEND_HOME_PATH = {cann_path}")
    else:
        print("[INFO] ASCEND_HOME_PATH not set (source set_env.sh if needed)")


def main():
    print("=== vLLM-Ascend Environment Check ===")
    print()

    print("--- Model Download ---")
    check_env_var("VLLM_USE_MODELSCOPE", recommended="true")
    check_env_var("HF_ENDPOINT")
    print()

    print("--- NPU Memory ---")
    check_env_var("PYTORCH_NPU_ALLOC_CONF", recommended="expandable_segments:True")
    print()

    print("--- Logging ---")
    check_env_var("ASCEND_LOG_PATH", recommended="/tmp/ascend/log")
    check_env_var("ASCEND_GLOBAL_LOG_LEVEL")
    print()

    print("--- CPU / OpenMP ---")
    check_env_var("OMP_PROC_BIND", recommended="false")
    check_env_var("OMP_NUM_THREADS", recommended="1")
    print()

    print("--- vLLM ---")
    check_env_var("VLLM_WORKER_MULTIPROC_METHOD", recommended="spawn")
    check_env_var("TASK_QUEUE_ENABLE", recommended="1")
    print()

    print("--- HCCL (Multi-card) ---")
    check_env_var("HCCL_BUFFSIZE", recommended="512")
    check_env_var("HCCL_ALGO")
    print()

    print("--- Hardware ---")
    check_npu()
    check_cann()


if __name__ == "__main__":
    main()
