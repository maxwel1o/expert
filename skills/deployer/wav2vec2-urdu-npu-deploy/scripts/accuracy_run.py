#!/usr/bin/env python3
"""
精度验证: NPU vs CPU logits 对比
运行流程:
  1. CPU 推理 -> 保存基线
  2. NPU 推理 -> 对比基线
  3. 输出精度报告

用法:
  python3 accuracy_run.py
"""
import os
import sys
import json
import time
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "accuracy_check")


def run_cpu_baseline():
    """在 CPU 上运行推理（独立进程，避免 transfer_to_npu 影响）"""
    print("=" * 60)
    print("Step 1: CPU Baseline Inference")
    print("=" * 60)

    cpu_script = os.path.join(SCRIPT_DIR, "run_cpu_baseline.py")
    result = subprocess.run(
        [sys.executable, cpu_script],
        capture_output=False,
        cwd=SCRIPT_DIR,
    )
    return result.returncode == 0


def run_npu_comparison():
    """在 NPU 上运行推理并对比"""
    print("\n" + "=" * 60)
    print("Step 2: NPU Inference & Accuracy Comparison")
    print("=" * 60)

    npu_script = os.path.join(SCRIPT_DIR, "run_npu.py")
    result = subprocess.run(
        [sys.executable, npu_script],
        capture_output=False,
        cwd=SCRIPT_DIR,
    )
    return result.returncode == 0


def print_report():
    """打印最终精度报告"""
    results_path = os.path.join(OUTPUT_DIR, "accuracy_results.json")
    if not os.path.exists(results_path):
        print("No accuracy results found.")
        return

    with open(results_path) as f:
        r = json.load(f)

    print("\n" + "=" * 60)
    print("精度验证报告")
    print("=" * 60)
    print(f"  模型: {r['model']}")
    print(f"  NPU: {r['npu_device']}")
    print(f"  最大绝对误差: {r['max_abs_diff']:.8f}")
    print(f"  平均绝对误差: {r['mean_abs_diff']:.8f}")
    if r['max_rel_diff_nonzero_pct'] > 0:
        print(f"  最大相对误差: {r['max_rel_diff_nonzero_pct']:.4f}%")
        print(f"  平均相对误差: {r['mean_rel_diff_nonzero_pct']:.4f}%")
    print(f"  Token预测一致率: {r['token_match_pct']:.2f}%")
    print(f"  Top-1 in Top-5: {r['top1_in_top5_pct']:.2f}%")
    print(f"  平均推理时间: {r['avg_inference_time_ms']:.2f} ms")
    print(f"\n  总体结果: {'✓ PASSED' if r['passed'] else '✗ FAILED'}")
    print("=" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: CPU baseline
    if not run_cpu_baseline():
        print("ERROR: CPU baseline failed")
        sys.exit(1)

    # Step 2: NPU comparison
    if not run_npu_comparison():
        print("ERROR: NPU comparison failed")
        sys.exit(1)

    # Report
    print_report()


if __name__ == "__main__":
    main()
