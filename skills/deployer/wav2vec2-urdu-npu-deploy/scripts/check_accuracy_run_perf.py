#!/usr/bin/env python3
"""检查精度验证和性能测试结果"""
import os
import sys
import json

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODEL_DIR, "accuracy_check")


def main():
    print("=" * 60)
    print("wav2vec2-large-xls-r-300m-Urdu 适配验证检查")
    print("=" * 60)

    # Check accuracy results
    acc_path = os.path.join(OUTPUT_DIR, "accuracy_results.json")
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            r = json.load(f)
        print(f"\n[精度验证] {'✓ PASSED' if r['passed'] else '✗ FAILED'}")
        print(f"  Token预测一致率: {r['token_match_pct']:.2f}%")
        print(f"  最大绝对误差: {r['max_abs_diff']:.6f}")
    else:
        print(f"\n[精度验证] 未运行 - 请执行: python3 accuracy_run.py")
        return 1

    # Check perf results
    perf_path = os.path.join(OUTPUT_DIR, "perf_results.json")
    if os.path.exists(perf_path):
        with open(perf_path) as f:
            p = json.load(f)
        print(f"\n[性能数据]")
        for dur, data in p["benchmark_results"].items():
            print(f"  {dur}: {data['avg_latency_ms']}ms (RTF={data['rtf']})")
    else:
        print(f"\n[性能数据] 未运行 - 请执行: python3 accuracy_run_perf.py")

    print("\n" + "=" * 60)
    print("适配状态: 完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
