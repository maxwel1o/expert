#!/usr/bin/env python3
"""
wav2vec2-large-xls-r-300m-Urdu NPU 性能基准测试
用法:
  python3 accuracy_run_perf.py          # 默认测试
  python3 accuracy_run_perf.py --dtype fp32  # 指定精度
"""
import os
import sys
import json
import time
import argparse
import numpy as np

import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch
from transformers import (
    AutoModelForCTC,
    AutoFeatureExtractor,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODEL_DIR, "accuracy_check")


def generate_audio(duration_sec):
    """生成指定长度的测试音频"""
    t = np.linspace(0, duration_sec, int(16000 * duration_sec), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 200 * t) +
             0.2 * np.sin(2 * np.pi * 400 * t) +
             0.1 * np.random.randn(len(t)))
    return (audio / np.max(np.abs(audio)) * 0.8).astype(np.float32)


def benchmark(audio_durations=[3, 5, 10, 30], num_warmup=3, num_runs=10):
    """在不同音频长度上测试性能"""
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print(f"Performance Benchmark - {torch.npu.get_device_name(0)}")
    print("=" * 60)

    # Load model
    print("Loading model...")
    model = AutoModelForCTC.from_pretrained(MODEL_DIR)
    model = model.to("cuda:0")  # mapped to npu:0 by transfer_to_npu
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    processor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)

    results = {}
    for duration in audio_durations:
        audio = generate_audio(duration)

        inputs = processor(
            audio, sampling_rate=16000,
            return_tensors="pt", return_attention_mask=True,
        )
        input_values = inputs.input_values.to("cuda:0")
        attention_mask = inputs.attention_mask.to("cuda:0")

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(input_values, attention_mask=attention_mask)

        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.npu.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = model(input_values, attention_mask=attention_mask)
            torch.npu.synchronize()
            times.append(time.time() - start)

        avg = np.mean(times) * 1000
        std = np.std(times) * 1000
        audio_len_s = len(audio) / 16000
        rtf = avg / 1000 / audio_len_s

        results[f"{duration}s"] = {
            "audio_length_s": audio_len_s,
            "avg_latency_ms": round(avg, 2),
            "std_ms": round(std, 2),
            "rtf": round(rtf, 4),
        }
        print(f"\n  Audio: {audio_len_s}s")
        print(f"  Latency: {avg:.2f} ± {std:.2f} ms")
        print(f"  RTF: {rtf:.4f}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {
        "model": "wav2vec2-large-xls-r-300m-Urdu",
        "device": torch.npu.get_device_name(0),
        "dtype": "fp32",
        "benchmark_results": results,
    }
    report_path = os.path.join(OUTPUT_DIR, "perf_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp32", choices=["fp32"])
    args = parser.parse_args()
    benchmark()
