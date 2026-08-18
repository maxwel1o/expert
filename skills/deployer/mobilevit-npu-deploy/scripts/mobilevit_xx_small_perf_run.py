#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend NPU Performance Benchmark for apple/mobilevit-xx-small
Measures inference latency and throughput on NPU.
"""

import os
import sys
import json
import time
import statistics

import torch

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    torch_npu = None

from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image


def run_perf(model_path: str, iterations: int = 50, warmup: int = 10):
    device = torch.device("npu") if torch_npu and torch.npu.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    processor = MobileViTImageProcessor.from_pretrained(model_path)
    model = MobileViTForImageClassification.from_pretrained(model_path).to(device).eval()

    dummy = Image.new("RGB", (256, 256), color=128)

    print(f"Warmup: {warmup} iterations...")
    for _ in range(warmup):
        inputs = processor(images=dummy, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            _ = model(pixel_values=pixel_values)
        if device.type == "npu":
            torch.npu.synchronize()

    latencies = []
    print(f"Benchmark: {iterations} iterations...")
    for _ in range(iterations):
        inputs = processor(images=dummy, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        if device.type == "npu":
            torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(pixel_values=pixel_values)
        if device.type == "npu":
            torch.npu.synchronize()
        latency = time.perf_counter() - start
        latencies.append(latency)

    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50 = statistics.median(latencies)
    p90 = sorted(latencies)[int(iterations * 0.9)] if iterations >= 10 else max_latency
    throughput = 1.0 / avg_latency

    report = {
        "model": "apple/mobilevit-xx-small",
        "device": str(device),
        "iterations": iterations,
        "warmup": warmup,
        "image_size": "256x256 (crop), 288x288 (resize)",
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "min_latency_ms": round(min_latency * 1000, 2),
        "max_latency_ms": round(max_latency * 1000, 2),
        "p50_latency_ms": round(p50 * 1000, 2),
        "p90_latency_ms": round(p90 * 1000, 2),
        "throughput_images_per_sec": round(throughput, 2),
        "raw_latencies_ms": [round(l * 1000, 2) for l in latencies],
    }
    return report


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./model"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    output = sys.argv[3] if len(sys.argv) > 3 else "perf_report.json"

    print("=" * 60)
    print("Ascend NPU Performance Benchmark")
    print("Model: apple/mobilevit-xx-small")
    print("=" * 60)

    report = run_perf(model_path, iterations=iterations)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
