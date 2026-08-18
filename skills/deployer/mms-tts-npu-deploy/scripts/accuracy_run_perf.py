#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend NPU Performance Benchmark for facebook/mms-tts-*
Measures latency, throughput, and RTF on NPU.
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

from transformers import AutoTokenizer, VitsModel

_TEST_TEXTS_ENV = os.environ.get("MMS_TEST_TEXTS")
if _TEST_TEXTS_ENV:
    TEST_TEXTS = [t.strip() for t in _TEST_TEXTS_ENV.split("||")]
else:
    TEST_TEXTS = [
        "Hello, welcome to the world of text to speech.",
        "This is a test of the English text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Today is a great day for technology.",
        "Machine learning models can generate natural speech.",
        "The weather is beautiful outside.",
        "She sells seashells by the seashore.",
        "Programming is both an art and a science.",
        "Thank you for using this text to speech model.",
    ]


def run_perf(model_path, iterations=10, warmup=3):
    device = torch.device("npu") if torch_npu and torch.npu.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path).to(device).eval()
    sr = model.config.sampling_rate

    for i in range(warmup):
        t = TEST_TEXTS[i % len(TEST_TEXTS)]
        inputs = tokenizer(t, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if device.type == "npu":
            torch.npu.synchronize()

    latencies = []
    total_samples = 0
    total_chars = 0

    for i in range(iterations):
        t = TEST_TEXTS[i % len(TEST_TEXTS)]
        total_chars += len(t)
        inputs = tokenizer(t, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if device.type == "npu":
            torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if device.type == "npu":
            torch.npu.synchronize()
        latency = time.perf_counter() - start

        latencies.append(latency)
        total_samples += outputs.waveform.shape[-1]

    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50 = statistics.median(latencies)
    p90 = sorted(latencies)[int(iterations * 0.9)] if iterations >= 10 else max_latency
    total_audio_sec = total_samples / sr
    rtf = avg_latency / (total_audio_sec / iterations)
    throughput_chars = total_chars / sum(latencies)

    report = {
        "device": str(device),
        "iterations": iterations,
        "avg_latency_sec": round(avg_latency, 4),
        "min_latency_sec": round(min_latency, 4),
        "max_latency_sec": round(max_latency, 4),
        "p50_latency_sec": round(p50, 4),
        "p90_latency_sec": round(p90, 4),
        "rtf": round(rtf, 4),
        "throughput_chars_per_sec": round(throughput_chars, 2),
        "raw_latencies": [round(l, 4) for l in latencies],
    }
    return report


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./model"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output = sys.argv[3] if len(sys.argv) > 3 else "perf_report.json"

    print("=" * 60)
    print("Ascend NPU Performance Benchmark (MMS-TTS)")
    print("=" * 60)

    report = run_perf(model_path, iterations=iterations)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
