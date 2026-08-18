#!/usr/bin/env python3
"""Performance benchmark: NPU vs CPU inference time for kotoba-whisper models.

Usage:
    python3 eval_performance.py --model kotoba-tech/kotoba-whisper-v1.0
"""

import argparse
import json
import time
import warnings
import numpy as np
import torch
import torch_npu
warnings.filterwarnings("ignore")


def generate_test_audio(duration_sec=3.0, sample_rate=16000):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    freq = np.linspace(200, 800, len(t))
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="kotoba-whisper performance eval")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    args = parser.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_short = args.model.split("/")[-1]
    print(f"Benchmarking: {args.model}")
    audio = generate_test_audio()

    results = {}

    for device in ["cpu", "npu"]:
        print(f"\n[{device.upper()}] Loading model...")
        model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
        proc = WhisperProcessor.from_pretrained(args.model)

        inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features.to(device)
        forced_ids = proc.get_decoder_prompt_ids(language="ja", task="transcribe")

        # Warmup
        print("  Warming up...")
        for _ in range(2):
            with torch.no_grad():
                _ = model.generate(feat, forced_decoder_ids=forced_ids, max_new_tokens=32)

        # Benchmark
        times = []
        for i in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                ids = model.generate(feat, forced_decoder_ids=forced_ids, max_new_tokens=128)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0]
            print(f"  Run {i+1}: {elapsed:.2f}s - {text[:60]}")

        results[device] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
        }
        del model

    speedup = results["cpu"]["mean"] / results["npu"]["mean"]
    print(f"\nSpeedup (NPU vs CPU): {speedup:.1f}x")

    results["speedup"] = speedup
    results["model"] = model_short
    results["device"] = torch.npu.get_device_name(0)

    output = f"{model_short}_perf.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
