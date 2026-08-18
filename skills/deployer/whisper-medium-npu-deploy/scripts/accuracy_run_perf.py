#!/usr/bin/env python3
"""whisper-medium Ascend NPU 性能基准测试"""
import os
import json
import time
import numpy as np
import torch
import torch_npu  # noqa: F401
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "./whisper-medium")
AUDIO_PATH = os.environ.get("TEST_AUDIO_PATH", "test_audio.wav")

device = torch.device("npu:0")
print(f"Device: {torch.npu.get_device_name(0)}")

processor = WhisperProcessor.from_pretrained(MODEL_PATH)

audio, sr = sf.read(AUDIO_PATH)
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

audio_duration = len(audio) / 16000
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.float32, low_cpu_mem_usage=True,
).to(device)
model.eval()

# Warmup
print("Warming up...")
for _ in range(5):
    with torch.no_grad():
        _ = model.generate(
            input_features.to(device), language="english", task="transcribe",
            return_timestamps=False, max_length=448, num_beams=1,
        )
torch.npu.synchronize()

# Benchmark
n_runs = 20
latencies = []
for i in range(n_runs):
    torch.npu.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            input_features.to(device), language="english", task="transcribe",
            return_timestamps=False, max_length=448, num_beams=1,
        )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    latencies.append(elapsed * 1000)  # ms

latencies = np.array(latencies)

# Encoder-only latency
torch.npu.synchronize()
start = time.perf_counter()
with torch.no_grad():
    _ = model.get_encoder()(input_features.to(device))
torch.npu.synchronize()
encoder_ms = (time.perf_counter() - start) * 1000

print(f"\n{'='*50}")
print("Performance Results")
print(f"{'='*50}")
print(f"Audio duration: {audio_duration:.2f}s")
print(f"Encoder latency: {encoder_ms:.1f}ms")
print(f"\nEnd-to-end latency ({n_runs} runs):")
print(f"  Mean:  {latencies.mean():.1f}ms")
print(f"  Median:{np.median(latencies):.1f}ms")
print(f"  Min:   {latencies.min():.1f}ms")
print(f"  Max:   {latencies.max():.1f}ms")
print(f"  Std:   {latencies.std():.1f}ms")
print(f"  P90:   {np.percentile(latencies, 90):.1f}ms")
print(f"  P99:   {np.percentile(latencies, 99):.1f}ms")
print(f"\nReal-time factor (RTF): {latencies.mean()/1000/audio_duration:.4f}")
print(f"Speedup vs real-time: {audio_duration/(latencies.mean()/1000):.1f}x")

results = {
    "audio_duration_s": round(audio_duration, 2),
    "encoder_latency_ms": round(encoder_ms, 1),
    "e2e_latency_mean_ms": round(float(latencies.mean()), 1),
    "e2e_latency_median_ms": round(float(np.median(latencies)), 1),
    "e2e_latency_min_ms": round(float(latencies.min()), 1),
    "e2e_latency_max_ms": round(float(latencies.max()), 1),
    "e2e_latency_std_ms": round(float(latencies.std()), 1),
    "e2e_latency_p90_ms": round(float(np.percentile(latencies, 90)), 1),
    "e2e_latency_p99_ms": round(float(np.percentile(latencies, 99)), 1),
    "rtf": round(float(latencies.mean() / 1000 / audio_duration), 4),
    "num_runs": n_runs,
}

with open("perf_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to perf_results.json")
