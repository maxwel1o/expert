#!/usr/bin/env python3
"""
Accuracy & Performance Evaluation for Pyannote Speaker Diarization on NPU.

Tests:
1. Accuracy: DER between NPU and CPU outputs (< 1% required)
2. Performance: RTF (Real-Time Factor) at various audio durations

Usage:
    pip install pyannote.audio==4.0.4 torch_npu soundfile pyannote.metrics
    python3 evaluate.py
"""

import os
import sys
import json
import time
import tempfile
import warnings

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

# Add inference.py to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Audio Generation
# ============================================================
def generate_speech_like_audio(duration=10.0, sample_rate=16000, num_speakers=2):
    """Generate multi-speaker-like audio with varying spectral content."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros(int(sample_rate * duration))
    seg_dur = duration / max(num_speakers, 1)
    for spk in range(num_speakers):
        start = spk * seg_dur
        end = min(start + seg_dur * 0.8, duration)
        s, e = int(start * sample_rate), int(end * sample_rate)
        if e - s <= 0:
            continue
        seg_t = t[s:e]
        base_freq = 100 + spk * 150
        seg = 0.3 * np.sin(2 * np.pi * base_freq * seg_t)
        seg += 0.15 * np.sin(2 * np.pi * base_freq * 1.5 * seg_t)
        seg += 0.1 * np.sin(2 * np.pi * base_freq * 2.0 * seg_t)
        envelope = np.sin(np.pi * np.linspace(0, 1, e - s)) ** 2
        audio[s:e] += seg * envelope
    noise = 0.02 * np.random.randn(int(sample_rate * duration))
    audio += noise
    peak = np.max(np.abs(audio) + 1e-8)
    if peak > 0:
        audio /= peak
    return audio.astype(np.float32)


# ============================================================
# Evaluation Functions
# ============================================================
def evaluate_accuracy(pipeline, test_wav):
    """Evaluate DER between NPU and CPU outputs."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    t0 = time.time()
    npu_out = pipeline(test_wav)
    npu_time = time.time() - t0

    pipeline.to(torch.device("cpu"))
    t0 = time.time()
    cpu_out = pipeline(test_wav)
    cpu_time = time.time() - t0

    if hasattr(torch, 'npu') and torch.npu.is_available():
        pipeline.to(torch.device("npu"))

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der = metric(cpu_out.speaker_diarization, npu_out.speaker_diarization)

    return {
        "der": float(der),
        "der_pct": round(float(der) * 100, 4),
        "npu_speakers": len(set(npu_out.speaker_diarization.labels())),
        "cpu_speakers": len(set(cpu_out.speaker_diarization.labels())),
        "npu_time_s": round(npu_time, 3),
        "cpu_time_s": round(cpu_time, 3),
    }


def evaluate_performance(pipeline, audio_path, audio_duration):
    """Evaluate RTF and throughput."""
    for _ in range(3):
        pipeline(audio_path)
    times = []
    for _ in range(5):
        t0 = time.time()
        pipeline(audio_path)
        times.append(time.time() - t0)
    avg = np.mean(times)
    rtf = avg / audio_duration if audio_duration > 0 else float('inf')
    return {
        "audio_duration_s": audio_duration,
        "avg_inference_time_s": round(avg, 3),
        "std_inference_time_s": round(np.std(times), 3),
        "rtf": round(rtf, 4),
        "throughput_x_realtime": round(1.0 / rtf, 2) if rtf > 0 else float('inf'),
        "inference_times_s": [round(t, 3) for t in times],
    }


# ============================================================
# Main
# ============================================================
def main():
    from inference import apply_patches, load_pipeline

    apply_patches()
    has_npu = torch.npu.is_available()
    device = torch.device("npu") if has_npu else torch.device("cpu")
    print(f"[EVAL] Device: {device}")

    results = {"device": str(device), "npu_available": has_npu, "models": {}}

    for model_name in ["community-1", "basic"]:
        print(f"\n{'=' * 60}")
        print(f"[EVAL] Loading {model_name} pipeline...")
        pipeline = load_pipeline(model=model_name, device=device)

        for duration in [10, 30]:
            print(f"\n  --- {model_name}: {duration}s audio ---")
            audio = generate_speech_like_audio(duration=duration)
            tmp = os.path.join(tempfile.gettempdir(), f"eval_{model_name}_{duration}s.wav")
            sf.write(tmp, audio, 16000)

            perf = evaluate_performance(pipeline, tmp, duration)
            print(f"  Performance: RTF={perf['rtf']}, avg={perf['avg_inference_time_s']}s")
            results["models"].setdefault(model_name, {}).setdefault("performance", {})[f"{duration}s"] = perf

            acc = evaluate_accuracy(pipeline, tmp)
            print(f"  Accuracy: DER={acc['der_pct']}%, NPU spkrs={acc['npu_speakers']}")
            results["models"].setdefault(model_name, {}).setdefault("accuracy", {})[f"{duration}s"] = acc

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for model_name, data in results["models"].items():
        print(f"\n--- {model_name} ---")
        for dur, acc in data.get("accuracy", {}).items():
            status = "PASS" if acc["der"] < 0.01 else "FAIL"
            print(f"  Accuracy ({dur}): DER={acc['der_pct']}% [{status}] "
                  f"NPU={acc['npu_time_s']}s CPU={acc['cpu_time_s']}s")
            if acc["der"] >= 0.01:
                all_pass = False
        for dur, perf in data.get("performance", {}).items():
            print(f"  Performance ({dur}): RTF={perf['rtf']}, {perf['throughput_x_realtime']}x realtime")

    if all_pass:
        print(f"\n[PASS] All accuracy checks passed (DER < 1%)")
    else:
        print(f"\n[FAIL] Some accuracy checks exceeded threshold")

    return results


if __name__ == "__main__":
    main()
