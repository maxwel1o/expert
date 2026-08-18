#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend NPU Accuracy Validation for facebook/mms-tts-*
Validates model correctness on NPU for generative VITS TTS.

VITS uses stochastic duration prediction (use_stochastic_duration_prediction=true)
and noise injection (noise_scale=0.667). This means:
  - Same text -> different waveform lengths/shapes per run (expected)
  - CPU and NPU will produce different outputs (expected)
  - Exact waveform comparison is NOT meaningful

Validation strategy:
  1. NPU self-consistency: mel-spectrogram distribution stability across runs
  2. CPU-NPU structural consistency: mel-spectrogram statistics alignment
  3. Output validity: waveform has expected range, non-zero values
"""

import os
import sys
import json

import torch
import numpy as np

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
        "This is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Today is a great day for technology.",
    ]


def synthesize_waveform(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return outputs.waveform[0].cpu().numpy()


def compute_mel_spectrogram(waveform, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    frames = []
    for i in range(0, len(waveform) - n_fft, hop_length):
        frame = waveform[i:i + n_fft]
        windowed = frame * np.hanning(n_fft)
        fft = np.fft.rfft(windowed)
        power = np.abs(fft) ** 2
        frames.append(power)
    if len(frames) == 0:
        return np.zeros((n_mels, 1))
    spec = np.stack(frames, axis=1)
    mel_bins = np.linspace(0, spec.shape[0], n_mels, endpoint=False, dtype=int)
    mel_spec = np.zeros((n_mels, spec.shape[1]))
    for i, b in enumerate(mel_bins):
        if b < spec.shape[0]:
            mel_spec[i] = spec[b]
    return np.log(mel_spec + 1e-10)


def compute_spectral_stats(waveform, sr=16000):
    mel = compute_mel_spectrogram(waveform, sr=sr)
    return {
        "mel_mean": float(np.mean(mel)),
        "mel_std": float(np.std(mel)),
        "wave_mean": float(np.mean(waveform)),
        "wave_std": float(np.std(waveform)),
        "wave_max": float(np.max(np.abs(waveform))),
        "waveform_length": int(len(waveform)),
    }


def validate_output_validity(waveform):
    checks = [
        ("non_zero", np.max(np.abs(waveform)) > 1e-6),
        ("finite", np.all(np.isfinite(waveform))),
        ("reasonable_range", np.max(np.abs(waveform)) < 10.0),
        ("min_length", len(waveform) > 1000),
    ]
    return all(c[1] for c in checks), checks


def validate_npu_consistency(tokenizer, model_npu, device, runs=3):
    print("\n=== NPU Self-Consistency Validation ===")
    all_pass = True
    results = []

    for text in TEST_TEXTS:
        runs_wave = []
        for _ in range(runs):
            wave = synthesize_waveform(text, tokenizer, model_npu, device)
            runs_wave.append(wave)

        valid, _ = validate_output_validity(runs_wave[0])
        stats = [compute_spectral_stats(w) for w in runs_wave]
        mel_means = [s["mel_mean"] for s in stats]
        mel_stds = [s["mel_std"] for s in stats]
        mean_variance = float(np.var(mel_means))
        std_variance = float(np.var(mel_stds))

        struct_pass = mean_variance < 3.0 and std_variance < 2.0 and valid
        status = "PASS" if struct_pass else "FAIL"
        if not struct_pass:
            all_pass = False

        print(f"\nText: {text}")
        print(f"  Waveform lengths: {[len(w) for w in runs_wave]}")
        print(f"  Mel mean variance: {mean_variance:.4f}")
        print(f"  Mel std variance: {std_variance:.4f}")
        print(f"  Status: {status}")

        results.append({
            "text": text,
            "status": status,
            "output_valid": valid,
            "mel_mean_variance": mean_variance,
            "mel_std_variance": std_variance,
            "waveform_lengths": [int(len(w)) for w in runs_wave],
            "spectral_stats": stats,
        })

    return all_pass, results


def validate_cpu_npu_structure(tokenizer, model_cpu, model_npu, cpu_device, npu_device):
    print("\n=== CPU vs NPU Output Validation ===")
    all_pass = True
    results = []

    for text in TEST_TEXTS:
        cpu_wave = synthesize_waveform(text, tokenizer, model_cpu, cpu_device)
        npu_wave = synthesize_waveform(text, tokenizer, model_npu, npu_device)
        cpu_valid, _ = validate_output_validity(cpu_wave)
        npu_valid, _ = validate_output_validity(npu_wave)
        cpu_stats = compute_spectral_stats(cpu_wave)
        npu_stats = compute_spectral_stats(npu_wave)
        mel_mean_diff = abs(cpu_stats["mel_mean"] - npu_stats["mel_mean"])
        mel_std_diff = abs(cpu_stats["mel_std"] - npu_stats["mel_std"])

        both_valid = cpu_valid and npu_valid
        struct_pass = both_valid and mel_mean_diff < 2.0 and mel_std_diff < 2.0
        status = "PASS" if struct_pass else "FAIL"
        if not struct_pass:
            all_pass = False

        print(f"\nText: {text}")
        print(f"  CPU: valid={cpu_valid} len={len(cpu_wave)}")
        print(f"  NPU: valid={npu_valid} len={len(npu_wave)}")
        print(f"  Mel mean diff: {mel_mean_diff:.4f}  Mel std diff: {mel_std_diff:.4f}")
        print(f"  Status: {status}")

        results.append({
            "text": text,
            "status": status,
            "cpu_stats": cpu_stats,
            "npu_stats": npu_stats,
            "mel_mean_diff": float(mel_mean_diff),
            "mel_std_diff": float(mel_std_diff),
        })

    return all_pass, results


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./model"
    output_report = sys.argv[2] if len(sys.argv) > 2 else "accuracy_report.json"

    print("=" * 60)
    print("Ascend NPU Accuracy Validation (MMS-TTS)")
    print("Note: Generative VITS TTS is non-deterministic by design.")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_cpu = VitsModel.from_pretrained(model_path).eval()

    model_npu = None
    if torch_npu is not None and torch.npu.is_available():
        model_npu = VitsModel.from_pretrained(model_path).to("npu").eval()
    else:
        print("NPU not available, skipping NPU validation.")

    npu_pass = True
    npu_consistency_results = []
    if model_npu is not None:
        npu_pass, npu_consistency_results = validate_npu_consistency(tokenizer, model_npu, torch.device("npu"))

    struct_pass = True
    struct_results = []
    if model_npu is not None:
        struct_pass, struct_results = validate_cpu_npu_structure(
            tokenizer, model_cpu, model_npu, torch.device("cpu"), torch.device("npu")
        )

    overall_pass = npu_pass and struct_pass
    report = {
        "model": model_path,
        "architecture": "VitsModel",
        "npu_available": model_npu is not None,
        "note": "VITS is a generative TTS model with stochastic duration prediction.",
        "overall_status": "PASS" if overall_pass else "FAIL",
        "npu_self_consistency": {
            "status": "PASS" if npu_pass else "FAIL",
            "results": npu_consistency_results,
        },
        "cpu_npu_structural": {
            "status": "PASS" if struct_pass else "FAIL",
            "results": struct_results,
        },
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {output_report}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
