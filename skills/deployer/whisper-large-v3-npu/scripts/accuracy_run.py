#!/usr/bin/env python3
"""
whisper-large-v3 NPU Accuracy Verification Script
Compares NPU output against CPU reference to verify <1% error rate.

Metrics:
- RMS relative error on logits (target: <1%)
- Top-1 token match rate (target: 100%)
- Generative token match rate (target: 100%)
- Cosine similarity of encoder hidden states (target: >0.999)

Usage:
    python accuracy_run.py --model /path/to/whisper-large-v3 --audio test.wav
    python accuracy_run.py --model /path/to/whisper-large-v3 --audio test.wav --tolerance 0.01
"""

import argparse
import os
import time
import sys
import warnings
import json

warnings.filterwarnings("ignore")
os.environ["HCCL_CONNECT_TIMEOUT"] = "1800"

import numpy as np
import torch
import torch_npu  # noqa: F401
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".wav", ".flac", ".ogg"):
        import soundfile as sf
        audio, sr = sf.read(audio_path)
    else:
        import librosa
        audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def compute_rms_rel_error(cpu_tensor: torch.Tensor, npu_tensor: torch.Tensor) -> float:
    """RMS of difference relative to RMS of reference signal."""
    diff = (cpu_tensor - npu_tensor).abs()
    rms_diff = torch.sqrt((diff ** 2).mean()).item()
    rms_signal = torch.sqrt((cpu_tensor ** 2).mean()).item()
    return rms_diff / (rms_signal + 1e-8)


def compute_cosine_similarity(cpu_tensor: torch.Tensor, npu_tensor: torch.Tensor) -> dict:
    """Per-position cosine similarity statistics."""
    cpu_norm = cpu_tensor / (cpu_tensor.norm(dim=-1, keepdim=True) + 1e-8)
    npu_norm = npu_tensor / (npu_tensor.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (cpu_norm * npu_norm).sum(dim=-1)
    return {
        "mean": cos_sim.mean().item(),
        "min": cos_sim.min().item(),
        "max": cos_sim.max().item(),
    }


def verify_accuracy(model_path: str, audio_path: str, tolerance: float = 0.01) -> dict:
    """Run full accuracy verification and return results dict."""
    results = {
        "model": model_path,
        "audio": audio_path,
        "tolerance": tolerance,
        "passed": True,
        "checks": {},
    }

    print("=" * 70)
    print("Whisper-large-v3 NPU Accuracy Verification")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Audio: {audio_path}")
    print(f"NPU device: {torch.npu.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    processor = WhisperProcessor.from_pretrained(model_path)
    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features_cpu = inputs.input_features
    print(f"Input shape: {input_features_cpu.shape}")

    # ---- Load CPU model ----
    print("\n[1/4] Loading CPU model (float32)...")
    model_cpu = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_cpu.eval()

    # ---- Load NPU model ----
    print("[2/4] Loading NPU model (float16)...")
    model_npu = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model_npu = model_npu.to("npu")
    model_npu.eval()

    # ---- Check 1: Generate text match ----
    print("[3/4] Running generate comparison...")
    input_npu = input_features_cpu.to("npu", dtype=torch.float16)
    with torch.no_grad():
        gen_cpu = model_cpu.generate(input_features_cpu, max_new_tokens=64, task="transcribe", language="en")
        gen_npu = model_npu.generate(input_npu, max_new_tokens=64, task="transcribe", language="en")

    cpu_text = processor.decode(gen_cpu[0], skip_special_tokens=True)
    npu_text = processor.decode(gen_npu[0].cpu(), skip_special_tokens=True)
    text_match = cpu_text == npu_text

    cpu_tokens = gen_cpu[0].tolist()
    npu_tokens = gen_npu[0].cpu().tolist()
    max_len = max(len(cpu_tokens), len(npu_tokens))
    for lst in [cpu_tokens, npu_tokens]:
        while len(lst) < max_len:
            lst.append(-1)
    token_matches = sum(1 for a, b in zip(cpu_tokens, npu_tokens) if a == b)
    token_rate = 100.0 * token_matches / max_len

    results["checks"]["generate"] = {
        "cpu_text": cpu_text,
        "npu_text": npu_text,
        "text_match": text_match,
        "token_match_rate": token_rate,
        "passed": text_match and token_rate >= 99.0,
    }
    print(f"  CPU text: {cpu_text}")
    print(f"  NPU text: {npu_text}")
    print(f"  Text match: {text_match}, Token rate: {token_rate:.1f}%")

    # ---- Check 2: Logits comparison (single forward pass) ----
    print("\n[4/4] Computing logits accuracy...")
    decoder_start = torch.tensor([[model_cpu.config.decoder_start_token_id]])
    with torch.no_grad():
        out_cpu = model_cpu(input_features_cpu, decoder_input_ids=decoder_start)
        out_npu = model_npu(input_npu, decoder_input_ids=decoder_start.to("npu"))

    cpu_logits = out_cpu.logits.float()
    npu_logits = out_npu.logits.float().cpu()

    rms_rel_logits = compute_rms_rel_error(cpu_logits, npu_logits)
    top1_cpu = cpu_logits.argmax(dim=-1)
    top1_npu = npu_logits.argmax(dim=-1)
    top1_match = (top1_cpu == top1_npu).float().mean().item()

    results["checks"]["logits"] = {
        "rms_relative_error": rms_rel_logits,
        "top1_match": top1_match,
        "passed": rms_rel_logits < tolerance and top1_match >= 0.99,
    }
    print(f"  Logits RMS relative error: {rms_rel_logits*100:.4f}% (target <{tolerance*100}%)")
    print(f"  Top-1 match: {top1_match*100:.2f}%")

    # ---- Check 3: Encoder hidden state cosine similarity ----
    with torch.no_grad():
        enc_cpu = model_cpu.model.encoder(input_features_cpu)
        enc_npu = model_npu.model.encoder(input_npu)

    enc_cpu_h = enc_cpu.last_hidden_state.float()
    enc_npu_h = enc_npu.last_hidden_state.float().cpu()
    cos_sim = compute_cosine_similarity(enc_cpu_h, enc_npu_h)

    results["checks"]["encoder"] = {
        "mean_cosine": cos_sim["mean"],
        "min_cosine": cos_sim["min"],
        "passed": cos_sim["mean"] > 0.999,
    }
    print(f"  Encoder mean cosine sim: {cos_sim['mean']:.6f} (target >0.999)")
    print(f"  Encoder min cosine sim:  {cos_sim['min']:.6f}")

    # ---- Overall verdict ----
    all_checks = results["checks"]
    results["passed"] = (
        all_checks["generate"]["passed"]
        and all_checks["logits"]["passed"]
        and all_checks["encoder"]["passed"]
    )

    print("\n" + "=" * 70)
    print("VERIFICATION RESULT")
    print("=" * 70)
    for name, check in all_checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\nOVERALL: {'PASS - Error < 1%' if results['passed'] else 'FAIL - Error > 1%'}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Whisper NPU accuracy verification")
    parser.add_argument("--model", default="/opt/atomgit/whisper-large-v3", help="Model path")
    parser.add_argument("--audio", required=True, help="Test audio file")
    parser.add_argument("--tolerance", type=float, default=0.01, help="RMS error tolerance (default 0.01 = 1%%)")
    parser.add_argument("--output", help="Save results as JSON")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: Model directory not found: {args.model}")
        sys.exit(1)
    if not torch.npu.is_available():
        print("ERROR: Ascend NPU not available")
        sys.exit(1)

    results = verify_accuracy(args.model, args.audio, args.tolerance)

    if args.output:
        # Convert non-serializable values
        out = {
            "model": results["model"],
            "audio": results["audio"],
            "tolerance": results["tolerance"],
            "passed": results["passed"],
            "checks": {
                name: {k: v for k, v in check.items() if k != "passed"}
                for name, check in results["checks"].items()
            },
        }
        out["checks_details"] = results["checks"]
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
