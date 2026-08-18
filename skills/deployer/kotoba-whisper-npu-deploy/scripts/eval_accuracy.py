#!/usr/bin/env python3
"""Accuracy evaluation: NPU vs CPU logit comparison for kotoba-whisper models.

Usage:
    python3 eval_accuracy.py --model kotoba-tech/kotoba-whisper-v1.0
"""

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="kotoba-whisper accuracy eval")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    args = parser.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_short = args.model.split("/")[-1]
    print(f"Evaluating: {args.model}")
    audio = generate_test_audio()

    # CPU baseline
    print("\n[1/4] Loading CPU model...")
    cpu_model = WhisperForConditionalGeneration.from_pretrained(args.model).to("cpu").eval()
    cpu_proc = WhisperProcessor.from_pretrained(args.model)

    inputs = cpu_proc(audio, sampling_rate=16000, return_tensors="pt")
    feat = inputs.input_features

    with torch.no_grad():
        enc = cpu_model.model.encoder(feat).last_hidden_state
    dec_in = torch.tensor([[cpu_model.config.decoder_start_token_id]])
    with torch.no_grad():
        dec = cpu_model.model.decoder(dec_in, encoder_hidden_states=enc).last_hidden_state
        cpu_logits = cpu_model.proj_out(dec).squeeze().float()
    del cpu_model

    # NPU inference
    print("[2/4] Loading NPU model...")
    npu_model = WhisperForConditionalGeneration.from_pretrained(args.model).to("npu").eval()
    feat_npu = feat.to("npu")
    dec_in_npu = dec_in.to("npu")
    with torch.no_grad():
        enc_npu = npu_model.model.encoder(feat_npu).last_hidden_state
        dec_npu = npu_model.model.decoder(dec_in_npu, encoder_hidden_states=enc_npu).last_hidden_state
        npu_logits = npu_model.proj_out(dec_npu).squeeze().cpu().float()
    del npu_model

    # Compare
    print("[3/4] Comparing logits...")
    abs_diff = (cpu_logits - npu_logits).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_logits.flatten().unsqueeze(0), npu_logits.flatten().unsqueeze(0)
    ).item()

    signal_range = (cpu_logits.max() - cpu_logits.min()).item()
    rel_err = abs_diff.max().item() / max(signal_range, 1e-8)

    result = {
        "model": model_short,
        "cosine_similarity": cos_sim,
        "max_abs_error": abs_diff.max().item(),
        "mean_abs_error": abs_diff.mean().item(),
        "relative_error_vs_range": rel_err,
        "passed": cos_sim > 0.99 and rel_err < 0.01,
    }

    print("[4/4] Results:")
    print(f"  Cosine similarity:    {cos_sim:.8f}")
    print(f"  Max abs error:        {result['max_abs_error']:.6f}")
    print(f"  Mean abs error:       {result['mean_abs_error']:.6f}")
    print(f"  Rel error vs range:   {rel_err:.4%}")
    print(f"  Result: {'PASS' if result['passed'] else 'FAIL'} (threshold cos>0.99, rel_err<1%)")

    output = f"{model_short}_eval.json"
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved to {output}")


if __name__ == "__main__":
    main()
