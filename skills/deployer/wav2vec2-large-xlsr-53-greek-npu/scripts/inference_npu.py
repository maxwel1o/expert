"""
NPU inference script for wav2vec2-large-xlsr-53-greek.
Uses torch_npu.contrib.transfer_to_npu for automatic CUDA->NPU migration.
"""
import os
import sys
import json
import numpy as np

# ============================================================
# 0. Auto-migration: must be imported BEFORE torch/transformers
# ============================================================
import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

MODEL_NAME = "./modelscope_cache/jonatasgrosman/wav2vec2-large-xlsr-53-greek"
SEED = 42

def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

def generate_test_audio(sr=16000, duration=3.0):
    """Generate deterministic synthetic audio for reproducibility."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    rng = np.random.default_rng(SEED)
    audio = audio + 0.01 * rng.standard_normal(audio.shape)
    return audio.astype(np.float32), sr

def main():
    set_seed()
    device = torch.device("npu:0")
    print(f"[NPU] torch_npu version: {torch_npu.__version__}")
    print(f"[NPU] transfer_to_npu loaded: {transfer_to_npu}")
    print(f"[NPU] NPU available: {torch.npu.is_available()}")
    print(f"[NPU] Device: {device}")

    print(f"[NPU] Loading model: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    print("[NPU] Preparing test audio...")
    audio, sr = generate_test_audio()
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    print(f"[NPU] input_values shape: {input_values.shape}, dtype: {input_values.dtype}, device: {input_values.device}")

    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits

    print(f"[NPU] logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}")

    # Save outputs
    npu_out = {
        "logits": logits.cpu().numpy().tolist(),
        "shape": list(logits.shape),
        "device": str(logits.device),
    }
    out_path = "/opt/atomgit/wav2vec2-npu-adapt/output_npu.json"
    with open(out_path, "w") as f:
        json.dump(npu_out, f)
    print(f"[NPU] Output saved to {out_path}")

    np.save("/opt/atomgit/wav2vec2-npu-adapt/logits_npu.npy", logits.cpu().numpy())
    print("[NPU] logits_npu.npy saved")

if __name__ == "__main__":
    main()
