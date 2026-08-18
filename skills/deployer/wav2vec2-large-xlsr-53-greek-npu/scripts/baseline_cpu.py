"""
Baseline inference script for wav2vec2-large-xlsr-53-greek on CPU.
Generates reference logits and hidden states for NPU accuracy comparison.
"""
import os
import sys
import json
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

MODEL_NAME = "./modelscope_cache/jonatasgrosman/wav2vec2-large-xlsr-53-greek"
SEED = 42

def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def generate_test_audio(sr=16000, duration=3.0):
    """Generate deterministic synthetic audio for reproducibility."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sum of sine waves at 440Hz and 880Hz
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    # Add small deterministic noise
    rng = np.random.default_rng(SEED)
    audio = audio + 0.01 * rng.standard_normal(audio.shape)
    return audio.astype(np.float32), sr

def main():
    set_seed()
    device = torch.device("cpu")
    print(f"[Baseline] Loading model: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    print("[Baseline] Preparing test audio...")
    audio, sr = generate_test_audio()
    # Resample if needed (librosa default is 22050, we want 16k)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    print(f"[Baseline] input_values shape: {input_values.shape}, dtype: {input_values.dtype}")

    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits

    print(f"[Baseline] logits shape: {logits.shape}, dtype: {logits.dtype}")

    # Save reference outputs
    ref = {
        "logits": logits.cpu().numpy().tolist(),
        "shape": list(logits.shape),
        "device": str(device),
    }
    out_path = "/opt/atomgit/wav2vec2-npu-adapt/ref_output_cpu.json"
    with open(out_path, "w") as f:
        json.dump(ref, f)
    print(f"[Baseline] Reference output saved to {out_path}")

    # Also save raw numpy for precise comparison
    np.save("/opt/atomgit/wav2vec2-npu-adapt/ref_logits_cpu.npy", logits.cpu().numpy())
    print("[Baseline] ref_logits_cpu.npy saved")

if __name__ == "__main__":
    main()
