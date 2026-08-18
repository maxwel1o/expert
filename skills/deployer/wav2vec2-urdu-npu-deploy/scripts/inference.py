#!/usr/bin/env python3
"""
wav2vec2-large-xls-r-300m-Urdu Inference Script for Ascend NPU
===============================================================
This script performs Urdu speech recognition using a fine-tuned
Wav2Vec2 model on Huawei Ascend NPU.

Usage:
  # Inference on single audio file
  python3 inference.py --audio /path/to/audio.wav

  # Inference on all audio files in a directory
  python3 inference.py --audio /path/to/dir/

  # Run with synthetic test audio
  python3 inference.py --test

Requirements:
  torch, torch_npu>=2.9.0, transformers>=4.16.0, soundfile, pyctcdecode
"""
import os
import sys
import glob
import time
import argparse
import warnings

import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch
import numpy as np
import soundfile as sf
from transformers import (
    AutoModelForCTC,
    AutoFeatureExtractor,
    AutoProcessor,
)

warnings.filterwarnings("ignore", message="Warning: The.*owner does not match")
warnings.filterwarnings("ignore", message="Permission mismatch")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda:0"  # transfer_to_npu maps to npu:0


def load_model():
    """Load Wav2Vec2 model and processor on NPU."""
    print(f"[INFO] Loading model from: {MODEL_DIR}")

    # Attempt to load processor with LM support if available
    try:
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        print(f"[INFO] Processor: {type(processor).__name__}")
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
        print(f"[INFO] Processor: {type(processor).__name__} (fallback, no LM)")

    model = AutoModelForCTC.from_pretrained(MODEL_DIR)
    model = model.to(DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: Wav2Vec2ForCTC ({n_params / 1e6:.1f}M params)")
    print(f"[INFO] Device: {next(model.parameters()).device}")

    return model, processor


def transcribe(model, processor, audio_array, sample_rate=16000):
    """
    Transcribe audio using the Wav2Vec2 model on NPU.

    Args:
        model: Wav2Vec2ForCTC model on NPU
        processor: Wav2Vec2Processor or FeatureExtractor
        audio_array: numpy array of audio samples
        sample_rate: audio sample rate (must be 16000)

    Returns:
        transcription text
    """
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_values = inputs.input_values.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    logits_np = logits.cpu().numpy()[0]

    try:
        text = processor.decode(logits_np)
        # Handle LM output wrapper
        if hasattr(text, "text"):
            text = text.text
    except AttributeError:
        pred_ids = np.argmax(logits_np, axis=-1)
        text = processor.decode(pred_ids)

    return text


def main():
    parser = argparse.ArgumentParser(
        description="wav2vec2-large-xls-r-300m-Urdu NPU Inference"
    )
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file or directory")
    parser.add_argument("--test", action="store_true",
                        help="Run with synthetic test audio")
    args = parser.parse_args()

    # Display NPU info
    print(f"[INFO] NPU: {torch.npu.get_device_name(0)}")
    print(f"[INFO] NPU Memory: "
          f"{torch.npu.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    model, processor = load_model()

    if args.test:
        duration = 3.0
        t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t)
            + 0.2 * np.sin(2 * np.pi * 400 * t)
            + 0.1 * np.random.randn(len(t))
        )
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio = audio.astype(np.float32)

        print(f"[INFO] Synthetic audio: {len(audio)} samples, "
              f"{len(audio) / 16000:.2f}s")
        text = transcribe(model, processor, audio)
        print(f"[RESULT] {text}")

    elif args.audio:
        path = args.audio
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            extensions = ("**/*.wav", "**/*.flac", "**/*.mp3")
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(path, ext), recursive=True))
            files = sorted(set(files))
        else:
            print(f"[ERROR] {path} not found")
            sys.exit(1)

        if not files:
            print(f"[ERROR] No audio files found in {path}")
            sys.exit(1)

        print(f"[INFO] Processing {len(files)} audio file(s)...")
        for fpath in files:
            try:
                audio, sr = sf.read(fpath)
                if sr != 16000:
                    print(f"[WARN] {os.path.basename(fpath)}: "
                          f"sample rate {sr}, resampling recommended")
                start = time.time()
                text = transcribe(model, processor, audio, sr)
                elapsed = time.time() - start
                print(f"[RESULT] {os.path.basename(fpath)}: {text} "
                      f"({elapsed * 1000:.1f}ms)")
            except Exception as e:
                print(f"[ERROR] {os.path.basename(fpath)}: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
