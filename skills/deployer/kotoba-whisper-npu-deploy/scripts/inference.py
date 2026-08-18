#!/usr/bin/env python3
"""NPU inference script for kotoba-whisper speech recognition models.

Usage:
    python3 inference.py --model kotoba-tech/kotoba-whisper-v1.0 --audio input.wav
"""

import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch_npu
from transformers import WhisperForConditionalGeneration, WhisperProcessor

warnings.filterwarnings("ignore")


def load_model(model_id, device="npu"):
    """Load Whisper model and processor."""
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def transcribe(audio_path, model, processor, device="npu", language="ja", task="transcribe"):
    """Transcribe an audio file using the NPU model."""
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000

    audio = audio.astype(np.float32)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    with torch.no_grad():
        start = time.perf_counter()
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=256,
        )
        elapsed = time.perf_counter() - start

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    audio_duration = len(audio) / sr
    rtf = elapsed / audio_duration

    return transcription, elapsed, audio_duration, rtf


def main():
    parser = argparse.ArgumentParser(description="kotoba-whisper NPU Inference")
    parser.add_argument("--model", default="kotoba-tech/kotoba-whisper-v1.0",
                        help="HuggingFace model ID")
    parser.add_argument("--audio", required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--language", default="ja", help="Language code (default: ja)")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--device", default="npu", choices=["npu", "cpu"])
    args = parser.parse_args()

    device_name = torch.npu.get_device_name(0) if args.device == "npu" else "CPU"

    print(f"Model: {args.model}")
    print(f"Device: {device_name}")
    print(f"Loading model...")

    model, processor = load_model(args.model, args.device)

    print(f"Transcribing: {args.audio}")
    transcription, elapsed, audio_dur, rtf = transcribe(
        args.audio, model, processor, args.device, args.language, args.task
    )

    print(f"\n{'='*60}")
    print(f"Transcription: {transcription}")
    print(f"{'='*60}")
    print(f"Audio duration:  {audio_dur:.1f}s")
    print(f"Process time:    {elapsed:.2f}s")
    print(f"RTF:             {rtf:.4f} ({'%.1f' % (audio_dur/elapsed)}x real-time)")
    print(f"Device:          {device_name}")


if __name__ == "__main__":
    main()
