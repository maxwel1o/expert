#!/usr/bin/env python3
"""whisper-medium Ascend NPU 推理脚本"""
import os
import argparse
import torch
import torch_npu  # noqa: F401
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "./whisper-medium")


def main():
    parser = argparse.ArgumentParser(description="whisper-medium NPU inference")
    parser.add_argument("audio_path", nargs="?", default="test_audio.wav",
                        help="Path to audio file (16kHz WAV recommended)")
    parser.add_argument("--language", default="english", help="Language (default: english)")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                        help="Task: transcribe or translate")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                        help="Precision")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = torch.device("npu:0")
    print(f"Device: {device} ({torch.npu.get_device_name(0)})")

    # Load processor and model
    print("Loading processor & model...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"Model: whisper-medium ({args.dtype})")

    # Load and preprocess audio
    audio, sr = sf.read(args.audio_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        generated = model.generate(
            input_features,
            language=args.language,
            task=args.task,
            return_timestamps=False,
            max_length=448,
        )

    torch.npu.synchronize()
    transcription = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"\nTranscription: {transcription}")


if __name__ == "__main__":
    main()
