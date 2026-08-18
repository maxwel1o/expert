#!/usr/bin/env python3
"""
whisper-large-v3 NPU Model Wrapper
Provides a simple interface for loading and running whisper-large-v3 on Ascend NPU.

Usage:
    from model_files import WhisperNPU
    model = WhisperNPU("/path/to/whisper-large-v3", device="npu")
    result = model.transcribe("audio.wav")
    print(result)
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch_npu  # noqa: F401


class WhisperNPU:
    """Whisper-large-v3 inference wrapper for Ascend NPU."""

    DEFAULT_MODEL = "/opt/atomgit/whisper-large-v3"

    def __init__(
        self,
        model_path: str = None,
        device: str = "npu",
        dtype: str = "float16",
        attn_implementation: str = "eager",
    ):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.model_path = model_path or self.DEFAULT_MODEL
        self.dtype = torch.float16 if dtype == "float16" else torch.float32

        if device == "npu" and not torch.npu.is_available():
            raise RuntimeError("Ascend NPU not available")

        print(f"[WhisperNPU] Loading model from: {self.model_path}")
        start = time.time()

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        if device == "npu":
            self.model = self.model.to("npu")
        self.model.eval()

        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        print(f"[WhisperNPU] Model loaded in {time.time() - start:.1f}s")

    @staticmethod
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

    def transcribe(
        self,
        audio,
        sampling_rate: int = 16000,
        language: str = "en",
        max_tokens: int = 256,
    ) -> dict:
        """Transcribe audio to text. Returns dict with text, tokens, time."""
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                max_new_tokens=max_tokens,
                task="transcribe",
                language=language,
            )
        elapsed = time.time() - start

        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return {
            "text": text,
            "tokens": outputs[0].cpu().tolist(),
            "time": elapsed,
            "speed_ms_per_token": 1000 * elapsed / len(outputs[0]),
        }

    def translate(
        self,
        audio,
        sampling_rate: int = 16000,
        language: str = "en",
        max_tokens: int = 256,
    ) -> dict:
        """Translate audio to English text."""
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                max_new_tokens=max_tokens,
                task="translate",
                language=language,
            )
        elapsed = time.time() - start

        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return {
            "text": text,
            "tokens": outputs[0].cpu().tolist(),
            "time": elapsed,
            "speed_ms_per_token": 1000 * elapsed / len(outputs[0]),
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_files/npu_wrapper.py <audio_file>")
        sys.exit(1)

    model = WhisperNPU()
    result = model.transcribe(sys.argv[1])
    print(f"Text: {result['text']}")
    print(f"Time: {result['time']:.2f}s, Speed: {result['speed_ms_per_token']:.1f}ms/token")
