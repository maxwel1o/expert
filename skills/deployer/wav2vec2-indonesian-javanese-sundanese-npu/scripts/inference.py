#!/usr/bin/env python3
"""
wav2vec2-indonesian-javanese-sundanese NPU Inference Script
Supports Ascend NPU (default), CPU, and CUDA backends.
"""

import argparse
import os
import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "1")


def get_device(backend="npu"):
    if backend == "npu":
        import torch_npu
        assert torch.npu.is_available(), "NPU not available"
        return torch.device("npu:0")
    elif backend == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def load_model_and_processor(model_path):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    return model, processor


def load_audio(audio_path, target_sr=16000):
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def transcribe(model, processor, audio_array, device, chunk_length_s=None):
    inputs = processor(
        audio_array, sampling_rate=16000, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        if chunk_length_s and chunk_length_s > 0:
            # Chunked inference for long audio
            sample_rate = 16000
            chunk_len = int(chunk_length_s * sample_rate)
            stride_len = int(chunk_length_s * sample_rate * 0.1)
            all_logits = []
            audio_len = input_values.shape[1]
            for start in range(0, audio_len, stride_len):
                end = min(start + chunk_len, audio_len)
                chunk = input_values[:, start:end]
                if chunk.shape[1] < 100:
                    continue
                chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
                output = model(chunk, attention_mask=chunk_mask)
                all_logits.append(output.logits.cpu().numpy())
            if not all_logits:
                output = model(input_values, attention_mask=attention_mask)
                logits = output.logits.cpu().numpy()
            else:
                logits = np.concatenate(all_logits, axis=1)
        else:
            output = model(input_values, attention_mask=attention_mask)
            logits = output.logits.cpu().numpy()

    if hasattr(processor, "decode"):
        result = processor.decode(logits[0])
        return result.text if hasattr(result, "text") else str(result)
    else:
        pred_ids = np.argmax(logits, axis=-1)
        return processor.batch_decode(pred_ids)[0]


def main():
    parser = argparse.ArgumentParser(
        description="wav2vec2-indonesian-javanese-sundanese NPU Inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/opt/atomgit/wav2vec2-indonesian-javanese-sundanese",
        help="Path to model directory",
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to input audio file (.wav, .mp3, .flac)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="npu",
        choices=["npu", "cpu", "cuda"],
        help="Inference backend (default: npu)",
    )
    parser.add_argument(
        "--chunk-length-s",
        type=float,
        default=None,
        help="Chunk length in seconds for long audio streaming inference",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed timing information"
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file not found: {args.audio}")
        sys.exit(1)

    # Load model
    device = get_device(args.backend)
    print(f"Device: {device}")
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model_and_processor(args.model_path)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: Wav2Vec2ForCTC ({param_count:,} params)")

    # Load and process audio
    print(f"Loading audio: {args.audio}")
    audio = load_audio(args.audio)
    print(f"Audio: {len(audio)/16000:.1f}s @ 16kHz")

    # Inference
    import time
    print("Running inference...")
    t0 = time.time()
    text = transcribe(model, processor, audio, device, args.chunk_length_s)
    elapsed = time.time() - t0

    print(f"\n=== Transcription Result ===")
    print(f"  Text     : {text}")
    if args.verbose:
        print(f"  Time     : {elapsed*1000:.1f} ms")
        print(f"  RTF      : {elapsed / (len(audio)/16000):.4f}")
        print(f"  Backend  : {args.backend}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
