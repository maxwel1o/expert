#!/usr/bin/env python3
"""
Qwen3-ASR-0.6B NPU Inference Script (vLLM backend on Ascend NPU)

Usage:
    export ASCEND_RT_VISIBLE_DEVICES=0
    python inference.py --audio /path/to/audio.wav --language Chinese
"""

import argparse
import os
import sys

# Apply vllm-ascend compatibility patches before importing qwen_asr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npu_patch  # noqa: F401

from qwen_asr import Qwen3ASRModel


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR-0.6B NPU Inference")
    parser.add_argument("--model", default="/opt/atomgit/QwenASR/models", help="Model path")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav)")
    parser.add_argument("--language", default=None, help="Force language (e.g. Chinese, English)")
    parser.add_argument("--dtype", default="bfloat16", help="Data type")
    parser.add_argument("--max-model-len", type=int, default=65536, help="Max model length")
    parser.add_argument("--max-num-seqs", type=int, default=4, help="Max num sequences")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory util")
    args = parser.parse_args()

    print("Loading Qwen3-ASR-0.6B on NPU...")
    asr = Qwen3ASRModel.LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("Model loaded. Running transcription...")

    results = asr.transcribe(
        audio=args.audio,
        language=args.language,
        return_time_stamps=False,
    )

    for i, r in enumerate(results):
        print(f"\n[Result {i}]")
        print(f"  Language: {r.language}")
        print(f"  Text: {r.text}")


if __name__ == "__main__":
    main()
