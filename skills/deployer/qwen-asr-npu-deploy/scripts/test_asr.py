#!/usr/bin/env python3
"""
Qwen3-ASR-0.6B NPU Adaptation Test Script

Tests both NPU (vLLM backend) and CPU (transformers backend) inference
and reports precision gap.
"""

import argparse
import os
import sys
import time

import torch


MODEL_PATH = "/opt/atomgit/QwenASR/models"


def test_npu(audio_path: str, language: str | None):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import npu_patch  # noqa: F401
    from qwen_asr import Qwen3ASRModel

    print("[NPU] Initializing vLLM backend...")
    t0 = time.time()
    asr = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=65536,
        max_num_seqs=4,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    init_t = time.time() - t0

    t0 = time.time()
    results = asr.transcribe(audio=audio_path, language=language, return_time_stamps=False)
    infer_t = time.time() - t0

    text = results[0].text
    lang = results[0].language
    print(f"[NPU] Init={init_t:.2f}s Inference={infer_t:.2f}s")
    print(f"[NPU] Language={lang!r} Text={text!r}")
    return text, infer_t


def test_cpu(audio_path: str, language: str | None):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
    from qwen_asr import Qwen3ASRModel

    print("[CPU] Initializing transformers backend...")
    t0 = time.time()
    asr = Qwen3ASRModel.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    init_t = time.time() - t0

    t0 = time.time()
    results = asr.transcribe(audio=audio_path, language=language, return_time_stamps=False)
    infer_t = time.time() - t0

    text = results[0].text
    lang = results[0].language
    print(f"[CPU] Init={init_t:.2f}s Inference={infer_t:.2f}s")
    print(f"[CPU] Language={lang!r} Text={text!r}")
    return text, infer_t


def compare(text_cpu: str, text_npu: str):
    if text_cpu == text_npu:
        print("\n[Comparison] CPU and NPU outputs are IDENTICAL")
        return 0.0
    else:
        print("\n[Comparison] CPU and NPU outputs DIFFER")
        print(f"  CPU:  {text_cpu}")
        print(f"  NPU:  {text_npu}")
        # Simple character-level diff ratio
        import difflib
        ratio = difflib.SequenceMatcher(None, text_cpu, text_npu).ratio()
        print(f"  Similarity: {ratio*100:.2f}%")
        return 1.0 - ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="/opt/atomgit/QwenASR/test_data/asr_zh.wav")
    parser.add_argument("--language", default=None)
    parser.add_argument("--backend", choices=["npu", "cpu", "both"], default="both")
    args = parser.parse_args()

    text_npu = None
    time_npu = None
    text_cpu = None
    time_cpu = None

    if args.backend in ("npu", "both"):
        text_npu, time_npu = test_npu(args.audio, args.language)

    if args.backend in ("cpu", "both"):
        text_cpu, time_cpu = test_cpu(args.audio, args.language)

    if args.backend == "both" and text_cpu and text_npu:
        gap = compare(text_cpu, text_npu)
        print(f"\n[Summary]")
        print(f"  CPU time: {time_cpu:.2f}s")
        print(f"  NPU time: {time_npu:.2f}s")
        print(f"  Speedup:  {time_cpu/time_npu:.2f}x")
        print(f"  Precision gap: {gap*100:.2f}%")


if __name__ == "__main__":
    main()
