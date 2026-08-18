#!/usr/bin/env python3
"""Combined NPU inference for all 3 FunASR Paraformer models.

Usage:
    python3 paraformer_npu_infer.py --model 1 --audio test.wav
    python3 paraformer_npu_infer.py --model 2 --audio test.wav --warmup 3 --benchmark 10
    python3 paraformer_npu_infer.py --model 3 --audio test.wav --no-spk
"""
import argparse
import time
import numpy as np
import soundfile as sf
import torch
import torch_npu
from functools import wraps
from funasr.auto.auto_model import AutoModel

# Model configurations
MODEL1 = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL2 = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL3 = "iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn"

VAD = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC2 = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
PUNC3 = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
SPK = "iic/speech_campplus_sv_zh-cn_16k-common"


def load_audio(wav_path: str) -> np.ndarray:
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def build_npu_model(model_id: str, device: str = "npu:0",
                    vad_model=None, punc_model=None, spk_model=None):
    """Build models on CPU, move ASR to NPU, patch inference."""
    kwargs = {"model": model_id, "hub": "ms", "device": "cpu",
              "disable_update": True, "disable_pbar": True}
    if vad_model:
        kwargs["vad_model"] = vad_model
    if punc_model:
        kwargs["punc_model"] = punc_model
    if spk_model:
        kwargs["spk_model"] = spk_model

    model = AutoModel(**kwargs)

    # Move ASR to NPU; sub-models stay on CPU
    model.model = model.model.to(device)
    model.model.eval()

    # Patch inference to inject NPU device (bypasses _reset_runtime_configs)
    orig_infer = model.model.inference
    @wraps(orig_infer)
    def patched(*args, **kw):
        kw["device"] = device
        kw["ngpu"] = 1
        return orig_infer(*args, **kw)
    model.model.inference = patched
    return model


def get_model_config(model_num: int):
    configs = {
        1: {"id": MODEL1, "vad": None, "punc": None, "spk": None,
            "name": "Paraformer ASR"},
        2: {"id": MODEL2, "vad": VAD, "punc": PUNC2, "spk": None,
            "name": "Paraformer ASR+VAD+Punc"},
        3: {"id": MODEL3, "vad": VAD, "punc": PUNC3, "spk": SPK,
            "name": "Paraformer ASR+VAD+Punc+SPK"},
    }
    return configs[model_num]


def main():
    parser = argparse.ArgumentParser(description="Paraformer NPU Inference")
    parser.add_argument("--model", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model-dir", default=None, help="Local model directory")
    parser.add_argument("--no-spk", action="store_true", help="Disable speaker")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--benchmark", type=int, default=0, help="Benchmark runs")
    args = parser.parse_args()

    cfg = get_model_config(args.model)
    spk_model = None if args.no_spk else cfg["spk"]
    model_id = args.model_dir or cfg["id"]

    print(f"Building {cfg['name']} on NPU...")
    model = build_npu_model(model_id, vad_model=cfg["vad"],
                            punc_model=cfg["punc"], spk_model=spk_model)

    audio = load_audio(args.audio)
    duration = len(audio) / 16000.0

    # Warmup
    for _ in range(args.warmup):
        model.generate(input=audio)

    # Transcription
    torch.npu.synchronize()
    t0 = time.time()
    result = model.generate(input=audio)
    torch.npu.synchronize()
    elapsed = time.time() - t0
    text = result[0].get("text", "") if result else ""
    spk_info = result[0].get("sentence_info", None) if result else None

    print(f"\n=== Transcription ===")
    print(f"Text: {text}")
    if spk_info:
        print(f"Speaker segments: {len(spk_info)}")
        for seg in spk_info[:3]:
            print(f"  [{seg.get('spk','?')}] {seg.get('text','')[:60]}")
    print(f"Audio: {duration:.2f}s | Inference: {elapsed:.3f}s | RTF: {elapsed/duration:.3f}")

    # Benchmark
    if args.benchmark > 0:
        times = []
        for i in range(args.benchmark):
            torch.npu.synchronize()
            t0 = time.time()
            model.generate(input=audio)
            torch.npu.synchronize()
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        print(f"\nBenchmark ({args.benchmark} runs): avg {avg:.3f}s | RTF {avg/duration:.3f}")


if __name__ == "__main__":
    main()
