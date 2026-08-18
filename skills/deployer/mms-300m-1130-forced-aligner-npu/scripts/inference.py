"""
mms-300m-1130-forced-aligner — Ascend NPU 推理脚本
===================================================
支持 3 种运行模式:
  1. basic     — 单一音频文件推理
  2. batch     — 批量音频文件推理
  3. benchmark — 精度 + 性能基准测试

用法:
  # 单文件推理
  python3 inference.py basic --audio test.wav

  # 批量推理
  python3 inference.py batch --audio-dir ./audios/

  # 基准测试
  python3 inference.py benchmark --audio test.wav
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

warnings.filterwarnings("ignore")

MODEL_PATH = os.environ.get(
    "MMS_MODEL_PATH",
    os.path.dirname(os.path.abspath(__file__)),
)
SAMPLE_RATE = 16000


def check_device():
    """自动选择设备: NPU > CUDA > CPU"""
    if torch.npu.is_available():
        device = "npu"
        device_name = torch.npu.get_device_name(0)
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"
    return device, device_name


def load_audio(path, target_sr=16000):
    """加载音频，支持 .wav / .flac / .mp3"""
    audio, sr = sf.read(path)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            # 简单线性重采样
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * target_sr / sr)),
                np.arange(len(audio)),
                audio,
            )
        sr = target_sr
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def load_model(device, model_path=None):
    """加载 Wav2Vec2 模型和 processor"""
    model_path = model_path or MODEL_PATH
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, processor


def run_inference(model, processor, audio, device):
    """执行模型推理，返回 logits 和解码文本"""
    inputs = processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return logits.cpu(), transcription


def cmd_basic(args):
    """单文件推理"""
    device, device_name = check_device()
    print(f"[INFO] Device: {device_name}")
    print(f"[INFO] Loading model from {MODEL_PATH}...")

    model, processor = load_model(device)
    audio = load_audio(args.audio)
    print(f"[INFO] Audio: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.2f}s")

    t0 = time.time()
    logits, transcription = run_inference(model, processor, audio, device)
    elapsed = time.time() - t0

    print(f"\n--- Results ---")
    print(f"  Transcription: '{transcription}'")
    print(f"  Logits shape:  {list(logits.shape)}")
    print(f"  Inference time: {elapsed:.3f}s")
    print(f"  Real-time factor: {elapsed/(len(audio)/SAMPLE_RATE):.3f}x")

    if args.output:
        result = {
            "audio": args.audio,
            "duration_s": len(audio) / SAMPLE_RATE,
            "transcription": transcription,
            "inference_time_s": round(elapsed, 3),
            "rtf": round(elapsed / (len(audio) / SAMPLE_RATE), 4),
            "device": device_name,
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Results saved to {args.output}")


def cmd_batch(args):
    """批量推理"""
    device, device_name = check_device()
    print(f"[INFO] Device: {device_name}")
    print(f"[INFO] Loading model...")

    model, processor = load_model(device)
    audio_dir = Path(args.audio_dir)
    extensions = ("*.wav", "*.flac", "*.mp3", "*.m4a")
    files = []
    for ext in extensions:
        files.extend(audio_dir.glob(ext))
    files = sorted(files)

    if not files:
        print(f"[ERROR] No audio files found in {audio_dir}")
        return

    print(f"[INFO] Found {len(files)} audio files")
    results = []

    for fpath in files:
        audio = load_audio(str(fpath))
        t0 = time.time()
        _, transcription = run_inference(model, processor, audio, device)
        elapsed = time.time() - t0
        results.append({
            "file": fpath.name,
            "duration_s": round(len(audio) / SAMPLE_RATE, 3),
            "transcription": transcription,
            "inference_time_s": round(elapsed, 3),
        })
        print(f"  [{len(results)}/{len(files)}] {fpath.name}: '{transcription[:40]}' ({elapsed:.2f}s)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] Batch results saved to {args.output}")


def cmd_benchmark(args):
    """精度 + 性能基准测试 (CPU vs NPU)"""
    print("=" * 60)
    print("  mms-300m-1130-forced-aligner — Benchmark Suite")
    print("=" * 60)

    # 加载音频
    audio = load_audio(args.audio)
    audio_len_s = len(audio) / SAMPLE_RATE
    print(f"\n[DATA] Audio: {len(audio)} samples, {audio_len_s:.2f}s")

    results = {
        "model": "mms-300m-1130-forced-aligner",
        "architecture": "Wav2Vec2ForCTC",
        "audio": args.audio,
        "audio_duration_s": audio_len_s,
        "devices": {},
    }

    # 测试目标设备
    devices_to_test = []

    if torch.npu.is_available():
        devices_to_test.append(("npu", "Ascend NPU"))
    if torch.cuda.is_available():
        devices_to_test.append(("cuda", "CUDA GPU"))
    devices_to_test.append(("cpu", "CPU"))

    baseline_logits = None
    baseline_trans = None
    baseline_name = ""

    for dev_id, dev_name in devices_to_test:
        print(f"\n{'=' * 50}")
        print(f"[BENCH] Device: {dev_name}")
        print(f"{'=' * 50}")

        model, processor = load_model(dev_id)
        model.eval()

        # Warmup (2次)
        warmup_audio = audio[:16000]  # 1s warmup
        warmup_inputs = processor(
            warmup_audio, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True,
        )
        warmup_inputs = {k: v.to(dev_id) for k, v in warmup_inputs.items()}
        with torch.no_grad():
            for _ in range(2):
                model(warmup_inputs["input_values"],
                      attention_mask=warmup_inputs.get("attention_mask"))

        # 性能测试 - 运行 5 次取中位数
        times = []
        for i in range(5):
            t0 = time.time()
            logits, trans = run_inference(model, processor, audio, dev_id)
            elapsed = time.time() - t0
            times.append(elapsed)

        times.sort()
        median_time = times[len(times) // 2]
        rtf = median_time / audio_len_s

        print(f"  Run times: {[f'{t:.3f}s' for t in times]}")
        print(f"  Median:    {median_time:.3f}s")
        print(f"  RTF:       {rtf:.3f}x")
        print(f"  Trans:     '{trans}'")

        # 精度对比
        if baseline_logits is not None:
            abs_diff = torch.abs(baseline_logits - logits)
            mse = torch.mean(abs_diff ** 2).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                baseline_logits.flatten().unsqueeze(0),
                logits.flatten().unsqueeze(0),
            ).item()
            trans_match = baseline_trans == trans

            print(f"\n  --- Accuracy vs {baseline_name} ---")
            print(f"  Cosine similarity: {cos_sim:.8f}")
            print(f"  MSE:               {mse:.8e}")
            print(f"  Max abs error:     {abs_diff.max().item():.6e}")
            print(f"  Trans match:       {trans_match}")
        else:
            cos_sim = 1.0
            mse = 0.0
            trans_match = True
            baseline_name = dev_name

        results["devices"][dev_id] = {
            "device_name": dev_name,
            "median_inference_time_s": round(median_time, 4),
            "rtf": round(rtf, 4),
            "transcription": trans,
            "cosine_similarity_vs_baseline": round(cos_sim, 8),
            "mse_vs_baseline": round(mse, 10),
            "transcription_match_vs_baseline": trans_match,
        }

        if baseline_logits is None:
            baseline_logits = logits
            baseline_trans = trans

    # 最终报告
    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    for dev_id, data in results["devices"].items():
        print(f"  {data['device_name']:15s}: {data['median_inference_time_s']:.3f}s "
              f"(RTF={data['rtf']:.3f}x)  "
              f"cos_sim={data['cosine_similarity_vs_baseline']:.6f}")
    print(f"{'=' * 60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] Benchmark results saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="mms-300m-1130-forced-aligner NPU 推理",
    )
    parser.add_argument("--model-path", default=None, help="模型路径")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # basic 子命令
    p_basic = subparsers.add_parser("basic", help="单音频推理")
    p_basic.add_argument("--audio", required=True, help="音频文件路径")
    p_basic.add_argument("--output", "-o", default=None, help="输出 JSON 路径")

    # batch 子命令
    p_batch = subparsers.add_parser("batch", help="批量推理")
    p_batch.add_argument("--audio-dir", required=True, help="音频目录")
    p_batch.add_argument("--output", "-o", default=None, help="输出 JSON 路径")

    # benchmark 子命令
    p_bench = subparsers.add_parser("benchmark", help="精度+性能基准测试")
    p_bench.add_argument("--audio", required=True, help="音频文件路径")
    p_bench.add_argument("--output", "-o", default=None, help="输出 JSON 路径")

    args = parser.parse_args()
    global MODEL_PATH
    if args.model_path:
        MODEL_PATH = args.model_path

    if args.cmd == "basic":
        cmd_basic(args)
    elif args.cmd == "batch":
        cmd_batch(args)
    elif args.cmd == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
