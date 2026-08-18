"""
mms-300m-1130-forced-aligner —   基准测试
==============================================
测：精度 + 性能 + 不同音频长度下的 RTF
输出：标准 JSON 结果 + 日志
"""

import os, sys, time, json, warnings
import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (
    load_model, load_audio, run_inference, check_device,
    MODEL_PATH, SAMPLE_RATE,
)

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


def generate_speech_like_audio(duration_s, sr=16000):
    """生成多段语音模拟音频"""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = np.zeros_like(t)

    # 多个共振峰音段模拟不同音素
    formants = [
        (300, 870, 2240, 0.8),   # /a/ like
        (400, 1200, 2600, 0.6),  # /i/ like
        (200, 600, 2100, 0.7),   # /u/ like
        (500, 1500, 2800, 0.5),  # /e/ like
        (350, 1000, 2400, 0.6),  # schwa
    ]

    seg_duration = duration_s / len(formants)
    for i, (f1, f2, f3, amp) in enumerate(formants):
        start = int(i * seg_duration * sr)
        end = int((i + 1) * seg_duration * sr)
        seg_t = t[start:end] - t[start]
        env = np.exp(-seg_t * 1.5)
        seg_audio = (
            amp * np.sin(2 * np.pi * f1 * seg_t) * env +
            amp * 0.6 * np.sin(2 * np.pi * f2 * seg_t) * env**0.7 +
            amp * 0.3 * np.sin(2 * np.pi * f3 * seg_t) * env**0.5
        )
        # 起落包络
        rise = np.minimum(1.0, seg_t / 0.02)
        fall = np.minimum(1.0, (seg_t[-1] - seg_t) / 0.05)
        seg_audio *= rise * fall
        audio[start:end] += seg_audio

    # 添加辅音噪声段
    for i in range(1, len(formants)):
        noise_start = int((i * seg_duration - 0.05) * sr)
        noise_end = int((i * seg_duration + 0.05) * sr)
        if noise_end < len(audio):
            audio[noise_start:noise_end] += np.random.randn(
                noise_end - noise_start
            ) * 0.05

    audio = audio / np.max(np.abs(audio) + 1e-6)
    return audio.astype(np.float32)


def main():
    results = {
        "model": "mms-300m-1130-forced-aligner",
        "architecture": "Wav2Vec2ForCTC",
        "framework": "transformers + torch_npu",
        "device_info": {},
        "precision": {},
        "performance": {},
        "multi_length": [],
    }

    # 设备信息
    device, device_name = check_device()
    results["device_info"] = {
        "device": device,
        "device_name": device_name,
    }
    print(f"[INFO] Device: {device_name}")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] Model path: {MODEL_PATH}")

    # =========================================================
    # Part 1: 精度测试 (CPU vs NPU)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("[PART 1] Precision Test (CPU vs NPU)")
    print(f"{'=' * 60}")

    audio_3s = generate_speech_like_audio(3.0)
    sf.write("/tmp/eval_audio_3s.wav", audio_3s, SAMPLE_RATE)

    cpu_logits, cpu_trans = None, None
    npu_logits, npu_trans = None, None

    # CPU baseline
    print("\n  [BASELINE] Running on CPU...")
    model_cpu, processor = load_model("cpu")
    cpu_logits, cpu_trans = run_inference(model_cpu, processor, audio_3s, "cpu")
    del model_cpu
    print(f"    Transcription: '{cpu_trans}'")
    print(f"    Logits shape: {list(cpu_logits.shape)}")

    # NPU target
    print("\n  [TARGET] Running on NPU...")
    model_npu, _ = load_model("npu")
    npu_logits, npu_trans = run_inference(model_npu, processor, audio_3s, "npu")
    del model_npu
    if torch.npu.is_available():
        torch.npu.empty_cache()
    print(f"    Transcription: '{npu_trans}'")
    print(f"    Logits shape: {list(npu_logits.shape)}")

    # 计算精度指标
    print("\n  [METRICS] Computing accuracy...")
    abs_diff = torch.abs(cpu_logits - npu_logits)
    mse = torch.mean(abs_diff ** 2).item()
    rmse = np.sqrt(mse).item()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_logits.flatten().unsqueeze(0),
        npu_logits.flatten().unsqueeze(0),
    ).item()
    trans_match = cpu_trans == npu_trans

    # 对大值的相对误差
    significant = torch.abs(cpu_logits) > 0.01
    if significant.any():
        rel_err = abs_diff[significant] / torch.abs(cpu_logits[significant])
        mean_rel_err = rel_err.mean().item()
        max_rel_err = rel_err.max().item()
        within_1pct = (rel_err < 0.01).float().mean().item()
    else:
        mean_rel_err = 0.0
        max_rel_err = 0.0
        within_1pct = 1.0

    results["precision"] = {
        "audio_duration_s": 3.0,
        "cpu_transcription": cpu_trans,
        "npu_transcription": npu_trans,
        "transcription_match": trans_match,
        "cosine_similarity": round(cos_sim, 8),
        "mse": round(mse, 10),
        "rmse": round(rmse, 8),
        "max_abs_error": round(max_abs_err, 8),
        "mean_abs_error": round(mean_abs_err, 8),
        "mean_rel_error_significant": round(mean_rel_err, 6),
        "max_rel_error_significant": round(max_rel_err, 6),
        "within_1pct_significant": round(within_1pct, 4),
    }

    print(f"    Cosine similarity:           {cos_sim:.8f}")
    print(f"    MSE:                         {mse:.8e}")
    print(f"    RMSE:                        {rmse:.8e}")
    print(f"    Max abs error:               {max_abs_err:.6e}")
    print(f"    Mean rel error (significant):{mean_rel_err:.6%}")
    print(f"    Transcription match:         {trans_match}")

    # =========================================================
    # Part 2: 性能测试 (多种音频长度)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("[PART 2] Performance Benchmark")
    print(f"{'=' * 60}")

    durations = [1.0, 3.0, 5.0, 10.0, 30.0]

    for dur in durations:
        print(f"\n  [PERF] Audio length: {dur}s")
        audio = generate_speech_like_audio(dur)
        model, _ = load_model(device)

        # Warmup
        w_audio = audio[:min(16000, len(audio))]
        w_inputs = processor(w_audio, sampling_rate=SAMPLE_RATE,
                             return_tensors="pt", padding=True)
        w_inputs = {k: v.to(device) for k, v in w_inputs.items()}
        with torch.no_grad():
            for _ in range(2):
                model(w_inputs["input_values"],
                      attention_mask=w_inputs.get("attention_mask"))

        # 3 轮计时
        times = []
        for i in range(3):
            t0 = time.time()
            _, trans = run_inference(model, processor, audio, device)
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"      Run {i+1}: {elapsed:.3f}s  trans='{trans[:30]}'")

        median_t = sorted(times)[len(times) // 2]
        rtf = median_t / dur

        entry = {
            "audio_duration_s": dur,
            "inference_times_s": [round(t, 4) for t in times],
            "median_time_s": round(median_t, 4),
            "rtf": round(rtf, 4),
        }
        results["multi_length"].append(entry)
        print(f"    => Median: {median_t:.3f}s, RTF: {rtf:.3f}x")

        del model
        if device == "npu":
            torch.npu.empty_cache()

    # Part 3: CPU 对比性能
    if device == "npu":
        print(f"\n  [PERF] CPU baseline (3.0s audio)...")
        audio_3s = generate_speech_like_audio(3.0)
        model_cpu, _ = load_model("cpu")
        cpu_times = []
        for i in range(3):
            t0 = time.time()
            _, _ = run_inference(model_cpu, processor, audio_3s, "cpu")
            elapsed = time.time() - t0
            cpu_times.append(elapsed)
        cpu_median = sorted(cpu_times)[len(cpu_times) // 2]
        print(f"    CPU median: {cpu_median:.3f}s (RTF: {cpu_median/3.0:.3f}x)")
        results["performance"]["cpu_baseline_3s"] = {
            "median_time_s": round(cpu_median, 4),
            "rtf": round(cpu_median / 3.0, 4),
        }
        del model_cpu

    # =========================================================
    # 汇总报告
    # =========================================================
    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    p = results["precision"]
    status = "PASS" if (p["cosine_similarity"] > 0.999 and p["mse"] < 5e-4 and p["transcription_match"]) else "FAIL"
    print(f"  Model:            mms-300m-1130-forced-aligner")
    print(f"  Device:           {device_name}")
    print(f"  Precision:        CosSim={p['cosine_similarity']:.6f}, MSE={p['mse']:.2e}")
    print(f"  Trans Match:      {p['transcription_match']}")
    print(f"  Status:           {status}")
    print(f"{'=' * 60}")

    # 保存结果
    results["status"] = status
    output_path = "/opt/atomgit/mms-300m-1130-forced-aligner/evaluation/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Results saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
