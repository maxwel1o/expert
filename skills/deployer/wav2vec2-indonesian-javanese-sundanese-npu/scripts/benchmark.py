#!/usr/bin/env python3
"""
Comprehensive NPU accuracy & performance evaluation for
wav2vec2-indonesian-javanese-sundanese on Ascend 910B4.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import torch
import torch_npu

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "1")

MODEL_PATH = "/opt/atomgit/wav2vec2-indonesian-javanese-sundanese"
RESULTS = {}
LOG_FILE = "eval/benchmark.log"
JSON_FILE = "eval/results.json"


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def create_test_audio(duration=3.0, sample_rate=16000):
    """Generate synthetic multi-tone speech-like audio."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 500 * t)
        + 0.15 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1200 * t)
        + 0.05 * np.random.randn(len(t))
    )
    return (audio / np.max(np.abs(audio))).astype(np.float32)


def env_check():
    log("=" * 60)
    log("1. ENVIRONMENT CHECK")
    log("=" * 60)
    info = {
        "pytorch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "npu_available": torch.npu.is_available(),
        "npu_count": torch.npu.device_count(),
        "npu_name": torch.npu.get_device_name(0) if torch.npu.is_available() else "N/A",
        "cann": os.environ.get("ASCEND_TOOLKIT_HOME", "N/A"),
        "soc_version": os.environ.get("SOC_VERSION", "N/A"),
        "transformers": __import__("transformers").__version__,
    }
    for k, v in info.items():
        log(f"  {k:20s}: {v}")
    RESULTS["environment"] = info
    return True


def model_load_test():
    log("\n" + "=" * 60)
    log("2. MODEL LOADING TEST")
    log("=" * 60)

    from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

    t0 = time.time()
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_PATH)
    t1 = time.time()
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    model.eval()
    t2 = time.time()

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Processor load time : {(t1-t0)*1000:.0f} ms")
    log(f"  Model load time     : {(t2-t1)*1000:.0f} ms")
    log(f"  Total params        : {n_params:,}")
    log(f"  Architecture        : {model.config.architectures[0]}")
    log(f"  Hidden size         : {model.config.hidden_size}")
    log(f"  Num layers          : {model.config.num_hidden_layers}")
    log(f"  Vocab size          : {model.config.vocab_size}")

    RESULTS["model"] = {
        "params": n_params,
        "architecture": model.config.architectures[0],
        "load_time_ms": (t2 - t1) * 1000,
    }
    return model, processor


def accuracy_test(model, processor):
    log("\n" + "=" * 60)
    log("3. ACCURACY TEST (NPU vs CPU)")
    log("=" * 60)

    durations = [1.0, 3.0, 5.0, 10.0]
    all_results = []

    for dur in durations:
        log(f"\n  --- Test: {dur}s audio ---")
        audio = create_test_audio(duration=dur, sample_rate=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        iv = inputs.input_values
        mask = inputs.get("attention_mask")

        # CPU
        model_cpu = model.cpu()
        with torch.no_grad():
            t0 = time.time()
            out_cpu = model_cpu(iv, attention_mask=mask)
            cpu_t = time.time() - t0
        cpu_logits = out_cpu.logits.numpy()

        # NPU
        model_npu = model.npu()
        iv_npu = iv.npu()
        mask_npu = mask.npu() if mask is not None else None
        with torch.no_grad():
            for _ in range(3):
                _ = model_npu(iv_npu, attention_mask=mask_npu)
            torch.npu.synchronize()
            t0 = time.time()
            out_npu = model_npu(iv_npu, attention_mask=mask_npu)
            torch.npu.synchronize()
            npu_t = time.time() - t0
        npu_logits = out_npu.logits.cpu().numpy()

        # Metrics
        abs_diff = np.abs(cpu_logits - npu_logits)
        cos_sim = float(
            np.dot(cpu_logits.flatten(), npu_logits.flatten())
            / (np.linalg.norm(cpu_logits.flatten()) * np.linalg.norm(npu_logits.flatten()) + 1e-10)
        )

        result = {
            "duration_s": dur,
            "input_shape": list(iv.shape),
            "output_shape": list(cpu_logits.shape),
            "cpu_time_ms": cpu_t * 1000,
            "npu_time_ms": npu_t * 1000,
            "speedup": cpu_t / npu_t,
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "cosine_similarity": cos_sim,
            "passed": cos_sim > 0.9999,
        }
        all_results.append(result)

        log(f"    Input shape        : {iv.shape}")
        log(f"    Output shape       : {cpu_logits.shape}")
        log(f"    CPU time           : {cpu_t*1000:.1f} ms")
        log(f"    NPU time           : {npu_t*1000:.1f} ms")
        log(f"    Speedup            : {cpu_t/npu_t:.1f}x")
        log(f"    Cosine similarity  : {cos_sim:.8f}")
        log(f"    Max abs diff       : {np.max(abs_diff):.6e}")
        log(f"    Result             : {'PASS' if result['passed'] else 'FAIL'}")

    RESULTS["accuracy"] = all_results
    return all(c["passed"] for c in all_results)


def performance_test(model, processor):
    log("\n" + "=" * 60)
    log("4. PERFORMANCE BENCHMARK")
    log("=" * 60)

    model_npu = model.npu()
    perf_results = []

    configs = [
        {"duration": 1.0, "desc": "1s short"},
        {"duration": 5.0, "desc": "5s medium"},
        {"duration": 10.0, "desc": "10s long"},
        {"duration": 30.0, "desc": "30s extended"},
    ]

    for cfg in configs:
        audio = create_test_audio(duration=cfg["duration"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        iv = inputs.input_values.npu()
        mask = inputs.get("attention_mask")
        mask = mask.npu() if mask is not None else None

        # Warmup
        for _ in range(5):
            _ = model_npu(iv, attention_mask=mask)
        torch.npu.synchronize()

        # Benchmark: 20 iterations
        times = []
        for i in range(20):
            torch.npu.synchronize()
            t0 = time.time()
            _ = model_npu(iv, attention_mask=mask)
            torch.npu.synchronize()
            times.append(time.time() - t0)

        times = np.array(times)
        perf = {
            "desc": cfg["desc"],
            "duration_s": cfg["duration"],
            "iterations": 20,
            "mean_ms": float(np.mean(times) * 1000),
            "median_ms": float(np.median(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "rtf": float(np.mean(times) / cfg["duration"]),
        }
        perf_results.append(perf)

        log(f"\n  {cfg['desc']} audio ({cfg['duration']}s):")
        log(f"    Mean latency   : {perf['mean_ms']:.1f} ms")
        log(f"    Median latency : {perf['median_ms']:.1f} ms")
        log(f"    P95 latency    : {np.percentile(times, 95)*1000:.1f} ms")
        log(f"    P99 latency    : {np.percentile(times, 99)*1000:.1f} ms")
        log(f"    RTF            : {perf['rtf']:.4f}")

    RESULTS["performance"] = perf_results
    return True


def batch_performance_test(model, processor):
    log("\n" + "=" * 60)
    log("5. BATCH PERFORMANCE BENCHMARK")
    log("=" * 60)

    model_npu = model.npu()
    batch_results = []

    for bs in [1, 2, 4, 8]:
        audios = [create_test_audio(duration=3.0) for _ in range(bs)]
        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        iv = inputs.input_values.npu()
        mask = inputs.get("attention_mask")
        mask = mask.npu() if mask is not None else None

        for _ in range(5):
            _ = model_npu(iv, attention_mask=mask)
        torch.npu.synchronize()

        times = []
        for _ in range(20):
            torch.npu.synchronize()
            t0 = time.time()
            _ = model_npu(iv, attention_mask=mask)
            torch.npu.synchronize()
            times.append(time.time() - t0)

        times = np.array(times)
        br = {
            "batch_size": bs,
            "mean_ms": float(np.mean(times) * 1000),
            "median_ms": float(np.median(times) * 1000),
            "per_item_ms": float(np.mean(times) * 1000 / bs),
        }
        batch_results.append(br)

        log(f"  batch_size={bs}: mean={br['mean_ms']:.1f}ms, "
            f"per_item={br['per_item_ms']:.1f}ms")

    RESULTS["batch_performance"] = batch_results


def decode_test(model, processor):
    log("\n" + "=" * 60)
    log("6. DECODER INTEGRATION TEST")
    log("=" * 60)

    audio = create_test_audio(duration=3.0)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    model_npu = model.npu()
    with torch.no_grad():
        output = model_npu(inputs.input_values.npu())
    logits = output.logits.cpu().numpy()

    # Decode with LM
    t0 = time.time()
    result = processor.decode(logits[0])
    decode_time = time.time() - t0
    text = result.text if hasattr(result, "text") else str(result)

    log(f"  Decoded text    : '{text}'")
    log(f"  Decode time     : {decode_time*1000:.1f} ms")
    log(f"  Logit score     : {result.logit_score:.4f}" if hasattr(result, "logit_score") else "")
    RESULTS["decode"] = {"text": text, "decode_time_ms": decode_time * 1000}


def cpu_vs_npu_comparison(model, processor):
    log("\n" + "=" * 60)
    log("7. CPU vs NPU FULL COMPARISON")
    log("=" * 60)

    audio = create_test_audio(duration=5.0)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # CPU
    model_cpu = model.cpu()
    with torch.no_grad():
        t0 = time.time()
        out_cpu = model_cpu(inputs.input_values, attention_mask=inputs.get("attention_mask"))
        cpu_tt = time.time() - t0
    cpu_logits = out_cpu.logits.numpy()

    # NPU
    model_npu = model.npu()
    iv = inputs.input_values.npu()
    m = inputs.get("attention_mask")
    m = m.npu() if m is not None else None

    for _ in range(3):
        _ = model_npu(iv, attention_mask=m)
    torch.npu.synchronize()
    t0 = time.time()
    out_npu = model_npu(iv, attention_mask=m)
    torch.npu.synchronize()
    npu_tt = time.time() - t0
    npu_logits = out_npu.logits.detach().cpu().numpy()

    # Decode both
    cpu_text = processor.decode(cpu_logits[0])
    cpu_str = cpu_text.text if hasattr(cpu_text, "text") else str(cpu_text)
    npu_text = processor.decode(npu_logits[0])
    npu_str = npu_text.text if hasattr(npu_text, "text") else str(npu_text)

    cos_sim = float(
        np.dot(cpu_logits.flatten(), npu_logits.flatten())
        / (np.linalg.norm(cpu_logits.flatten()) * np.linalg.norm(npu_logits.flatten()) + 1e-10)
    )

    comp = {
        "cpu_total_ms": cpu_tt * 1000,
        "npu_total_ms": npu_tt * 1000,
        "npu_speedup": cpu_tt / npu_tt,
        "cosine_similarity": cos_sim,
        "cpu_decoded": cpu_str,
        "npu_decoded": npu_str,
        "decode_match": cpu_str == npu_str,
    }

    log(f"  CPU total time     : {cpu_tt*1000:.1f} ms")
    log(f"  NPU total time     : {npu_tt*1000:.1f} ms")
    log(f"  NPU speedup        : {cpu_tt/npu_tt:.1f}x")
    log(f"  Cosine similarity  : {cos_sim:.8f}")
    log(f"  CPU decoded        : '{cpu_str}'")
    log(f"  NPU decoded        : '{npu_str}'")
    log(f"  Decode match       : {cpu_str == npu_str}")

    RESULTS["full_comparison"] = comp
    return True


def main():
    # Init log
    with open(LOG_FILE, "w") as f:
        f.write(f"# wav2vec2-indonesian-javanese-sundanese NPU Evaluation Log\n")
        f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Device: Ascend 910B4\n\n")

    log("wav2vec2-indonesian-javanese-sundanese NPU Evaluation")
    log(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_passed = True

    # 1. Environment
    all_passed &= env_check()

    # 2. Model loading
    model, processor = model_load_test()

    # 3. Accuracy
    acc_ok = accuracy_test(model, processor)
    all_passed &= acc_ok

    # 4. Performance
    perf_ok = performance_test(model, processor)
    all_passed &= perf_ok

    # 5. Batch performance
    batch_performance_test(model, processor)

    # 6. Decoder test
    decode_test(model, processor)

    # 7. Full comparison
    cpu_npu_ok = cpu_vs_npu_comparison(model, processor)
    all_passed &= cpu_npu_ok

    # Summary
    log("\n" + "=" * 60)
    log("8. SUMMARY")
    log("=" * 60)
    RESULTS["overall_status"] = "PASS" if all_passed else "FAIL"
    log(f"  Overall status : {RESULTS['overall_status']}")
    log(f"  Accuracy       : {'PASS' if acc_ok else 'FAIL'}")
    log(f"  Performance    : {'PASS' if perf_ok else 'FAIL'}")
    log(f"  Full compare   : {'PASS' if cpu_npu_ok else 'FAIL'}")

    # Save JSON results
    with open(JSON_FILE, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    log(f"\n  Results saved to: {JSON_FILE}")
    log(f"  Log saved to    : {LOG_FILE}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
