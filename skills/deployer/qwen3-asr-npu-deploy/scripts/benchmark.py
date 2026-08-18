"""
Qwen3-ASR & ForcedAligner Benchmark on Ascend NPU
===================================================
Measures: CER/WER, timestamp validity, latency (P50/P95), throughput
Usage:
    python benchmark.py [--asr-path <path>] [--aligner-path <path>]
"""
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import numpy as np
import soundfile as sf
import torch, torch_npu

# Register model types
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import AutoConfig, AutoModel, AutoProcessor
AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from qwen_asr.inference.utils import parse_asr_output, SAMPLE_RATE


def cer(ref, hyp):
    if not ref and not hyp: return 0.0
    if not hyp: return 1.0
    import Levenshtein
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r and not h: return 0.0
    if not h: return 1.0
    import Levenshtein
    return Levenshtein.distance(r, h) / max(len(r), 1)


TEST_CASES = {
    "zh": {
        "audio": "test_audio_zh.wav",
        "ref": "甚至出现交易几乎停滞的情况。",
        "lang": "Chinese",
    },
    "en": {
        "audio": "test_audio_en.wav",
        "ref": "Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.",
        "lang": "English",
    },
}


def benchmark_asr(model_path, device="npu:0"):
    print("=" * 60)
    print("ASR Benchmark")
    print("=" * 60)

    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map=None, low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    asr = Qwen3ASRModel(
        backend="transformers", model=model, processor=processor,
        max_inference_batch_size=-1, max_new_tokens=512,
    )
    asr.device = torch.device(device)
    asr.dtype = torch.bfloat16

    results = {}
    for name, case in TEST_CASES.items():
        if not os.path.exists(case["audio"]):
            print(f"  Skip {name}: {case['audio']} not found")
            continue

        # Warmup
        asr.transcribe(audio=case["audio"], language=case["lang"])

        # Bench
        latencies = []
        texts = []
        for _ in range(5):
            t0 = time.perf_counter()
            r = asr.transcribe(audio=case["audio"], language=case["lang"])
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            texts.append(r[0].text)

        avg = np.mean(latencies)
        results[name] = {
            "cer": round(cer(case["ref"], texts[0]), 4),
            "wer": round(wer(case["ref"], texts[0]), 4),
            "avg_latency_s": round(avg, 3),
            "p50_s": round(np.percentile(latencies, 50), 3),
            "p95_s": round(np.percentile(latencies, 95), 3),
            "throughput": round(1.0 / avg, 2),
            "asr_text": texts[0],
        }
        print(f"  {name.upper()}: CER={results[name]['cer']}, WER={results[name]['wer']}, "
              f"latency={avg:.3f}s, throughput={results[name]['throughput']}/s")

    return results


def benchmark_aligner(model_path, device="npu:0"):
    print("\n" + "=" * 60)
    print("ForcedAligner Benchmark")
    print("=" * 60)

    model = Qwen3ForcedAligner.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=None,
    )
    model.model = model.model.to(device).eval()

    results = {}
    for name, case in TEST_CASES.items():
        if not os.path.exists(case["audio"]):
            continue

        # Warmup
        model.align(audio=case["audio"], text=case["ref"], language=case["lang"])

        # Bench
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            r = model.align(audio=case["audio"], text=case["ref"], language=case["lang"])
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        avg = np.mean(latencies)
        ts_out = r[0]
        dur = len(sf.read(case["audio"])[0]) / SAMPLE_RATE
        ts_ok = all(0 <= t.start_time <= t.end_time <= dur for t in ts_out)

        results[name] = {
            "num_tokens": len(ts_out),
            "timestamps_valid": ts_ok,
            "avg_latency_s": round(avg, 3),
            "p50_s": round(np.percentile(latencies, 50), 3),
            "p95_s": round(np.percentile(latencies, 95), 3),
            "throughput": round(1.0 / avg, 2),
        }
        print(f"  {name.upper()}: {len(ts_out)} tokens, valid={ts_ok}, "
              f"latency={avg:.3f}s, throughput={results[name]['throughput']}/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen3 ASR/ForcedAligner Benchmark")
    parser.add_argument("--asr-path", default="./Qwen3-ASR-0.6B")
    parser.add_argument("--aligner-path", default="./Qwen3-ForcedAligner-0.6B")
    args = parser.parse_args()

    report = {}
    report["asr"] = benchmark_asr(args.asr_path)
    report["aligner"] = benchmark_aligner(args.aligner_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for model, results in report.items():
        for lang, data in results.items():
            print(f"  {model} ({lang}): OK" if data.get("timestamps_valid", True) else f"  {model} ({lang}): WARN")

    with open("benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
