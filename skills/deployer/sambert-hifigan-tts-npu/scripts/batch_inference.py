#!/usr/bin/env python3
"""
Batch inference for all downloaded Sambert-HifiGAN TTS models.
Runs inference on CPU + NPU for each model and compares outputs.
"""

import os
import sys
import json
import time
import argparse
import warnings
import subprocess
import numpy as np

warnings.filterwarnings("ignore")


MODELS_CACHE = "/opt/atomgit/.cache/modelscope/hub/models/iic"


def find_downloaded_models():
    """Find all downloaded models with voices/ directory."""
    models = []
    if not os.path.isdir(MODELS_CACHE):
        return models
    for name in sorted(os.listdir(MODELS_CACHE)):
        model_dir = os.path.join(MODELS_CACHE, name)
        voices_dir = os.path.join(model_dir, "voices")
        if os.path.isdir(model_dir) and os.path.isdir(voices_dir):
            models.append(name)
    return models


def get_voices(model_dir):
    """Get available voices for a model."""
    voices_dir = os.path.join(model_dir, "voices")
    voices = [d for d in os.listdir(voices_dir)
              if os.path.isdir(os.path.join(voices_dir, d))]
    return sorted(voices)


def run_inference(model_name, voice, text, device, output_wav):
    """Run inference for a single model/voice/device."""
    model_dir = os.path.join(MODELS_CACHE, model_name)
    cmd = [
        sys.executable, "/opt/atomgit/models/direct_inference.py",
        "--model-dir", model_dir,
        "--voice", voice,
        "--text", text,
        "--device", device,
        "--output", output_wav,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="Batch TTS inference")
    parser.add_argument("--text", type=str, default="北京今天天气怎么样",
                        help="Input text for TTS")
    parser.add_argument("--output-dir", type=str, default="/opt/atomgit/models/results",
                        help="Output directory for results")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip CPU inference")
    parser.add_argument("--skip-npu", action="store_true",
                        help="Skip NPU inference")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "audio"), exist_ok=True)

    # Find models
    models = find_downloaded_models()
    if not models:
        print("No downloaded models found!")
        print("Run download_models.sh first or check MODELS_CACHE path.")
        sys.exit(1)

    print(f"Found {len(models)} models:")
    for m in models:
        voices = get_voices(os.path.join(MODELS_CACHE, m))
        print(f"  {m}: {voices}")

    # Results database
    results_db = os.path.join(args.output_dir, "results.json")
    results = {}

    for model_name in models:
        model_dir = os.path.join(MODELS_CACHE, model_name)
        voices = get_voices(model_dir)

        for voice in voices:
            print(f"\n{'='*60}")
            print(f"Model: {model_name} | Voice: {voice}")
            print(f"{'='*60}")

            # Output filenames
            safe_model = model_name.replace("/", "_")
            prefix = f"{safe_model}_{voice}"
            cpu_wav = os.path.join(args.output_dir, "audio", f"{prefix}_cpu.wav")
            npu_wav = os.path.join(args.output_dir, "audio", f"{prefix}_npu.wav")

            entry = {
                "model": model_name,
                "voice": voice,
                "text": args.text,
                "cpu_time": None,
                "npu_time": None,
                "snr": None,
                "correlation": None,
                "cpu_samples": None,
                "npu_samples": None,
                "status": "pending",
            }

            # CPU inference
            if not args.skip_cpu:
                print(f"\n  [CPU] Running inference...")
                result, elapsed = run_inference(
                    model_name, voice, args.text, "cpu", cpu_wav
                )
                entry["cpu_time"] = round(elapsed, 2)
                if result.returncode == 0:
                    import soundfile as sf
                    data, _ = sf.read(cpu_wav)
                    entry["cpu_samples"] = len(data)
                    print(f"  [CPU] OK ({elapsed:.1f}s, {len(data)} samples)")
                else:
                    entry["status"] = f"cpu_error: {result.stderr[-200:]}"
                    print(f"  [CPU] FAILED: {result.stderr[-200:]}")
                    continue

            # NPU inference
            if not args.skip_npu:
                print(f"  [NPU] Running inference...")
                result, elapsed = run_inference(
                    model_name, voice, args.text, "npu", npu_wav
                )
                entry["npu_time"] = round(elapsed, 2)
                if result.returncode == 0:
                    import soundfile as sf
                    data, _ = sf.read(npu_wav)
                    entry["npu_samples"] = len(data)
                    print(f"  [NPU] OK ({elapsed:.1f}s, {len(data)} samples)")
                else:
                    entry["status"] = f"npu_error: {result.stderr[-200:]}"
                    print(f"  [NPU] FAILED: {result.stderr[-200:]}")
                    continue

            # Compare CPU vs NPU
            if not args.skip_cpu and not args.skip_npu:
                import soundfile as sf
                cpu_data, _ = sf.read(cpu_wav)
                npu_data, _ = sf.read(npu_wav)
                min_len = min(len(cpu_data), len(npu_data))
                cpu_a = cpu_data[:min_len]
                npu_a = npu_data[:min_len]

                diff = np.abs(cpu_a - npu_a)
                signal_power = np.mean(cpu_a ** 2)
                noise_power = np.mean((cpu_a - npu_a) ** 2)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
                corr = float(np.corrcoef(cpu_a, npu_a)[0, 1])
                entry["snr"] = round(snr, 2)
                entry["correlation"] = round(corr, 6)
                entry["status"] = "pass" if snr > 20 else "low_snr"
                print(f"  [COMPARE] SNR={snr:.1f}dB, Corr={corr:.4f}")

            results[f"{model_name}_{voice}"] = entry

            # Save incremental results
            with open(results_db, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v["status"] == "pass")
    failed = sum(1 for v in results.values() if v["status"] != "pass")

    print(f"Total: {len(results)} models/voices")
    print(f"Passed: {passed}")
    print(f"Failed/Error: {failed}")

    for key, entry in results.items():
        status = "PASS" if entry["status"] == "pass" else entry["status"]
        cpu_t = entry.get("cpu_time", "N/A")
        npu_t = entry.get("npu_time", "N/A")
        snr = entry.get("snr", "N/A")
        print(f"  {key}: {status} | CPU={cpu_t}s NPU={npu_t}s SNR={snr}")

    print(f"\nResults saved to {results_db}")


if __name__ == "__main__":
    main()
