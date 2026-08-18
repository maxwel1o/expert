#!/usr/bin/env python3
"""Comprehensive evaluation for all 3 Paraformer models on NPU."""
import os, sys, time, json
import torch, torch_npu
import numpy as np
import soundfile as sf
from functools import wraps

from funasr.auto.auto_model import AutoModel

# === Paths ===
WAV = "/opt/atomgit/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"

MODELS = {
    "model1": {
        "name": "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "dir": "/opt/atomgit/.cache/modelscope/hub/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": None, "punc": None, "spk": None,
        "output_dir": "/opt/atomgit/npu_ref/iic/paraformer/eval_logs/model1",
    },
    "model2": {
        "name": "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "dir": "/opt/atomgit/.cache/modelscope/hub/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "spk": None,
        "output_dir": "/opt/atomgit/npu_ref/iic/paraformer/eval_logs/model2",
    },
    "model3": {
        "name": "speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
        "dir": "/opt/atomgit/.cache/modelscope/hub/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "spk": None,
        "output_dir": "/opt/atomgit/npu_ref/iic/paraformer/eval_logs/model3",
    },
}


def load_audio(wav_path):
    data, sr = sf.read(wav_path)
    if data.ndim > 1: data = data.mean(axis=1)
    return data.astype(np.float32)


def build_npu_model(cfg, device="npu:0"):
    kw = {"model": cfg["dir"], "hub": "ms", "device": "cpu",
          "disable_update": True, "disable_pbar": True}
    if cfg["vad"]: kw["vad_model"] = cfg["vad"]
    if cfg["punc"]: kw["punc_model"] = cfg["punc"]
    if cfg["spk"]: kw["spk_model"] = cfg["spk"]

    model_inst = AutoModel(**kw)
    model_inst.model = model_inst.model.to(device)
    model_inst.model.eval()
    orig_infer = model_inst.model.inference
    @wraps(orig_infer)
    def patched(*a, **kw):
        kw["device"] = device; kw["ngpu"] = 1; return orig_infer(*a, **kw)
    model_inst.model.inference = patched
    return model_inst


def build_cpu_model(cfg):
    kw = {"model": cfg["dir"], "hub": "ms", "device": "cpu",
          "disable_update": True, "disable_pbar": True}
    if cfg["vad"]: kw["vad_model"] = cfg["vad"]
    if cfg["punc"]: kw["punc_model"] = cfg["punc"]
    if cfg["spk"]: kw["spk_model"] = cfg["spk"]
    return AutoModel(**kw)


def eval_model(label, cfg, audio):
    print(f"\n{'='*60}")
    print(f"  Evaluating {cfg['name']}")
    print(f"{'='*60}")

    # === NPU timing ===
    print("\n  [NPU Timing]")
    model_npu = build_npu_model(cfg)
    npu_times = []
    for i in range(3):
        torch.npu.empty_cache(); torch.npu.synchronize()
        t0 = time.time()
        r = model_npu.generate(input=audio)
        torch.npu.synchronize()
        elapsed = time.time() - t0
        text = r[0].get("text", "") if r else ""
        npu_times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.3f}s  text='{text[:60]}'")

    npu_steady = sum(npu_times[1:]) / len(npu_times[1:])
    npu_text = text

    # === CPU timing ===
    print("\n  [CPU Timing]")
    model_cpu = build_cpu_model(cfg)
    torch.set_num_threads(16)
    t0 = time.time()
    r = model_cpu.generate(input=audio)
    cpu_time = time.time() - t0
    cpu_text = r[0].get("text", "") if r else ""
    print(f"    CPU: {cpu_time:.3f}s  text='{cpu_text[:60]}'")

    # === Accuracy ===
    print("\n  [Accuracy]")
    acc = 100.0 if cpu_text == npu_text else 0.0
    if cpu_text and npu_text and cpu_text != npu_text:
        common = sum(1 for a, b in zip(cpu_text, npu_text) if a == b)
        acc = 100 * common / max(len(cpu_text), len(npu_text))
    status = "PASS" if acc >= 99.0 else "FAIL"
    print(f"    {status}: {acc:.2f}%  (CPU='{cpu_text[:40]}' / NPU='{npu_text[:40]}')")

    # === Summary ===
    result = {
        "model": cfg["name"],
        "npu_1st_run_s": round(npu_times[0], 3),
        "npu_steady_state_s": round(npu_steady, 3),
        "cpu_time_s": round(cpu_time, 3),
        "speedup_vs_cpu": round(cpu_time / npu_steady, 2),
        "accuracy_pct": round(acc, 2),
        "accuracy_status": status,
        "cpu_text": cpu_text,
        "npu_text": npu_text,
    }
    print(f"\n  SUMMARY: {result}")

    # Save to file
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "eval_results.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save raw logs
    with open(os.path.join(cfg["output_dir"], "eval_log.txt"), "w") as f:
        f.write(f"Model: {cfg['name']}\n")
        f.write(f"NPU Runs: {npu_times}\n")
        f.write(f"NPU 1st: {npu_times[0]:.3f}s\n")
        f.write(f"NPU Steady: {npu_steady:.3f}s\n")
        f.write(f"CPU Time: {cpu_time:.3f}s\n")
        f.write(f"Speedup: {cpu_time/npu_steady:.2f}x\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"CPU Text: {cpu_text}\n")
        f.write(f"NPU Text: {npu_text}\n")

    return result


if __name__ == "__main__":
    audio = load_audio(WAV)
    results = {}
    for key, cfg in MODELS.items():
        try:
            results[key] = eval_model(key, cfg, audio)
        except Exception as e:
            print(f"\n  [ERROR] {cfg['name']}: {e}")
            import traceback; traceback.print_exc()

    # Summary table
    print("\n\n")
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<50} {'CPU(s)':<8} {'NPU(s)':<8} {'Speedup':<8} {'Acc':<6}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for key, r in results.items():
        print(f"  {r['model'][:48]:<50} {r['cpu_time_s']:<8.3f} {r['npu_steady_state_s']:<8.3f} {r['speedup_vs_cpu']:<8.2f}x {r['accuracy_pct']:<6.2f}%")
    print()

    # Save combined results
    combined = {"environment": {"npu": "Ascend910B4", "torch_npu": "2.9.0"},
                "results": results}
    with open("/opt/atomgit/npu_ref/iic/paraformer/eval_logs/combined_results.json", "w") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
