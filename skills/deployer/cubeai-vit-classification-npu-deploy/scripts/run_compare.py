#!/usr/bin/env python3
"""CPU vs NPU precision comparison for ViT image classification models."""

import os
import sys
import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


@torch.no_grad()
def infer(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    outputs = model(**inputs)
    elapsed = time.perf_counter() - t0
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_idx = logits.argmax(-1).item()
    return logits.cpu(), probs.cpu(), pred_idx, elapsed


def find_sample(model_path):
    img_dir = os.path.join(model_path, "images")
    if os.path.isdir(img_dir):
        imgs = sorted(os.listdir(img_dir))
        if imgs:
            return os.path.join(img_dir, imgs[0])
    return None


def compare(model_path, image_path=None):
    import torch_npu  # noqa: F401

    model_name = os.path.basename(model_path.rstrip("/"))
    print(f"Model: {model_name}")

    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    num_labels = model.config.num_labels
    print(f"Architecture: ViTForImageClassification, {num_labels} classes")

    sample = image_path or find_sample(model_path)
    if not sample or not os.path.exists(sample):
        print("ERROR: No test image found.")
        return None
    print(f"Test image: {sample}")

    # CPU inference
    print("\n--- CPU Inference ---")
    model_cpu = model.to("cpu").eval()
    logits_cpu, probs_cpu, pred_cpu, t_cpu = infer(model_cpu, processor, sample, "cpu")
    label_cpu = model_cpu.config.id2label[pred_cpu]
    print(f"  Time: {t_cpu*1000:.2f} ms")
    print(f"  Top-1: [{pred_cpu}] {label_cpu}")
    print(f"  Confidence: {probs_cpu[0, pred_cpu].item():.6f}")

    # NPU inference
    print("\n--- NPU Inference ---")
    model_npu = model.to("npu:0").eval()
    logits_npu, probs_npu, pred_npu, t_npu = infer(model_npu, processor, sample, "npu:0")
    label_npu = model_npu.config.id2label[pred_npu]
    print(f"  Time: {t_npu*1000:.2f} ms")
    print(f"  Top-1: [{pred_npu}] {label_npu}")
    print(f"  Confidence: {probs_npu[0, pred_npu].item():.6f}")

    # Compare
    print("\n" + "=" * 50)
    print("Comparison Results")
    print("=" * 50)

    logits_cpu_np = logits_cpu.numpy()
    logits_npu_np = logits_npu.numpy()
    probs_cpu_np = probs_cpu.numpy()
    probs_npu_np = probs_npu.numpy()

    results = {
        "model": model_name,
        "cpu_time_ms": round(t_cpu * 1000, 2),
        "npu_time_ms": round(t_npu * 1000, 2),
        "speedup": round(t_cpu / t_npu, 2) if t_npu > 0 else float('inf'),
        "cpu_top1": int(pred_cpu),
        "cpu_top1_label": label_cpu,
        "npu_top1": int(pred_npu),
        "npu_top1_label": label_npu,
        "top1_match": pred_cpu == pred_npu,
        "max_logits_ae": round(float(np.max(np.abs(logits_cpu_np - logits_npu_np))), 6),
        "mean_logits_ae": round(float(np.mean(np.abs(logits_cpu_np - logits_npu_np))), 6),
        "max_probs_ae": round(float(np.max(np.abs(probs_cpu_np - probs_npu_np))), 6),
        "mean_probs_ae": round(float(np.mean(np.abs(probs_cpu_np - probs_npu_np))), 6),
        "cosine_similarity": round(float(np.dot(logits_cpu_np[0], logits_npu_np[0]) / (
            np.linalg.norm(logits_cpu_np[0]) * np.linalg.norm(logits_npu_np[0]))), 6),
        "l2_error": round(float(np.sqrt(np.mean((logits_cpu_np - logits_npu_np) ** 2))), 6),
    }
    results["max_error_pct"] = round(results["max_probs_ae"] * 100, 4)
    results["passed"] = results["max_error_pct"] < 1.0

    for k, v in results.items():
        print(f"  {k}: {v}")

    # Cleanup
    del model_npu
    torch.npu.empty_cache()
    import gc; gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="CPU vs NPU Precision Comparison")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = compare(args.model_path, args.image)
    if results:
        print(f"\nPrecision Check: {'PASS' if results['passed'] else 'FAIL'}")
        print(f"Max Probability Error: {results['max_error_pct']}%")
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
