#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend NPU Accuracy Validation for apple/mobilevit-xx-small
Validates CPU vs NPU numerical consistency (logits, probabilities, top-1 labels).
Threshold: relative error < 1% for classification confidence scores.
"""

import os
import sys
import json
import warnings

import torch
import numpy as np

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    torch_npu = None

from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image


TEST_IMAGES = [
    {
        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "description": "two cats on a bed",
    },
    {
        "url": "http://images.cocodataset.org/val2017/000000039770.jpg",
        "description": "cat on a bed",
    },
]


def load_test_images():
    import requests
    images = []
    for item in TEST_IMAGES:
        try:
            resp = requests.get(item["url"], stream=True, timeout=30)
            img = Image.open(resp.raw).convert("RGB")
            images.append(img)
            print(f"  Loaded: {item['description']}")
        except Exception as e:
            print(f"  Failed to load {item['url']}: {e}")
    return images


def get_predictions(processor, model, images, device):
    all_logits = []
    all_probs = []
    all_top1 = []
    with torch.no_grad():
        for img in images:
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_top1.append(logits.argmax(-1).item())
    return all_logits, all_probs, all_top1


def compute_metrics(cpu_logits, npu_logits, cpu_probs, npu_probs, cpu_top1, npu_top1, id2label):
    results = []
    max_errors = []
    for i in range(len(cpu_logits)):
        logit_diff = torch.abs(cpu_logits[i] - npu_logits[i])
        logit_rel_error = logit_diff / (torch.abs(cpu_logits[i]) + 1e-8)
        max_logit_rel_error = logit_rel_error.max().item()
        mean_logit_rel_error = logit_rel_error.mean().item()

        prob_diff = torch.abs(cpu_probs[i] - npu_probs[i])
        max_prob_diff = prob_diff.max().item()
        mean_prob_diff = prob_diff.mean().item()

        label_match = cpu_top1[i] == npu_top1[i]

        cos_sim = torch.nn.functional.cosine_similarity(
            cpu_logits[i].flatten().unsqueeze(0),
            npu_logits[i].flatten().unsqueeze(0)
        ).item()

        max_errors.append(max_logit_rel_error)

        cpu_top1_label = id2label.get(str(cpu_top1[i]), f"class_{cpu_top1[i]}")
        npu_top1_label = id2label.get(str(npu_top1[i]), f"class_{npu_top1[i]}")

        result = {
            "image_index": i,
            "cpu_top1_label": cpu_top1_label,
            "npu_top1_label": npu_top1_label,
            "top1_match": label_match,
            "max_logit_relative_error": round(max_logit_rel_error, 6),
            "mean_logit_relative_error": round(mean_logit_rel_error, 6),
            "max_probability_difference": round(max_prob_diff, 6),
            "mean_probability_difference": round(mean_prob_diff, 6),
            "logit_cosine_similarity": round(cos_sim, 6),
        }
        results.append(result)

        status = "PASS" if (label_match and cos_sim > 0.999) else "FAIL"
        print(f"\n  Image {i}: {status}")
        print(f"    CPU top-1: {cpu_top1_label}")
        print(f"    NPU top-1: {npu_top1_label}")
        print(f"    Label match: {label_match}")
        print(f"    Max logit relative error: {max_logit_rel_error:.6f} ({max_logit_rel_error*100:.4f}%)")
        print(f"    Mean logit relative error: {mean_logit_rel_error:.6f}")
        print(f"    Max prob difference: {max_prob_diff:.6f}")
        print(f"    Logit cosine similarity: {cos_sim:.6f}")

    all_match = all(r["top1_match"] for r in results)
    min_cos_sim = min(r["logit_cosine_similarity"] for r in results)
    overall_pass = all_match and min_cos_sim > 0.999
    return results, overall_pass, min_cos_sim


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./model"
    output_report = sys.argv[2] if len(sys.argv) > 2 else "accuracy_report.json"

    print("=" * 60)
    print("Ascend NPU Accuracy Validation")
    print("Model: apple/mobilevit-xx-small (MobileViTForImageClassification)")
    print("=" * 60)

    processor = MobileViTImageProcessor.from_pretrained(model_path)
    model_cpu = MobileViTForImageClassification.from_pretrained(model_path)
    model_cpu.eval()

    npu_available = torch_npu is not None and torch.npu.is_available()
    model_npu = None
    if npu_available:
        print("Loading model on NPU...")
        model_npu = MobileViTForImageClassification.from_pretrained(model_path)
        model_npu = model_npu.to("npu")
        model_npu.eval()
    else:
        print("NPU not available. Skipping NPU validation.")

    print("\nLoading test images...")
    images = load_test_images()
    if not images:
        print("ERROR: No test images loaded!")
        return 1
    print(f"Loaded {len(images)} test images.")

    print("\nRunning inference on CPU...")
    cpu_logits, cpu_probs, cpu_top1 = get_predictions(processor, model_cpu, images, torch.device("cpu"))

    npu_results = None
    overall_pass = True
    if npu_available and model_npu is not None:
        print("Running inference on NPU...")
        npu_logits, npu_probs, npu_top1 = get_predictions(processor, model_npu, images, torch.device("npu"))

        print("\n=== CPU vs NPU Comparison ===")
        id2label = model_cpu.config.id2label
        npu_results, overall_pass, max_error = compute_metrics(
            cpu_logits, npu_logits, cpu_probs, npu_probs, cpu_top1, npu_top1, id2label
        )

        print(f"\n  Min cosine similarity: {max_error:.6f}")
        print(f"  Overall status: {'PASS' if overall_pass else 'FAIL'}")
    else:
        print("Skipping NPU comparison (NPU unavailable).")
        overall_pass = False

    report = {
        "model": model_path,
        "architecture": "MobileViTForImageClassification",
        "npu_available": npu_available,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "threshold": "cosine_similarity > 0.999 and top-1 label match",
        "test_images": len(images),
        "cpu_npu_comparison": {
            "status": "PASS" if overall_pass else "FAIL",
            "results": npu_results,
        },
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Report saved to: {output_report}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 60}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
