#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend NPU Inference Script for apple/mobilevit-xx-small
Model: MobileViTForImageClassification (ImageNet-1k pretrained)
Hardware: Huawei Ascend 910B NPU
"""

import os
import sys
import time
import argparse
import warnings
from typing import Optional

import torch
import numpy as np

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    torch_npu = None

from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image


def get_device():
    if torch_npu is not None and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device):
    print(f"Loading model from: {model_path}")
    print(f"Target device: {device}")

    processor = MobileViTImageProcessor.from_pretrained(model_path)
    model = MobileViTForImageClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {param_count:,}")
    return processor, model


def predict(
    image_path: str,
    processor,
    model,
    device: torch.device,
    top_k: int = 5,
    benchmark: bool = False,
):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        if benchmark:
            if device.type == "npu":
                torch.npu.synchronize()
            start = time.perf_counter()
            outputs = model(pixel_values=pixel_values)
            if device.type == "npu":
                torch.npu.synchronize()
            latency = time.perf_counter() - start
        else:
            outputs = model(pixel_values=pixel_values)
            latency = None

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        label = model.config.id2label.get(str(idx), f"unknown_{idx}")
        results.append({"label": label, "score": prob, "index": idx})

    return results, latency, image


def print_results(results, latency=None):
    print("\n=== Classification Results ===")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['label']} (score: {r['score']:.4f})")
    if latency is not None:
        print(f"\nInference latency: {latency*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Ascend NPU MobileViT-XX-Small Inference")
    parser.add_argument("--model_path", type=str, default="./model",
                        help="Path to model directory")
    parser.add_argument("--image", type=str, default=None,
                        help="Input image path")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k predictions to show")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode with timing")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations")
    args = parser.parse_args()

    device = get_device()
    processor, model = load_model(args.model_path, device)

    if args.image:
        results, latency, _ = predict(args.image, processor, model, device,
                                       top_k=args.top_k, benchmark=args.benchmark)
        print_results(results, latency)
    elif args.benchmark:
        print("\n=== Benchmark Mode ===")
        print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
        dummy = Image.new("RGB", (256, 256), color=128)
        for i in range(args.warmup):
            inputs = processor(images=dummy, return_tensors="pt")
            model(inputs["pixel_values"].to(device))
        latencies = []
        for i in range(args.iterations):
            inputs = processor(images=dummy, return_tensors="pt")
            if device.type == "npu":
                torch.npu.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model(inputs["pixel_values"].to(device))
            if device.type == "npu":
                torch.npu.synchronize()
            latencies.append(time.perf_counter() - start)
        avg_lat = sum(latencies) / len(latencies)
        print(f"Avg latency: {avg_lat*1000:.2f} ms")
        print(f"Min latency: {min(latencies)*1000:.2f} ms")
        print(f"Max latency: {max(latencies)*1000:.2f} ms")
        print(f"Throughput: {1.0/avg_lat:.2f} images/sec")
    else:
        print("Please provide --image or use --benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()
