#!/usr/bin/env python3
"""LeViT model inference on CPU or NPU.

Usage:
  python3 run_inference.py --model levit-128 --device npu --image test.jpg
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import LevitForImageClassificationWithTeacher, LevitImageProcessor


MODEL_NAMES = {
    "128": "levit-128", "128s": "levit-128S", "192": "levit-192",
    "256": "levit-256", "384": "levit-384",
    "levit-128": "levit-128", "levit-128S": "levit-128S",
    "levit-192": "levit-192", "levit-256": "levit-256", "levit-384": "levit-384",
}


def resolve_model(model_arg: str) -> str:
    if model_arg in MODEL_NAMES:
        return MODEL_NAMES[model_arg]
    return model_arg


def get_device(device: str):
    if device == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("NPU is not available")
        return torch.device("npu:0")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="LeViT NPU inference")
    parser.add_argument("--model", default="levit-128", help="Model name")
    parser.add_argument("--image", default="test.jpg", help="Input image path")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--model-dir", default=None, help="Model directory (optional)")
    args = parser.parse_args()

    model_name = resolve_model(args.model)
    device = get_device(args.device)
    model_dir = args.model_dir or os.path.join(
        os.path.expanduser("~"), ".cache/modelscope/hub/models/facebook", model_name
    )

    print(f"Loading model: {model_name}")
    processor = LevitImageProcessor.from_pretrained(model_dir)
    model = LevitForImageClassificationWithTeacher.from_pretrained(model_dir)
    model = model.float().to(device)
    model.eval()

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Warmup + timed runs
    with torch.no_grad():
        _ = model(pixel_values)
    if device.type == "npu":
        torch.npu.synchronize()

    num_runs = 10
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(pixel_values)
    if device.type == "npu":
        torch.npu.synchronize()
    elapsed_ms = (time.time() - start) / num_runs * 1000

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    probs = torch.nn.functional.softmax(logits, dim=-1)

    top5 = probs.squeeze().topk(5)
    print(f"\n  Results ({device.type.upper()}):")
    print(f"  Time: {elapsed_ms:.2f}ms")
    for i, (p, idx) in enumerate(zip(top5[0].tolist(), top5[1].tolist()), 1):
        print(f"    {i}. class {idx}: {p*100:.2f}%")

    output = {
        "model": model_name, "device": device.type, "time_ms": elapsed_ms,
        "logits": logits.cpu().numpy().tolist(),
        "probabilities": probs.cpu().numpy().tolist(),
    }
    out_path = f"/tmp/{model_name}_{device.type}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
