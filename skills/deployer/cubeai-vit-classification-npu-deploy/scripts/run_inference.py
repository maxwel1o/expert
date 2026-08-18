#!/usr/bin/env python3
"""Generic ViT image classification inference on CPU/Ascend NPU."""

import os
import sys
import argparse
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
    return pred_idx, probs.cpu(), elapsed


def main():
    parser = argparse.ArgumentParser(description="ViT Image Classification on CPU/NPU")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--image", type=str, default=None, help="Input image path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"])
    args = parser.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
        device = torch.device("npu:0")
        print(f"  NPU: {torch_npu.npu.get_device_name(0)}")
    else:
        device = torch.device("cpu")
    print(f"Device: {args.device}")

    # Find sample image
    img_path = args.image
    if img_path is None:
        img_dir = os.path.join(args.model_path, "images")
        if os.path.isdir(img_dir):
            imgs = sorted(os.listdir(img_dir))
            if imgs:
                img_path = os.path.join(img_dir, imgs[0])
        if img_path is None or not os.path.exists(img_path):
            print("No image specified and no sample found in model directory.")
            sys.exit(1)

    print(f"Loading model from: {args.model_path}")
    processor = ViTImageProcessor.from_pretrained(args.model_path)
    model = ViTForImageClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    print(f"Model: ViTForImageClassification, num_labels={model.config.num_labels}")

    pred_idx, probs, elapsed = infer(model, processor, img_path, device)
    label = model.config.id2label[pred_idx]
    confidence = probs[0, pred_idx].item()

    print(f"\nResults:")
    print(f"  Inference time: {elapsed*1000:.2f} ms")
    print(f"  Top-1 class: [{pred_idx}] {label}")
    print(f"  Top-1 confidence: {confidence:.4f}")

    top5 = probs.topk(min(5, model.config.num_labels), dim=-1)
    print(f"\nTop-{top5.indices.size(1)} predictions:")
    for i in range(top5.indices.size(1)):
        idx = top5.indices[0, i].item()
        score = top5.values[0, i].item()
        print(f"  [{idx}] {model.config.id2label[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()
