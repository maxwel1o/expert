#!/usr/bin/env python3
"""
Inference script for ConvFormer model on CPU or NPU.

Usage:
    python3 inference.py --device cpu    # CPU inference
    python3 inference.py --device npu    # NPU inference (Ascend)
"""

import argparse
import time
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def parse_args():
    parser = argparse.ArgumentParser(description="ConvFormer NPU Inference")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"],
                        help="Device to run inference on (cpu or npu)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. convformer_b36.sail_in1k_384)")
    parser.add_argument("--image", type=str, default="test.jpg",
                        help="Input image path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    print(f"Loading model: {args.model}")
    model = timm.create_model(args.model, pretrained=True)
    model.eval()

    if device == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            print("ERROR: NPU is not available on this system.")
            return
        model = model.npu()
        print("Model moved to NPU (Ascend 910)")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    img = Image.open(args.image).convert("RGB")
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    input_tensor = transform(img).unsqueeze(0)

    if device == "npu":
        input_tensor = input_tensor.npu()

    print(f"Input shape: {input_tensor.shape}")

    # Warmup (NPU only)
    if device == "npu":
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        torch.npu.synchronize()

    # Inference
    print("Running inference...")
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        if device == "npu":
            torch.npu.synchronize()
        elapsed = time.time() - start

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5 = probs.topk(5)

    print(f"\n=== Results ===")
    print(f"Device: {device.upper()}")
    print(f"Inference time: {elapsed:.4f}s")
    print(f"Top-5 predictions:")
    for i in range(5):
        print(f"  {i+1}. class={top5.indices[i].item():5d}  prob={top5.values[i].item():.6f}")
    print()


if __name__ == "__main__":
    main()
