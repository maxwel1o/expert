#!/usr/bin/env python3
"""Sequencer2D inference on CPU and NPU using timm with local weights."""
import argparse
import os
import time
import numpy as np
from PIL import Image

import torch
import torch_npu

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sequencer2d_s.in1k", help="timm model name")
    parser.add_argument("--weights", default=None, help="path to local weights file (.bin or .safetensors)")
    parser.add_argument("--image", default=None, help="path to input image (optional)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "npu"], help="inference device")
    parser.add_argument("--dump", default=None, help="dump output logits to .npy file")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Creating model: {args.model}")
    model = timm.create_model(args.model, pretrained=False)

    if args.weights:
        print(f"Loading weights from: {args.weights}")
        if args.weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.weights)
        else:
            state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully")
    else:
        model = timm.create_model(args.model, pretrained=True)

    model.eval()
    model.to(device)
    print(f"Model loaded on {device}")

    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config)
    print(f"Input config: {data_config}")

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    input_tensor = transform(img).unsqueeze(0).to(device)

    for _ in range(3):
        _ = model(input_tensor)

    if args.device == "npu":
        torch.npu.synchronize(device)
    start = time.time()
    for _ in range(10):
        output = model(input_tensor)
    if args.device == "npu":
        torch.npu.synchronize(device)
    end = time.time()

    logits = output.cpu().numpy()
    latencies = (end - start) / 10

    print(f"\nResults on {args.device}:")
    print(f"  Output shape: {logits.shape}")
    top1_idx = np.argmax(logits[0])
    print(f"  Top-1 class: {top1_idx}")
    print(f"  Top-1 prob:  {torch.softmax(output, dim=1)[0, top1_idx].item():.6f}")
    top5_indices = np.argsort(logits[0])[-5:][::-1]
    print(f"  Top-5 classes: {top5_indices}")
    print(f"  Avg latency: {latencies*1000:.2f} ms per sample")

    if args.dump:
        np.save(args.dump, logits)
        print(f"Logits saved to {args.dump}")

    return logits


if __name__ == "__main__":
    main()
