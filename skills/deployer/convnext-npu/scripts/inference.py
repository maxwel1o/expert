#!/usr/bin/env python3
"""
Inference script for ConvNeXt model on CPU or NPU.

Usage:
    python3 inference.py --device cpu    # CPU inference
    python3 inference.py --device npu    # NPU inference (Ascend)
"""

import argparse
import os
import time
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def parse_args():
    parser = argparse.ArgumentParser(description="ConvNeXt NPU Inference")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"],
                        help="Device to run inference on (cpu or npu)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. convnext_nano.in12k_ft_in1k)")
    parser.add_argument("--image", type=str, default="test.jpg",
                        help="Input image path")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to local model weights (.safetensors)")
    return parser.parse_args()


def load_model_weights(model, model_name, model_path=""):
    """Load model weights from local file or ModelScope cache."""
    if model_path and os.path.exists(model_path):
        from safetensors.torch import load_file
        print(f"Loading weights from: {model_path}")
        sd = load_file(model_path)
        model.load_state_dict(sd, strict=False)
        print("Weights loaded successfully")
        return True

    # Try ModelScope cache
    modelscope_name = model_name.replace(".", "___")
    cache_paths = [
        f"/opt/atomgit/convnext_workspace/modelscope_cache/timm/{modelscope_name}/model.safetensors",
        f"/opt/atomgit/convformer_workspace/modelscope_cache/timm/{modelscope_name}/model.safetensors",
    ]
    for p in cache_paths:
        if os.path.exists(p):
            from safetensors.torch import load_file
            print(f"Loading weights from ModelScope cache: {p}")
            sd = load_file(p)
            model.load_state_dict(sd, strict=False)
            print("Weights loaded successfully")
            return True

    print("ERROR: Could not find model weights. Use --model-path or download from ModelScope first.")
    return False


def main():
    args = parse_args()
    device = args.device

    print(f"Loading model: {args.model}")
    model = timm.create_model(args.model, pretrained=False)
    model.eval()

    if not load_model_weights(model, args.model, args.model_path):
        return

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
