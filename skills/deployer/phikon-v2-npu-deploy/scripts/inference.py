#!/usr/bin/env python3
"""phikon-v2 NPU inference script.

Model: owkin/phikon-v2 (Dinov2Model / ViT-L/16, 303M params)
Device: Ascend NPU (910B4)
"""
import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image


def get_device() -> str:
    """Auto-detect NPU/CUDA/CPU."""
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    model_path: str,
    use_fp16: bool = True,
    device: Optional[str] = None,
):
    """Load phikon-v2 model on target device."""
    if device is None:
        device = get_device()

    from transformers import AutoImageProcessor, Dinov2Model

    device_str = f"{device}:0" if device in ("npu", "cuda") else "cpu"
    torch_dtype = torch.float16 if (use_fp16 and device != "cpu") else torch.float32

    print(f"Loading model from: {model_path}")
    model = Dinov2Model.from_pretrained(model_path, torch_dtype=torch_dtype)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = model.to(device_str).eval()

    print(f"  device={device_str}, dtype={torch_dtype}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, processor, device_str


@torch.no_grad()
def extract_features(
    model,
    processor,
    image: Image.Image,
    device_str: str,
) -> torch.Tensor:
    """Extract CLS token features from a single image.

    Args:
        model: Dinov2Model loaded on NPU.
        processor: HuggingFace image processor.
        image: PIL Image (will be resized to 224x224).
        device_str: Target device string.

    Returns:
        CLS token embedding [1, 1024].
    """
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device_str)

    if device_str.startswith("npu"):
        pixel_values = pixel_values.half()
        with torch.autocast(device_type="npu", dtype=torch.float16):
            outputs = model(pixel_values=pixel_values)
    else:
        outputs = model(pixel_values=pixel_values)

    return outputs.pooler_output  # [1, 1024]


@torch.no_grad()
def extract_features_batch(
    model,
    processor,
    images: list[Image.Image],
    device_str: str,
) -> torch.Tensor:
    """Extract features from a batch of images."""
    pixel_values_list = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        pixel_values_list.append(inputs["pixel_values"])

    pixel_values = torch.cat(pixel_values_list, dim=0).to(device_str)

    if device_str.startswith("npu"):
        pixel_values = pixel_values.half()
        with torch.autocast(device_type="npu", dtype=torch.float16):
            outputs = model(pixel_values=pixel_values)
    else:
        outputs = model(pixel_values=pixel_values)

    return outputs.pooler_output  # [batch, 1024]


def benchmark(model, processor, device_str: str, batch_sizes: list[int] = None):
    """Run benchmark."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    print(f"\nBenchmark ({device_str}):")
    print(f"{'Batch':>6} | {'Latency':>10} | {'Throughput':>12}")
    print("-" * 32)

    for bs in batch_sizes:
        imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(bs)]
        pv_list = [processor(images=img, return_tensors="pt")["pixel_values"] for img in imgs]
        pixel_values = torch.cat(pv_list, dim=0).to(device_str)
        if device_str.startswith("npu"):
            pixel_values = pixel_values.half()

        # Warmup
        for _ in range(5):
            model(pixel_values=pixel_values)
        if device_str.startswith("npu"):
            torch.npu.synchronize()

        num_runs = 100 if bs <= 4 else 50
        start = time.perf_counter()
        for _ in range(num_runs):
            model(pixel_values=pixel_values)
        if device_str.startswith("npu"):
            torch.npu.synchronize()
        end = time.perf_counter()

        avg_ms = (end - start) / num_runs * 1000
        img_per_sec = bs / avg_ms * 1000
        print(f"{bs:>6} | {avg_ms:>8.2f} ms | {img_per_sec:>10.1f} img/s")


def main():
    parser = argparse.ArgumentParser(description="phikon-v2 NPU inference")
    parser.add_argument("--model-path", default="/opt/atomgit/phikon-v2-npu/model_files")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, processor, device_str = load_model(args.model_path, use_fp16=args.fp16)

    if args.benchmark:
        benchmark(model, processor, device_str)
        return

    if args.image:
        image = Image.open(args.image).convert("RGB")
        features = extract_features(model, processor, image, device_str)
        print(f"Features shape: {features.shape}")
        print(f"Feature norm: {features.norm().item():.4f}")
        print(f"First 10 dims: {features[0, :10].tolist()}")
    else:
        # Demo with random image
        print("No image provided; running demo with random image...")
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        features = extract_features(model, processor, image, device_str)
        print(f"Demo feature shape: {features.shape}")
        print(f"Feature norm: {features.norm().item():.4f}")
        print("✓ Inference successful")


if __name__ == "__main__":
    main()
