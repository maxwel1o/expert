#!/usr/bin/env python3
"""WebSSL MAE NPU Inference Script (通用版)"""
import os
import sys
import time
import argparse
import json

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np


def get_model_config():
    """Return model-specific configuration based on MODEL_NAME env var or default."""
    model_name = os.environ.get("MODEL_NAME", "facebook/webssl-mae300m-full2b-224")
    configs = {
        "facebook/webssl-mae300m-full2b-224": {"resolution": 224, "params": "300M"},
        "facebook/webssl-mae700m-full2b-224": {"resolution": 224, "params": "700M"},
        "facebook/webssl-mae1b-full2b-224": {"resolution": 224, "params": "1B"},
    }
    cfg = configs.get(model_name, {"resolution": 224, "params": "unknown"})
    cfg["model_name"] = model_name
    return cfg


def create_test_image(resolution=224, seed=42):
    """Create a reproducible synthetic RGB image."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (resolution, resolution, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description="WebSSL MAE NPU Inference")
    parser.add_argument("--device", type=str, default="npu", choices=["npu", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--save-output", type=str, default=None, help="Path to save output tensor (.pt)")
    parser.add_argument("--cache-dir", type=str, default="model_cache", help="Local cache dir for weights")
    args = parser.parse_args()

    cfg = get_model_config()
    model_name = cfg["model_name"]
    resolution = cfg["resolution"]

    print("=" * 70)
    print(f"WebSSL MAE Inference on Ascend NPU")
    print(f"Model: {model_name}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Device: {args.device}")
    print("=" * 70)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    if args.device == "npu":
        device_name = torch.npu.get_device_name(0)
        print(f"NPU Device: {device_name}")
    print(f"torch version: {torch.__version__}")
    print(f"torch_npu version: {torch_npu.__version__}")

    print(f"\n[1/4] Loading processor and model...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")

    print(f"\n[2/4] Moving model to {args.device}...")
    if args.device in ("npu", "cuda"):
        model = model.cuda()
    else:
        model = model.cpu()
    print(f"  Model device: {next(model.parameters()).device}")

    print(f"\n[3/4] Preparing test image ({resolution}x{resolution})...")
    image = create_test_image(resolution)
    inputs = processor(images=image, return_tensors="pt")
    if args.device in ("npu", "cuda"):
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.cpu() for k, v in inputs.items()}
    print(f"  Input shape: {inputs['pixel_values'].shape}")

    print(f"\n[4/4] Running inference (warmup={args.warmup}, runs={args.runs})...")
    with torch.no_grad():
        for i in range(args.warmup):
            _ = model(**inputs)
        if args.device == "npu":
            torch.npu.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.runs):
            outputs = model(**inputs)
        if args.device == "npu":
            torch.npu.synchronize()
    total_time = time.time() - t0
    avg_time = total_time / args.runs

    last_hidden_state = outputs.last_hidden_state
    print(f"  Output shape: {last_hidden_state.shape}")
    print(f"  Output norm: {last_hidden_state.norm().item():.4f}")
    print(f"  Average inference time ({args.runs} runs): {avg_time*1000:.2f} ms")

    if args.device == "npu":
        mem_allocated = torch.npu.memory_allocated(0) / 1024**2
        mem_reserved = torch.npu.memory_reserved(0) / 1024**2
        print(f"  NPU Memory: allocated={mem_allocated:.1f}MB, reserved={mem_reserved:.1f}MB")

    if args.save_output:
        torch.save(last_hidden_state.cpu(), args.save_output)
        print(f"  Output saved to {args.save_output}")

    metrics = {
        "model_name": model_name,
        "resolution": resolution,
        "device": args.device,
        "output_shape": list(last_hidden_state.shape),
        "output_norm": float(last_hidden_state.norm().item()),
        "avg_latency_ms": round(avg_time * 1000, 4),
        "load_time_s": round(load_time, 2),
    }
    metrics_path = args.save_output.replace(".pt", "_metrics.json") if args.save_output else "inference_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
