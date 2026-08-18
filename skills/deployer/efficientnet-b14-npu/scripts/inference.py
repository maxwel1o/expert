#!/usr/bin/env python3
"""Unified NPU inference script for timm EfficientNet classification models.

Usage: python3 inference.py <model_name> [--device cpu|npu]
"""
import argparse
import json
import os
import sys
import time

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import requests
from io import BytesIO

from modelscope.hub.snapshot_download import snapshot_download


def get_model_name(url_or_name: str) -> str:
    """Extract timm model name from URL or direct name."""
    name = url_or_name.strip()
    if "modelscope.cn/models/timm/" in name:
        name = name.rsplit("timm/", 1)[-1]
    name = name.split("?")[0].strip("/")
    return name


def download_from_modelscope(model_name: str, cache_dir: str = None) -> str:
    """Download model weights from ModelScope."""
    repo_id = f"timm/{model_name}"
    print(f"[INFO] Downloading from ModelScope: {repo_id}")
    model_path = snapshot_download(repo_id, cache_dir=cache_dir)
    print(f"[INFO] Downloaded to: {model_path}")
    return model_path


def load_state_dict(model_path: str):
    """Load state dict from downloaded model files."""
    pytorch_file = os.path.join(model_path, "pytorch_model.bin")
    safetensors_file = os.path.join(model_path, "model.safetensors")

    if os.path.exists(pytorch_file):
        print(f"[INFO] Loading weights from: {pytorch_file}")
        state_dict = torch.load(pytorch_file, map_location="cpu", weights_only=True)
    elif os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        print(f"[INFO] Loading safetensors from: {safetensors_file}")
        state_dict = load_file(safetensors_file, device="cpu")
    else:
        raise FileNotFoundError(f"No weight file found in {model_path}")

    # Handle timm export format where weights are under 'model' key
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    return state_dict


def run_inference(model_name: str, device: str = "cpu"):
    """Run inference for a single model on specified device."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "modelscope_cache")
    output_dir = os.path.join(base_dir, "outputs", model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[INFO] Running inference for: {model_name}")
    print(f"[INFO] Device: {device}")
    print(f"{'='*60}")

    # Download model from ModelScope
    model_path = download_from_modelscope(model_name, cache_dir=cache_dir)

    # Read config for input size
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        input_size = config.get("input_size", [3, 224, 224])
        img_size = input_size[-1] if len(input_size) == 3 else input_size[-1]
    else:
        img_size = 224

    print(f"[INFO] Input image size: {img_size}")

    # Load model with classifier
    print(f"[INFO] Creating timm model: {model_name}")
    model = timm.create_model(model_name, pretrained=False)
    state_dict = load_state_dict(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print(f"[INFO] Model loaded on {device}")

    # Get image transform
    data_config = resolve_data_config({}, model=model)
    print(f"[INFO] Data config: {data_config}")
    transform = create_transform(**data_config)

    # Prepare test input
    try:
        url = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/cats.png"
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        print(f"[INFO] Using test image from URL, size: {img.size}")
    except Exception:
        img = Image.new("RGB", (img_size, img_size), color=(128, 128, 128))
        print(f"[INFO] Using synthetic test image")

    input_tensor = transform(img).unsqueeze(0)

    if device == "npu":
        input_tensor = input_tensor.to(device)

    print(f"[INFO] Input tensor shape: {list(input_tensor.shape)}")

    # Warm-up
    print("[INFO] Warm-up inference...")
    with torch.no_grad():
        _ = model(input_tensor)

    # Timed inference
    print(f"[INFO] Running timed inference ({10} runs)...")
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = model(input_tensor)
    end = time.time()
    avg_time = (end - start) / 10

    output_cpu = output.cpu()
    probs = torch.nn.functional.softmax(output_cpu[0], dim=0)
    top5_values, top5_indices = torch.topk(probs, 5)

    # Get ImageNet class names
    try:
        url_labels = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        resp_labels = requests.get(url_labels, timeout=5)
        labels = [l.strip() for l in resp_labels.text.strip().split("\n")]
    except Exception:
        labels = [str(i) for i in range(1000)]

    results = {
        "model_name": model_name,
        "device": device,
        "avg_inference_time_s": round(avg_time, 4),
        "avg_inference_time_ms": round(1000 * avg_time, 2),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output_cpu.shape),
        "top5_predictions": []
    }

    print(f"\n[RESULTS] Model: {model_name}")
    print(f"[RESULTS] Device: {device}")
    print(f"[RESULTS] Avg inference time: {avg_time:.4f}s ({1000*avg_time:.2f}ms)")
    print(f"[RESULTS] Output shape: {list(output_cpu.shape)}")
    print(f"[RESULTS] Top-5 predictions:")
    for i in range(5):
        idx = top5_indices[i].item()
        label = labels[idx] if idx < len(labels) else str(idx)
        val = top5_values[i].item()
        print(f"  {i+1}. class {idx} ({label}): {val:.6f}")
        results["top5_predictions"].append({
            "rank": i + 1, "class": idx, "label": label, "probability": round(val, 6)
        })

    # Save output tensor
    output_path = os.path.join(output_dir, f"output_{device}.pt")
    torch.save(output_cpu, output_path)
    print(f"[INFO] Output saved to: {output_path}")

    # Save results JSON
    results_path = os.path.join(output_dir, f"results_{device}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved to: {results_path}")

    # Save timing
    timing_path = os.path.join(output_dir, f"timing_{device}.txt")
    with open(timing_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Average inference time over 10 runs: {avg_time:.4f}s ({1000*avg_time:.2f}ms)\n")
    print(f"[INFO] Timing saved to: {timing_path}")

    # Free memory
    del model, input_tensor, output
    if device == "npu":
        torch.npu.empty_cache()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run timm model inference on CPU or NPU")
    parser.add_argument("model_name", help="Model name or ModelScope URL")
    parser.add_argument("--device", default="cpu", choices=["cpu", "npu"], help="Device to run on")
    args = parser.parse_args()

    model_name = get_model_name(args.model_name)
    run_inference(model_name, args.device)


if __name__ == "__main__":
    main()
