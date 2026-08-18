#!/usr/bin/env python3
"""SPNASNet_100 Inference on CPU and NPU."""

import argparse
import json
import os
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from timm.data import resolve_model_data_config, create_transform

import warnings
warnings.filterwarnings("ignore")

MODEL_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".modelscope_cache/models/timm/spnasnet_100___rmsp_in1k"
)


def load_local_model(model_name, cache_dir=None):
    """Load model from local cache (ModelScope download)."""
    if cache_dir is None:
        cache_dir = MODEL_CACHE

    model = timm.create_model(model_name, pretrained=False)
    model = model.eval()

    safetensors_path = os.path.join(cache_dir, "model.safetensors")
    pt_path = os.path.join(cache_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        print(f"  Loaded weights from: {safetensors_path}")
    elif os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        print(f"  Loaded weights from: {pt_path}")
    else:
        print("  Local weights not found, trying HF download...")
        model = timm.create_model(model_name, pretrained=True)
        return model

    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            print(f"  Skipping key: {k}")

    model.load_state_dict(filtered, strict=False)
    print(f"  Loaded {len(filtered)}/{len(model_state)} keys")
    return model


def load_image(image_path=None):
    """Load a test image for inference."""
    if image_path and Path(image_path).exists():
        return Image.open(image_path).convert("RGB")

    import numpy as np
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def run_inference(device, model_name="spnasnet_100.rmsp_in1k",
                  image_path=None, num_runs=10, cache_dir=None):
    """Run inference on specified device."""
    print(f"\n=== Running on {device.upper()} ===")

    print(f"Loading model '{model_name}'...")
    model = load_local_model(model_name, cache_dir)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if device == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            print("ERROR: NPU not available!")
            sys.exit(1)
        model = model.to("npu")
        torch.npu.synchronize()
    elif device == "cpu":
        model = model.to("cpu")

    data_config = resolve_model_data_config(model)
    transforms = create_transform(**data_config, is_training=False)
    print(f"Config: {data_config}")

    img = load_image(image_path)
    input_tensor = transforms(img).unsqueeze(0)

    if device == "npu":
        input_tensor = input_tensor.to("npu")
        torch.npu.synchronize()
    else:
        input_tensor = input_tensor.to("cpu")

    # Warmup
    for _ in range(3):
        _ = model(input_tensor)
    if device == "npu":
        torch.npu.synchronize()

    # Timed inference
    times = []
    outputs = None
    for i in range(num_runs):
        if device == "npu":
            torch.npu.synchronize()
        start = time.perf_counter()
        outputs = model(input_tensor)
        if device == "npu":
            torch.npu.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    probs = F.softmax(outputs[0], dim=0)
    top5_values, top5_indices = torch.topk(probs * 100, k=5)

    try:
        from timm.data import IMAGENET_1k
        id2label = IMAGENET_1k
    except Exception:
        id2label = {i: f"class_{i}" for i in range(1000)}

    results = {
        "device": device,
        "model": model_name,
        "num_runs": num_runs,
        "avg_time_ms": sum(times) / len(times) * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "top5": [
            {
                "class": int(top5_indices[i]),
                "label": id2label.get(int(top5_indices[i]), f"class_{int(top5_indices[i])}"),
                "probability": float(top5_values[i]),
            }
            for i in range(5)
        ],
    }

    logits = outputs[0]
    results["logits_stats"] = {
        "mean": float(logits.mean()),
        "std": float(logits.std()),
        "min": float(logits.min()),
        "max": float(logits.max()),
    }

    print(f"\nResults ({device.upper()}):")
    print(f"  Avg inference time: {results['avg_time_ms']:.2f} ms")
    print(f"  Logits: mean={results['logits_stats']['mean']:.4f}, "
          f"std={results['logits_stats']['std']:.4f}")
    for i, pred in enumerate(results["top5"]):
        print(f"  Top-{i+1}: {pred['label']} ({pred['probability']:.2f}%)")

    return logits.cpu(), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spnasnet_100.rmsp_in1k")
    parser.add_argument("--image", default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    cache_dir = args.cache_dir or MODEL_CACHE

    # Run CPU inference
    cpu_logits, cpu_results = run_inference("cpu", args.model, args.image,
                                            args.runs, cache_dir)

    print("\n" + "=" * 60)

    # Run NPU inference
    npu_logits, npu_results = run_inference("npu", args.model, args.image,
                                            args.runs, cache_dir)

    # Save results
    output = {"cpu": cpu_results, "npu": npu_results}
    Path("results").mkdir(exist_ok=True)
    torch.save({"cpu": cpu_logits, "npu": npu_logits.cpu()}, "results/logits.pt")
    with open("results/inference_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to results/inference_results.json")
    print(f"\n{'='*60}")
    print("CPU vs NPU comparison summary:")
    print(f"  CPU avg time: {cpu_results['avg_time_ms']:.2f} ms")
    print(f"  NPU avg time: {npu_results['avg_time_ms']:.2f} ms")
    print(f"  Speedup: {cpu_results['avg_time_ms'] / npu_results['avg_time_ms']:.2f}x")


if __name__ == "__main__":
    main()
