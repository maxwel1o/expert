#!/usr/bin/env python3
"""WebSSL MAE NPU vs CPU Accuracy Verification Script (通用版)"""
import os
import sys
import json
import argparse

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


def run_inference(model, processor, image, device):
    """Run inference on specified device and return last_hidden_state."""
    inputs = processor(images=image, return_tensors="pt")
    if device in ("npu", "cuda"):
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if device == "npu":
        torch.npu.synchronize()

    return outputs.last_hidden_state.cpu()


def compare_outputs(cpu_output, npu_output):
    """Compare CPU and NPU outputs and return metrics."""
    diff = torch.abs(cpu_output - npu_output)
    max_abs_error = float(diff.max().item())
    mean_abs_error = float(diff.mean().item())
    mean_relative_error = float((diff / (torch.abs(cpu_output) + 1e-8)).mean().item()) * 100.0
    l2_relative_error = float(torch.norm(npu_output - cpu_output).item() / torch.norm(cpu_output).item()) * 100.0
    norm_relative_error = abs(float(npu_output.norm().item()) - float(cpu_output.norm().item())) / float(cpu_output.norm().item()) * 100.0

    cpu_flat = cpu_output.flatten()
    npu_flat = npu_output.flatten()
    cos_sim = float(torch.nn.functional.cosine_similarity(cpu_flat.unsqueeze(0), npu_flat.unsqueeze(0)).item())

    return {
        "max_abs_error": round(max_abs_error, 6),
        "mean_abs_error": round(mean_abs_error, 6),
        "mean_relative_error_percent": round(mean_relative_error, 4),
        "l2_relative_error_percent": round(l2_relative_error, 4),
        "norm_relative_error_percent": round(norm_relative_error, 4),
        "cosine_similarity": round(cos_sim, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="WebSSL MAE NPU Accuracy Verification")
    parser.add_argument("--cache-dir", type=str, default="model_cache", help="Local cache dir for weights")
    parser.add_argument("--threshold", type=float, default=1.0, help="Relative error threshold in %")
    args = parser.parse_args()

    cfg = get_model_config()
    model_name = cfg["model_name"]
    resolution = cfg["resolution"]

    print("=" * 70)
    print(f"WebSSL MAE Accuracy Verification: NPU vs CPU")
    print(f"Model: {model_name}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Threshold: {args.threshold}%")
    print("=" * 70)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    image = create_test_image(resolution)

    print(f"\n[1/5] Loading processor and model...")
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
    print(f"  Model loaded: {model_name}")

    print(f"\n[2/5] Running inference on NPU...")
    model_npu = model.cuda()
    npu_output = run_inference(model_npu, processor, image, "npu")
    print(f"  NPU output shape: {npu_output.shape}")
    print(f"  NPU output norm: {npu_output.norm().item():.4f}")

    del model_npu
    torch.npu.empty_cache()

    print(f"\n[3/5] Running inference on CPU (baseline)...")
    model_cpu = model.cpu()
    cpu_output = run_inference(model_cpu, processor, image, "cpu")
    print(f"  CPU output shape: {cpu_output.shape}")
    print(f"  CPU output norm: {cpu_output.norm().item():.4f}")

    print(f"\n[4/5] Comparing outputs...")
    metrics = compare_outputs(cpu_output, npu_output)
    print(f"  Max absolute error: {metrics['max_abs_error']}")
    print(f"  Mean absolute error: {metrics['mean_abs_error']}")
    print(f"  Mean relative error: {metrics['mean_relative_error_percent']}%")
    print(f"  L2 relative error: {metrics['l2_relative_error_percent']}%")
    print(f"  Norm relative error: {metrics['norm_relative_error_percent']}%")
    print(f"  Cosine similarity: {metrics['cosine_similarity']}")

    passed = metrics["l2_relative_error_percent"] < args.threshold
    status = "PASS" if passed else "FAIL"
    print(f"\n[5/5] Result: {status} (threshold: {args.threshold}%, metric: L2 relative error)")

    results = {
        "model_name": model_name,
        "resolution": resolution,
        "cpu_output_norm": float(cpu_output.norm().item()),
        "npu_output_norm": float(npu_output.norm().item()),
        **metrics,
        "threshold_percent": args.threshold,
        "status": status,
    }

    report_path = "accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAccuracy report saved to {report_path}")

    if results["status"] == "FAIL":
        sys.exit(1)

    print("\nVerification completed!")


if __name__ == "__main__":
    main()
