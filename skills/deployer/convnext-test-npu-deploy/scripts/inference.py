#!/usr/bin/env python3
"""ConvNeXt image classification inference on CPU and NPU (Ascend)."""

import os
import sys
import json
import time
import torch
import numpy as np
from PIL import Image

# Check NPU availability
NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()

MODEL_NAME = "test_convnext.r160_in1k"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "timm", "test_convnext___r160_in1k")
TEST_IMAGE = os.path.join(MODEL_PATH, "test", "test_owl.jpg")


def load_model(device="cpu"):
    """Load model on specified device."""
    import timm

    # Check for local checkpoint
    ckpt_path = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(ckpt_path):
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            checkpoint_path=ckpt_path,
        )
    else:
        model = timm.create_model(MODEL_NAME, pretrained=True)

    model = model.to(device)
    model.eval()
    return model


def preprocess(image_path, data_config):
    """Preprocess image using timm transforms."""
    import timm
    img = Image.open(image_path).convert("RGB")
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms(img).unsqueeze(0)  # add batch dimension


def run_inference(model, input_tensor, device, num_runs=5, warmup=3):
    """Run inference and measure latency."""
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Timed runs
    torch.cuda.synchronize() if device == "cuda" else (
        torch.npu.synchronize() if device == "npu" else None
    )
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
    torch.cuda.synchronize() if device == "cuda" else (
        torch.npu.synchronize() if device == "npu" else None
    )
    end = time.perf_counter()

    avg_latency = (end - start) / num_runs * 1000  # ms
    return output, avg_latency


def main():
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}")

    # Check model path
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Load model config
    import timm
    model = load_model("cpu")
    data_config = timm.data.resolve_model_data_config(model)
    print(f"\nInput config: {json.dumps(data_config, indent=2)}")

    # Load and preprocess test image
    if os.path.exists(TEST_IMAGE):
        print(f"\nTest image: {TEST_IMAGE}")
        input_tensor = preprocess(TEST_IMAGE, data_config)
    else:
        print("\nNo test image found, using random input.")
        input_tensor = torch.randn(1, 3, data_config["input_size"][1],
                                   data_config["input_size"][2])

    print(f"Input shape: {input_tensor.shape}")
    print(f"Input device: {input_tensor.device}")

    # ===== CPU Inference =====
    print(f"\n{'─'*40}")
    print("CPU Inference")
    print(f"{'─'*40}")
    cpu_model = model.to("cpu")
    cpu_model.eval()
    cpu_output, cpu_latency = run_inference(cpu_model, input_tensor, "cpu")
    cpu_probs = torch.softmax(cpu_output, dim=1)
    cpu_top5 = torch.topk(cpu_probs, k=5)
    print(f"CPU latency: {cpu_latency:.2f} ms / sample")
    print(f"CPU output shape: {cpu_output.shape}")
    print(f"CPU Top-5 probabilities: {cpu_top5.values[0].tolist()}")
    print(f"CPU Top-5 indices: {cpu_top5.indices[0].tolist()}")

    # ===== NPU Inference =====
    if NPU_AVAILABLE:
        print(f"\n{'─'*40}")
        print("NPU Inference")
        print(f"{'─'*40}")
        npu_model = load_model("npu")
        npu_model.eval()
        npu_output, npu_latency = run_inference(npu_model, input_tensor, "npu")
        npu_probs = torch.softmax(npu_output.cpu(), dim=1)
        npu_top5 = torch.topk(npu_probs, k=5)
        print(f"NPU latency: {npu_latency:.2f} ms / sample")
        print(f"NPU output shape: {npu_output.shape}")
        print(f"NPU Top-5 probabilities: {npu_top5.values[0].tolist()}")
        print(f"NPU Top-5 indices: {npu_top5.indices[0].tolist()}")

        # ===== Result Summary =====
        print(f"\n{'='*60}")
        print("Result Summary")
        print(f"{'='*60}")
        print(f"CPU latency:  {cpu_latency:.2f} ms")
        print(f"NPU latency:  {npu_latency:.2f} ms")
        print(f"Speedup:      {cpu_latency / npu_latency:.2f}x")
    else:
        print("\n[INFO] NPU not available. CPU inference only.")

    # Save results
    results = {
        "model": MODEL_NAME,
        "cpu_latency_ms": round(cpu_latency, 2),
        "cpu_top5_probs": [round(float(v), 6) for v in cpu_top5.values[0]],
        "cpu_top5_indices": [int(i) for i in cpu_top5.indices[0]],
    }
    if NPU_AVAILABLE:
        results.update({
            "npu_latency_ms": round(npu_latency, 2),
            "npu_top5_probs": [round(float(v), 6) for v in npu_top5.values[0]],
            "npu_top5_indices": [int(i) for i in npu_top5.indices[0]],
            "speedup": round(cpu_latency / npu_latency, 2),
        })

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "inference_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    if NPU_AVAILABLE:
        import gc
        gc.collect()
        torch.npu.empty_cache()

    return results


if __name__ == "__main__":
    main()
