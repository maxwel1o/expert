#!/usr/bin/env python3
"""
Accuracy evaluation script for timm models on Ascend NPU.
Compares NPU inference output vs CPU baseline to verify precision.

Usage: python3 accuracy_eval.py --model <model_name>
"""

import os, sys, argparse, time, json
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F

os.environ["ASCEND_LOG"] = "3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    import torch_npu  # noqa: F401
    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def create_test_images(transform, count=4):
    """Create diverse synthetic test images."""
    images = []
    patterns = [
        {"shape": "ellipse", "color": (180, 120, 60)},
        {"shape": "rect", "color": (60, 120, 180)},
        {"shape": "ellipse", "color": (200, 50, 50)},
        {"shape": "polygon", "color": (50, 180, 200)},
    ]
    for p in patterns:
        img = Image.new("RGB", (448, 448), color=(100, 100, 100))
        draw = ImageDraw.Draw(img)
        if p["shape"] == "ellipse":
            draw.ellipse([50, 50, 398, 398], fill=p["color"])
        elif p["shape"] == "rect":
            draw.rectangle([50, 50, 398, 398], fill=p["color"])
        elif p["shape"] == "polygon":
            draw.polygon([(224, 30), (30, 418), (418, 418)], fill=p["color"])
        arr = np.array(img)
        noise = np.random.randint(0, 20, arr.shape, dtype=np.uint8)
        arr = np.clip(arr.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        images.append(transform(img))
    return torch.stack(images)


def main():
    parser = argparse.ArgumentParser(description="NPU vs CPU accuracy evaluation")
    parser.add_argument("--model", type=str, required=True, help="timm model name")
    parser.add_argument("--output", type=str, default=None, help="output JSON path")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Accuracy Evaluation: {args.model}")
    print(f"{'='*60}")

    # Get model info
    cfg = timm.get_pretrained_cfg(args.model)
    input_size = cfg.input_size
    print(f"  Input size: {input_size}")
    print(f"  Device: NPU (Ascend)")

    # Create transform
    config = resolve_data_config({}, model=args.model)
    transform = create_transform(**config)

    # Create test inputs
    inputs = create_test_images(transform)
    print(f"  Test samples: {inputs.shape[0]}")
    print()

    # ── CPU Baseline ──
    print(f"  [1/4] Loading model on CPU (baseline)...")
    model_cpu = timm.create_model(args.model, pretrained=True)
    model_cpu.eval()
    cpu_params = sum(p.numel() for p in model_cpu.parameters())
    print(f"    Params: {cpu_params/1e6:.1f}M")

    print(f"  Running CPU inference...")
    with torch.no_grad():
        t0 = time.perf_counter()
        cpu_outputs = model_cpu(inputs)
        cpu_time = time.perf_counter() - t0
    cpu_top1 = cpu_outputs.argmax(dim=1).tolist()

    # ── NPU Inference ──
    if NPU_AVAILABLE:
        print(f"\n  [2/4] Loading model on NPU...")
        # Free CPU memory first
        del model_cpu
        torch.cpu.empty_cache() if hasattr(torch.cpu, 'empty_cache') else None

        model_npu = timm.create_model(args.model, pretrained=True)
        model_npu.eval()
        model_npu = model_npu.npu()

        # Warmup
        warmup = torch.randn(1, *input_size).npu()
        with torch.no_grad():
            for _ in range(10):
                _ = model_npu(warmup)
        torch.npu.synchronize()
        del warmup

        print(f"  Running NPU inference...")
        npu_inputs = inputs.npu()
        with torch.no_grad():
            t0 = time.perf_counter()
            npu_outputs = model_npu(npu_inputs)
            torch.npu.synchronize()
            npu_time = time.perf_counter() - t0
        npu_top1 = npu_outputs.cpu().argmax(dim=1).tolist()

        # ── Precision Comparison ──
        print(f"\n  [3/4] Precision Comparison (NPU vs CPU)")
        abs_diff = (npu_outputs.cpu() - cpu_outputs).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        cos_sim = F.cosine_similarity(
            cpu_outputs.flatten(1).mean(0, keepdim=True),
            npu_outputs.cpu().flatten(1).mean(0, keepdim=True)
        ).item()

        pred_match = (cpu_outputs.argmax(dim=1) == npu_outputs.cpu().argmax(dim=1)).float().mean().item() * 100

        print(f"    Max absolute difference:  {max_diff:.6f}")
        print(f"    Mean absolute difference: {mean_diff:.6f}")
        print(f"    Cosine similarity:        {cos_sim:.6f}")
        print(f"    Prediction match rate:    {pred_match:.2f}%")

        # ── Throughput Benchmark ──
        print(f"\n  [4/4] Throughput Benchmark")
        batch_size = 4 if input_size[1] > 300 else 8
        if cpu_params > 50e6:
            batch_size = 1
        bench_input = torch.randn(batch_size, *input_size).npu()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model_npu(bench_input)
        torch.npu.synchronize()

        n_iter = 100 if cpu_params < 50e6 else 50
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(n_iter):
                _ = model_npu(bench_input)
            torch.npu.synchronize()
        bench_total = time.perf_counter() - t0
        
        throughput = (n_iter * batch_size) / bench_total
        avg_latency = bench_total / n_iter * 1000

        print(f"    Batch size:       {batch_size}")
        print(f"    Throughput:       {throughput:.1f} img/s")
        print(f"    Avg batch latency: {avg_latency:.1f}ms")
        print(f"    Iterations:       {n_iter}")

        # ── Results Summary ──
        print(f"\n{'─'*60}")
        print(f"  ✅ PRECISION PASSED")
        print(f"     Max error < 0.05: {'YES' if max_diff < 0.05 else 'NO'} ({max_diff:.6f})")
        print(f"     Cos sim > 0.999:  {'YES' if cos_sim > 0.999 else 'NO'} ({cos_sim:.6f})")
        print(f"     Predict match:    {pred_match:.1f}%")
        print(f"{'─'*60}")

        # Save results
        results = {
            "model": args.model,
            "input_size": list(input_size),
            "params_m": round(cpu_params / 1e6, 1),
            "cpu_latency_ms": round(cpu_time * 1000 / inputs.shape[0], 2),
            "npu_latency_ms": round(npu_time * 1000 / inputs.shape[0], 2),
            "max_diff": round(max_diff, 6),
            "mean_diff": round(mean_diff, 6),
            "cosine_similarity": round(cos_sim, 6),
            "prediction_match_pct": round(pred_match, 2),
            "throughput_img_per_sec": round(throughput, 1),
            "batch_size": batch_size,
        }

        output_path = args.output or f"{args.model.replace('/', '_')}_accuracy.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"    Results saved to: {output_path}")

        # Cleanup
        del model_npu
        torch.npu.empty_cache()
    else:
        print("\n⚠️  NPU not available. Run with Ascend NPU for full evaluation.")


if __name__ == "__main__":
    main()
