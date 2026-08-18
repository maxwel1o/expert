#!/usr/bin/env python3
"""
CPU vs NPU precision comparison script for ConvNeXt model.

Usage:
    python3 compare_cpu_npu.py --model convformer_b36.sail_in1k_384 --image test.jpg
"""

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def parse_args():
    parser = argparse.ArgumentParser(description="CPU vs NPU Precision Comparison")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name")
    parser.add_argument("--image", type=str, default="test.jpg",
                        help="Input image path")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to local model weights (.safetensors)")
    return parser.parse_args()


def load_model_weights(model, model_name, model_path=""):
    """Load model weights from local file or ModelScope cache."""
    if model_path and os.path.exists(model_path):
        from safetensors.torch import load_file
        sd = load_file(model_path)
        model.load_state_dict(sd, strict=False)
        return True
    modelscope_name = model_name.replace(".", "___")
    cache_paths = [
        f"/opt/atomgit/convnext_workspace/modelscope_cache/timm/{modelscope_name}/model.safetensors",
        f"/opt/atomgit/convformer_workspace/modelscope_cache/timm/{modelscope_name}/model.safetensors",
    ]
    for p in cache_paths:
        if os.path.exists(p):
            from safetensors.torch import load_file
            sd = load_file(p)
            model.load_state_dict(sd, strict=False)
            return True
    return False


def main():
    args = parse_args()
    model_name = args.model
    img_path = args.image

    print(f"Model: {model_name}")
    print(f"Image: {img_path}")
    print()

    # Load image
    img = Image.open(img_path).convert("RGB")

    # ====== CPU Inference ======
    print("=" * 50)
    print("CPU Inference")
    print("=" * 50)

    model_cpu = timm.create_model(model_name, pretrained=False)
    model_cpu.eval()
    if not load_model_weights(model_cpu, model_name, args.model_path):
        print("ERROR: Could not load model weights")
        return

    config = resolve_data_config({}, model=model_cpu)
    transform = create_transform(**config)
    input_cpu = transform(img).unsqueeze(0)

    with torch.no_grad():
        start = time.time()
        output_cpu = model_cpu(input_cpu)
        cpu_time = time.time() - start

    cpu_probs = F.softmax(output_cpu[0], dim=0)
    cpu_top5 = cpu_probs.topk(5)

    print(f"Inference time: {cpu_time:.4f}s")
    print(f"Top-5:")
    for i in range(5):
        print(f"  {i+1}. class={cpu_top5.indices[i].item():5d}  prob={cpu_top5.values[i].item():.6f}")
    print()

    # ====== NPU Inference ======
    print("=" * 50)
    print("NPU Inference")
    print("=" * 50)

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("ERROR: NPU not available!")
        return

    model_npu = timm.create_model(model_name, pretrained=False)
    model_npu = model_npu.npu()
    model_npu.eval()
    if not load_model_weights(model_npu, model_name, args.model_path):
        print("ERROR: Could not load model weights for NPU")
        return

    input_npu = transform(img).unsqueeze(0).npu()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model_npu(input_npu)
    torch.npu.synchronize()

    with torch.no_grad():
        start = time.time()
        output_npu = model_npu(input_npu)
        torch.npu.synchronize()
        npu_time = time.time() - start

    npu_probs = F.softmax(output_npu[0], dim=0)
    npu_top5 = npu_probs.topk(5)

    print(f"Inference time: {npu_time:.4f}s")
    print(f"Top-5:")
    for i in range(5):
        print(f"  {i+1}. class={npu_top5.indices[i].item():5d}  prob={npu_top5.values[i].item():.6f}")
    print()

    # ====== Comparison ======
    print("=" * 50)
    print("Precision Comparison")
    print("=" * 50)

    cpu_out = output_cpu[0].cpu()
    npu_out = output_npu[0].cpu()

    mae = torch.abs(cpu_out - npu_out).mean().item()
    mse = torch.pow(cpu_out - npu_out, 2).mean().item()
    max_err = torch.abs(cpu_out - npu_out).max().item()
    cos_sim = F.cosine_similarity(cpu_out.unsqueeze(0), npu_out.unsqueeze(0)).item()

    # Compute relative error on top-100 logits by magnitude (avoid near-zero division)
    cpu_mag = cpu_out.abs()
    top100_idx = cpu_mag.argsort(descending=True)[:100]
    rel_err_top100 = (torch.abs(cpu_out[top100_idx] - npu_out[top100_idx]) / (torch.abs(cpu_out[top100_idx]) + 1e-8)).mean().item() * 100

    # Also compute normalized MAE (divide by max absolute value)
    max_abs = max(cpu_out.abs().max().item(), npu_out.abs().max().item())
    norm_mae = mae / max_abs * 100 if max_abs > 0 else 0

    cpu_top1_idx = cpu_top5.indices[0].item()
    npu_top1_idx = npu_top5.indices[0].item()
    cpu_top1_prob = cpu_top5.values[0].item()
    npu_top1_prob = npu_top5.values[0].item()

    cpu_top5_set = set(cpu_top5.indices.tolist())
    npu_top5_set = set(npu_top5.indices.tolist())
    top5_overlap = len(cpu_top5_set & npu_top5_set)

    print(f"MAE:               {mae:.10f}")
    print(f"MSE:               {mse:.10f}")
    print(f"Max Absolute Error: {max_err:.10f}")
    print(f"Cosine Similarity: {cos_sim:.10f}")
    print(f"Mean Relative Error (top-100): {rel_err_top100:.6f}%")
    print(f"Normalized MAE:    {norm_mae:.6f}%")
    print(f"Top-5 Overlap:     {top5_overlap}/5")
    print(f"Top-1 Match:       {cpu_top1_idx == npu_top1_idx}")
    print(f"  CPU Top-1:       class={cpu_top1_idx:5d}  prob={cpu_top1_prob:.6f}")
    print(f"  NPU Top-1:       class={npu_top1_idx:5d}  prob={npu_top1_prob:.6f}")
    print()

    # Conclusion
    passed = rel_err_top100 < 1.0
    print(f"Requirement: NPU vs CPU error < 1%")
    print(f"Result: {rel_err_top100:.6f}% - {'PASSED' if passed else 'FAILED'}")
    print()

    # Save results
    results = {
        "model": model_name,
        "mae": round(mae, 10),
        "mse": round(mse, 10),
        "max_abs_error": round(max_err, 10),
        "cosine_similarity": round(cos_sim, 10),
        "mean_relative_error_pct": round(rel_err_top100, 6),
        "normalized_mae_pct": round(norm_mae, 6),
        "top5_overlap": top5_overlap,
        "top1_match": cpu_top1_idx == npu_top1_idx,
        "cpu_top1": {"class": cpu_top1_idx, "prob": round(cpu_top1_prob, 6)},
        "npu_top1": {"class": npu_top1_idx, "prob": round(npu_top1_prob, 6)},
        "cpu_time_s": round(cpu_time, 4),
        "npu_time_s": round(npu_time, 4),
        "speedup": round(cpu_time / npu_time, 2) if npu_time > 0 else 0,
        "passed": passed
    }

    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved to comparison_results.json")


if __name__ == "__main__":
    main()
