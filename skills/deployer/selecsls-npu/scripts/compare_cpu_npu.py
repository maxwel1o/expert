#!/usr/bin/env python3
"""Compare CPU vs NPU inference results for timm classification models."""
import argparse
import json
import gc
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="timm model name")
    parser.add_argument("--image", default="test_image.jpg", help="input image path")
    parser.add_argument("--output", default="compare_results.json")
    args = parser.parse_args()

    device_cpu = torch.device("cpu")
    device_npu = torch.device("npu")
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("NPU not available, cannot compare")
        return

    # Load image
    img = Image.open(args.image).convert("RGB")

    # ---------- CPU inference ----------
    print(f"=== CPU Inference: {args.model} ===")
    model_cpu = timm.create_model(args.model, pretrained=True)
    model_cpu = model_cpu.to(device_cpu)
    model_cpu = model_cpu.eval()

    data_config = timm.data.resolve_model_data_config(model_cpu)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_tensor = transforms(img).unsqueeze(0)

    with torch.no_grad():
        output_cpu = model_cpu(input_tensor)
    probs_cpu = F.softmax(output_cpu[0], dim=0)

    print(f"CPU top-5: {torch.topk(probs_cpu, k=5).indices.tolist()}")
    print(f"CPU top-5 probs: {[round(p.item(), 6) for p in torch.topk(probs_cpu, k=5).values]}")

    # Save CPU output
    cpu_output = {
        "logits": output_cpu[0].tolist(),
        "top5_indices": torch.topk(probs_cpu, k=5).indices.tolist(),
        "top5_probs": [round(p.item(), 6) for p in torch.topk(probs_cpu, k=5).values],
    }

    # Clean CPU model
    del model_cpu
    gc.collect()

    # ---------- NPU inference ----------
    print(f"=== NPU Inference: {args.model} ===")
    model_npu = timm.create_model(args.model, pretrained=True)
    model_npu = model_npu.to(device_npu)
    model_npu = model_npu.eval()

    input_tensor_npu = transforms(img).unsqueeze(0).to(device_npu)

    with torch.no_grad():
        output_npu = model_npu(input_tensor_npu)

    output_npu_cpu = output_npu.cpu()
    probs_npu = F.softmax(output_npu_cpu[0], dim=0)

    print(f"NPU top-5: {torch.topk(probs_npu, k=5).indices.tolist()}")
    print(f"NPU top-5 probs: {[round(p.item(), 6) for p in torch.topk(probs_npu, k=5).values]}")

    npu_output = {
        "logits": output_npu_cpu[0].tolist(),
        "top5_indices": torch.topk(probs_npu, k=5).indices.tolist(),
        "top5_probs": [round(p.item(), 6) for p in torch.topk(probs_npu, k=5).values],
    }

    # ---------- Comparison ----------
    logits_cpu = torch.tensor(cpu_output["logits"])
    logits_npu = torch.tensor(npu_output["logits"])

    abs_diff = torch.abs(logits_cpu - logits_npu)
    rel_diff = abs_diff / (torch.abs(logits_cpu) + 1e-8)

    # Classification-specific metrics
    top1_cpu = cpu_output["top5_indices"][0]
    top1_npu = npu_output["top5_indices"][0]
    top1_match = top1_cpu == top1_npu
    top5_cpu_set = set(cpu_output["top5_indices"])
    top5_npu_set = set(npu_output["top5_indices"])
    top5_overlap = len(top5_cpu_set & top5_npu_set)

    cos_sim = F.cosine_similarity(logits_cpu.unsqueeze(0), logits_npu.unsqueeze(0)).item()

    comparison = {
        "model": args.model,
        "max_abs_diff": round(abs_diff.max().item(), 8),
        "mean_abs_diff": round(abs_diff.mean().item(), 8),
        "max_rel_diff": round(rel_diff.max().item(), 8),
        "mean_rel_diff": round(rel_diff.mean().item(), 8),
        "cosine_similarity": round(cos_sim, 8),
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "cpu_top1": top1_cpu,
        "npu_top1": top1_npu,
        "cpu_top5": cpu_output["top5_indices"],
        "npu_top5": npu_output["top5_indices"],
        "max_diff_pct": round((abs_diff.max().item() / (torch.abs(logits_cpu).max().item() + 1e-8) * 100), 6),
    }

    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n=== Comparison Results ===")
    print(f"Max absolute diff: {comparison['max_abs_diff']}")
    print(f"Mean absolute diff: {comparison['mean_abs_diff']}")
    print(f"Cosine similarity: {comparison['cosine_similarity']}")
    print(f"Top-1 match: {comparison['top1_match']} (CPU: {top1_cpu}, NPU: {top1_npu})")
    print(f"Top-5 overlap: {top5_overlap}/5")
    print(f"Max diff %: {comparison['max_diff_pct']}%")
    print(f"\nResults saved to {args.output}")

    # Clean NPU model
    del model_npu, input_tensor_npu, output_npu, output_npu_cpu
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()

if __name__ == "__main__":
    main()
