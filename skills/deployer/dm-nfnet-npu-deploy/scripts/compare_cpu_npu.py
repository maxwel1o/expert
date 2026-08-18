#!/usr/bin/env python3
"""Compare dm_nfnet CPU vs NPU inference results."""
import argparse
import json
import os
import time
import torch
import timm
from safetensors.torch import load_file
from PIL import Image
from timm.data import create_transform, resolve_data_config

MODELS = [
    "dm_nfnet_f0.dm_in1k",
    "dm_nfnet_f1.dm_in1k",
    "dm_nfnet_f2.dm_in1k",
    "dm_nfnet_f3.dm_in1k",
    "dm_nfnet_f4.dm_in1k",
    "dm_nfnet_f5.dm_in1k",
    "dm_nfnet_f6.dm_in1k",
]


def get_model_path(model_name):
    cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
    path = os.path.join(cache_dir, f"timm/{model_name}/model.safetensors")
    if os.path.exists(path):
        return path
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f == "model.safetensors" and model_name.replace(".", "_") in root.replace(".", "_"):
                return os.path.join(root, f)
    return None


def load_model(model_name, device, weights_path):
    model = timm.create_model(model_name, pretrained=False)
    model.eval()
    if weights_path and os.path.exists(weights_path):
        sd = load_file(weights_path)
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(filtered_sd, strict=False)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--image", default="test_img.jpg")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    weights_path = args.weights or get_model_path(args.model)
    if not weights_path or not os.path.exists(weights_path):
        print("ERROR: No weights found.")
        return

    # Load image
    img = Image.open(args.image).convert("RGB")

    # ---- CPU inference ----
    print("Running CPU inference...")
    model_cpu = load_model(args.model, "cpu", weights_path)
    config = resolve_data_config({}, model=model_cpu)
    transform = create_transform(**config)
    input_cpu = transform(img).unsqueeze(0).to("cpu")

    torch.set_num_threads(16)
    start = time.time()
    with torch.no_grad():
        cpu_logits = model_cpu(input_cpu)
    cpu_time = time.time() - start
    cpu_probs = torch.nn.functional.softmax(cpu_logits[0], dim=0)
    cpu_top1 = cpu_probs.argmax().item()
    cpu_top5 = cpu_probs.topk(5).indices.cpu().tolist()

    # Free CPU model
    del model_cpu
    torch.set_num_threads(1)

    # ---- NPU inference ----
    print("Running NPU inference...")
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("ERROR: NPU not available.")
        return

    model_npu = load_model(args.model, "npu:0", weights_path)
    input_npu = transform(img).unsqueeze(0).to("npu:0")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model_npu(input_npu)
    torch.npu.synchronize()

    start = time.time()
    with torch.no_grad():
        npu_logits = model_npu(input_npu)
    torch.npu.synchronize()
    npu_time = time.time() - start

    npu_probs = torch.nn.functional.softmax(npu_logits[0], dim=0)
    npu_top1 = npu_probs.argmax().item()
    npu_top5 = npu_probs.topk(5).indices.cpu().tolist()

    cpu_logits = cpu_logits[0].float().cpu()
    npu_logits = npu_logits[0].float().cpu()
    cpu_probs = cpu_probs.float().cpu()
    npu_probs = npu_probs.float().cpu()

    # Compute metrics
    abs_diff = (cpu_logits - npu_logits).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mse = ((cpu_logits - npu_logits) ** 2).mean().item()

    rel_diff = abs_diff / (cpu_logits.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item() * 100
    mean_rel_diff = rel_diff.mean().item() * 100

    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_logits.unsqueeze(0), npu_logits.unsqueeze(0)).item()

    # Top-100 relative diff (for classification relevance)
    top100_val, top100_idx = cpu_probs.topk(min(100, len(cpu_probs)))
    cpu_top100 = cpu_logits[top100_idx]
    npu_top100 = npu_logits[top100_idx]
    top100_rel_diff = ((cpu_top100 - npu_top100).abs() / (cpu_top100.abs() + 1e-8)).mean().item() * 100

    top1_match = cpu_top1 == npu_top1
    top5_overlap = len(set(cpu_top5) & set(npu_top5))

    precision_pass = top100_rel_diff < 1.0

    # Speedup
    speedup = cpu_time / npu_time if npu_time > 0 else 0

    results = {
        "model_name": args.model,
        "cpu_time_s": round(cpu_time, 4),
        "npu_time_s": round(npu_time, 4),
        "speedup": round(speedup, 2),
        "max_abs_diff": round(max_abs_diff, 8),
        "mean_abs_diff": round(mean_abs_diff, 8),
        "mse": round(mse, 8),
        "max_rel_diff_pct": round(max_rel_diff, 6),
        "mean_rel_diff_pct": round(mean_rel_diff, 6),
        "cosine_similarity": round(cos_sim, 8),
        "top100_mean_rel_diff_pct": round(top100_rel_diff, 6),
        "top1_match": top1_match,
        "top1_cpu": cpu_top1,
        "top1_npu": npu_top1,
        "top5_cpu": cpu_top5,
        "top5_npu": npu_top5,
        "top5_overlap": top5_overlap,
        "precision_pass": precision_pass,
    }

    # Save results
    output_dir = f"models/{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")
    print(f"Precision pass: {precision_pass} (top100_rel_diff={top100_rel_diff:.4f}%)")
    print(f"Speedup: {speedup:.2f}x (CPU: {cpu_time:.4f}s, NPU: {npu_time:.4f}s)")

    # Cleanup
    del model_npu
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
