#!/usr/bin/env python3
"""
DINO ViT-S/8 NPU Accuracy Evaluation Script
精度验证脚本：对比 NPU 与 CPU 推理结果
"""
import os
import sys
import time
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

# 昇腾NPU自动迁移注入
import torch_npu
from torch_npu.contrib import transfer_to_npu

import vision_transformer as vits


def load_model(arch, patch_size, pretrained_weights, device):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def run_inference(model, tensor, warmup=3, runs=10):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tensor)
        if tensor.device.type == "npu":
            torch.npu.synchronize()
        times = []
        for _ in range(runs):
            start = time.time()
            out = model(tensor)
            if tensor.device.type == "npu":
                torch.npu.synchronize()
            times.append((time.time() - start) * 1000)
        return out, sum(times) / len(times), times


def compute_error(ref, target):
    ref = ref.detach().cpu()
    target = target.detach().cpu()
    abs_diff = torch.abs(ref - target)
    l2_diff = torch.norm(ref - target, p=2).item()
    l2_ref = torch.norm(ref, p=2).item()
    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "l2_diff": l2_diff,
        "l2_relative": l2_diff / (l2_ref + 1e-8),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", required=True, type=str)
    parser.add_argument("--image", default="", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--runs", default=10, type=int)
    args = parser.parse_args()

    print("=" * 60)
    print("DINO ViT-S/8 NPU Accuracy Evaluation")
    print("=" * 60)

    # CPU baseline
    print("\n[CPU] Loading model...")
    model_cpu = load_model("vit_small", 8, args.pretrained_weights, torch.device("cpu"))

    # NPU target
    print("[NPU] Loading model...")
    model_npu = load_model("vit_small", 8, args.pretrained_weights, torch.device("npu:0"))

    if args.image:
        print(f"\nUsing image: {args.image}")
        transform = get_transform()
        img = Image.open(args.image).convert("RGB")
        tensor_cpu = transform(img).unsqueeze(0).to("cpu")
        tensor_npu = transform(img).unsqueeze(0).to("npu:0")
    else:
        print(f"\nUsing random input (seed={args.seed})")
        torch.manual_seed(args.seed)
        tensor_cpu = torch.randn(1, 3, 224, 224).to("cpu")
        tensor_npu = tensor_cpu.to("npu:0")

    out_cpu, t_cpu, _ = run_inference(model_cpu, tensor_cpu, args.warmup, args.runs)
    out_npu, t_npu, _ = run_inference(model_npu, tensor_npu, args.warmup, args.runs)

    print(f"\nCPU inference time: {t_cpu:.2f} ms")
    print(f"NPU inference time: {t_npu:.2f} ms")
    print(f"Speedup: {t_cpu / t_npu:.1f}x")

    err = compute_error(out_cpu, out_npu)
    print(f"\nAccuracy Comparison:")
    print(f"  Max abs diff:  {err['max_abs_diff']:.6e}")
    print(f"  Mean abs diff: {err['mean_abs_diff']:.6e}")
    print(f"  L2 relative:   {err['l2_relative']:.6e} ({err['l2_relative'] * 100:.2f}%)")

    passed = err["l2_relative"] < 0.01
    print(f"\nPrecision check (L2 relative < 1%): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
