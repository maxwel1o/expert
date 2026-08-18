#!/usr/bin/env python3
"""
vit_base_patch16_224.dino — Ascend NPU 推理脚本

用法:
  python3 inference.py                           # 合成数据推理
  python3 inference.py --image /path/to/img.jpg  # 真实图片推理
  python3 inference.py --help                    # 查看全部参数
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ["NPU_IGNORE_PERMISSION_MISMATCH"] = "1"

from torchvision import transforms
from PIL import Image


def load_model(device):
    """加载 DINO ViT-B/16 模型到指定设备。"""
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    model.eval().to(device)
    return model


def get_transform():
    """DINO 标准图片预处理流水线。"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image(path, transform):
    """加载单张图片并预处理。"""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def infer(model, x):
    """执行推理，返回特征向量 [B, D]。"""
    return model(x)


def main():
    parser = argparse.ArgumentParser(description="DINO ViT-B/16 Ascend NPU 推理")
    parser.add_argument("--image", type=str, default=None, help="输入图片路径")
    parser.add_argument("--device", type=str, default="npu:0",
                        choices=["cpu", "npu:0"], help="推理设备")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 加载模型...")
    model = load_model(device)
    print(f"[INFO] 模型参数: {sum(p.numel() for p in model.parameters()):,}")

    transform = get_transform()
    if args.image:
        print(f"[INFO] 加载图片: {args.image}")
        x = load_image(args.image, transform)
    else:
        print(f"[INFO] 使用合成数据 (batch_size={args.batch_size})")
        torch.manual_seed(42)
        x = torch.randn(args.batch_size, 3, 224, 224) * 0.2 + 0.5
        x = x.clamp(0, 1)
        transform_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = transform_tensor(x)

    x = x.to(device)
    print(f"[INFO] 输入 shape: {x.shape}")

    # Warmup
    for _ in range(5):
        _ = infer(model, x)
    if device.type == "npu":
        torch.npu.synchronize()

    # Benchmark
    n_repeat = 50
    start = time.perf_counter()
    for _ in range(n_repeat):
        infer(model, x)
    if device.type == "npu":
        torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) / n_repeat * 1000

    # Final inference
    out = infer(model, x)
    if device.type == "npu":
        out = out.cpu()

    print(f"\n[结果]")
    print(f"  特征向量 shape: {out.shape}")
    print(f"  特征 norm:  {out.norm().item():.4f}")
    print(f"  推理延迟:    {elapsed:.2f} ms/iter")
    print(f"  top-5 维度: {out.topk(5).indices[0].tolist()}")


if __name__ == "__main__":
    main()
