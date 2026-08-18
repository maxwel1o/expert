#!/usr/bin/env python3
"""
DINO ViT-S/8 NPU Inference Script
基于昇腾NPU的模型推理脚本
"""
import os
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

# 昇腾NPU自动迁移注入（必须在其他import之前）
import torch_npu
from torch_npu.contrib import transfer_to_npu

import vision_transformer as vits


def main():
    parser = argparse.ArgumentParser(description="DINO ViT-S/8 NPU Inference")
    parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
    parser.add_argument("--patch_size", default=8, type=int, help="Patch size")
    parser.add_argument("--pretrained_weights", required=True, type=str, help="Path to pretrained weights")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--img_size", default=224, type=int, help="Input image size")
    parser.add_argument("--device", default="npu", type=str, choices=["cpu", "cuda", "npu"], help="Device")
    args = parser.parse_args()

    # 加载模型
    print(f"Loading model: {args.arch} patch_size={args.patch_size}")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 设置设备
    if args.device == "npu":
        device = torch.device("npu:0")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(args.img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = Image.open(args.image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        features = model(img_tensor)

    print(f"Input shape:  {img_tensor.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Device:       {device}")
    print(f"First 10 features: {features[0, :10].cpu().numpy()}")

    return features


if __name__ == "__main__":
    main()
