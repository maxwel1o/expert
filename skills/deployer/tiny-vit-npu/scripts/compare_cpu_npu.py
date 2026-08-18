#!/usr/bin/env python3
"""TinyViT CPU vs NPU accuracy comparison script for Ascend NPU deployment skill."""
import argparse
import gc
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from timm import create_model
from modelscope import snapshot_download
from safetensors.torch import load_file

import torch_npu


def parse_args():
    parser = argparse.ArgumentParser(description="TinyViT CPU/NPU accuracy comparison")
    parser.add_argument("--model-name", type=str, required=True, help="timm model name")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    return parser.parse_args()


def get_test_image(size):
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    return transform(img).unsqueeze(0)


def load_model(model_name):
    ms_path = "timm/" + model_name
    local_path = snapshot_download(ms_path)
    model = create_model(model_name, pretrained=False)
    model.eval()
    safetensors_path = os.path.join(local_path, "model.safetensors")
    pytorch_bin_path = os.path.join(local_path, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(pytorch_bin_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def main():
    args = parse_args()
    model_name = args.model_name
    img_size = args.image_size

    print(f"Model: {model_name}")
    print(f"Image size: {img_size}")
    print("=" * 60)

    # CPU
    model_cpu = load_model(model_name)
    torch.manual_seed(42)
    input_cpu = get_test_image(img_size)

    print("\n--- CPU Inference ---")
    cpu_start = time.time()
    for _ in range(10):
        output_cpu = model_cpu(input_cpu)
    cpu_elapsed = time.time() - cpu_start
    cpu_probs = torch.softmax(output_cpu, dim=1)
    cpu_top5_probs, cpu_top5_idx = torch.topk(cpu_probs, 5, dim=1)
    print(f"Average time: {cpu_elapsed / 10 * 1000:.2f} ms")

    # NPU
    print("\n--- NPU Inference ---")
    model_npu = load_model(model_name)
    model_npu = model_npu.to("npu:0").float().eval()
    input_npu = get_test_image(img_size).float().to("npu:0")

    for _ in range(3):
        _ = model_npu(input_npu)
    torch.npu.synchronize()

    npu_start = time.time()
    for _ in range(10):
        output_npu = model_npu(input_npu)
    torch.npu.synchronize()
    npu_elapsed = time.time() - npu_start

    output_npu = output_npu.cpu()
    npu_probs = torch.softmax(output_npu, dim=1)
    npu_top5_probs, npu_top5_idx = torch.topk(npu_probs, 5, dim=1)
    print(f"Average time: {npu_elapsed / 10 * 1000:.2f} ms")

    # Comparison
    print("\n" + "=" * 60)
    print("CPU vs NPU Accuracy Comparison")
    print("=" * 60)

    cos_sim = torch.nn.CosineSimilarity(dim=1)(output_cpu, output_npu).item()
    top1_match = (torch.argmax(output_cpu, dim=1) == torch.argmax(output_npu, dim=1)).item()
    norm_prob_diff = torch.norm(cpu_probs - npu_probs)
    prob_rel_error = (norm_prob_diff / torch.norm(cpu_probs)).item()

    print(f"Logits Max Abs Error:  {torch.abs(output_cpu - output_npu).max().item():.6e}")
    print(f"Logits Mean Abs Error: {torch.abs(output_cpu - output_npu).mean().item():.6e}")
    print(f"Probs Max Abs Error:   {torch.abs(cpu_probs - npu_probs).max().item():.6e}")
    print(f"Probs Mean Abs Error:  {torch.abs(cpu_probs - npu_probs).mean().item():.6e}")
    print(f"Cosine Similarity:     {cos_sim:.8f}")
    print(f"Prob Relative Error:   {prob_rel_error:.4f}%")
    print(f"Top-1 Class Match:     {top1_match}")
    print(f"CPU: {cpu_elapsed/10*1000:.2f} ms | NPU: {npu_elapsed/10*1000:.2f} ms")
    print(f"Speedup: {cpu_elapsed / npu_elapsed:.2f}x")

    passed = prob_rel_error < 0.01
    print(f"\nCONCLUSION: error={prob_rel_error*100:.4f}% {'PASS' if passed else 'FAIL'}")

    del model_cpu, model_npu, input_cpu, input_npu, output_cpu, output_npu
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
