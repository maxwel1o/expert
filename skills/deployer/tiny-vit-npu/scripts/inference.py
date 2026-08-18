#!/usr/bin/env python3
"""TinyViT NPU inference script for Ascend NPU deployment skill."""
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
    parser = argparse.ArgumentParser(description="TinyViT NPU inference")
    parser.add_argument("--model-name", type=str, required=True, help="timm model name")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--device", type=str, default="npu", choices=["cpu", "npu"])
    parser.add_argument("--output", type=str, default=None, help="Output logits file")
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


def run_inference(model_name, device="npu", image_size=224, num_runs=10):
    model = load_model(model_name)
    model = model.to(device)
    model = model.float()
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    input_tensor = get_test_image(image_size)
    input_tensor = input_tensor.float().to(device)

    for _ in range(3):
        _ = model(input_tensor)
    if device == "npu":
        torch.npu.synchronize()

    start = time.time()
    for _ in range(num_runs):
        output = model(input_tensor)
    if device == "npu":
        torch.npu.synchronize()
    elapsed = time.time() - start

    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)

    print(f"Average time: {elapsed / num_runs * 1000:.2f} ms")
    print(f"Top-5 predictions:")
    for i in range(5):
        print(f"  {i+1}. class {top5_indices[0][i].item()} (prob: {top5_probs[0][i].item():.6f})")

    del model, input_tensor, output
    gc.collect()
    if device == "npu":
        torch.npu.empty_cache()

    return probs, top5_indices


@torch.no_grad()
def main():
    args = parse_args()
    run_inference(args.model_name, args.device, args.image_size)


if __name__ == "__main__":
    main()
