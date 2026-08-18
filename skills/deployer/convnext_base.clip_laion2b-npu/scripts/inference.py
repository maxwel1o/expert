"""
convnext_base.clip_laion2b - Ascend NPU Inference Script
========================================================
Usage:
    # NPU inference
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_RT_VISIBLE_DEVICES=0
    python3 convnext_infer.py --device npu --image test_input.jpg

    # CPU inference
    python3 convnext_infer.py --device cpu --image test_input.jpg
"""
import argparse
import time
import torch
import numpy as np
from safetensors.torch import load_file
from PIL import Image

# NPU injection (required for CUDA → NPU API mapping)
import torch_npu
from torch_npu.contrib import transfer_to_npu

from timm.models.convnext import convnext_base
from timm.data import resolve_data_config, create_transform

MODEL_NAME = "convnext_base.clip_laion2b"
WEIGHT_PATH = "/opt/atomgit/convnext_base.clip_laion2b_modelscope/timm/convnext_base.clip_laion2b/model.safetensors"
NUM_CLASSES = 640


def load_model(device: str, weight_path: str = WEIGHT_PATH):
    device = torch.device(device)
    model = convnext_base(pretrained=False, num_classes=NUM_CLASSES)
    model.eval()

    state_dict = load_file(weight_path)
    # Strip "model." prefix if present
    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model = model.to(device)
    return model


def get_transform():
    """Get preprocessing pipeline matching model's training setup (CLIP-style)."""
    cfg = resolve_data_config({}, model=convnext_base(pretrained=False))
    return create_transform(**cfg), cfg


def preprocess_image(image_path: str, transform):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dim


def infer(model, input_tensor, device: str, num_runs: int = 10):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        # Warmup
        _ = model(input_tensor)
        if device == "npu":
            torch.npu.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_runs):
            output = model(input_tensor)
        if device == "npu":
            torch.npu.synchronize()
        elapsed = time.perf_counter() - start

    return output, elapsed / num_runs


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt Base inference on Ascend NPU")
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--image", type=str, help="Path to input image")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    if args.device == "npu":
        print(f"NPU available: {torch.npu.is_available()}, "
              f"Device: {torch.npu.get_device_name(0)}")

    model = load_model(args.device)
    transform, config = get_transform()
    print(f"Input config: {config}")

    if args.image:
        input_tensor = preprocess_image(args.image, transform)
        print(f"Image: {args.image} → {tuple(input_tensor.shape)}")
    else:
        input_tensor = torch.randn(1, *config["input_size"])
        print(f"Random input: {tuple(input_tensor.shape)}")

    output, avg_time = infer(model, input_tensor, args.device)
    print(f"Avg inference time: {avg_time * 1000:.2f} ms")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output stats: min={output.min().item():.4f}, "
          f"max={output.max().item():.4f}, "
          f"mean={output.mean().item():.4f}")
    print(f"Top-5 indices: {output[0].topk(5).indices.tolist()}")


if __name__ == "__main__":
    main()
