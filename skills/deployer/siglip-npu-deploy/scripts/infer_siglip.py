#!/usr/bin/env python3
"""
SigLIP/SigLIP2 Unified NPU Inference Script.
Usage:
    python infer_siglip.py --model ViT-B-16-SigLIP --image test.jpg
    python infer_siglip.py --model ViT-L-16-SigLIP2-384 --image test.jpg --device cpu
"""
import argparse
import os
import time
import torch
import open_clip
from safetensors.torch import load_file
from PIL import Image

MODELS = [
    "ViT-B-16-SigLIP", "ViT-B-16-SigLIP-256", "ViT-B-16-SigLIP-384", "ViT-B-16-SigLIP-512",
    "ViT-B-16-SigLIP-i18n-256",
    "ViT-B-16-SigLIP2", "ViT-B-16-SigLIP2-256", "ViT-B-16-SigLIP2-384", "ViT-B-16-SigLIP2-512",
    "ViT-B-32-SigLIP2-256",
    "ViT-L-16-SigLIP-256", "ViT-L-16-SigLIP-384",
    "ViT-L-16-SigLIP2-256", "ViT-L-16-SigLIP2-384", "ViT-L-16-SigLIP2-512",
]


def get_embed_dim(model_name):
    if "L-16" in model_name:
        return 1024
    return 768


def main():
    parser = argparse.ArgumentParser(description="SigLIP/SigLIP2 NPU Inference")
    parser.add_argument("--model", type=str, default="ViT-B-16-SigLIP",
                        choices=MODELS, help="Model name")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to safetensors weights (optional, auto-detect)")
    parser.add_argument("--device", type=str, default="npu",
                        choices=["cpu", "npu"], help="Device")
    args = parser.parse_args()

    device = args.device
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("NPU not available, falling back to CPU")
        device = "cpu"

    # Auto-detect weights path
    weights_path = args.weights
    if weights_path is None:
        candidates = [
            f"open_clip_model.safetensors",
            f"../modelscope/timm/{args.model}/open_clip_model.safetensors",
            os.path.expanduser(f"~/.cache/modelscope/timm/{args.model}/open_clip_model.safetensors"),
        ]
        for p in candidates:
            if os.path.exists(p):
                weights_path = p
                break
        if weights_path is None:
            print(f"Weights not found for {args.model}. Download from ModelScope:")
            print(f"  pip install modelscope")
            print(f"  python -c \"from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('timm/{args.model}')\"")
            return

    print(f"Loading {args.model} on {device}...")
    print(f"Weights: {weights_path}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=False)
    sd = load_file(weights_path)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    embed_dim = get_embed_dim(args.model)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Expected output dim: {embed_dim}")
    print("Running inference...")

    if device == "npu":
        torch.npu.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    if device == "npu":
        torch.npu.synchronize()
    elapsed = time.time() - start

    features = output[0]  # image embeddings
    print(f"\nInference time: {elapsed:.4f}s")
    print(f"Feature shape: {features.shape}")
    print(f"Feature[:10]: {features[0, :10]}")
    print("Done.")


if __name__ == "__main__":
    main()
