#!/usr/bin/env python3
"""dm_nfnet NPU inference script."""
import argparse
import time
import torch
import timm
from safetensors.torch import load_file
from PIL import Image
from timm.data import create_transform, resolve_data_config
import os

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


def main():
    parser = argparse.ArgumentParser(description="dm_nfnet NPU Inference")
    parser.add_argument("--model", type=str, default="dm_nfnet_f0.dm_in1k",
                        choices=MODELS, help="Model name")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file (optional, auto-detect)")
    parser.add_argument("--device", type=str, default="npu",
                        choices=["cpu", "npu"], help="Device")
    args = parser.parse_args()

    device = args.device
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("NPU not available, falling back to CPU")
        device = "cpu"

    weights_path = args.weights
    if weights_path is None:
        weights_path = get_model_path(args.model)
    if weights_path and os.path.exists(weights_path):
        print(f"Weights: {weights_path}")
    else:
        print("WARNING: No weights found, using random initialization")

    print(f"Loading {args.model} on {device}...")
    model = timm.create_model(args.model, pretrained=False)
    model.eval()

    if weights_path and os.path.exists(weights_path):
        sd = load_file(weights_path)
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        if len(filtered_sd) != len(model_sd):
            print(f"Warning: loaded {len(filtered_sd)}/{len(model_sd)} parameters")
        model.load_state_dict(filtered_sd, strict=False)

    model = model.to(device)

    # Preprocess image
    img = Image.open(args.image).convert("RGB")
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    input_tensor = transform(img).unsqueeze(0).to(device)

    print(f"Input shape: {input_tensor.shape}")
    print("Running inference...")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    if device == "npu":
        torch.npu.synchronize()

    # Timed inference
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    if device == "npu":
        torch.npu.synchronize()
    elapsed = time.time() - start

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_idx = probs.topk(5).indices.cpu().tolist()
    top5_probs = probs.topk(5).values.cpu().tolist()

    print(f"\nInference time: {elapsed:.4f}s")
    print(f"Output shape: {list(output.shape)}")
    print("Top-5 predictions:")
    for i, (idx, p) in enumerate(zip(top5_idx, top5_probs)):
        print(f"  {i+1}. class {idx}: {p:.6f}")
    print(f"Logits[:10]: {output[0][:10].tolist()}")
    print("Done.")


if __name__ == "__main__":
    main()
