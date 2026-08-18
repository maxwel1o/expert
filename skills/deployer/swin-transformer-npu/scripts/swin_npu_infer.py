#!/usr/bin/env python3
"""Swin Transformer NPU inference script.

Usage:
    python3 swin_npu_infer.py --model swin_tiny_patch4_window7_224.ms_in1k --device npu
    python3 swin_npu_infer.py --model swin_tiny_patch4_window7_224.ms_in1k --device cpu --image input.jpg
"""

import argparse
import json
import time
import torch
import timm
from PIL import Image, ImageDraw
from timm.data import resolve_data_config, create_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Swin Transformer NPU Inference")
    parser.add_argument("--model", type=str, required=True, help="timm model name")
    parser.add_argument("--device", type=str, default="npu", choices=["cpu", "npu"])
    parser.add_argument("--image", type=str, default=None, help="Input image path")
    return parser.parse_args()


@torch.no_grad()
def run_inference(model_name, device, image_path=None):
    if device == "npu":
        if not torch.npu.is_available():
            print("ERROR: NPU not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("npu:0")
    else:
        device = torch.device("cpu")

    print(f"Model: {model_name}")
    print(f"Device: {device}")

    # Load model
    t0 = time.time()
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Prepare input
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config)
    input_size = data_config.get("input_size", (3, 224, 224))
    actual_size = input_size[-1]

    if image_path:
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (actual_size, actual_size), color=(40, 40, 60))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, actual_size // 2, actual_size // 2], fill=(255, 80, 80))
        draw.ellipse([actual_size // 4, actual_size // 4, 3 * actual_size // 4, 3 * actual_size // 4], fill=(80, 255, 80))
        draw.polygon([(actual_size // 2, 10), (actual_size - 10, actual_size // 2), (actual_size // 2, actual_size - 10)], fill=(80, 80, 255))

    input_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Input shape: {input_tensor.shape}")

    # Warmup
    for _ in range(3):
        _ = model(input_tensor)

    # Timed inference
    if device.type == "npu":
        torch.npu.synchronize()
    t1 = time.time()
    output = model(input_tensor)
    if device.type == "npu":
        torch.npu.synchronize()
    t2 = time.time()

    infer_ms = (t2 - t1) * 1000
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)

    # Load ImageNet labels
    idx_to_label = {}
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json",
            timeout=5
        )
        class_idx = json.loads(resp.read())
        idx_to_label = {int(v[0]): v[1] for v in class_idx.values()}
    except Exception:
        pass

    print(f"\n=== Results ({str(device).upper()}) ===")
    print(f"Inference time: {infer_ms:.2f} ms")
    print(f"Output shape: {output.shape}")
    for i in range(5):
        idx = top5_idx[i].item()
        label = idx_to_label.get(idx, f"class_{idx}")
        print(f"  {i + 1}. {label}: {top5_prob[i].item() * 100:.2f}%")

    return {
        "output": output.detach().cpu(),
        "time_ms": infer_ms,
        "top5_idx": top5_idx.detach().cpu(),
        "top5_prob": top5_prob.detach().cpu(),
    }


def main():
    args = parse_args()
    result = run_inference(args.model, args.device, args.image)

    # Save output
    safe_name = args.model.replace("/", "_")
    torch.save(result["output"], f"{safe_name}_{args.device}_output.pt")
    print(f"\nSaved: {safe_name}_{args.device}_output.pt")
    print(f"Inference time: {result['time_ms']:.2f}ms")


if __name__ == "__main__":
    main()
