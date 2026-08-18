#!/usr/bin/env python3
"""SE-Net NPU inference example for all models in batch 24."""
import argparse
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

MODELS = [
    "seresnet50.a1_in1k",
    "seresnet50.ra2_in1k",
    "seresnet50.a3_in1k",
    "seresnet50.a2_in1k",
    "seresnet33ts.ra2_in1k",
    "seresnet152d.ra2_in1k",
    "senet154.gluon_in1k",
    "sehalonet33ts.ra2_in1k",
    "sebotnet33ts_256.a1h_in1k",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="seresnet50.a1_in1k",
                        choices=MODELS)
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--image", default="test_image.jpg")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "npu" or torch.npu.is_available() else "cpu")

    print(f"Loading {args.model_name}...")
    model = timm.create_model(args.model_name, pretrained=True)
    model.eval().to(device)
    print(f"Model loaded on {device}")

    img = Image.open(args.image).convert("RGB")
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5 = torch.topk(probs, k=5)

    print("\nTop-5 predictions:")
    for i in range(5):
        print(f"  {top5.indices[i].item()}: {top5.values[i].item():.4f}")


if __name__ == "__main__":
    main()
