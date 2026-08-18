#!/usr/bin/env python3
"""
Example: XCiT NPU Inference
Run XCiT image classification on Ascend NPU.
"""
import torch
import timm
from PIL import Image
from timm.data import create_transform, resolve_model_data_config


def run_xcit_inference(model_name="xcit_tiny_24_p8_384.fb_dist_in1k",
                       image_path="test.jpg", device="npu:0"):
    # Load model
    model = timm.create_model(model_name, pretrained=False)

    # Load local weights
    # torch.load("pytorch_model.bin" or "model.safetensors")
    model = model.eval()
    model.to(device)

    # Preprocess
    img = Image.open(image_path).convert("RGB")
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Results
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs, 5)
    print(f"Top-5 predictions:")
    for i in range(5):
        print(f"  {i+1}. class {top5_indices[0][i].item()}: {top5_probs[0][i].item()*100:.2f}%")

    return output


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "xcit_tiny_24_p8_384.fb_dist_in1k"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"
    device = sys.argv[3] if len(sys.argv) > 3 else "npu:0"
    run_xcit_inference(model_name, image_path, device)
