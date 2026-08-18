#!/usr/bin/env python3
"""Run timm SelecSLS model inference on CPU or NPU."""
import argparse
import json
import time
import torch
import timm
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="timm model name")
    parser.add_argument("--image", default="test_image.jpg", help="input image path")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--output", default="results.json", help="output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("NPU not available, falling back to CPU")
        device = torch.device("cpu")

    # Load model
    print(f"Loading model: {args.model} on {args.device}...")
    model = timm.create_model(args.model, pretrained=True)
    model = model.to(device)
    model = model.eval()

    # Get transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Load and preprocess image
    img = Image.open(args.image).convert("RGB")
    input_tensor = transforms(img).unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)

    # Inference
    torch.cuda.synchronize() if args.device == "cuda" else None
    if args.device == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()

    times = []
    output = None
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        if args.device == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_idx = torch.topk(probs, k=5).indices.tolist()
    top5_probs = torch.topk(probs, k=5).values.tolist()

    result = {
        "model": args.model,
        "device": args.device,
        "mean_inference_time": sum(times) / len(times),
        "top5_indices": top5_idx,
        "top5_probabilities": [round(p, 6) for p in top5_probs],
        "output_shape": list(output.shape),
        "output_sample": output[0, :5].tolist(),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {args.output}")
    print(f"Mean inference time: {result['mean_inference_time']:.4f}s")
    print(f"Top-5 indices: {top5_idx}")
    print(f"Top-5 probs: {[round(p, 6) for p in top5_probs]}")

    # Clean up
    del model, input_tensor, output
    if args.device == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()

if __name__ == "__main__":
    main()
