#!/usr/bin/env python3
"""Inference script for timm Wide ResNet models - loading from local cache."""
import argparse
import json
import time
import os
import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model
from timm.data import resolve_data_config, create_transform

def get_modelscope_path(model_name):
    """Get modelscope cache path for a timm model."""
    encoded_name = model_name.replace('.', '___')
    return f"/opt/atomgit/.cache/modelscope/hub/models/timm/{encoded_name}"

def load_model_local(model_name, device):
    """Load model from local modelscope cache."""
    print(f"Loading {model_name} on {device}...")
    start = time.time()
    model = create_model(model_name, pretrained=False)

    cache_dir = get_modelscope_path(model_name)
    safetensors_path = os.path.join(cache_dir, 'model.safetensors')
    bin_path = os.path.join(cache_dir, 'pytorch_model.bin')

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f"No weight file found in {cache_dir}")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    return model

def get_model_info(model_name):
    """Print model information."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    from timm import list_models
    available = list_models(pretrained=True)
    matching = [m for m in available if m.startswith(model_name.split('.')[0])]
    print(f"Matching models in timm registry: {len(matching)}")
    for m in matching[:5]:
        print(f"  - {m}")
    print(f"Task type: Image Classification")
    print(f"Framework: PyTorch + timm")

def preprocess_image(image_path, model):
    """Preprocess image for model input."""
    img = Image.open(image_path).convert('RGB')
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, config['input_size']

def run_inference(model, input_tensor, device):
    """Run inference and return logits and probabilities."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        inference_time = time.time() - start
    logits = output.cpu()
    probs = F.softmax(logits, dim=1)
    return logits, probs, inference_time

def get_topk(probs, k=5):
    """Get top-k predictions."""
    top_probs, top_indices = torch.topk(probs, k=k, dim=1)
    return top_indices[0].tolist(), top_probs[0].tolist()

def load_imagenet_labels():
    """Load ImageNet class labels."""
    try:
        import json, urllib.request
        url = "https://github.com/anishathalye/imagenet-simple-labels/raw/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=10) as f:
            labels = json.load(f)
        return labels
    except:
        return [f"class_{i}" for i in range(1000)]

def main():
    parser = argparse.ArgumentParser(description='Wide ResNet NPU Inference (local)')
    parser.add_argument('--model', type=str, required=True, help='timm model name')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'npu'], help='Device')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of inference runs for timing')
    args = parser.parse_args()

    get_model_info(args.model)

    device = torch.device(args.device)

    if args.image is None:
        from urllib.request import urlretrieve
        test_img_path = "/tmp/test_cat.jpg"
        try:
            urlretrieve(
                "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01882714_koala.JPEG",
                test_img_path
            )
            print(f"Downloaded test image to {test_img_path}")
        except:
            img = Image.new('RGB', (224, 224), color=(100, 100, 200))
            img.save(test_img_path)
            print(f"Created dummy test image at {test_img_path}")
        args.image = test_img_path

    model = load_model_local(args.model, device)

    input_tensor, input_size = preprocess_image(args.image, model)
    print(f"Input size: {input_size}")
    print(f"Batch shape: {input_tensor.shape}")

    logits, probs, _ = run_inference(model, input_tensor, device)
    print(f"Output shape: {logits.shape}")
    print(f"Inference: WARMUP done")

    times = []
    for i in range(args.num_runs):
        _, _, t = run_inference(model, input_tensor, device)
        times.append(t)

    avg_time = sum(times) / len(times)
    print(f"\nInference times ({args.num_runs} runs):")
    print(f"  Average: {avg_time*1000:.2f} ms")
    print(f"  Min: {min(times)*1000:.2f} ms")
    print(f"  Max: {max(times)*1000:.2f} ms")

    top_indices, top_probs = get_topk(probs, k=5)
    labels = load_imagenet_labels()

    print(f"\nTop-5 predictions:")
    print(f"{'Rank':<6} {'Class ID':<10} {'Probability':<12} {'Label'}")
    print(f"{'-'*60}")
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"{i+1:<6} {idx:<10} {prob:.6f}    {label}")

    results = {
        "model": args.model,
        "device": args.device,
        "input_size": list(input_tensor.shape),
        "output_shape": list(logits.shape),
        "top5_indices": top_indices,
        "top5_probs": [round(p, 6) for p in top_probs],
        "top5_labels": [labels[i] if i < len(labels) else f"class_{i}" for i in top_indices],
        "avg_inference_time_ms": round(avg_time * 1000, 2),
        "num_runs": args.num_runs,
    }

    with open(f"results_{args.device}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to results_{args.device}.json")

    # Also save output for comparison
    torch.save(logits, f"{args.device}_output.pt")
    import numpy as np
    np.save(f"{args.device}_output.npy", logits.numpy())

    # Cleanup
    del model
    import gc
    gc.collect()
    if args.device == 'npu':
        try:
            torch.npu.empty_cache()
        except:
            pass

    return results

if __name__ == "__main__":
    main()
