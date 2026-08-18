#!/usr/bin/env python3
"""Inference script for VOLO models on Ascend NPU - loading from local cache."""
import argparse
import json
import time
import os
import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model
from timm.data import resolve_data_config, create_transform

CACHE_DIR = "/opt/atomgit/.cache/modelscope/timm"
INDEX_FILE = "/opt/atomgit/.cache/modelscope/volo_models_index.json"

def get_weight_path(model_name):
    """Find local weight file for a model."""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE) as f:
            idx = json.load(f)
        wp = idx.get(model_name)
        if wp and os.path.exists(wp):
            return wp
    local_dir = os.path.join(CACHE_DIR, model_name)
    for fn in ["model.safetensors", "pytorch_model.bin"]:
        fp = os.path.join(local_dir, fn)
        if os.path.exists(fp):
            return fp
    return None

def load_model_local(model_name, device):
    """Load model from local ModelScope cache."""
    print(f"Loading {model_name} on {device}...")
    start = time.time()
    model = create_model(model_name, pretrained=False)

    weight_path = get_weight_path(model_name)
    if weight_path is None:
        raise FileNotFoundError(f"No weight file found for {model_name} in {CACHE_DIR}")

    if weight_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    return model

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
    parser = argparse.ArgumentParser(description='VOLO NPU Inference')
    parser.add_argument('--model', type=str, required=True, help='timm model name')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'npu'], help='Device')
    parser.add_argument('--num_runs', type=int, default=10, help='Inference runs for timing')
    args = parser.parse_args()

    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Task type: Image Classification")
    print(f"Framework: PyTorch + timm")
    print(f"Architecture: VOLO (Vision Outlooker)")

    device = torch.device(args.device)

    if args.image is None:
        from urllib.request import urlretrieve
        test_img_path = "/tmp/test_volo.jpg"
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

    model = load_model_local(model_name, device)
    input_tensor, input_size = preprocess_image(args.image, model)
    print(f"Input size: {input_size}")
    print(f"Batch shape: {input_tensor.shape}")

    logits, probs, _ = run_inference(model, input_tensor, device)
    print(f"Output shape: {logits.shape}")

    times = []
    for i in range(args.num_runs):
        _, _, t = run_inference(model, input_tensor, device)
        times.append(t)

    avg_time = sum(times) / len(times)
    print(f"\nInference times ({args.num_runs} runs):")
    print(f"  Average: {avg_time*1000:.2f} ms")
    print(f"  Min: {min(times)*1000:.2f} ms")
    print(f"  Max: {max(times)*1000:.2f} ms")

    top_probs, top_indices = torch.topk(probs, k=5, dim=1)
    labels = load_imagenet_labels()

    print(f"\nTop-5 predictions:")
    print(f"{'Rank':<6} {'Class ID':<10} {'Probability':<12} {'Label'}")
    print(f"{'-'*60}")
    for i in range(5):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"{i+1:<6} {idx:<10} {prob:.6f}    {label}")

    results = {
        "model": model_name,
        "device": args.device,
        "input_size": list(input_size),
        "output_shape": list(logits.shape),
        "top5_indices": top_indices[0].tolist(),
        "top5_probs": [round(p, 6) for p in top_probs[0].tolist()],
        "top5_labels": [labels[i] if i < len(labels) else f"class_{i}" for i in top_indices[0].tolist()],
        "avg_inference_time_ms": round(avg_time * 1000, 2),
        "num_runs": args.num_runs,
    }

    with open(f"results_{args.device}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to results_{args.device}.json")

    torch.save(logits, f"{args.device}_output.pt")

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
