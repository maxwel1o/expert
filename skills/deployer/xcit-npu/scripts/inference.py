#!/usr/bin/env python3
"""
xcit_tiny_24_p8_384.fb_dist_in1k - NPU Inference Script
Run inference on Ascend NPU using timm model.
"""
import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image

# Model path from ModelScope cache
MODEL_DIR = os.path.expanduser("~/.cache/modelscope/hub/models/timm/xcit_tiny_24_p8_384___fb_dist_in1k")
MODEL_FILE = os.path.join(MODEL_DIR, "pytorch_model.bin")
os.environ["TORCH_HOME"] = os.path.dirname(MODEL_DIR)
os.environ["TIMM_USE_LOCAL"] = "1"

def load_image(image_path, transform):
    """Load and transform an image for the model."""
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dimension

def create_transform(data_config):
    """Create transform pipeline from data config."""
    from timm.data import create_transform
    return create_transform(**data_config, is_training=False)

def load_imagenet_labels():
    """Load ImageNet class labels."""
    try:
        import requests
        url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
        response = requests.get(url, timeout=10)
        class_idx = json.loads(response.text)
        labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
        return labels
    except Exception:
        return [f"class_{i}" for i in range(1000)]

def run_inference(device="cpu", model_name="xcit_tiny_24_p8_384.fb_dist_in1k",
                  image_path="test_input.jpg", num_runs=5):
    """Run model inference on specified device."""
    import timm

    print(f"\n{'='*60}")
    print(f"Running inference on {device.upper()}...")
    print(f"{'='*60}")

    # Load model from local checkpoint
    t0 = time.time()
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval()
    model.to(device)
    load_time = time.time() - t0
    print(f"Model loaded on {device}: {load_time:.2f}s")

    # Get model-specific transforms
    data_config = timm.data.resolve_model_data_config(model)
    transform = create_transform(data_config)

    # Load and preprocess image
    t0 = time.time()
    input_tensor = load_image(image_path, transform)
    input_tensor = input_tensor.to(device)
    pre_time = time.time() - t0
    print(f"Image preprocessed: {pre_time:.2f}s, shape={input_tensor.shape}")

    # Warm-up run
    with torch.no_grad():
        _ = model(input_tensor)

    # Benchmark runs
    times = []
    output = None
    for i in range(num_runs):
        t0 = time.time()
        with torch.no_grad():
            out = model(input_tensor)
        elapsed = time.time() - t0
        times.append(elapsed)
        output = out
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms")

    avg_time = np.mean(times) * 1000
    print(f"\nAverage inference time: {avg_time:.2f}ms")

    # Post-process
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs, 5)

    # Load labels
    labels = load_imagenet_labels()

    print(f"\nTop-5 predictions ({device.upper()}):")
    for i in range(5):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item() * 100
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  {i+1}. {label} ({idx}): {prob:.2f}%")

    result = {
        "device": device,
        "model": model_name,
        "avg_inference_time_ms": avg_time,
        "load_time_s": load_time,
        "preprocess_time_s": pre_time,
        "top5_indices": top5_indices[0].cpu().tolist(),
        "top5_probs": [round(p * 100, 4) for p in top5_probs[0].cpu().tolist()],
        "top5_labels": [labels[idx] if idx < len(labels) else f"class_{idx}" for idx in top5_indices[0].cpu().tolist()],
        "logits": output.cpu().numpy().flatten().tolist(),
    }

    # Clean up
    del model, input_tensor, output
    import gc
    gc.collect()
    if "npu" in device:
        torch.npu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_input.jpg"

    # Run on CPU
    cpu_result = run_inference(device="cpu", image_path=image_path)

    # Run on NPU
    npu_result = run_inference(device="npu:0", image_path=image_path)

    # Save results
    output = {"cpu": cpu_result, "npu": npu_result}
    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nResults saved to inference_results.json")
