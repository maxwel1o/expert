#!/usr/bin/env python3
"""SwinV2 Image Classification Inference on CPU and NPU.

Loads model from local ModelScope cache, runs CPU and NPU inference.
"""
import os
import sys
import json
import time

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

MODEL_NAME = os.environ.get("MODEL_NAME", "")
if not MODEL_NAME:
    print("ERROR: MODEL_NAME environment variable is required")
    sys.exit(1)

MODELSCOPE_DIR = os.environ.get("MODELSCOPE_DIR", "/opt/atomgit/batch19/modelscope_cache")
TEST_IMAGE = os.environ.get("TEST_IMAGE", "/opt/atomgit/batch19/test_image.jpg")

# Check NPU availability
NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
DEVICE_COUNT = torch.npu.device_count() if NPU_AVAILABLE else 0
print(f"PyTorch version: {torch.__version__}")
print(f"NPU available: {NPU_AVAILABLE}, Device count: {DEVICE_COUNT}")
if NPU_AVAILABLE:
    print(f"NPU device: {torch.npu.get_device_name(0)}")
print(f"Model: {MODEL_NAME}")

def load_local_model(model_name, modelscope_dir):
    """Load model from local ModelScope cache."""
    # Find model directory in cache
    import glob as glob_mod
    model_path = None
    for d in glob_mod.glob(os.path.join(modelscope_dir, "timm", f"{model_name.replace('/', '___')}*")):
        model_path = d
        break
    if not model_path:
        for d in glob_mod.glob(os.path.join(modelscope_dir, "timm", f"{model_name.split('.')[0]}*")):
            model_path = d
            break
    if not model_path:
        # Try direct name
        for d in glob_mod.glob(os.path.join(modelscope_dir, "timm", "*")):
            if model_name.replace("/", "_") in d or model_name.replace(".", "_") in d:
                model_path = d
                break

    print(f"Looking for model in: {modelscope_dir}/timm/")
    print(f"Model path: {model_path}")

    if not model_path or not os.path.exists(model_path):
        print(f"ERROR: Model not found in local cache: {model_name}")
        sys.exit(1)

    # Look for weight files
    safetensors_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")

    # Load config
    config_path = os.path.join(model_path, "config.json")

    # Create model without pretrained weights first
    model = timm.create_model(model_name, pretrained=False, checkpoint_path="")

    # Load pretrained weights
    if os.path.exists(safetensors_path):
        print(f"Loading from safetensors: {safetensors_path}")
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        print(f"Loading from pytorch_model.bin: {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        print(f"ERROR: No weight files found in {model_path}")
        for f in os.listdir(model_path):
            print(f"  {f}")
        sys.exit(1)

    # Filter state dict - only keep keys that match model
    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v

    missing = set(model_state.keys()) - set(filtered.keys())
    if missing:
        print(f"Warning: Missing keys: {len(missing)}")

    model.load_state_dict(filtered, strict=False)
    print(f"Loaded weights from local cache ({len(filtered)}/{len(model_state)} keys)")
    return model

def load_image(img_path):
    """Load and preprocess an image."""
    img = Image.open(img_path).convert("RGB")
    return img

def run_inference(device):
    """Run model inference on specified device."""
    print(f"\n--- Running inference on {device} ---")

    # Load model from local cache
    model = load_local_model(MODEL_NAME, MODELSCOPE_DIR)

    # Get the model's data config
    data_config = resolve_data_config({}, model=model)
    print(f"Data config: {data_config}")

    model = model.to(device)
    model.eval()

    transform = create_transform(**data_config)
    img = load_image(TEST_IMAGE)
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    # Synchronize before timing
    if device == "npu":
        torch.npu.synchronize()

    # Timed inference
    num_runs = 10
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
    if device == "npu":
        torch.npu.synchronize()
    elapsed = time.time() - start

    avg_time = elapsed / num_runs
    print(f"Average inference time over {num_runs} runs: {avg_time*1000:.2f} ms")

    probs = torch.nn.functional.softmax(output[0], dim=0)

    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5)

    result = {
        "model": MODEL_NAME,
        "device": device,
        "avg_inference_time_ms": round(avg_time * 1000, 2),
        "logits": output[0].detach().cpu().tolist(),
        "top5_indices": top5_indices.detach().cpu().tolist(),
        "top5_probs": [round(p.item(), 6) for p in top5_probs],
    }

    print(f"Top-5 predictions (device={device}):")
    for i in range(5):
        print(f"  {top5_indices[i].item()}: {top5_probs[i].item():.6f}")

    return model, result, input_tensor

# Run CPU inference
cpu_model, cpu_result, cpu_input = run_inference("cpu")

# Run NPU inference
if NPU_AVAILABLE:
    npu_model, npu_result, npu_input = run_inference("npu")

# Save results
output = {"cpu": cpu_result}
if NPU_AVAILABLE:
    output["npu"] = npu_result

os.makedirs("results", exist_ok=True)
with open("results/inference_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nInference results saved to results/inference_results.json")
