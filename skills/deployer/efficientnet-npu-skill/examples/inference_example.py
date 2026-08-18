#!/usr/bin/env python3
"""Example: Run NPU inference for EfficientNet models (batch_14)."""
from timm import create_model
from timm.data import create_transform, resolve_data_config
from PIL import Image
import torch
import torch.nn.functional as F

MODEL_NAME = "tf_efficientnet_b0.in1k"


def load_local_model(model_name, device="cpu"):
    """Load model from local modelscope cache."""
    import os
    model = create_model(model_name, pretrained=False)

    # Try common cache paths
    cache_paths = [
        f"/opt/atomgit/.cache/modelscope/hub/models/timm/{model_name}",
        f"/opt/atomgit/batch_14/modelscope_cache/timm/{model_name}",
    ]
    model_path = None
    for p in cache_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(f"Model not found in cache: {model_name}")

    safetensors_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def main():
    device = "npu:0" if torch.npu.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = load_local_model(MODEL_NAME, device=device)

    # Create test image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # Preprocess
    data_config = resolve_data_config(model=model)
    transform = create_transform(**data_config)
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Get predictions
    probs = F.softmax(output.cpu(), dim=1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)

    print(f"\n=== Top-5 Predictions ({device}) ===")
    for i in range(5):
        print(f"  {i+1}. [{top5_indices[0][i].item()}] prob: {top5_probs[0][i].item():.6f}")

    # Memory cleanup
    del model, input_tensor, output
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
