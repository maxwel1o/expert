#!/usr/bin/env python3
"""Example: Run NPU inference for SEMNASNet models."""

from timm import create_model
from timm.data import create_transform, resolve_data_config
from PIL import Image
import torch
import torch.nn.functional as F

MODEL_NAME = "semnasnet_100.rmsp_in1k"
MODEL_PATH = f"/opt/atomgit/.cache/modelscope/hub/models/timm/{MODEL_NAME}"


def load_local_model(model_name, device="cpu"):
    """Load model from local modelscope cache."""
    import os
    model = create_model(model_name, pretrained=False)
    safetensors_path = os.path.join(
        f"/opt/atomgit/.cache/modelscope/hub/models/timm/{model_name}",
        "model.safetensors"
    )
    bin_path = safetensors_path.replace("model.safetensors", "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

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


if __name__ == "__main__":
    main()
