#!/usr/bin/env python3
"""Quick start example for ConvNeXt NPU inference."""
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

model_name = "convnext_nano.in12k_ft_in1k"
device = "npu" if torch.npu.is_available() else "cpu"

# Load model
model = timm.create_model(model_name, pretrained=True)
model = model.eval()
if device == "npu":
    model = model.npu()

# Load and preprocess image
img = Image.open("test.jpg").convert("RGB")
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
input_tensor = transform(img).unsqueeze(0)
if device == "npu":
    input_tensor = input_tensor.npu()

# Inference
with torch.no_grad():
    output = model(input_tensor)

# Get top-5 predictions
probs = torch.nn.functional.softmax(output[0], dim=0)
top5 = probs.topk(5)
print(f"Device: {device}")
print(f"Top-5 predictions:")
for i in range(5):
    print(f"  {i+1}. class={top5.indices[i].item():5d}  prob={top5.values[i].item():.6f}")
