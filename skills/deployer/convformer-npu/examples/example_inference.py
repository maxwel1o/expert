#!/usr/bin/env python3
"""Example: Run ConvFormer NPU inference."""
import torch, timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

model = timm.create_model("convformer_b36.sail_in1k_384", pretrained=True).npu().eval()
img = Image.open("test.jpg").convert("RGB")
config = resolve_data_config({}, model=model)
tensor = create_transform(**config)(img).unsqueeze(0).npu()

with torch.no_grad():
    out = model(tensor)
    probs = torch.nn.functional.softmax(out[0], dim=0)
    top5 = probs.topk(5)
    for i in range(5):
        print(f"{i+1}. class={top5.indices[i].item()} prob={top5.values[i].item():.6f}")
