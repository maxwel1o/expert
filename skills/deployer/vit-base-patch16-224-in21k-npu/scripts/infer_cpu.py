#!/usr/bin/env python3
"""CPU Baseline 推理脚本"""
import os
import sys
import pickle
import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 固定随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)

model_path_file = '/opt/atomgit/vit-base-patch16-224-npu/model_weights/model_path.txt'
if os.path.exists(model_path_file):
    with open(model_path_file) as f:
        model_path = f.read().strip()
else:
    model_path = 'google/vit-base-patch16-224-in21k'

print(f"[CPU] Loading model from: {model_path}")

processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

device = torch.device('cpu')
model = model.to(device)

# 构造随机输入（模拟 224x224 RGB 图像）
random_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=random_image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"[CPU] Output logits shape: {logits.shape}")
print(f"[CPU] Output logits sample: {logits[0, :5].tolist()}")

# 保存结果用于精度对比
result = {
    'logits': logits.cpu().numpy(),
    'device': 'cpu'
}
output_path = '/opt/atomgit/vit-base-patch16-224-npu/scripts/cpu_output.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(result, f)
print(f"[CPU] Result saved to: {output_path}")
