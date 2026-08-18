#!/usr/bin/env python3
"""NPU 推理脚本 - 注入 transfer_to_npu 自动迁移"""
import os
import sys
import pickle
import numpy as np

# ============================================================
# 昇腾 NPU 自动迁移注入（必须在所有其他 import 之前）
# ============================================================
import torch_npu
from torch_npu.contrib import transfer_to_npu

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

print(f"[NPU] Loading model from: {model_path}")
print(f"[NPU] torch_npu available: {torch_npu.npu.is_available()}")

processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

device = torch.device('npu:0')
model = model.to(device)

# 构造与 CPU 完全相同的随机输入
random_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=random_image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"[NPU] Output logits shape: {logits.shape}")
print(f"[NPU] Output logits sample: {logits[0, :5].cpu().tolist()}")

# 保存结果用于精度对比
result = {
    'logits': logits.cpu().numpy(),
    'device': 'npu'
}
output_path = '/opt/atomgit/vit-base-patch16-224-npu/scripts/npu_output.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(result, f)
print(f"[NPU] Result saved to: {output_path}")
