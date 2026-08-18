#!/usr/bin/env python3
"""Example: Deploy and run VOLO model inference on Ascend NPU."""
import torch
from timm import create_model
from timm.data import resolve_data_config, create_transform
from PIL import Image

# 1. Configure NPU device
device = torch.device('npu:0')
print(f'Using device: {device}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
print(f'Device name: {torch.npu.get_device_name(0)}')

# 2. Load model from local safetensors
model_name = 'volo_d1_224.sail_in1k'
print(f'\nLoading model: {model_name}')

# Create model architecture (without pretrained weights download)
model = create_model(model_name, pretrained=False)

# Load weights from local cache (modelscope timm path)
cache_path = f'/opt/atomgit/.cache/modelscope/timm/{model_name}'
import os
safetensors_path = os.path.join(cache_path, 'model.safetensors')
if os.path.exists(safetensors_path):
    from safetensors.torch import load_file
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
else:
    # Fallback: load pretrained
    model = create_model(model_name, pretrained=True)

model = model.to(device)
model.eval()
print('Model loaded successfully!')

# 3. Prepare input
img = Image.new('RGB', (224, 224), color=(100, 100, 200))
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
input_tensor = transform(img).unsqueeze(0).to(device)
print(f'Input shape: {input_tensor.shape}')
print(f'Input size: {config["input_size"]}')

# 4. Run inference
with torch.no_grad():
    output = model(input_tensor)

import torch.nn.functional as F
probs = F.softmax(output, dim=1)
top_probs, top_indices = torch.topk(probs, k=5, dim=1)

print('\nTop-5 predictions:')
for i in range(5):
    print(f'  {i+1}. class {top_indices[0][i].item():4d}  prob: {top_probs[0][i].item():.6f}')

# 5. Cleanup
del model
import gc
gc.collect()
torch.npu.empty_cache()
print('\nResources released.')
