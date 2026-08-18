#!/usr/bin/env python3
"""
AIMv2 通用权重转换脚本
将 Apple 原始 safetensors 格式转换为 transformers wrapper 模型格式
支持所有 12 个变体（large/huge/1B/3B × 224/336/448）

用法: MODEL_SIZE=3B IMG_SIZE=224 python3 aimv2_weight_convert.py
"""
import json, os, sys
from safetensors.torch import load_file
import torch

# ============ 配置 ============
MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = os.environ.get('IMG_SIZE', '224')
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'

MODEL_PATH = os.path.expanduser(
    f'~/.cache/modelscope/hub/models/apple/{MODEL_NAME}'
)
OUTPUT_PATH = os.path.join(MODEL_PATH, 'converted_model.pth')

# Layer 数映射
NUM_LAYERS_MAP = {'large': 16, 'huge': 24, '1B': 24, '3B': 24}
NUM_LAYERS = NUM_LAYERS_MAP.get(MODEL_SIZE, 24)

# ============ 验证路径 ============
if not os.path.exists(MODEL_PATH):
    print(f"! Error: {MODEL_PATH} not found")
    print(f"! Please download the model first:")
    print(f"  modelscope download --model apple/{MODEL_NAME}")
    sys.exit(1)

index_path = os.path.join(MODEL_PATH, 'model.safetensors.index.json')
if not os.path.exists(index_path):
    print(f"! Error: {index_path} not found")
    print(f"! Please ensure the safetensors weights exist in {MODEL_PATH}")
    sys.exit(1)

print(f"Converting {MODEL_NAME} ({NUM_LAYERS} layers)")
print(f"Model path: {MODEL_PATH}")

# 1. 加载所有 safetensors 文件
with open(index_path) as f:
    index = json.load(f)

weight_map = index['weight_map']
unique_files = set(weight_map.values())
print(f"Found safetensors files: {unique_files}")

all_tensors = {}
for fname in sorted(unique_files):
    fpath = os.path.join(MODEL_PATH, fname)
    print(f"  Loading {fname}...")
    tensors = load_file(fpath)
    all_tensors.update(tensors)

print(f"Loaded {len(all_tensors)} tensors from safetensors")

# 2. 构造映射
new_state_dict = {}

# 2a. Patch embedding
new_state_dict['preprocessor.patchifier.proj.weight'] = all_tensors['embeddings.patch_embed.weight']
new_state_dict['preprocessor.patchifier.proj.bias'] = all_tensors['embeddings.patch_embed.bias']

# 2b. Patch norm (RMSNorm)
new_state_dict['preprocessor.patchifier.norm.weight'] = all_tensors['embeddings.rms_norm.weight']

# 2c. Position embedding
new_state_dict['preprocessor.pos_embed'] = all_tensors['embeddings.position_embedding.weight'].unsqueeze(0)

# 2d. Transformer layers
for i in range(NUM_LAYERS):
    prefix_safe = f'encoder.layers.{i}'
    prefix_model = f'trunk.blocks.{i}'

    # Attention: q_proj, k_proj, v_proj → cat → qkv
    q = all_tensors[f'{prefix_safe}.attention.q_proj.weight']
    k = all_tensors[f'{prefix_safe}.attention.k_proj.weight']
    v = all_tensors[f'{prefix_safe}.attention.v_proj.weight']
    qkv = torch.cat([q, k, v], dim=0)
    new_state_dict[f'{prefix_model}.attn.qkv.weight'] = qkv

    # Output projection
    new_state_dict[f'{prefix_model}.attn.proj.weight'] = all_tensors[f'{prefix_safe}.attention.out_proj.weight']

    # MLP: gate_proj→fc1, down_proj→fc2, up_proj→fc3
    new_state_dict[f'{prefix_model}.mlp.fc1.weight'] = all_tensors[f'{prefix_safe}.ffn.gate_proj.weight']
    new_state_dict[f'{prefix_model}.mlp.fc2.weight'] = all_tensors[f'{prefix_safe}.ffn.down_proj.weight']
    new_state_dict[f'{prefix_model}.mlp.fc3.weight'] = all_tensors[f'{prefix_safe}.ffn.up_proj.weight']

    # Norms
    new_state_dict[f'{prefix_model}.norm_1.weight'] = all_tensors[f'{prefix_safe}.rms_norm1.weight']
    new_state_dict[f'{prefix_model}.norm_2.weight'] = all_tensors[f'{prefix_safe}.rms_norm2.weight']

# 2e. Post trunk norm
new_state_dict['trunk.post_trunk_norm.weight'] = all_tensors['rms_norm.weight']

# 3. 验证
expected_num = 1 + 1 + 1 + 1 + NUM_LAYERS * (1 + 1 + 3 + 2) + 1
print(f"\nConverted {len(new_state_dict)}/{expected_num} parameter groups")

# 4. 保存
torch.save(new_state_dict, OUTPUT_PATH)
print(f"Saved converted weights to {OUTPUT_PATH}")
file_size = os.path.getsize(OUTPUT_PATH) / 1024**3
print(f"Size: {file_size:.2f} GB")
print("✓ Weight conversion complete")
