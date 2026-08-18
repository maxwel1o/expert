#!/usr/bin/env bash
# SigLIP/SigLIP2 单模型 NPU 推理示例
# 用法: bash examples/run_single_model.sh <model_name>

MODEL_NAME="${1:-ViT-B-16-SigLIP}"

echo "=== SigLIP/SigLIP2 Single Model NPU Inference ==="
echo "Model: $MODEL_NAME"
echo ""

# Step 1: 下载模型权重
echo "Step 1: Downloading model from ModelScope..."
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
path = snapshot_download('timm/${MODEL_NAME}')
print(f'Downloaded to: {path}')
"

# Step 2: NPU 推理
echo ""
echo "Step 2: Running NPU inference..."
python3 scripts/infer_siglip.py \
    --model "$MODEL_NAME" \
    --image test.jpg \
    --device npu

# Step 3: CPU 推理对比
echo ""
echo "Step 3: Running CPU inference for comparison..."
python3 scripts/infer_siglip.py \
    --model "$MODEL_NAME" \
    --image test.jpg \
    --device cpu

echo ""
echo "=== Done ==="
