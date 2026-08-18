#!/usr/bin/env bash
# dm_nfnet 单模型 NPU 推理示例
# 用法: bash examples/run_single_model.sh <model_name>

MODEL_NAME="${1:-dm_nfnet_f0.dm_in1k}"

echo "=== dm_nfnet Single Model NPU Inference ==="
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
python3 scripts/infer_nfnet.py \
    --model "$MODEL_NAME" \
    --image test.jpg \
    --device npu

# Step 3: CPU/NPU 精度对比
echo ""
echo "Step 3: Running CPU/NPU comparison..."
python3 scripts/compare_cpu_npu.py \
    --model "$MODEL_NAME" \
    --image test.jpg

# Step 4: 清理显存
python3 -c "
import torch
import gc
gc.collect()
torch.npu.empty_cache()
"

echo ""
echo "=== Done ==="
