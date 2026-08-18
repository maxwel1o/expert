#!/usr/bin/env bash
# 串行运行所有 SKNet 模型，避免 NPU 显存爆炸
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$SCRIPT_DIR/.."

MODELS=(
    "skresnext50_32x4d.ra_in1k"
    "skresnet34.ra_in1k"
    "skresnet18.ra_in1k"
)

echo "============================================"
echo "SKNet 模型串行 NPU 部署验证"
echo "共 ${#MODELS[@]} 个模型"
echo "============================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "开始处理: $model"
    echo "============================================"

    # 1. CPU 推理
    echo "[1/5] CPU 推理..."
    python3 "$SKILL_DIR/../inference.py" --model "$model" --device cpu 2>&1 | grep -v "WARNING\|Warning\|path"

    # 2. NPU 推理
    echo "[2/5] NPU 推理..."
    python3 "$SKILL_DIR/../inference.py" --model "$model" --device npu 2>&1 | grep -v "WARNING\|Warning\|path"

    # 3. 精度对比
    echo "[3/5] CPU/NPU 精度对比..."
    python3 "$SKILL_DIR/../compare_cpu_npu.py" --model "$model" 2>&1 | grep -v "WARNING\|Warning\|path"

    # 4. 截图生成
    echo "[4/5] 生成终端截图..."
    safe_name=$(echo "$model" | tr '/' '_')
    txt_file="$SKILL_DIR/../${safe_name}_screenshot.txt"
    png_file="$SKILL_DIR/../${model}-npu/terminal_screenshot.png"
    mkdir -p "$(dirname "$png_file")"
    python3 /opt/atomgit/terminal_screenshot.py --input "$txt_file" --output "$png_file" 2>&1 | grep -v WARNING

    # 5. 释放资源
    echo "[5/5] 释放资源..."
    python3 -c "
import gc
gc.collect()
import torch
if hasattr(torch, 'npu'):
    torch.npu.empty_cache()
    print('NPU cache cleared')
"

    echo "完成: $model"
    echo ""
done

echo "============================================"
echo "所有模型处理完成！"
echo "============================================"
