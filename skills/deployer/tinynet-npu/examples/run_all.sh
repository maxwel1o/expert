#!/bin/bash
# TinyNet 全量模型 NPU 推理与精度对比示例脚本
# 按 tinynet_e → d → c → b → a 顺序串行执行

set -e

MODELS="tinynet_e tinynet_d tinynet_c tinynet_b tinynet_a"
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"

echo "============================================"
echo "TinyNet NPU 全量模型推理与精度对比"
echo "============================================"
echo ""

for model in $MODELS; do
    echo ""
    echo "============================================"
    echo "处理模型: $model"
    echo "============================================"
    
    echo ""
    echo "--- NPU 推理 ---"
    python3 "$SCRIPT_DIR/inference.py" --model "$model" --device npu
    
    echo ""
    echo "--- CPU vs NPU 精度对比 ---"
    python3 "$SCRIPT_DIR/compare_cpu_npu.py" --model "$model"
    
    echo ""
    echo "--- 生成终端截图 ---"
    python3 "$SCRIPT_DIR/generate_screenshot.py" --model "$model"
    
    echo ""
    echo "--- 释放 NPU 显存 ---"
    python3 -c "import gc; gc.collect(); import torch; torch.npu.empty_cache()"
    
    echo "完成: $model"
done

echo ""
echo "============================================"
echo "所有模型处理完成！"
echo "============================================"
