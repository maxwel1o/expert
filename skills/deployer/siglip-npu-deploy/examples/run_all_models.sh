#!/usr/bin/env bash
# SigLIP/SigLIP2 所有模型串行 NPU 推理示例
# 用法: bash examples/run_all_models.sh

MODELS=(
    "ViT-B-16-SigLIP"
    "ViT-B-16-SigLIP-256"
    "ViT-B-16-SigLIP-384"
    "ViT-B-16-SigLIP-512"
    "ViT-B-16-SigLIP-i18n-256"
    "ViT-B-16-SigLIP2"
    "ViT-B-16-SigLIP2-256"
    "ViT-B-16-SigLIP2-384"
    "ViT-B-16-SigLIP2-512"
    "ViT-B-32-SigLIP2-256"
    "ViT-L-16-SigLIP-256"
    "ViT-L-16-SigLIP-384"
    "ViT-L-16-SigLIP2-256"
    "ViT-L-16-SigLIP2-384"
    "ViT-L-16-SigLIP2-512"
)

echo "=== SigLIP/SigLIP2 Batch NPU Inference (All 15 Models) ==="
echo ""

for model in "${MODELS[@]}"; do
    echo "============================================"
    echo "Processing: $model"
    echo "============================================"

    # NPU 推理
    python3 scripts/infer_siglip.py \
        --model "$model" \
        --image test.jpg \
        --device npu

    # 清理显存
    python3 -c "
import torch
import gc
gc.collect()
torch.npu.empty_cache()
"

    echo ""
done

echo "=== All models processed ==="
