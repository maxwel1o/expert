#!/bin/bash
# Example: Run TinyViT model inference on NPU

echo "=== TinyViT NPU Inference Example ==="
echo ""

# 1. NPU inference for a single model
echo "1. Running NPU inference for tiny_vit_5m_224.in1k"
python3 ../scripts/inference.py \
    --model-name tiny_vit_5m_224.in1k \
    --device npu \
    --image-size 224

echo ""
echo "2. Running CPU/NPU accuracy comparison"
python3 ../scripts/compare_cpu_npu.py \
    --model-name tiny_vit_5m_224.in1k \
    --image-size 224

echo ""
echo "=== Example Complete ==="
