#!/bin/bash
# Batch run all TinyViT models for CPU/NPU accuracy comparison
# Serial execution to avoid NPU OOM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS=(
    "tiny_vit_5m_224.dist_in22k:224"
    "tiny_vit_5m_224.in1k:224"
    "tiny_vit_5m_224.dist_in22k_ft_in1k:224"
    "tiny_vit_21m_512.dist_in22k_ft_in1k:512"
    "tiny_vit_21m_384.dist_in22k_ft_in1k:384"
    "tiny_vit_21m_224.dist_in22k_ft_in1k:224"
    "tiny_vit_21m_224.in1k:224"
    "tiny_vit_21m_224.dist_in22k:224"
    "tiny_vit_11m_224.dist_in22k:224"
    "tiny_vit_11m_224.in1k:224"
    "tiny_vit_11m_224.dist_in22k_ft_in1k:224"
)

for entry in "${MODELS[@]}"; do
    IFS=':' read -r model_name img_size <<< "$entry"
    echo ""
    echo "========================================"
    echo "Processing: $model_name (img=$img_size)"
    echo "========================================"
    python3 "$SCRIPT_DIR/compare_cpu_npu.py" \
        --model-name "$model_name" \
        --image-size "$img_size" 2>&1
    echo "Sleeping 2 seconds before next model..."
    sleep 2
done

echo ""
echo "========== ALL MODELS COMPLETED =========="
