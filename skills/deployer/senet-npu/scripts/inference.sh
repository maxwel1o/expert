#!/bin/bash
# SE-Net NPU Deployment Script
# Usage: bash scripts/inference.sh <model_name> [device]
# Example: bash scripts/inference.sh seresnet50.a1_in1k npu

set -e

MODEL_NAME="${1:-seresnet50.a1_in1k}"
DEVICE="${2:-npu}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Set HF mirror
export HF_ENDPOINT=https://hf-mirror.com

echo "========================================="
echo "SE-Net NPU Inference"
echo "Model: $MODEL_NAME"
echo "Device: $DEVICE"
echo "========================================="

# Run inference
python3 "$BASE_DIR/examples/example.py" \
    --model-name "$MODEL_NAME" \
    --device "$DEVICE"

echo "========================================="
echo "Inference complete!"
echo "========================================="
