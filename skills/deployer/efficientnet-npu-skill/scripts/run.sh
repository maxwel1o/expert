#!/usr/bin/env bash
# EfficientNet NPU Deployment Skill - Main Script
# Usage: bash run.sh <model_name> [device]
set -e

MODEL_NAME="${1:-tf_efficientnet_b0.in1k}"
DEVICE="${2:-npu}"

echo "========================================"
echo "EfficientNet NPU Deployment Skill"
echo "Model: ${MODEL_NAME}"
echo "Device: ${DEVICE}"
echo "========================================"

# Step 1: Check environment
echo ""
echo "[Step 1] Checking environment..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'NPU available: {torch.npu.is_available() if hasattr(torch, \"npu\") else False}')
import timm
print(f'timm: {timm.__version__}')
"

# Step 2: Run Inference
echo ""
echo "[Step 2] Running ${DEVICE} inference..."
python3 inference.py "${MODEL_NAME}" --device "${DEVICE}"

# Step 3: If NPU, also run CPU and compare
if [ "${DEVICE}" = "npu" ]; then
    echo ""
    echo "[Step 3] Running CPU inference for comparison..."
    python3 inference.py "${MODEL_NAME}" --device "cpu"

    echo ""
    echo "[Step 4] Running CPU vs NPU accuracy comparison..."
    python3 compare_cpu_npu.py "${MODEL_NAME}"
fi

# Step 4: Memory cleanup
echo ""
echo "[Step 5] Cleaning up..."
python3 -c "
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
        print('NPU cache cleared')
except Exception:
    pass
"

echo ""
echo "========================================"
echo "Done! Model ${MODEL_NAME} completed."
echo "========================================"
