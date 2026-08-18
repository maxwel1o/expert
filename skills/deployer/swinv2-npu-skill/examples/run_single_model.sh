#!/usr/bin/env bash
# Example: Run single SwinV2 model inference on NPU
# Usage: bash run_single_model.sh <model_name> [test_image_path]

set -e

MODEL_NAME="${1:-swinv2_tiny_window8_256.ms_in1k}"
TEST_IMAGE="${2:-https://github.com/pytorch/hub/raw/master/images/dog.jpg}"
MODELSCOPE_DIR="${MODELSCOPE_DIR:-./modelscope_cache}"

echo "=== SwinV2 NPU Inference ==="
echo "Model: ${MODEL_NAME}"
echo "Test image: ${TEST_IMAGE}"
echo "ModelScope cache: ${MODELSCOPE_DIR}"
echo ""

# Step 1: Run inference (CPU + NPU)
echo "--- Step 1: Inference ---"
export MODEL_NAME="${MODEL_NAME}"
export TEST_IMAGE="${TEST_IMAGE}"
export MODELSCOPE_DIR="${MODELSCOPE_DIR}"
python3 ../scripts/inference.py

echo ""
echo "--- Step 2: CPU/NPU Precision Comparison ---"
python3 ../scripts/compare_cpu_npu.py

echo ""
echo "=== Done ==="
echo "Results saved to:"
echo "  - results/inference_results.json"
echo "  - results/comparison_results.json"
