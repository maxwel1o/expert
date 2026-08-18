#!/usr/bin/env bash
# SKNet NPU Deployment Script
# Usage: bash run.sh <model_name> [action]
#   model_name: skresnext50_32x4d.ra_in1k | skresnet34.ra_in1k | skresnet18.ra_in1k
#   action: all | inference | compare | readme | screenshot | publish

set -euo pipefail

MODEL=${1:-}
ACTION=${2:-all}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_name> [action]"
    echo "Available models:"
    echo "  - skresnext50_32x4d.ra_in1k"
    echo "  - skresnet34.ra_in1k"
    echo "  - skresnet18.ra_in1k"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/../.."
WORK_DIR="$PROJECT_DIR/$MODEL-npu"

echo "============================================"
echo "SKNet NPU Deployment Script"
echo "Model: $MODEL"
echo "Action: $ACTION"
echo "Work dir: $WORK_DIR"
echo "============================================"

# Ensure dependencies
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm torchvision Pillow numpy safetensors 2>&1 | tail -1

check_npu() {
    python3 -c "import torch; print('NPU available:', torch.npu.is_available())" 2>/dev/null | grep -v WARNING || echo "NPU not available"
}

download_model() {
    local model=$1
    python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('timm/$model', cache_dir='/opt/atomgit/.cache/modelscope')
print('Downloaded: $model')
" 2>&1 | grep -v "WARNING\|Warning\|path\|Permission"
}

run_inference() {
    local model=$1
    local device=$2
    python3 "$PROJECT_DIR/inference.py" --model "$model" --device "$device" 2>&1 | grep -v "WARNING\|Warning\|path\|Permission"
}

run_compare() {
    local model=$1
    python3 "$PROJECT_DIR/compare_cpu_npu.py" --model "$model" 2>&1 | grep -v "WARNING\|Warning\|path\|Permission"
}

generate_screenshot() {
    local model=$1
    local safe_name=$(echo "$model" | tr '/' '_')
    local text_file="$PROJECT_DIR/${safe_name}_screenshot.txt"
    local png_file="$WORK_DIR/terminal_screenshot.png"

    if [ -f "$text_file" ] && [ -f "/opt/atomgit/terminal_screenshot.py" ]; then
        python3 /opt/atomgit/terminal_screenshot.py --input "$text_file" --output "$png_file" 2>&1 | grep -v WARNING
        echo "Screenshot saved to $png_file"
    fi
}

# Execute based on action
case "$ACTION" in
    all)
        echo "Step 1: Checking NPU environment..."
        check_npu

        echo "Step 2: Downloading model..."
        download_model "$MODEL"

        echo "Step 3: Running CPU inference..."
        run_inference "$MODEL" "cpu"

        echo "Step 4: Running NPU inference..."
        run_inference "$MODEL" "npu"

        echo "Step 5: Running CPU/NPU comparison..."
        run_compare "$MODEL"

        echo "Step 6: Generating screenshot..."
        generate_screenshot "$MODEL"

        echo "All steps completed for $MODEL"
        ;;
    inference)
        run_inference "$MODEL" "cpu"
        run_inference "$MODEL" "npu"
        ;;
    compare)
        run_compare "$MODEL"
        ;;
    screenshot)
        generate_screenshot "$MODEL"
        ;;
    *)
        echo "Unknown action: $ACTION"
        exit 1
        ;;
esac
