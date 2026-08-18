#!/bin/bash
# SPNASNet_100: serial execution on Ascend NPU
# Usage: bash run_model.sh

set -e

MODEL_NAME="spnasnet_100.rmsp_in1k"
WORK_DIR="./${MODEL_NAME}"

echo ""
echo "========================================"
echo "Processing: ${MODEL_NAME}"
echo "========================================"

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Download model from ModelScope
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('timm/${MODEL_NAME}', cache_dir='./model')
"

# Copy inference scripts
SCRIPT_DIR="$(dirname "$0")/../scripts"
cp "${SCRIPT_DIR}/inference.py" ./
cp "${SCRIPT_DIR}/compare_cpu_npu.py" ./
cp "${SCRIPT_DIR}/requirements.txt" ./

# Install dependencies if needed
pip install -r requirements.txt -q

# Create a test image
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (224, 224), (128, 128, 128))
draw = ImageDraw.Draw(img)
draw.ellipse([50, 50, 174, 174], fill=(200, 100, 50))
draw.rectangle([20, 20, 100, 100], fill=(50, 150, 200))
img.save('test_image.jpg')
print('Test image created')
"

# Run inference (CPU + NPU)
echo ""
echo "--- Running inference ---"
python3 inference.py

# Run accuracy comparison
echo ""
echo "--- CPU/NPU Precision Comparison ---"
python3 compare_cpu_npu.py

# Generate terminal screenshot
echo ""
echo "--- Generating terminal screenshot ---"
python3 /opt/atomgit/terminal_screenshot.py \
    --text "SPNASNet_100 NPU Inference Results\n" \
    --output terminal_screenshot.png

# Cleanup model weights to save space
rm -rf ./model

cd ..

echo ""
echo "========================================"
echo "Completed: ${MODEL_NAME}"
echo "========================================"
