#!/bin/bash
# ConvNeXt test models: serial execution on Ascend NPU
# Usage: bash run_all_models.sh

set -e

MODELS=(
    "test_convnext.r160_in1k"
    "test_convnext2.r160_in1k"
    "test_convnext3.r160_in1k"
)

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing: $model"
    echo "========================================"

    MODEL_DIR="./${model}"
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    # Download model
    python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('timm/${model}', cache_dir='./model')
"

    # Copy inference scripts
    cp ../scripts/inference.py ./
    cp ../scripts/compare_cpu_npu.py ./
    cp ../scripts/requirements.txt ./

    # Patch model name
    sed -i "s/test_convnext\.r160_in1k/${model}/g" inference.py
    sed -i "s/test_convnext___r160_in1k/${model//./_}_r160_in1k/g" inference.py
    sed -i "s/test_convnext\.r160_in1k/${model}/g" compare_cpu_npu.py

    # Run inference
    python3 inference.py

    # Run accuracy comparison
    python3 compare_cpu_npu.py

    # Generate terminal screenshot
    python3 /opt/atomgit/terminal_screenshot.py \
        --input screenshot_terminal.txt \
        --output terminal_screenshot.png

    # Cleanup model weights
    rm -rf ./model

    cd ..

    echo ""
    echo "========================================"
    echo "Completed: $model"
    echo "========================================"
done

echo ""
echo "All models completed successfully!"
