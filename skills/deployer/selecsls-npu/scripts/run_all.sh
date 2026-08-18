#!/bin/bash
# 串行执行所有 SelecSLS 模型的 NPU 推理和精度验证
# 每个模型完成后释放资源再处理下一个

set -e

MODELS=("selecsls60b.in1k" "selecsls60.in1k" "selecsls42b.in1k")

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "[$(date +'%H:%M:%S')] Processing: $model"
    echo "=========================================="

    mkdir -p "$model"
    cd "$model"

    echo "[STEP 1/4] CPU Inference..."
    python3 ../scripts/inference.py --model "$model" --image ../test_image.jpg --device cpu --output cpu_results.json

    echo "[STEP 2/4] NPU Inference..."
    python3 ../scripts/inference.py --model "$model" --image ../test_image.jpg --device npu --output npu_results.json

    echo "[STEP 3/4] CPU/NPU Precision Comparison..."
    python3 ../scripts/compare_cpu_npu.py --model "$model" --image ../test_image.jpg --output compare_results.json

    echo "[STEP 4/4] Cleaning up resources..."
    python3 -c "
import gc, torch
gc.collect()
if hasattr(torch, 'npu'):
    torch.npu.empty_cache()
print('Memory and NPU cache released.')
"

    cd ..
    echo "[DONE] $model completed."
    echo ""
done

echo "=========================================="
echo "All models processed successfully!"
echo "=========================================="
