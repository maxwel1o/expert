#!/bin/bash
# Full pipeline: Run all LeViT models on NPU, compare accuracy, generate READMEs
# Usage: bash examples/full_pipeline.sh

set -e

echo "=========================================="
echo "  LeViT NPU Full Pipeline"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "[Step 1] Installing dependencies..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch torchvision transformers Pillow numpy requests

# Step 2: Verify NPU
echo ""
echo "[Step 2] Verifying NPU..."
python3 -c "
import torch
if hasattr(torch, 'npu') and torch.npu.is_available():
    print(f'NPU available: {torch.npu.get_device_name(0)}')
    print(f'NPU count: {torch.npu.device_count()}')
else:
    print('ERROR: NPU not available')
    exit(1)
"

# Step 3: Download models
echo ""
echo "[Step 3] Downloading models..."
for model in facebook/levit-128 facebook/levit-128S facebook/levit-192 facebook/levit-256 facebook/levit-384; do
    python3 -c "from modelscope import snapshot_download; snapshot_download('${model}')" 2>&1 | tail -1
done

# Step 4: Create test image
echo ""
echo "[Step 4] Creating test image..."
python3 -c "
from PIL import Image
Image.new('RGB', (224, 224), color=(73, 109, 137)).save('test.jpg')
echo 'Created test.jpg'
" 2>&1

# Step 5: Run all models
echo ""
echo "[Step 5] Running batch inference..."
python3 scripts/batch_run.py --models all --device npu 2>&1

# Step 6: Summary
echo ""
echo "[Step 6] Summary of results..."
python3 -c "
import json
models = ['levit-128', 'levit-128S', 'levit-192', 'levit-256', 'levit-384']
print(f'{\"Model\":<15} {\"CPU(ms)\":<10} {\"NPU(ms)\":<10} {\"Speedup\":<10} {\"Error%\":<10}')
print('-'*55)
for m in models:
    try:
        cpu = json.load(open(f'/tmp/{m}_cpu_results.json'))
        npu = json.load(open(f'/tmp/{m}_npu_results.json'))
        comp = json.load(open(f'/tmp/{m}_comparison.json'))
        print(f'{m:<15} {cpu[\"time_ms\"]:<10.2f} {npu[\"time_ms\"]:<10.2f} {cpu[\"time_ms\"]/npu[\"time_ms\"]:<10.1f}x {comp[\"probs_error_pct\"]:<10.6f}%')
    except Exception as e:
        print(f'{m:<15} ERROR: {e}')
" 2>&1

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
