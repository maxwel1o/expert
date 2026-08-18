#!/usr/bin/env bash
# Deploy and test Sambert-HifiGAN TTS on Ascend NPU
# Usage: bash run_all.sh [model_name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/.."
cd "${WORK_DIR}"

MODEL="${1:-all}"
TEXT="${2:-北京今天天气怎么样}"
DEVICE="${3:-npu}"

echo "==========================================="
echo "Sambert-HifiGAN TTS NPU Deployment"
echo "==========================================="
echo "Model: ${MODEL}"
echo "Text: ${TEXT}"
echo "Device: ${DEVICE}"
echo "==========================================="

# 1. Install dependencies
echo "[Step 1] Installing dependencies..."
pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple/

# 2. Setup ttsfrd stub
echo "[Step 2] Setting up ttsfrd stub..."
mkdir -p /tmp/ttsfrd_stub/ttsfrd
cp ttsfrd_stub.py /tmp/ttsfrd_stub/ttsfrd/__init__.py

# 3. Download model
echo "[Step 3] Downloading model..."
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('iic/${MODEL}', cache_dir='/path/to/modelscope/hub')
"

# 4. CPU inference
echo "[Step 4] Running CPU inference..."
python3 inference.py \
  --model "${MODEL}" \
  --voice zhitian_emo \
  --text "${TEXT}" \
  --device cpu \
  --output "results/${MODEL}_cpu.wav" \
  2>&1 | tee "logs/${MODEL}_cpu.log"

# 5. NPU inference
echo "[Step 5] Running NPU inference..."
python3 inference.py \
  --model "${MODEL}" \
  --voice zhitian_emo \
  --text "${TEXT}" \
  --device npu \
  --output "results/${MODEL}_npu.wav" \
  2>&1 | tee "logs/${MODEL}_npu.log"

# 6. Compare CPU vs NPU
echo "[Step 6] Comparing CPU vs NPU..."
python3 compare_cpu_npu.py \
  --model "${MODEL}" \
  --voice zhitian_emo \
  --text "${TEXT}" \
  2>&1 | tee "logs/${MODEL}_compare.log"

# 7. Resource cleanup
echo "[Step 7] Releasing resources..."
python3 -c "
import gc
gc.collect()
import torch
if hasattr(torch, 'npu'):
    torch.npu.empty_cache()
"

echo "==========================================="
echo "Done: ${MODEL}"
echo "==========================================="
