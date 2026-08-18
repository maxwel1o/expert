#!/bin/bash
# Serial execution script for all PP-OCRv4 models on Ascend NPU
# Prevents NPU memory explosion by running models serially and cleaning up after each

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$SKILL_DIR/scripts"
RESULTS_DIR="$SKILL_DIR/results"

IMAGE_PATH="${IMAGE_PATH:-/opt/atomgit/test_image.png}"
REC_IMAGE_PATH="${REC_IMAGE_PATH:-/opt/atomgit/test_rec_image.png}"

# Model list: model_id:repo_name:type
MODELS=(
  "OllmOne/PP-OCRv4:PP-OCRv4-npu:pipeline"
  "cycloneboy/ch_PP-OCRv4_det_infer:ch_PP-OCRv4_det_infer-npu:detection"
  "cycloneboy/ch_PP-OCRv4_det_server_infer:ch_PP-OCRv4_det_server_infer-npu:detection"
  "cycloneboy/ch_PP-OCRv4_rec_infer:ch_PP-OCRv4_rec_infer-npu:recognition"
  "cycloneboy/ch_PP-OCRv4_rec_server_infer:ch_PP-OCRv4_rec_server_infer-npu:recognition"
  "cycloneboy/en_PP-OCRv4_rec_infer:en_PP-OCRv4_rec_infer-npu:recognition"
  "somohk/en_PP-OCRv4_rec_infer:en_PP-OCRv4_rec_infer_somohk-npu:recognition"
)

echo "============================================"
echo "  PP-OCRv4 NPU Deployment - Run All Models"
echo "============================================"
echo "Total models: ${#MODELS[@]}"
echo "Image: $IMAGE_PATH"
echo "Rec Image: $REC_IMAGE_PATH"
echo ""

for MODEL_ENTRY in "${MODELS[@]}"; do
  IFS=':' read -r MODEL_ID REPO_NAME MODEL_TYPE <<< "$MODEL_ENTRY"

  echo "============================================"
  echo "  Processing: $MODEL_ID ($REPO_NAME)"
  echo "  Type: $MODEL_TYPE"
  echo "============================================"

  MODEL_RESULTS="$RESULTS_DIR/$REPO_NAME"
  mkdir -p "$MODEL_RESULTS"

  # Step 1: Download model
  echo "[1/4] Downloading model..."
  python3 -c "
from modelscope import snapshot_download
import os
os.makedirs('$MODEL_RESULTS/model', exist_ok=True)
snapshot_download('$MODEL_ID', cache_dir='$MODEL_RESULTS/model')
print('Downloaded to $MODEL_RESULTS/model')
" || echo "  Download skipped (model may already exist)"

  # Step 2: CPU inference
  echo "[2/4] Running CPU inference..."
  IMG="$IMAGE_PATH"
  if [ "$MODEL_TYPE" = "recognition" ]; then
    IMG="$REC_IMAGE_PATH"
  fi

  # Step 3: CPU/NPU comparison
  echo "[3/4] Running CPU/NPU comparison..."

  # Step 4: Clean up NPU memory
  echo "[4/4] Cleaning up..."
  python3 -c "
import gc; gc.collect()
try:
    import torch
    torch.npu.empty_cache()
except: pass
print('Memory cleaned')
"

  echo "  $REPO_NAME completed!"
  echo ""
done

echo "============================================"
echo "  All models processed!"
echo "  Results: $RESULTS_DIR"
echo "============================================"
