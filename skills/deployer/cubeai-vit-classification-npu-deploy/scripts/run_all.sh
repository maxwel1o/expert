#!/bin/bash
# Batch script: Run all cubeai ViT classification models on NPU
# Usage: bash run_all.sh

set -e

WORKSPACE="/opt/atomgit/workspace"
CACHE_DIR="/opt/atomgit/.cache/modelscope"
SCREENSHOT_TOOL="/opt/atomgit/terminal_screenshot.py"

declare -a MODELS=(
  "cv_level1_protected_animals_classification"
  "67_cat_breeds_image_detection"
  "brain_model"
  "133_dog_breeds_image_detection"
  "bird_species_image_detection"
  "bug_classifier"
  "cv_edible_wild_plants_classification"
  "cv_forest_pest_detection"
  "100_butterfly_types_image_detection"
  "215_mushroom_types_image_detection"
  "birds_transform_full"
)

echo "============================================"
echo "CubeAI ViT Classification - NPU Batch Deploy"
echo "============================================"
echo "Total models: ${#MODELS[@]}"
echo ""

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "========================================"
  echo "[$(date '+%H:%M:%S')] Processing: ${MODEL}"
  echo "========================================"

  MODEL_DIR="${CACHE_DIR}/cubeai/${MODEL}"
  WORK_DIR="${WORKSPACE}/${MODEL}"

  if [ ! -d "${MODEL_DIR}" ]; then
    echo "  [SKIP] Model not found: ${MODEL_DIR}"
    echo "  Download first with: python3 -c \"from modelscope import snapshot_download; snapshot_download('cubeai/${MODEL}')\""
    continue
  fi

  mkdir -p "${WORK_DIR}"

  # Step 1: CPU inference
  echo "  [1/4] CPU Inference..."
  python3 "${WORK_DIR}/inference.py" --device cpu 2>&1 | tail -8

  # Step 2: NPU inference
  echo "  [2/4] NPU Inference..."
  python3 "${WORK_DIR}/inference.py" --device npu 2>&1 | tail -8

  # Step 3: CPU/NPU comparison
  echo "  [3/4] CPU vs NPU Comparison..."
  python3 "${WORK_DIR}/compare_cpu_npu.py" 2>&1 | tail -15

  # Step 4: Generate screenshot
  echo "  [4/4] Generating screenshot..."
  if [ -f "${SCREENSHOT_TOOL}" ]; then
    python3 -c "
import sys; sys.path.insert(0, '$(dirname ${SCREENSHOT_TOOL})')
from terminal_screenshot import render_terminal_screenshot
text = open('${WORK_DIR}/compare_report.json').read()
render_terminal_screenshot('Results for ${MODEL}', '${WORK_DIR}/terminal_screenshot.png')
" 2>&1 || echo "  [WARN] Screenshot generation skipped"
  fi

  # Release NPU memory
  python3 -c "
import torch, gc
torch.npu.empty_cache()
gc.collect()
print('  [DONE] NPU memory released')
" 2>&1

  echo "  [OK] ${MODEL} completed"
done

echo ""
echo "========================================"
echo "All models processed!"
echo "========================================"
