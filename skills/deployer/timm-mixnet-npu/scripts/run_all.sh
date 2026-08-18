#!/bin/bash
# Run all MixNet models sequentially to avoid NPU OOM
# Usage: bash run_all.sh <models_base_dir>

set -e

MODELS_DIR="${1:-./models}"
SKIP_DOWNLOAD="${2:-false}"

MODELS=(
  "tf_mixnet_s.in1k"
  "tf_mixnet_m.in1k"
  "tf_mixnet_l.in1k"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "timm MixNet NPU Batch Verification"
echo "=========================================="
echo "Models base dir: $MODELS_DIR"
echo ""

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "=========================================="
  echo "Processing: $MODEL"
  echo "=========================================="

  MODEL_DIR="$MODELS_DIR/$MODEL"
  mkdir -p "$MODEL_DIR"

  # Step 1: CPU inference
  echo "[Step 1] CPU inference for $MODEL"
  python3 "$SCRIPT_DIR/inference.py" \
    --model "$MODEL" \
    --checkpoint-dir "$MODEL_DIR" \
    --device cpu

  # Step 2: NPU inference
  echo "[Step 2] NPU inference for $MODEL"
  python3 "$SCRIPT_DIR/inference.py" \
    --model "$MODEL" \
    --checkpoint-dir "$MODEL_DIR" \
    --device npu

  # Step 3: Compare
  echo "[Step 3] CPU/NPU comparison for $MODEL"
  python3 "$SCRIPT_DIR/compare_cpu_npu.py"

  # Step 4: Save results
  echo "[Step 4] Saving results"
  mkdir -p "$MODEL_DIR/results"
  mv logits_*.pt output_*.json comparison_results.json "$MODEL_DIR/results/" 2>/dev/null || true

  # Step 5: Release memory
  echo "[Step 5] Releasing memory"
  python3 -c "
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
except Exception:
    pass
print('Memory released')
"

  echo "[Done] $MODEL completed"
done

echo ""
echo "=========================================="
echo "All models completed!"
echo "=========================================="
