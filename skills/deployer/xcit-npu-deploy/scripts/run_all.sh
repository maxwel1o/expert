#!/bin/bash
"""
run_all.sh — Serial execution of all XCiT model NPU inference and CPU/NPU comparison.

Usage:
  bash run_all.sh                        # Run all models serially
  bash run_all.sh --skip-benchmark       # Skip benchmark, only run comparison
  bash run_all.sh --model xcit_tiny_12_p16_384  # Run single model
"""

set -e

MODELS=(
  "xcit_tiny_12_p16_384"
  "xcit_large_24_p8_224"
  "xcit_medium_24_p16_384"
  "xcit_tiny_12_p8_384"
  "xcit_small_12_p8_224"
)

run_single() {
  local model=$1
  echo ""
  echo "========================================"
  echo "  Processing model: $model"
  echo "========================================"

  echo ""
  echo "--- Step 1: CPU vs NPU precision comparison ---"
  python3 scripts/run_compare.py --model "$model"

  echo ""
  echo "--- Step 2: NPU inference benchmark ---"
  python3 scripts/run_inference.py --model "$model" --device npu:0 --warmup 3 --iters 20

  # Release memory
  python3 -c "
import gc
try:
    import torch
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
except: pass
gc.collect()
"
  echo "--- Completed: $model ---"
}

# Main
SKIP_BENCH=false
SINGLE_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-benchmark) SKIP_BENCH=true ;;
    --model) SINGLE_MODEL="$2"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

if [ -n "$SINGLE_MODEL" ]; then
  run_single "$SINGLE_MODEL"
else
  for model in "${MODELS[@]}"; do
    run_single "$model"
    # Inter-model delay to prevent NPU OOM
    sleep 5
  done
fi

echo ""
echo "All models completed successfully!"
