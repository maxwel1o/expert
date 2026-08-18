#!/bin/bash
# Generate datasets for acs-bench
# Usage: bash generate_datasets.sh <WORKDIR> [TOKENIZER_PATH] [LABELS...]
# Example: bash generate_datasets.sh ./workdir ./tokenizer/Qwen3-32B 90k 150k 200k
#          bash generate_datasets.sh ./workdir 90k 150k 200k  (uses default Qwen3-32B)
#
# LABELS are the user-facing length labels (e.g., 90k, 150k, 200k).
# They are used as-is for directory naming (dataset_90k/).
# The actual token count is resolved by expanding 'k' → ×1000.

set -e

WORKDIR=${1:?Usage: $0 <WORKDIR> [TOKENIZER_PATH] [LABELS...]}
shift

# Default tokenizer: Qwen3-32B
DEFAULT_TOKENIZER="Qwen/Qwen3-32B"
TOKENIZER_PATH="${1:-$DEFAULT_TOKENIZER}"

# If the first arg after WORKDIR looks like a label (contains 'k' or is pure digits), use default tokenizer
if [[ "$TOKENIZER_PATH" == *k ]] || [[ "$TOKENIZER_PATH" =~ ^[0-9]+$ ]]; then
  LABELS=("$TOKENIZER_PATH" "$@")
  TOKENIZER_PATH="$DEFAULT_TOKENIZER"
else
  shift
  LABELS=("$@")
fi

if [ ${#LABELS[@]} -eq 0 ]; then
  LABELS=(90k 150k 200k)
fi

mkdir -p "$WORKDIR"

# Resolve a label like "90k" or "150k" to an actual token count (90000, 150000).
# Pure numbers pass through unchanged.
resolve_length() {
  local label="$1"
  if [[ "$label" == *k ]]; then
    echo "${label%k}000"
  else
    echo "$label"
  fi
}

for LABEL in "${LABELS[@]}"; do
  LEN=$(resolve_length "$LABEL")
  DIR="dataset_${LABEL}"
  if [ -d "$WORKDIR/$DIR" ] && [ "$(ls -A "$WORKDIR/$DIR" 2>/dev/null)" ]; then
    echo "[$(date)] dataset_$LABEL 已存在，跳过"
    continue
  fi
  echo "=== [$(date)] 生成 dataset_$LABEL (input_length=$LEN) ==="
  acs-bench generate dataset \
    --tokenizer "$TOKENIZER_PATH" \
    --dataset-type random \
    --output-path "$WORKDIR/$DIR" \
    --input-length "$LEN" \
    --num-requests 1000
  echo "=== [$(date)] dataset_$LABEL 完成 ==="
done

echo "=== [$(date)] 所有数据集生成完毕 ==="
