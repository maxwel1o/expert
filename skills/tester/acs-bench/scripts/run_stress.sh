#!/bin/bash
# Run stress test matrix for acs-bench with retry on rate limit
# Usage: bash run_stress.sh <WORKDIR> <PROVIDER_CONFIG> <RESULTDIR> [LABELS] [NUM_REQS] [CONCS] [OUTPUT_LEN]
#
# LABELS are the user-facing length labels (e.g., 90k, 150k, 200k).
# They must match the labels used in generate_datasets.sh.
# Directory names and result tags use labels as-is (e.g., 90k_nr1_cc1).

WORKDIR=${1:?Usage: $0 <WORKDIR> <PROVIDER> <RESULTDIR> [LABELS] [NUM_REQS] [CONCS] [OUTPUT_LEN]}
PROVIDER_CONFIG=${2:?}
RESULTDIR=${3:?}
shift 3

# Defaults
IFS=' ' read -ra LABELS <<< "${1:-90k 150k 200k}"
IFS=' ' read -ra NUM_REQUESTS <<< "${2:-1 2 4 8}"
IFS=' ' read -ra CONCURRENCIES <<< "${3:-1 2 4 8}"
OUTPUT_LEN=${4:-100}

# Retry settings
MAX_RETRIES=3
RETRY_WAIT=120  # seconds to wait on rate limit

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

mkdir -p "$RESULTDIR"

for LABEL in "${LABELS[@]}"; do
  LEN=$(resolve_length "$LABEL")
  INPUT_PATH="$WORKDIR/dataset_${LABEL}/"

  if [ ! -d "$INPUT_PATH" ]; then
    echo "⚠️  数据集 $INPUT_PATH 不存在，跳过 label=$LABEL (length=$LEN)"
    continue
  fi

  for NR in "${NUM_REQUESTS[@]}"; do
    for CC in "${CONCURRENCIES[@]}"; do
      TAG="${LABEL}_nr${NR}_cc${CC}"
      OUTDIR="$RESULTDIR/$TAG"
      mkdir -p "$OUTDIR"

      echo "=== [$(date)] 压测 $TAG (input=$LEN, nr=$NR, cc=$CC, out=$OUTPUT_LEN) ==="

      RETRY=0
      while [ $RETRY -le $MAX_RETRIES ]; do
        acs-bench prof \
          --provider "$PROVIDER_CONFIG" \
          --dataset-type custom --input-path "$INPUT_PATH" \
          --concurrency-backend threading-pool \
          --backend openai-chat \
          --warmup 1 --epochs 2 \
          --num-requests "$NR" --concurrency "$CC" \
          --input-length "$LEN" --output-length "$OUTPUT_LEN" \
          --benchmark-save-path "$OUTDIR/" 2>&1 | tail -5

        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
          echo "=== [$(date)] $TAG 完成 ✅ ==="
          break
        fi

        # Check if failure is due to rate limit
        RETRY=$((RETRY + 1))
        if [ $RETRY -le $MAX_RETRIES ]; then
          echo "⚠️  [$(date)] $TAG 失败 (exit $EXIT_CODE)，${RETRY}/${MAX_RETRIES} 次重试，等待 ${RETRY_WAIT}s..."
          sleep $RETRY_WAIT
          # Exponential backoff
          RETRY_WAIT=$((RETRY_WAIT * 2))
        else
          echo "❌ [$(date)] $TAG 最终失败 (已重试 ${MAX_RETRIES} 次)"
        fi
      done
    done
  done
done

echo ""
echo "=== [$(date)] 所有压测完毕 ==="
echo "结果目录: $RESULTDIR"
