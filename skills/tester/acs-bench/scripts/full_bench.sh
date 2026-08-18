#!/bin/bash
# One-click full benchmark workflow
# Usage: bash full_bench.sh <WORKDIR> [TOKENIZER_PATH] <PROVIDER_CONFIG> [LABELS] [NUM_REQS] [CONCS]
#
# LABELS are the user-facing length labels (e.g., 90k, 150k, 200k).
# They flow through to generate_datasets.sh and run_stress.sh as-is.
# TOKENIZER_PATH defaults to Qwen/Qwen3-32B if omitted.
#
# Example:
#   bash full_bench.sh ./bench ./tokenizer/Qwen3-32B ./providers.yaml "90k 150k" "1 2 4" "1 2 4"
#   bash full_bench.sh ./bench ./providers.yaml "90k 150k"  (uses default Qwen3-32B)

set -e

WORKDIR=${1:?Usage: $0 <WORKDIR> [TOKENIZER] <PROVIDER> [LABELS] [NUM_REQS] [CONCS]}
shift

DEFAULT_TOKENIZER="Qwen/Qwen3-32B"
TOKENIZER_PATH="${1:-$DEFAULT_TOKENIZER}"

# If the first arg looks like a provider config (contains .yaml/.yml), use default tokenizer
if [[ "$TOKENIZER_PATH" == *.yaml ]] || [[ "$TOKENIZER_PATH" == *.yml ]]; then
  PROVIDER_CONFIG="$TOKENIZER_PATH"
  TOKENIZER_PATH="$DEFAULT_TOKENIZER"
else
  PROVIDER_CONFIG="${2:?Provider config required}"
  shift
fi
shift

RESULTDIR="$WORKDIR/results"

LABELS_STR="${1:-90k 150k 200k}"
NUM_REQS_STR="${2:-1 2 4 8}"
CONCS_STR="${3:-1 2 4 8}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================="
echo "  ACS-Bench Full Benchmark Workflow"
echo "========================================="
echo "Workdir:      $WORKDIR"
echo "Tokenizer:    $TOKENIZER_PATH"
echo "Provider:     $PROVIDER_CONFIG"
echo "Labels:       $LABELS_STR"
echo "Num requests: $NUM_REQS_STR"
echo "Concurrency:  $CONCS_STR"
echo "========================================="

# Step 1: Generate datasets
echo ""
echo ">>> Step 1: 生成数据集"
bash "$SCRIPT_DIR/generate_datasets.sh" "$WORKDIR" "$TOKENIZER_PATH" $LABELS_STR

# Step 2: Run stress tests
echo ""
echo ">>> Step 2: 压力测试"
bash "$SCRIPT_DIR/run_stress.sh" "$WORKDIR" "$PROVIDER_CONFIG" "$RESULTDIR" "$LABELS_STR" "$NUM_REQS_STR" "$CONCS_STR"

# Step 3: Parse results & generate report
echo ""
echo ">>> Step 3: 生成报告"
python3 "$SCRIPT_DIR/parse_results.py" "$RESULTDIR" "$WORKDIR/压测报告.md"

# Step 4: Compress & Deliver results
echo ""
echo ">>> Step 4: 压缩产物"
cd "$WORKDIR"
python3 -c "
import zipfile, os
zf = zipfile.ZipFile('results.zip', 'w', zipfile.ZIP_DEFLATED)
for root, dirs, files in os.walk('results'):
    for f in files:
        fp = os.path.join(root, f)
        zf.write(fp, fp)
zf.write('压测报告.md', '压测报告.md')
zf.close()
print(f'压缩包: results.zip ({os.path.getsize(\"results.zip\")//1024}KB)')
"

echo ""
echo ">>> Step 5: 交付报告"
python3 "$SCRIPT_DIR/deliver_results.py" "$WORKDIR/压测报告.md" "$WORKDIR/results.zip"

echo ""
echo "========================================="
echo "  完成！"
echo "  报告: $WORKDIR/压测报告.md"
echo "  数据: $WORKDIR/results.zip"
echo "========================================="
