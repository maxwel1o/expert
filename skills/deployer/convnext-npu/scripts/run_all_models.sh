#!/bin/bash
# Batch run all ConvNeXt models serially
set -e

WORK_DIR="/opt/atomgit/convnext_workspace"
CACHE_DIR="${WORK_DIR}/modelscope_cache"
TEST_IMAGE="${WORK_DIR}/test_image.jpg"
RESULTS_CSV="${WORK_DIR}/all_results.csv"
MODELSCOPE_CACHE="${CACHE_DIR}"

echo "Model,Status,CPU_Top1,NPU_Top1,Top1_Match,Top5_Overlap,Rel_Error%,Cosine_Sim,CPU_Time(s),NPU_Time(s),Speedup" > "$RESULTS_CSV"

declare -a MODELS=(
    "convnext_nano.in12k_ft_in1k"
    "convnext_nano.d1h_in1k"
    "convnext_nano.in12k"
    "convnext_nano.r384_in12k_ft_in1k"
    "convnext_nano.r384_ad_in12k"
    "convnext_nano.r384_in12k"
    "convnext_nano_ols.d1h_in1k"
    "convnext_pico.d1_in1k"
    "convnext_pico_ols.d1_in1k"
    "convnext_small.in12k"
    "convnext_small.in12k_ft_in1k"
    "convnext_small.fb_in22k_ft_in1k"
    "convnext_small.fb_in1k"
    "convnext_small.fb_in22k"
    "convnext_small.fb_in22k_ft_in1k_384"
    "convnext_small.in12k_ft_in1k_384"
    "convnext_tiny.fb_in22k_ft_in1k"
    "convnext_tiny.fb_in22k"
    "convnext_tiny.in12k_ft_in1k"
    "convnext_tiny.fb_in1k"
    "convnext_tiny.in12k_ft_in1k_384"
    "convnext_tiny.fb_in22k_ft_in1k_384"
    "convnext_tiny.in12k"
    "convnext_tiny_hnf.a2h_in1k"
)

run_model() {
    local MODEL_NAME="$1"
    local MODEL_DIR="${WORK_DIR}/models/${MODEL_NAME}"
    local MODEL_DOTNAME="${MODEL_NAME//./___}"

    echo ""
    echo "========================================"
    echo "Processing: ${MODEL_NAME}"
    echo "Started: $(date)"
    echo "========================================"

    mkdir -p "$MODEL_DIR"

    # Step 1: Download from ModelScope
    echo "[Step 1] Downloading from ModelScope..."
    python3 << PYEOF 2>&1 | tail -5
import os
os.environ['MODELSCOPE_CACHE'] = '${CACHE_DIR}'
from modelscope.hub.snapshot_download import snapshot_download
model_id = "timm/${MODEL_NAME}"
try:
    model_dir = snapshot_download(model_id, cache_dir="${CACHE_DIR}")
    print(f"Downloaded to: {model_dir}")
except Exception as e:
    print(f"Download error: {e}")
PYEOF

    local MODEL_PATH="${CACHE_DIR}/timm/${MODEL_DOTNAME}/model.safetensors"
    if [ ! -f "$MODEL_PATH" ]; then
        MODEL_PATH="${CACHE_DIR}/timm/${MODEL_NAME}/model.safetensors"
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        echo "FAILED: Model weights not found after download"
        echo "${MODEL_NAME},FAILED-DOWNLOAD,,,,,,,," >> "$RESULTS_CSV"
        return 1
    fi

    # Step 2: CPU Inference
    echo "[Step 2] CPU Inference..."
    python3 "${WORK_DIR}/templates/inference.py" \
        --model "$MODEL_NAME" --device cpu --image "$TEST_IMAGE" \
        --model-path "$MODEL_PATH" 2>&1 | tee "${MODEL_DIR}/cpu_output.log"

    # Step 3: NPU Inference
    echo "[Step 3] NPU Inference..."
    python3 "${WORK_DIR}/templates/inference.py" \
        --model "$MODEL_NAME" --device npu --image "$TEST_IMAGE" \
        --model-path "$MODEL_PATH" 2>&1 | tee "${MODEL_DIR}/npu_output.log"

    # Step 4: CPU/NPU Comparison
    echo "[Step 4] CPU/NPU Comparison..."
    python3 "${WORK_DIR}/templates/compare_cpu_npu.py" \
        --model "$MODEL_NAME" --image "$TEST_IMAGE" \
        --model-path "$MODEL_PATH" 2>&1 | tee "${MODEL_DIR}/comparison.log"

    # Step 5: Copy artifacts
    cp "${WORK_DIR}/templates/inference.py" "${MODEL_DIR}/inference.py"
    cp "${WORK_DIR}/templates/compare_cpu_npu.py" "${MODEL_DIR}/compare_cpu_npu.py"

    cat > "${MODEL_DIR}/requirements.txt" << 'EOF'
torch>=2.0.0
torch-npu>=2.0.0
timm>=0.9.0
Pillow>=10.0.0
modelscope>=1.0.0
safetensors>=0.4.0
EOF

    # Step 6: Parse results
    if [ -f "${MODEL_DIR}/comparison_results.json" ]; then
        python3 << PYEOF
import json
with open("${MODEL_DIR}/comparison_results.json") as f:
    r = json.load(f)
top1_match = "Y" if r["top1_match"] else "N"
print(f"{r['model']},OK,{r['cpu_top1']['class']},{r['npu_top1']['class']},{top1_match},{r['top5_overlap']}/5,{r['mean_relative_error_pct']:.4f}%,{r['cosine_similarity']:.8f},{r['cpu_time_s']:.4f},{r['npu_time_s']:.4f},{r['speedup']:.2f}x")
PYEOF
    fi

    # Step 7: Generate screenshot
    python3 /tmp/acend-/terminal_screenshot.py \
        --text "$(cat "${MODEL_DIR}/cpu_output.log" "${MODEL_DIR}/npu_output.log" "${MODEL_DIR}/comparison.log" 2>/dev/null)" \
        --output "${MODEL_DIR}/screenshot.png" 2>/dev/null || echo "Screenshot skipped"

    # Step 8: Memory cleanup
    python3 -c "
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
except: pass
" 2>/dev/null

    echo "========================================"
    echo "Completed: ${MODEL_NAME} at $(date)"
    echo "========================================"
}

# Run all models serially
for model in "${MODELS[@]}"; do
    # Parse result and append to CSV
    result=$(run_model "$model" 2>&1)
    echo "$result"

    # Extract JSON result line for CSV
    csv_line=$(echo "$result" | grep "^${model},OK," || true)
    if [ -n "$csv_line" ]; then
        echo "$csv_line" >> "$RESULTS_CSV"
    else
        echo "${model},FAILED,,,,,,,," >> "$RESULTS_CSV"
    fi

    echo ""
    echo "--- Waiting 2s before next model ---"
    sleep 2
done

echo ""
echo "========================================"
echo "All models completed!"
echo "Results: ${RESULTS_CSV}"
cat "$RESULTS_CSV"
echo "========================================"
