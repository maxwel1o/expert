#!/bin/bash
# Download all Sambert-HifiGAN TTS models from ModelScope
# Usage: bash download_models.sh
# This script first verifies which models exist, then downloads them.

set -e

CACHE_DIR="${MODELSCOPE_CACHE:-/opt/atomgit/.cache/modelscope/hub}"
MODELS_DIR="$CACHE_DIR/models/iic"

download_model() {
    local model_id="$1"
    local target_dir="$MODELS_DIR/$model_id"
    if [ -d "$target_dir/voices" ] || [ -d "$target_dir/am" ] 2>/dev/null; then
        echo "[SKIP] $model_id already downloaded"
        return 0
    fi
    echo "[DOWNLOAD] $model_id"
    python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
try:
    snapshot_download('iic/$model_id')
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1 | tail -1
}

# Candidate model IDs based on web search
CANDIDATES=(
    # Chinese (zh-cn) models
    "speech_sambert-hifigan_tts_zh-cn_16k"
    "speech_sambert-hifigan_tts_zhida_zh-cn_16k"
    "speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k"
    "speech_sambert-hifigan_tts_zhiyue_zh-cn_16k"
    "speech_sambert-hifigan_tts_zhisha_zh-cn_16k"
    "speech_sambert-hifigan_tts_zhizhe_emo_zh-cn_16k"
    "speech_sambert-hifigan_tts_zhitian_emo_zh-cn_16k"
    # English models
    "speech_sambert-hifigan_tts_en-us_16k"
    "speech_sambert-hifigan_tts_andy_en-us_16k"
    "speech_sambert-hifigan_tts_annie_en-us_16k"
    "speech_sambert-hifigan_tts_luca_en-gb_16k"
    "speech_sambert-hifigan_tts_luna_en-gb_16k"
    # Special models
    "speech_sambert-hifigan_nsf_tts_cally_en-us_24k"
    "speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k"
)

echo "Step 1: Verifying which models exist on ModelScope..."
echo
EXISTING=()
for model in "${CANDIDATES[@]}"; do
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "https://modelscope.cn/models/iic/$model/summary" 2>/dev/null)
    if [ "$http_code" = "200" ]; then
        echo "  [EXISTS] $model"
        EXISTING+=("$model")
    else
        echo "  [NOT FOUND] $model (HTTP $http_code)"
    fi
done

echo
echo "Step 2: Downloading ${#EXISTING[@]} existing models..."
echo
for model in "${EXISTING[@]}"; do
    download_model "$model"
done

echo
echo "Download complete! Downloaded models:"
for model in "${CANDIDATES[@]}"; do
    target_dir="$MODELS_DIR/$model"
    if [ -d "$target_dir" ] && ls "$target_dir/voices" &>/dev/null 2>&1; then
        echo "  [OK] $model"
    else
        echo "  [--] $model (not available)"
    fi
done
