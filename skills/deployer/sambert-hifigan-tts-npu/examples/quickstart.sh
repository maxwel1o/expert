#!/usr/bin/env bash
# Quick start example: run zh-cn model on CPU
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/.."
cd "${WORK_DIR}"

echo "=== Quick Start: Sambert-HifiGAN TTS ==="
echo ""
echo "1. CPU Inference (zh-cn)"
python3 scripts/inference.py \
  --model speech_sambert-hifigan_tts_zh-cn_16k \
  --voice zhitian_emo \
  --text "北京今天天气怎么样" \
  --device cpu \
  --output /tmp/tts_quickstart_cpu.wav

echo ""
echo "2. CPU Inference (en-us)"
python3 scripts/inference.py \
  --model speech_sambert-hifigan_tts_en-us_16k \
  --voice andy \
  --text "Hello world" \
  --device cpu \
  --output /tmp/tts_quickstart_en.wav

echo ""
echo "3. CPU vs NPU Comparison"
python3 scripts/compare_cpu_npu.py \
  --model speech_sambert-hifigan_tts_zh-cn_16k \
  --voice zhitian_emo \
  --text "北京今天天气怎么样"

echo ""
echo "Done! Audio saved to /tmp/tts_quickstart_*.wav"
