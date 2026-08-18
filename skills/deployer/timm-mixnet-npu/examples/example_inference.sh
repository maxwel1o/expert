#!/bin/bash
# Example: Run NPU inference for tf_mixnet_s.in1k
# Usage: bash example_inference.sh

MODEL_NAME="tf_mixnet_s.in1k"
CHECKPOINT_DIR="./checkpoints/$MODEL_NAME"

# Download model from ModelScope
echo "Downloading $MODEL_NAME..."
python3 -c "
from modelscope import snapshot_download
model_path = snapshot_download('timm/$MODEL_NAME', cache_dir='./models')
print('Downloaded to:', model_path)
"

# Run NPU inference
echo "Running NPU inference..."
python3 ../scripts/inference.py \
  --model "$MODEL_NAME" \
  --checkpoint-dir "./models/timm/${MODEL_NAME//./_}" \
  --device npu

echo "Done!"
