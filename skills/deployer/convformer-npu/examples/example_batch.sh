#!/bin/bash
# Example: Run multiple ConvFormer models serially on NPU
MODELS=(
    "convformer_b36.sail_in1k_384"
    "convformer_b36.sail_in1k"
    "convformer_s18.sail_in1k"
)
for model in "${MODELS[@]}"; do
    echo "Processing: $model"
    python3 inference.py --model "$model" --device npu
    python3 -c "import gc, torch; gc.collect(); torch.npu.empty_cache()"
done
