#!/usr/bin/env bash
# dm_nfnet 所有模型串行 NPU 推理示例
# 用法: bash examples/run_all_models.sh

WORKDIR="/opt/atomgit/dm_nfnet_workspace"

MODELS=(
    "dm_nfnet_f0.dm_in1k"
    "dm_nfnet_f1.dm_in1k"
    "dm_nfnet_f2.dm_in1k"
    "dm_nfnet_f3.dm_in1k"
    "dm_nfnet_f4.dm_in1k"
    "dm_nfnet_f5.dm_in1k"
    "dm_nfnet_f6.dm_in1k"
)

echo "=== dm_nfnet Batch NPU Inference (All 7 Models) ==="
echo ""

for model in "${MODELS[@]}"; do
    echo "============================================"
    echo "Processing: $model"
    echo "============================================"

    # NPU 推理
    python3 scripts/infer_nfnet.py \
        --model "$model" \
        --image test.jpg \
        --device npu

    # CPU/NPU 精度对比
    python3 scripts/compare_cpu_npu.py \
        --model "$model" \
        --image test.jpg

    # 清理显存
    python3 -c "
import torch
import gc
gc.collect()
torch.npu.empty_cache()
"

    echo ""
done

echo "=== All models processed ==="
