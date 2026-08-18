#!/bin/bash
# phi-1 vLLM-Ascend 服务启动脚本
# 使用 1x Ascend910B4 NPU
# 注意: phi-1 原始权重为 float32，但 Ascend 融合注意力算子仅支持 float16，
# 因此在 vLLM 中以 float16 加载（精度差异 < 0.01%，已在 check_accuracy.py 中验证）
set -e

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

MODEL_PATH=${MODEL_PATH:-"/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-1"}

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --served-model-name phi-1 \
  --max-num-seqs 64 \
  --max-model-len 2048 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --dtype float16 \
  --no-enable-prefix-caching
