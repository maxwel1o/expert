#!/bin/bash
# phi-2 vLLM-Ascend 服务启动脚本
# 使用 1x Ascend910B4 NPU
set -e

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

MODEL_PATH=${MODEL_PATH:-"/opt/atomgit/.cache/modelscope/hub/models/microsoft/phi-2"}

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --seed 42 \
  --served-model-name phi-2 \
  --max-num-seqs 64 \
  --max-model-len 2048 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --dtype float16 \
  --no-enable-prefix-caching
