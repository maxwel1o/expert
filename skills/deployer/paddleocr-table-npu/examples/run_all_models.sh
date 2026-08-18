#!/bin/bash
# 串行跑 4 个 PaddleOCR 表格模型的 NPU 部署,每步释放 NPU 显存。
# 严格按仓里 README/infer.py 路径,走 PaddleX 框架,不在脚本里写自定义 preprocess/postprocess。

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$SKILL_DIR/scripts"
RESULTS_DIR="$SKILL_DIR/results"
mkdir -p "$RESULTS_DIR"

# 模型列表: model_key : cli_args_for_run_inference
# 4 个模型按仓里的 run 方式调对应的 run_inference.py
declare -A MODEL_CMDS

MODEL_CMDS[doclayout]="python3 ${SCRIPTS_DIR}/models/doclayout/run_inference.py --model_dir /workspace/PP-DocLayout_plus-L --image_dir <DEMO_IMG> --out_dir <OUT>"
MODEL_CMDS[seal_det]="python3 ${SCRIPTS_DIR}/models/seal_det/run_inference.py --model_dir /workspace/PP-OCRv4_server_seal_det --image_dir <DEMO_IMG> --out_dir <OUT>"
MODEL_CMDS[table_cls]="python3 ${SCRIPTS_DIR}/models/table_cls/run_inference.py --model_dir /workspace/PP-LCNet_x1_0_table_cls --image <DEMO_IMG> --out_dir <OUT>"
MODEL_CMDS[table_cell]="python3 ${SCRIPTS_DIR}/models/table_cell/run_inference.py --model_dir /workspace/RT-DETR-L_wired_table_cell_det --image <DEMO_IMG> --out_dir <OUT>"

# demo 图
declare -A DEMO_IMG
DEMO_IMG[doclayout]=/workspace/PP-DocLayout_plus-L-sact/test_Doclayout_plus.png
DEMO_IMG[seal_det]=/workspace/PP-OCRv4_server_seal_det-OM/test_seal.png
DEMO_IMG[table_cls]=/workspace/PP-DocLayout_plus-L-sact/PP-LCNet_x1_0_table_cls/test_table.png
DEMO_IMG[table_cell]=/workspace/PP-DocLayout_plus-L-sact/RT-DETR-L_wired_table_cell_det-OM/test_rt.png

MODELS=("doclayout" "seal_det" "table_cls" "table_cell")

echo "============================================"
echo "  paddleocr-table-npu: 4 models serial run"
echo "============================================"
echo "Skill dir: $SKILL_DIR"
echo "Results:   $RESULTS_DIR"
echo ""

source /usr/local/Ascend/ascend-toolkit/set_env.sh

for MODEL_KEY in "${MODELS[@]}"; do
    echo ""
    echo "############################################"
    echo "#  Processing: $MODEL_KEY"
    echo "############################################"

    MODEL_OUT_DIR="$RESULTS_DIR/${MODEL_KEY}_out"
    mkdir -p "$MODEL_OUT_DIR"
    TEMPLATE="${MODEL_CMDS[$MODEL_KEY]}"
    DEMO="${DEMO_IMG[$MODEL_KEY]}"
    CMD="${TEMPLATE//<DEMO_IMG>/$DEMO}"
    CMD="${CMD//<OUT>/$MODEL_OUT_DIR}"

    echo "  cmd: $CMD"
    if eval "$CMD"; then
        echo "  [OK]"
    else
        echo "  [FAIL]"
    fi

    # 释放 NPU 显存(每个模型跑完都清)
    python3 -c "
import gc; gc.collect()
try:
    import torch
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()
except Exception: pass
" 2>/dev/null || true
    echo "  [NPU cache cleaned]"
done

echo ""
echo "============================================"
echo "  All models done!"
echo "  Results: $RESULTS_DIR"
echo "============================================"
ls -la "$RESULTS_DIR"
