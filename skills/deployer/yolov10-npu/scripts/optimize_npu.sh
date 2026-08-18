#!/bin/bash
# ====================================================================
# YOLOv10 Ascend NPU 推理性能优化启动脚本
# 作用: 设置运行时优化环境变量，启动推理/评测脚本
# ====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    echo "用法: $0 [options] [python_args...]"
    echo ""
    echo "选项:"
    echo "  --script SCRIPT   要运行的脚本 (默认: yolo_npu_evaluate.py)"
    echo "  --no-task-queue   不启用 TASK_QUEUE_ENABLE"
    echo "  --no-cpu-affinity 不启用 CPU 绑核"
    echo "  --no-tcmalloc     不启用 tcmalloc"
    echo "  --task-queue-mode MODE  TASK_QUEUE_ENABLE 模式 (1 或 2, 默认: 2)"
    echo "  --help            显示此帮助"
    echo ""
    echo "默认运行时优化:"
    echo "  TASK_QUEUE_ENABLE=2           # Host-bound 下发优化"
    echo "  PYTORCH_NPU_ALLOC_CONF=...    # NPU 内存分配优化"
    echo "  CPU_AFFINITY_CONF=2           # 细粒度 CPU 绑核"
    echo "  LD_PRELOAD=libtcmalloc.so     # 高性能内存分配器"
    echo "  COMBINED_ENABLE=1             # NPU 算子融合"
    echo "  ACL_OP_COMPILER_CACHE_MODE=1  # 算子编译缓存"
}

SCRIPT="evaluate.py"
USE_TASK_QUEUE=true
USE_CPU_AFFINITY=true
USE_TCMALLOC=true
TASK_QUEUE_MODE=2

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --script) SCRIPT="$2"; shift 2 ;;
        --no-task-queue) USE_TASK_QUEUE=false; shift ;;
        --no-cpu-affinity) USE_CPU_AFFINITY=false; shift ;;
        --no-tcmalloc) USE_TCMALLOC=false; shift ;;
        --task-queue-mode) TASK_QUEUE_MODE="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        --) shift; POSITIONAL+=("$@"); break ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

echo "============================================"
echo " YOLOv10 Ascend NPU 推理优化"
echo "============================================"

# ---- 1. NPU 运行时优化 ----
if [ "$USE_TASK_QUEUE" = true ]; then
    export TASK_QUEUE_ENABLE=$TASK_QUEUE_MODE
    echo "  [OK] TASK_QUEUE_ENABLE=$TASK_QUEUE_MODE"
fi
if [ "$USE_CPU_AFFINITY" = true ]; then
    export CPU_AFFINITY_CONF=2
    echo "  [OK] CPU_AFFINITY_CONF=2"
fi
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
echo "  [OK] PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
export COMBINED_ENABLE=1
export ACL_OP_COMPILER_CACHE_MODE=1
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_op_cache
mkdir -p /tmp/npu_op_cache
echo "  [OK] 算子编译缓存: /tmp/npu_op_cache"

# ---- 2. tcmalloc ----
if [ "$USE_TCMALLOC" = true ]; then
    TCMALLOC_PATH=$(find / -name "libtcmalloc.so*" 2>/dev/null | head -1)
    if [ -n "$TCMALLOC_PATH" ]; then
        export LD_PRELOAD="$TCMALLOC_PATH${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "  [OK] LD_PRELOAD=$TCMALLOC_PATH"
    else
        echo "  [WARN] tcmalloc 未找到, 跳过 (apt install libgoogle-perftools-dev)"
    fi
fi

echo "--------------------------------------------"

cd "$SCRIPT_DIR"
PYTHON_SCRIPT="${POSITIONAL[0]:-$SCRIPT}"
POSITIONAL=("${POSITIONAL[@]:1}")
echo "运行: python $PYTHON_SCRIPT ${POSITIONAL[@]}"
echo "============================================"
python "$PYTHON_SCRIPT" "${POSITIONAL[@]}"
