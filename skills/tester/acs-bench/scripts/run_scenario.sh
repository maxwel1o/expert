#!/bin/bash
# ============================================================
# acs-bench 场景化压测执行脚本
# ============================================================
# 读取场景YAML配置，自动构建acs-bench命令
# 用法:
#   bash scripts/run_scenario.sh -s fixedlen-10k-400-deepseek-v3 -c 400 -r 26
#   bash scripts/run_scenario.sh -s varlen-10k-600-deepseek-v3 -c 350 -r 20 -l peak
#   bash scripts/run_scenario.sh -s varlen-3838-600-deepseek-v3 -c 200 -r 15 -p  # dry-run
# ============================================================

set -euo pipefail

# ==================== 默认参数 ====================
SCENARIO=""
CONCURRENCY=""
REQUEST_RATE=""
EPOCHS=""
WARMUP=""
NUM_REQUESTS=""
LOG_PREFIX=""
DATASET_OVERRIDE=""
DRY_RUN=false
# 跑坡(climb)参数
USE_CLIMB=false
CLIMB_MODE=""
GROWTH_RATE=""
GROWTH_INTERVAL=""
INIT_CONCURRENCY=""

# ==================== 固定环境 ====================
PROF_ROOT="${PROF_ROOT:-/root/prof}"
ACS_ENV="${WORK_ROOT:-work_dir}/${CONDA_ENV:-conda_env}/bin/activate"
ACS_BIN="${WORK_ROOT:-work_dir}/${CONDA_ENV:-conda_env}/bin/acs-bench"

# ==================== 颜色 ====================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ==================== 用法 ====================
usage() {
    cat <<EOF
用法: $0 -s SCENARIO -c CONCURRENCY -r REQUEST_RATE [选项]

必填参数:
  -s SCENARIO       场景名（对应 conf/scenarios/<name>.yaml）
  -c CONCURRENCY    并发数

可选参数:
  -r REQUEST_RATE   请求速率 (req/s)，省略则不限速
  -d DATASET        覆盖数据集名（支持注入数据集，如 data_n3838_avg11944_r01_uid）
  -e EPOCHS         测试轮数（覆盖YAML默认）
  -w WARMUP         预热轮数（覆盖YAML默认）
  -n NUM_REQUESTS   总请求数（覆盖YAML默认）
  -l LOG_PREFIX     日志前缀（如 peak/stability/ramp）
  -p                仅打印命令不执行（dry-run）
  -h                显示帮助

跑坡(climb)参数:
  --climb           启用climb跑坡模式（--use-climb）
  --climb-mode MODE 跑坡模式: linear(线性)/static/dynamic, 默认linear
  --growth-rate N   每步并发增长量, 默认0
  --growth-interval MS  增长间隔(毫秒), 默认1000
  --init-concurrency N  初始并发数, 默认等于-c

示例:
  $0 -s fixedlen-10k-400-deepseek-v3 -c 400 -r 26
  $0 -s varlen-10k-600-deepseek-v3 -c 350 -r 20 -l peak
  $0 -s varlen-3838-600-deepseek-v3 -c 200 -r 15 -p
  $0 -s varlen-3838-600-deepseek-v3 -c 1000 -r 50 -d data_n3838_avg11944_r01_uid -l peak
  # 跑坡: 从c=10起步, 每1s增加20并发, 直到c=400
  $0 -s varlen-3838-600-deepseek-v3 -c 400 --climb --growth-rate 20 --growth-interval 1000 --init-concurrency 10 -l ramp
EOF
    exit 0
}

# ==================== 参数解析（短选项+长选项） ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) SCENARIO="$2"; shift 2 ;;
        -c) CONCURRENCY="$2"; shift 2 ;;
        -r) REQUEST_RATE="$2"; shift 2 ;;
        -d) DATASET_OVERRIDE="$2"; shift 2 ;;
        -e) EPOCHS="$2"; shift 2 ;;
        -w) WARMUP="$2"; shift 2 ;;
        -n) NUM_REQUESTS="$2"; shift 2 ;;
        -l) LOG_PREFIX="$2"; shift 2 ;;
        -p) DRY_RUN=true; shift ;;
        -h) usage ;;
        --climb)          USE_CLIMB=true; shift ;;
        --climb-mode)     CLIMB_MODE="$2"; shift 2 ;;
        --growth-rate)    GROWTH_RATE="$2"; shift 2 ;;
        --growth-interval) GROWTH_INTERVAL="$2"; shift 2 ;;
        --init-concurrency) INIT_CONCURRENCY="$2"; shift 2 ;;
        *) err "未知选项: $1"; usage ;;
    esac
done

# ==================== 参数校验 ====================
if [[ -z "$SCENARIO" || -z "$CONCURRENCY" ]]; then
    err "必填参数缺失: -s SCENARIO, -c CONCURRENCY"
    usage
fi

# ==================== 读取场景配置 ====================
SCENARIO_FILE="${PROF_ROOT}/conf/scenarios/${SCENARIO}.yaml"
if [[ ! -f "$SCENARIO_FILE" ]]; then
    err "场景配置不存在: $SCENARIO_FILE"
    exit 1
fi

info "读取场景配置: $SCENARIO_FILE"

# 使用Python解析YAML
parse_yaml() {
    python3 -c "
import yaml
with open('${SCENARIO_FILE}') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
val = cfg
for k in keys:
    if isinstance(val, dict):
        val = val.get(k, '')
    else:
        val = ''
        break
print(val)
"
}

DATASET_TYPE=$(parse_yaml "dataset_type")
DATASET=$(parse_yaml "dataset")
OUTPUT_LENGTH=$(parse_yaml "output_length")
IGNORE_EOS=$(parse_yaml "ignore_eos")
TOKENIZER=$(parse_yaml "tokenizer")
PROVIDER=$(parse_yaml "provider")
YAML_NUM_REQUESTS=$(parse_yaml "num_requests")
YAML_EPOCHS=$(parse_yaml "epochs")
YAML_WARMUP=$(parse_yaml "warmup")
TRUST_REMOTE_CODE=$(parse_yaml "trust_remote_code")

# CLI覆盖YAML默认值
[[ -z "${NUM_REQUESTS:-}" ]] && NUM_REQUESTS="${YAML_NUM_REQUESTS:-1000}"
[[ -z "${EPOCHS:-}" ]] && EPOCHS="${YAML_EPOCHS:-1}"
[[ -z "${WARMUP:-}" ]] && WARMUP="${YAML_WARMUP:-0}"
[[ -z "${TRUST_REMOTE_CODE:-}" || "$TRUST_REMOTE_CODE" == "True" ]] && TRUST_REMOTE_CODE_FLAG="--trust-remote-code"

# Provider路径：如果是相对路径，基于PROF_ROOT
if [[ "$PROVIDER" != /* ]]; then
    # 去掉开头的 ./
    PROVIDER="${PROVIDER#./}"
    PROVIDER="${PROF_ROOT}/${PROVIDER}"
fi

# ==================== 确定数据集路径 ====================
# -d 参数覆盖YAML中的数据集名（用于注入数据集）
if [[ -n "$DATASET_OVERRIDE" ]]; then
    DATASET="$DATASET_OVERRIDE"
    info "数据集覆盖为: $DATASET (来自 -d 参数)"
fi

if [[ "$DATASET_TYPE" == "fixedlen" ]]; then
    # 定长数据集：检查DSV3专用目录
    DATASET_DIR="${PROF_ROOT}/dataset/fixed_length/${DATASET}"
    DSV3_DIR="${PROF_ROOT}/dataset/fixed_length/${DATASET}_dsv30324"

    # 优先使用DSV3 tokenizer版本（如果tokenizer是DSV3且目录存在）
    if [[ "$TOKENIZER" == *"DeepSeek-V3-0324"* && -d "$DSV3_DIR" ]]; then
        INPUT_DIR="$DSV3_DIR"
        info "使用DSV3专用数据集: $DSV3_DIR"
    else
        INPUT_DIR="$DATASET_DIR"
    fi

    # 查找数据集JSON文件（扁平结构：数据文件直接在数据集目录下）
    JSON_FILE=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | head -1)
    if [[ -z "$JSON_FILE" ]]; then
        err "未找到数据集JSON文件: $INPUT_DIR"
        exit 1
    fi
    INPUT_PATH="$JSON_FILE"
    info "数据集文件: $INPUT_PATH"
else
    # 变长数据集
    INPUT_PATH="${PROF_ROOT}/dataset/mt_dataset/${DATASET}.json"
fi

if [[ ! -f "$INPUT_PATH" ]]; then
    err "数据集文件不存在: $INPUT_PATH"
    exit 1
fi

if [[ ! -f "$PROVIDER" ]]; then
    err "Provider配置不存在: $PROVIDER"
    exit 1
fi

# ==================== 检查残留进程 ====================
if ! $DRY_RUN; then
    RESIDUAL=$(ps aux | grep acs-bench | grep -v grep | grep -v "$$" || true)
    if [[ -n "$RESIDUAL" ]]; then
        warn "检测到残留 acs-bench 进程:"
        echo "$RESIDUAL"
        warn "建议先清理再执行，5秒后继续..."
        sleep 5
    fi
fi

# ==================== 激活环境 ====================
if ! $DRY_RUN; then
    info "激活 Conda 环境..."
    source "$ACS_ENV"
    if ! command -v acs-bench &>/dev/null; then
        err "acs-bench 不可用，请检查环境"
        exit 1
    fi
fi

# ==================== 构建日志路径 ====================
TIMESTAMP=$(date +%Y%m%d_%H%M)
RATE_STR="${REQUEST_RATE:-nolimit}"
if [[ -n "$LOG_PREFIX" ]]; then
    LOG_FILE="${PROF_ROOT}/log/${LOG_PREFIX}_${DATASET}_c${CONCURRENCY}_r${RATE_STR}_${TIMESTAMP}.log"
else
    LOG_FILE="${PROF_ROOT}/log/run_${DATASET}_c${CONCURRENCY}_r${RATE_STR}_${TIMESTAMP}.log"
fi

# ==================== 打印配置 ====================
echo ""
info "=========================================="
info "acs-bench 场景化压测配置"
info "=========================================="
echo "  场景:           $SCENARIO"
echo "  并发数:         $CONCURRENCY"
echo "  请求速率:       ${REQUEST_RATE:-不限速(nolimit)}"
echo "  数据集类型:     $DATASET_TYPE"
echo "  数据集:         $DATASET"
echo "  输入路径:       $INPUT_PATH"
echo "  输出长度:       $OUTPUT_LENGTH"
echo "  总请求数:       $NUM_REQUESTS"
echo "  Epochs:         $EPOCHS"
echo "  Warmup:         $WARMUP"
echo "  Ignore-EOS:     $IGNORE_EOS"
echo "  Tokenizer:      $TOKENIZER"
echo "  Provider:       $PROVIDER"
echo "  日志文件:       $LOG_FILE"
echo "  结果目录:       ${PROF_ROOT}/result/csv/"
if $USE_CLIMB; then
echo "  --- 跑坡(Climb)模式 ---"
echo "  模式:           ${CLIMB_MODE:-linear}"
echo "  初始并发:       ${INIT_CONCURRENCY:-$CONCURRENCY}"
echo "  增长速率:       ${GROWTH_RATE:-2} 并发/步"
echo "  增长间隔:       ${GROWTH_INTERVAL:-1000} ms"
fi
info "=========================================="
echo ""

# ==================== 构建命令（数组方式，避免引号转义问题） ====================
CMD_ARGS=(
    acs-bench prof
    --tokenizer "$TOKENIZER"
    $TRUST_REMOTE_CODE_FLAG
    --benchmark-save-path "${PROF_ROOT}/result/csv/"
    --epochs $EPOCHS
    --warmup $WARMUP
    --num-requests $NUM_REQUESTS
    --concurrency-backend threading-pool
    --backend openai-chat
    --input-path "$INPUT_PATH"
    --output-length $OUTPUT_LENGTH
    --ignore-eos $IGNORE_EOS
    --concurrency $CONCURRENCY
)
if [[ -n "$REQUEST_RATE" ]]; then
    CMD_ARGS+=(--request-rate "$REQUEST_RATE")
fi
if $USE_CLIMB; then
    CMD_ARGS+=(--use-climb)
    [[ -n "$CLIMB_MODE" ]] && CMD_ARGS+=(--climb-mode "$CLIMB_MODE")
    [[ -n "$GROWTH_RATE" ]] && CMD_ARGS+=(--growth-rate "$GROWTH_RATE")
    [[ -n "$GROWTH_INTERVAL" ]] && CMD_ARGS+=(--growth-interval "$GROWTH_INTERVAL")
    [[ -n "$INIT_CONCURRENCY" ]] && CMD_ARGS+=(--init-concurrency "$INIT_CONCURRENCY")
fi
CMD_ARGS+=(--provider "$PROVIDER" -D)

# ==================== 执行或打印 ====================
if $DRY_RUN; then
    info "=== Dry Run ==="
    echo ""
    echo "${CMD_ARGS[*]}"
    echo ""
    echo "Log: $LOG_FILE"
else
    info "启动压测 (nohup 后台)..."
    nohup "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    ok "压测已启动"
    echo "  PID:    $PID"
    echo "  日志:   $LOG_FILE"
    echo ""
    info "跟踪日志: tail -f $LOG_FILE"
    info "检查进程: ps -p $PID"
fi
