#!/bin/bash
# ============================================================
# vLLM-Ascend 模型部署脚本
# 
# 工作流程:
# 1. 用户提供镜像 tag + ModelScope 模型名
# 2. 检查本地镜像，不存在则从 quay.io/ascend/vllm-ascend 拉取
# 3. 检查本地模型，不存在则从 modelscope.cn 下载
# 4. 创建容器并在容器内启动 vLLM 服务
#
# 用法:
#   ./deploy_model.sh <预定义别名>
#   ./deploy_model.sh --image <tag> --model <ModelScope模型ID> [选项]
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/model_registry.sh"

# 默认配置
VLLM_REGISTRY="quay.io/ascend/vllm-ascend"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/sfs_turbo/models}"
CONTAINER_NAME_PREFIX="vllm-deploy"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_hint() { echo -e "${CYAN}[HINT]${NC} $1"; }

# 显示帮助
show_help() {
    cat << 'EOF'
用法: ./deploy_model.sh <命令|选项>

命令:
  <预定义别名>           使用预定义配置快速部署
                         例如: ./deploy_model.sh qwen3-32b

  --list                 列出所有预定义模型

  --status               查看当前运行的容器和服务

  --stop <容器名|端口>   停止指定的部署

自定义部署:
  --image <tag>          vLLM-Ascend 镜像 tag (查看: https://quay.io/repository/ascend/vllm-ascend?tab=tags)
  --model <ModelScopeID> ModelScope 模型 ID (查看: https://www.modelscope.cn/models)
                         例如: --model Qwen/Qwen3-32B

部署选项:
  --tp <数量>            tensor parallel 大小 (默认: 1)
  --dp <数量>            data parallel 大小 (默认: 1)
  --port <端口>          服务端口 (默认: 8000)
  --max-len <长度>       max_model_len (默认: 32768)
  --max-seqs <数量>      max_num_seqs (默认: 128)
  --gpu-util <比例>      GPU 显存利用率 (默认: 0.90)

其他选项:
  --dry-run              只打印命令不执行
  --no-pull-image        跳过镜像拉取
  --no-pull-model        跳过模型下载
  --help                 显示此帮助

示例:
  # 使用预定义配置
  ./deploy_model.sh qwen3-32b

  # 自定义部署
  ./deploy_model.sh --image openai-cpu-poc --model Qwen/Qwen3-32B --tp 8 --port 8006

  # 预览命令
  ./deploy_model.sh qwen3-32b --dry-run

参考链接:
  镜像列表: https://quay.io/repository/ascend/vllm-ascend?tab=tags
  模型列表: https://www.modelscope.cn/models
EOF
    exit 0
}

# 显示运行状态
show_status() {
    echo "=== 运行中的 vLLM 部署容器 ==="
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}" | grep -E "vllm|NAME" || echo "(无运行容器)"
    echo ""
    echo "=== 监听的服务端口 ==="
    ss -tlnp 2>/dev/null | grep -E "800[0-9]|State" || echo "(无服务端口)"
}

# 检查 Docker 是否可用
check_docker() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker 未安装或不可用"
        log_hint "请先安装 Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        log_error "无法连接 Docker daemon"
        log_hint "请检查 Docker 服务是否启动，或当前用户是否有 docker 组权限"
        exit 1
    fi
}

# 拉取 vLLM-Ascend 镜像
pull_image() {
    local tag=$1
    local image="${VLLM_REGISTRY}:${tag}"
    
    log_step "检查推理镜像: $image"
    
    if docker image inspect "$image" &>/dev/null; then
        log_info "镜像已存在本地"
        return 0
    fi
    
    if [[ $NO_PULL_IMAGE -eq 1 ]]; then
        log_warn "跳过镜像拉取，镜像不存在将导致部署失败"
        return 1
    fi
    
    log_info "从 quay.io 拉取镜像..."
    log_hint "镜像列表: https://quay.io/repository/ascend/vllm-ascend?tab=tags"
    
    if docker pull "$image"; then
        log_info "镜像拉取成功"
        return 0
    else
        log_error "镜像拉取失败"
        log_hint "请确认 tag 是否正确，访问 https://quay.io/repository/ascend/vllm-ascend?tab=tags 查看可用标签"
        return 1
    fi
}

# 从 ModelScope 下载模型
pull_model() {
    local modelscope_id=$1
    local model_name=$(basename "$modelscope_id")
    local model_path="${MODEL_CACHE_DIR}/${model_name}"
    
    log_step "检查模型: $modelscope_id"
    log_info "本地路径: $model_path"
    
    # 检查模型是否已存在
    if [[ -e "$model_path/config.json" ]] || [[ -e "$model_path/model.safetensors.index.json" ]]; then
        log_info "模型已存在本地"
        echo "$model_path"
        return 0
    fi
    
    if [[ $NO_PULL_MODEL -eq 1 ]]; then
        log_warn "跳过模型下载，模型不存在将导致部署失败"
        echo "$model_path"
        return 1
    fi
    
    log_info "从 ModelScope 下载模型..."
    log_hint "模型页面: https://www.modelscope.cn/models/${modelscope_id}"
    
    # 使用 modelscope 库下载
    mkdir -p "$model_path"
    
    if python3 -c "
from modelscope import snapshot_download
snapshot_download('${modelscope_id}', cache_dir='${MODEL_CACHE_DIR}')
" 2>&1; then
        log_info "模型下载成功"
        echo "$model_path"
        return 0
    else
        log_error "模型下载失败"
        log_hint "请确认 ModelScope ID 是否正确，访问 https://www.modelscope.cn/models 搜索模型"
        log_hint "或手动下载: pip install modelscope && modelscope download --model ${modelscope_id}"
        rm -rf "$model_path"
        return 1
    fi
}

# 生成容器名
generate_container_name() {
    local model_name=$1
    local port=$2
    echo "${CONTAINER_NAME_PREFIX}-${model_name}-${port}"
}

# 构建并执行部署
deploy() {
    local image_tag=$1
    local model_path=$2
    local model_name=$(basename "$model_path")
    local tp=$3
    local dp=$4
    local port=$5
    local max_len=$6
    local max_seqs=$7
    local gpu_util=$8
    local extra_args=$9
    
    local image="${VLLM_REGISTRY}:${image_tag}"
    local container_name=$(generate_container_name "$model_name" "$port")
    
    log_step "准备部署配置"
    echo "  镜像: $image"
    echo "  模型: $model_path"
    echo "  TP=$tp, DP=$dp, Port=$port"
    echo "  max_len=$max_len, max_seqs=$max_seqs"
    
    # 构建启动命令
    local launch_cmd="vllm serve /model \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size $tp \
        --data-parallel-size $dp \
        --max-model-len $max_len \
        --max-num-seqs $max_seqs \
        --gpu-memory-utilization $gpu_util \
        --trust-remote-code \
        --served-model-name auto \
        $extra_args"
    
    # 构建容器挂载
    local mount_args="-v ${model_path}:/model:ro"
    
    # 构建 docker run 命令
    local docker_cmd="docker run -d \
        --name ${container_name} \
        --network host \
        --privileged \
        ${mount_args} \
        -e VLLM_USE_V1=1 \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        $image \
        bash -c \"$launch_cmd\""
    
    if [[ $DRY_RUN -eq 1 ]]; then
        echo ""
        echo "===== DRY RUN ====="
        echo "Docker 命令:"
        echo "$docker_cmd"
        echo ""
        echo "容器内启动命令:"
        echo "$launch_cmd"
        echo "==================="
        return 0
    fi
    
    # 检查端口是否被占用
    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
        log_error "端口 $port 已被占用"
        log_hint "使用 --port 指定其他端口，或先停止占用端口的服务"
        return 1
    fi
    
    # 检查容器是否已存在
    if docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        log_warn "容器 $container_name 已存在，先删除..."
        docker rm -f "$container_name"
    fi
    
    log_step "启动容器..."
    eval "$docker_cmd"
    
    log_info "容器已启动: $container_name"
    log_info "查看日志: docker logs -f $container_name"
    log_info "进入容器: docker exec -it $container_name bash"
    log_info "测试服务: curl http://localhost:$port/v1/models"
}

# 主函数
main() {
    DRY_RUN=0
    NO_PULL_IMAGE=0
    NO_PULL_MODEL=0
    
    # 默认值
    IMAGE_TAG=""
    MODEL_SCOPE_ID=""
    TP_SIZE=1
    DP_SIZE=1
    PORT=8000
    MAX_MODEL_LEN=32768
    MAX_NUM_SEQS=128
    GPU_UTIL=0.90
    EXTRA_ARGS=""
    MODEL_ALIAS=""
    
    if [[ $# -eq 0 ]]; then
        show_help
    fi
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --list)
                list_models
                exit 0
                ;;
            --status)
                show_status
                exit 0
                ;;
            --stop)
                shift
                docker rm -f "$1" 2>/dev/null && log_info "已停止: $1" || log_error "容器不存在: $1"
                exit 0
                ;;
            --help|-h)
                show_help
                ;;
            --image)
                shift
                IMAGE_TAG="$1"
                ;;
            --model)
                shift
                MODEL_SCOPE_ID="$1"
                ;;
            --tp)
                shift
                TP_SIZE="$1"
                ;;
            --dp)
                shift
                DP_SIZE="$1"
                ;;
            --port)
                shift
                PORT="$1"
                ;;
            --max-len)
                shift
                MAX_MODEL_LEN="$1"
                ;;
            --max-seqs)
                shift
                MAX_NUM_SEQS="$1"
                ;;
            --gpu-util)
                shift
                GPU_UTIL="$1"
                ;;
            --dry-run)
                DRY_RUN=1
                ;;
            --no-pull-image)
                NO_PULL_IMAGE=1
                ;;
            --no-pull-model)
                NO_PULL_MODEL=1
                ;;
            -*)
                log_error "未知选项: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
            *)
                if [[ -z "$MODEL_ALIAS" ]]; then
                    MODEL_ALIAS="$1"
                fi
                ;;
        esac
        shift
    done
    
    # 检查 Docker
    check_docker
    
    # 如果使用预定义别名
    if [[ -n "$MODEL_ALIAS" ]]; then
        log_step "加载预定义配置: $MODEL_ALIAS"
        CONFIG=$(get_model_config "$MODEL_ALIAS")
        if [[ $? -ne 0 ]]; then
            log_error "未知的模型别名: $MODEL_ALIAS"
            echo ""
            list_models
            exit 1
        fi
        IFS='|' read -r IMAGE_TAG MODEL_SCOPE_ID TP_SIZE DP_SIZE MAX_MODEL_LEN PORT EXTRA_ARGS <<< "$CONFIG"
        log_info "ModelScope ID: $MODEL_SCOPE_ID"
    fi
    
    # 验证必要参数
    if [[ -z "$IMAGE_TAG" ]]; then
        log_error "未指定镜像 tag"
        log_hint "使用 --image <tag> 指定，或使用预定义别名"
        log_hint "查看可用镜像: https://quay.io/repository/ascend/vllm-ascend?tab=tags"
        exit 1
    fi
    
    if [[ -z "$MODEL_SCOPE_ID" ]]; then
        log_error "未指定 ModelScope 模型 ID"
        log_hint "使用 --model <ModelScopeID> 指定，或使用预定义别名"
        log_hint "查看可用模型: https://www.modelscope.cn/models"
        exit 1
    fi
    
    # 拉取镜像
    pull_image "$IMAGE_TAG" || [[ $DRY_RUN -eq 1 ]] || exit 1
    
    # 下载模型
    MODEL_PATH=$(pull_model "$MODEL_SCOPE_ID") || [[ $DRY_RUN -eq 1 ]] || exit 1
    
    # 执行部署
    deploy "$IMAGE_TAG" "$MODEL_PATH" "$TP_SIZE" "$DP_SIZE" "$PORT" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$GPU_UTIL" "$EXTRA_ARGS"
}

main "$@"
