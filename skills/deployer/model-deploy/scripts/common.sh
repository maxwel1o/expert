1|#!/bin/bash
2|# ============================================================
3|# 公共环境配置 - 从 vllm-ascend-tools 学习的最佳实践
4|# 参考: ${VLLM_TOOLS_SCRIPT_ROOT:-/opt/vllm-ascend-tools}/common.sh
5|# 参考: ${VLLM_TOOLS_ROOT:-/opt/vllm-ascend-tools}/
6|# ============================================================
7|
8|# ============================================================
9|# vLLM 基础配置
10|# ============================================================
11|export VLLM_USE_V1=1
12|export VLLM_VERSION=0.18.0
13|export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
14|
15|# OpenTelemetry 配置
16|export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
17|export PROMETHEUS_MULTIPROC_DIR=/tmp/
18|
19|# 通用启动参数
20|export COMMON_ARGS="
21|    --trust-remote-code
22|    --served-model-name auto
23|    --distributed-executor-backend mp
24|    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
25|    --enable-log-requests
26|    --enable-prompt-tokens-details
27|"
28|
29|# ============================================================
30|# 环境检测与配置
31|# ============================================================
32|
33|detect_env() {
34|    # 检测 A2 还是 A3 环境
35|    if [[ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ]]; then
36|        local version=$(cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | grep "Version" | awk '{print $3}')
37|        if [[ "$version" == *"8.2.RC1"* ]] || [[ "$version" == *"A2"* ]]; then
38|            echo "a2"
39|        else
40|            echo "a3"
41|        fi
42|    else
43|        # 默认 A3
44|        echo "a3"
45|    fi
46|}
47|
48|# 设置网络配置
49|setup_network() {
50|    local env_type="${1:-a3}"
51|    local nic_name="${2:-}"
52|    
53|    # 自动检测网卡
54|    if [[ -z "$nic_name" ]]; then
55|        if [[ "$env_type" == "a2" ]]; then
56|            nic_name="bond0"
57|        else
58|            # 尝试检测可用网卡
59|            for nic in eth0 enp23s0f3 bond0; do
60|                if ip link show "$nic" &>/dev/null; then
61|                    nic_name="$nic"
62|                    break
63|                fi
64|            done
65|            [[ -z "$nic_name" ]] && nic_name="eth0"
66|        fi
67|    fi
68|    
69|    export NET_CARD_NAME="$nic_name"
70|    
71|    if [[ "$env_type" == "a2" ]]; then
72|        export HCCL_INTRA_PCIE_ENABLE=1
73|        export HCCL_INTRA_ROCE_ENABLE=0
74|    else
75|        export HCCL_INTRA_PCIE_ENABLE=0
76|        export HCCL_INTRA_ROCE_ENABLE=1
77|    fi
78|    
79|    # 通信接口配置
80|    export GLOO_SOCKET_IFNAME=${NET_CARD_NAME}
81|    export TP_SOCKET_IFNAME=${NET_CARD_NAME}
82|    export HCCL_SOCKET_IFNAME=${NET_CARD_NAME}
83|    
84|    # HCCL 优化
85|    export HCCL_OP_EXPANSION_MODE=AIV
86|    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
87|    
88|    # 内存优化
89|    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
90|}
91|
92|# 设置模型特定环境变量
93|setup_model_env() {
94|    local model_alias="$1"
95|    
96|    case "$model_alias" in
97|        deepseek-v4-flash|deepseek-v4)
98|            # DeepSeek V4 特定配置
99|            export VLLM_ASCEND_APPLY_DSV4_PATCH=1
100|            export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
101|            export VLLM_ASCEND_ENABLE_FUSED_MC2=1
102|            ;;
103|        qwen3-32b|qwen3-235b-a22b)
104|            # Qwen3 Dense 优化
105|            export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
106|            export VLLM_ASCEND_ENABLE_FLASHCOMM=1
107|            ;;
108|        qwen3.5-397b|qwen3.5-122b)
109|            # Qwen3.5 MoE 优化
110|            export VLLM_ASCEND_ENABLE_NZ=1
111|            export VLLM_ASCEND_ENABLE_FUSED_MC2=1
112|            ;;
113|        glm-5|glm5.1)
114|            # GLM-5 优化
115|            export ASCEND_AGGREGATE_ENABLE=1
116|            export ASCEND_A3_ENABLE=1
117|            export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
118|            ;;
119|        *)
120|            # 默认配置
121|            export VLLM_ASCEND_ENABLE_NZ=1
122|            ;;
123|    esac
124|}
125|
126|# ============================================================
127|# HCCL Buffer Size 计算 (简化版)
128|# ============================================================
129|
130|calculate_hccl_buffsize() {
131|    local model_path="$1"
132|    local env_type="$2"
133|    local dp="$3"
134|    local tp="$4"
135|    local max_seqs="${5:-128}"
136|    
137|    # 默认值
138|    local default_size=512
139|    
140|    # 尝试从 config.json 读取
141|    local config_file="$model_path/config.json"
142|    if [[ ! -f "$config_file" ]]; then
143|        echo "$default_size"
144|        return
145|    fi
146|    
147|    # 简化计算: 基于 max_seqs 和并行度
148|    local ep_world_size=$((dp * tp))
149|    local result=$((max_seqs * ep_world_size / 16))
150|    
151|    # 确保最小值和最大值
152|    if [[ $result -lt 200 ]]; then
153|        result=200
154|    fi
155|    if [[ $result -gt 4096 ]]; then
156|        result=4096
157|    fi
158|    
159|    echo "$result"
160|}
161|
162|# ============================================================
163|# 颜色输出
164|# ============================================================
165|
166|RED='\033[0;31m'
167|GREEN='\033[0;32m'
168|YELLOW='\033[1;33m'
169|BLUE='\033[0;34m'
170|CYAN='\033[0;36m'
171|NC='\033[0m'
172|
173|log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
174|log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
175|log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
176|log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
177|log_hint() { echo -e "${CYAN}[HINT]${NC} $1"; }
178|
179|# ============================================================
180|# Docker 辅助函数
181|# ============================================================
182|
183|check_docker() {
184|    if ! command -v docker &> /dev/null; then
185|        log_error "Docker 未安装"
186|        return 1
187|    fi
188|    
189|    if ! docker info &> /dev/null; then
190|        log_error "Docker daemon 未运行或无权限"
191|        log_hint "尝试: sudo usermod -aG docker \$USER 或使用 sudo"
192|        return 1
193|    fi
194|    
195|    return 0
196|}
197|
198|check_image() {
199|    local image="$1"
200|    if docker image inspect "$image" &> /dev/null; then
201|        return 0
202|    else
203|        return 1
204|    fi
205|}
206|
207|pull_image() {
208|    local image="$1"
209|    log_info "拉取镜像: $image"
210|    docker pull "$image"
211|}
212|
213|# ============================================================
214|# 模型下载辅助函数
215|# ============================================================
216|
217|check_model() {
218|    local model_path="$1"
219|    if [[ -d "$model_path" ]] && [[ -f "$model_path/config.json" ]]; then
220|        return 0
221|    else
222|        return 1
223|    fi
224|}
225|
226|download_model() {
227|    local modelscope_id="$1"
228|    local cache_dir="$2"
229|    
230|    log_info "从 ModelScope 下载: $modelscope_id"
231|    
232|    # 确保 modelscope 已安装
233|    pip install modelscope -q 2>/dev/null || true
234|    
235|    # 下载模型
236|    python3 -c "
237|from modelscope import snapshot_download
238|import os
239|os.makedirs('$cache_dir', exist_ok=True)
240|snapshot_download('$modelscope_id', cache_dir='$cache_dir')
241|" 2>&1
242|}
243|
244|# ============================================================
245|# 服务测试
246|# ============================================================
247|
248|test_service() {
249|    local port="$1"
250|    local max_retries="${2:-30}"
251|    local retry=0
252|    
253|    while [[ $retry -lt $max_retries ]]; do
254|        if curl -s "http://localhost:$port/v1/models" &> /dev/null; then
255|            return 0
256|        fi
257|        sleep 2
258|        ((retry++))
259|    done
260|    
261|    return 1
262|}
263|
264|# ============================================================
265|# 生成 cudagraph_capture_sizes
266|# ============================================================
267|
268|generate_cudagraph_sizes() {
269|    local max_seqs="$1"
270|    # 生成常用的 cudagraph capture sizes
271|    local sizes="1,2,4,6,8,12,16,24,32"
272|    if [[ $max_seqs -ge 64 ]]; then
273|        sizes="$sizes,48,64,72,96,128"
274|    fi
275|    if [[ $max_seqs -ge 192 ]]; then
276|        sizes="$sizes,160,192,224,256"
277|    fi
278|    if [[ $max_seqs -ge 384 ]]; then
279|        sizes="$sizes,320,352,384,448,512"
280|    fi
281|    echo "$sizes"
282|}
283|