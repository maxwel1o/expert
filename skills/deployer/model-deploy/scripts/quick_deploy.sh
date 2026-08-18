1|#!/bin/bash
2|# ============================================================
3|# 一键部署脚本 - 使用默认配置快速部署模型
4|# 
5|# 参考:
6|#   - ${VLLM_TOOLS_SCRIPT_ROOT:-/opt/vllm-ascend-tools}/
7|#   - ${VLLM_TOOLS_ROOT:-/opt/vllm-ascend-tools}/
8|#   - https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/tutorials/models/
9|#
10|# 用法:
11|#   ./quick_deploy.sh <模型别名>
12|#   ./quick_deploy.sh qwen3-32b
13|#   ./quick_deploy.sh deepseek-v3 --env a2
14|#
15|# 此脚本会:
16|#   1. 从注册表读取模型配置
17|#   2. 设置环境变量 (网络、HCCL、模型特定优化等)
18|#   3. 检查/拉取镜像
19|#   4. 检查/下载模型
20|#   5. 创建容器并启动服务
21|# ============================================================
22|
23|set -e
24|
25|SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
26|source "$SCRIPT_DIR/model_registry.sh"
27|source "$SCRIPT_DIR/common.sh"
28|
29|# 默认配置
30|VLLM_REGISTRY="quay.io/ascend/vllm-ascend"
31|MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/sfs_turbo/models}"
32|CONTAINER_NAME_PREFIX="vllm-quick"
33|
34|# 显示帮助
35|show_help() {
36|    cat << 'EOF'
37|一键部署 vLLM-Ascend 模型
38|
39|用法:
40|  ./quick_deploy.sh <模型别名> [选项]
41|
42|模型别名:
43|  Qwen:     qwen3-32b, qwen3-30b-a3b, qwen3-235b-a22b, qwen2.5-72b/32b/14b/7b, qwq-32b
44|  DeepSeek: deepseek-v3, deepseek-r1, deepseek-v4-flash
45|  GLM:      glm-4-9b, glm-4-9b-chat, glm-5, glm-5.1
46|  其他:     llama3-70b, llama3-8b, kimi-k2, kimi-k2.5, minimax-m2.5
47|
48|选项:
49|  --env <环境>        环境类型: a2, a3 (默认: 自动检测)
50|  --nic <网卡>        网卡名称 (默认: 自动检测)
51|  --dry-run           只打印命令不执行
52|  --list              列出所有可用模型
53|  --node-ip <IP>      节点 IP (多节点部署)
54|
55|示例:
56|  ./quick_deploy.sh qwen3-32b
57|  ./quick_deploy.sh deepseek-v3 --env a2
58|  ./quick_deploy.sh qwen3-235b-a22b --dry-run
59|  ./quick_deploy.sh glm-5 --nic enp23s0f3
60|EOF
61|}
62|
63|# 解析参数
64|MODEL_ALIAS=""
65|ENV_TYPE=""
66|NIC_NAME=""
67|DRY_RUN=false
68|NODE_IP=""
69|
70|while [[ $# -gt 0 ]]; do
71|    case $1 in
72|        --help|-h)
73|            show_help
74|            exit 0
75|            ;;
76|        --list)
77|            list_models_by_category
78|            exit 0
79|            ;;
80|        --env)
81|            ENV_TYPE="$2"
82|            shift 2
83|            ;;
84|        --nic)
85|            NIC_NAME="$2"
86|            shift 2
87|            ;;
88|        --dry-run)
89|            DRY_RUN=true
90|            shift
91|            ;;
92|        --node-ip)
93|            NODE_IP="$2"
94|            shift 2
95|            ;;
96|        -*)
97|            log_error "未知选项: $1"
98|            show_help
99|            exit 1
100|            ;;
101|        *)
102|            if [[ -z "$MODEL_ALIAS" ]]; then
103|                MODEL_ALIAS="$1"
104|            fi
105|            shift
106|            ;;
107|    esac
108|done
109|
110|# 检查模型别名
111|if [[ -z "$MODEL_ALIAS" ]]; then
112|    log_error "请指定模型别名"
113|    echo ""
114|    list_models_by_category
115|    exit 1
116|fi
117|
118|# 获取模型配置
119|if ! CONFIG=$(get_model_config "$MODEL_ALIAS"); then
120|    log_error "未知模型: $MODEL_ALIAS"
121|    echo ""
122|    list_models_by_category
123|    exit 1
124|fi
125|
126|# 解析配置: 镜像tag|ModelScopeID|tp|dp|max_len|port|vllm_args
127|IFS='|' read -r IMAGE_TAG MODELSCOPE_ID TP DP MAX_LEN PORT VLLM_ARGS <<< "$CONFIG"
128|
129|# 获取模型特定配置
130|SPECULATIVE_CONFIG=""
131|TOKENIZER_CONFIG=""
132|ADDITIONAL_CONFIG=""
133|
134|if get_speculative_config "$MODEL_ALIAS" &>/dev/null; then
135|    SPECULATIVE_CONFIG=$(get_speculative_config "$MODEL_ALIAS")
136|fi
137|if get_tokenizer_config "$MODEL_ALIAS" &>/dev/null; then
138|    TOKENIZER_CONFIG=$(get_tokenizer_config "$MODEL_ALIAS")
139|fi
140|if get_additional_config "$MODEL_ALIAS" &>/dev/null; then
141|    ADDITIONAL_CONFIG=$(get_additional_config "$MODEL_ALIAS")
142|fi
143|
144|# 自动检测环境
145|if [[ -z "$ENV_TYPE" ]]; then
146|    ENV_TYPE=$(detect_env)
147|fi
148|
149|log_info "=========================================="
150|log_info "一键部署: $MODEL_ALIAS"
151|log_info "=========================================="
152|echo ""
153|echo "配置信息:"
154|echo "  镜像 tag:    $IMAGE_TAG"
155|echo "  模型 ID:     $MODELSCOPE_ID"
156|echo "  TP:          $TP"
157|echo "  DP:          $DP"
158|echo "  max_len:     $MAX_LEN"
159|echo "  端口:        $PORT"
160|echo "  环境:        $ENV_TYPE"
161|[[ -n "$VLLM_ARGS" ]] && echo "  vLLM 参数:   $VLLM_ARGS"
162|[[ -n "$SPECULATIVE_CONFIG" ]] && echo "  MTP 配置:    已配置"
163|[[ -n "$TOKENIZER_CONFIG" ]] && echo "  Tokenizer:   已配置"
164|echo ""
165|
166|# 设置网络配置
167|setup_network "$ENV_TYPE" "$NIC_NAME"
168|
169|# 设置模型特定环境变量
170|setup_model_env "$MODEL_ALIAS"
171|
172|# 构建容器名称
173|MODEL_NAME=$(echo "$MODELSCOPE_ID" | sed 's/.*\///' | sed 's/-/_/g')
174|CONTAINER_NAME="${CONTAINER_NAME_PREFIX}-${MODEL_NAME}-${PORT}"
175|
176|# 检查 Docker
177|if ! check_docker; then
178|    exit 1
179|fi
180|
181|# 检查镜像
182|IMAGE_FULL="${VLLM_REGISTRY}:${IMAGE_TAG}"
183|log_step "检查镜像: $IMAGE_FULL"
184|
185|if ! check_image "$IMAGE_FULL"; then
186|    if [[ "$DRY_RUN" == false ]]; then
187|        pull_image "$IMAGE_FULL"
188|    else
189|        echo "[DRY-RUN] docker pull $IMAGE_FULL"
190|    fi
191|else
192|    log_info "镜像已存在"
193|fi
194|
195|# 检查模型
196|MODEL_PATH="${MODEL_CACHE_DIR}/${MODEL_NAME}"
197|log_step "检查模型: $MODEL_PATH"
198|
199|if ! check_model "$MODEL_PATH"; then
200|    if [[ "$DRY_RUN" == false ]]; then
201|        download_model "$MODELSCOPE_ID" "$MODEL_CACHE_DIR"
202|    else
203|        echo "[DRY-RUN] modelscope download $MODELSCOPE_ID -> $MODEL_PATH"
204|    fi
205|else
206|    log_info "模型已存在"
207|fi
208|
209|# 计算 HCCL Buffer Size
210|HCCL_BUFFSIZE=$(calculate_hccl_buffsize "$MODEL_PATH" "$ENV_TYPE" "$DP" "$TP" 128)
211|log_info "HCCL_BUFFSIZE: $HCCL_BUFFSIZE"
212|
213|# 构建 vLLM 启动命令
214|VLLM_CMD="vllm serve /model \
215|    --port $PORT \
216|    --tensor-parallel-size $TP \
217|    --data-parallel-size $DP \
218|    --max-model-len $MAX_LEN \
219|    $COMMON_ARGS \
220|    $VLLM_ARGS \
221|    $SPECULATIVE_CONFIG \
222|    $TOKENIZER_CONFIG \
223|    $ADDITIONAL_CONFIG"
224|
225|# 构建环境变量列表
226|ENV_VARS="-e VLLM_USE_V1=1 \
227|    -e VLLM_VERSION=0.18.0 \
228|    -e TASK_QUEUE_ENABLE=1 \
229|    -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
230|    -e NET_CARD_NAME=${NET_CARD_NAME} \
231|    -e GLOO_SOCKET_IFNAME=${NET_CARD_NAME} \
232|    -e TP_SOCKET_IFNAME=${NET_CARD_NAME} \
233|    -e HCCL_SOCKET_IFNAME=${NET_CARD_NAME} \
234|    -e HCCL_BUFFSIZE=${HCCL_BUFFSIZE} \
235|    -e HCCL_OP_EXPANSION_MODE=AIV \
236|    -e OMP_NUM_THREADS=6 \
237|    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
238|
239|# 添加模型特定环境变量
240|case "$MODEL_ALIAS" in
241|    deepseek-v4-flash)
242|        ENV_VARS="$ENV_VARS \
243|            -e VLLM_ASCEND_APPLY_DSV4_PATCH=1 \
244|            -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1"
245|        ;;
246|    qwen3-32b)
247|        ENV_VARS="$ENV_VARS \
248|            -e VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1 \
249|            -e VLLM_ASCEND_ENABLE_FLASHCOMM=1"
250|        ;;
251|    qwen3.5-397b|qwen3.5-122b)
252|        ENV_VARS="$ENV_VARS \
253|            -e VLLM_ASCEND_ENABLE_NZ=1 \
254|            -e VLLM_ASCEND_ENABLE_FUSED_MC2=1"
255|        ;;
256|    glm-5|glm-5.1)
257|        ENV_VARS="$ENV_VARS \
258|            -e ASCEND_AGGREGATE_ENABLE=1 \
259|            -e ASCEND_A3_ENABLE=1 \
260|            -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1"
261|        ;;
262|esac
263|
264|# 构建 Docker 命令
265|DOCKER_CMD="docker run -d \
266|    --name $CONTAINER_NAME \
267|    --network host \
268|    --privileged \
269|    -v /usr/local/Ascend:/usr/local/Ascend \
270|    -v /home:/home \
271|    -v ${MODEL_PATH}:/model:ro \
272|    $ENV_VARS \
273|    $IMAGE_FULL \
274|    bash -c \"$VLLM_CMD\""
275|
276|echo ""
277|log_step "启动命令:"
278|echo "$DOCKER_CMD"
279|echo ""
280|
281|if [[ "$DRY_RUN" == true ]]; then
282|    log_info "[DRY-RUN] 跳过执行"
283|    exit 0
284|fi
285|
286|# 执行部署
287|log_step "创建容器并启动服务..."
288|eval "$DOCKER_CMD"
289|
290|# 等待服务启动
291|log_step "等待服务启动..."
292|sleep 5
293|
294|# 检查状态
295|if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
296|    log_info "=========================================="
297|    log_info "部署成功!"
298|    log_info "=========================================="
299|    echo ""
300|    echo "容器名称: $CONTAINER_NAME"
301|    echo "服务地址: http://localhost:$PORT"
302|    echo ""
303|    echo "测试命令:"
304|    echo "  curl http://localhost:$PORT/v1/models"
305|    echo ""
306|    echo "查看日志:"
307|    echo "  docker logs -f $CONTAINER_NAME"
308|    echo ""
309|    echo "停止服务:"
310|    echo "  docker stop $CONTAINER_NAME"
311|else
312|    log_error "部署失败，请检查日志:"
313|    echo "  docker logs $CONTAINER_NAME"
314|    exit 1
315|fi
316|