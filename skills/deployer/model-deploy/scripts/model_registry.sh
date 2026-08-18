1|#!/bin/bash
2|# ============================================================
3|# 模型注册表 - 从 vllm-ascend-tools 学习的配置
4|# 
5|# 参考:
6|#   - ${VLLM_TOOLS_SCRIPT_ROOT:-/opt/vllm-ascend-tools}/
7|#   - ${VLLM_TOOLS_ROOT:-/opt/vllm-ascend-tools}/
8|#   - https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/tutorials/models/
9|#
10|# 格式: MODEL_CONFIGS[别名]="镜像tag|ModelScope模型ID|tp|dp|max_len|port|vllm_args"
11|#
12|# 注意: ModelScope 模型 ID 需要与用户确认具体版本！
13|# 查询地址: https://www.modelscope.cn/models
14|# ============================================================
15|
16|declare -A MODEL_CONFIGS=(
17|    # ==================== Qwen3 系列 ====================
18|    # Qwen3-32B (Dense)
19|    ["qwen3-32b"]="openai-cpu-poc|Qwen/Qwen3-32B|8|1|32768|8006|--max-num-seqs 384 --block-size 128 --gpu-memory-utilization 0.9"
20|    
21|    # Qwen3-30B-A3B (MoE) - 已验证于 ${TEST_HOST:-localhost}, NPU 4-7, 端口 8007
22|    ["qwen3-30b-a3b"]="release_0.18.0|Qwen/Qwen3-30B-A3B|4|1|32768|8007|--max-num-seqs 168 --no-enforce-eager --enable-expert-parallel --gpu-memory-utilization 0.90"
23|    
24|    # Qwen3-235B-A22B (MoE 大模型)
25|    ["qwen3-235b-a22b"]="openai-cpu-poc|Qwen/Qwen3-235B-A22B|16|1|262144|8006|--max-num-seqs 48 --enforce-eager --quantization ascend --enable-expert-parallel --async-scheduling"
26|    
27|    # ==================== Qwen2.5 系列 ====================
28|    ["qwen2.5-72b"]="openai-cpu-poc|Qwen/Qwen2.5-72B-Instruct|8|1|32768|8006|--max-num-seqs 128 --block-size 128 --gpu-memory-utilization 0.9"
29|    ["qwen2.5-32b"]="openai-cpu-poc|Qwen/Qwen2.5-32B-Instruct|8|1|32768|8006|--max-num-seqs 128 --block-size 128 --gpu-memory-utilization 0.9"
30|    ["qwen2.5-14b"]="openai-cpu-poc|Qwen/Qwen2.5-14B-Instruct|4|1|32768|8006|--max-num-seqs 256 --block-size 128 --gpu-memory-utilization 0.9"
31|    ["qwen2.5-7b"]="openai-cpu-poc|Qwen/Qwen2.5-7B-Instruct|2|1|32768|8006|--max-num-seqs 256 --block-size 128 --gpu-memory-utilization 0.9"
32|    
33|    # Qwen2.5-VL (多模态)
34|    ["qwen2.5-vl-72b"]="openai-cpu-poc|Qwen/Qwen2.5-VL-72B-Instruct|8|1|32768|8006|--max-num-seqs 64 --mm-processor-cache-type shm --mm-processor-cache-gb 4"
35|    
36|    # ==================== Qwen3.5 MoE 系列 ====================
37|    # Qwen3.5-397B-A17B (需要 MTP)
38|    ["qwen3.5-397b"]="openai-cpu-poc|Qwen/Qwen3.5-397B-A17B|16|1|262144|8006|--max-num-seqs 128 --quantization ascend --enable-expert-parallel --async-scheduling --no-enable-prefix-caching"
39|    
40|    # Qwen3.5-122B-A10B (需要 MTP)
41|    ["qwen3.5-122b"]="openai-cpu-poc|Qwen/Qwen3.5-122B-A10B|8|1|131072|8006|--max-num-seqs 64 --quantization ascend --enable-expert-parallel --async-scheduling"
42|    
43|    # ==================== QwQ 推理模型 ====================
44|    ["qwq-32b"]="openai-cpu-poc|Qwen/QwQ-32B|8|1|32768|8006|--max-num-seqs 128 --block-size 128 --gpu-memory-utilization 0.9"
45|    
46|    # ==================== DeepSeek 系列 ====================
47|    # DeepSeek-V3
48|    ["deepseek-v3"]="openai-cpu-poc|deepseek-ai/DeepSeek-V3|8|2|65536|8006|--max-num-seqs 96 --quantization ascend --enable-expert-parallel --enforce-eager"
49|    
50|    # DeepSeek-R1 (推理)
51|    ["deepseek-r1"]="openai-cpu-poc|deepseek-ai/DeepSeek-R1|8|1|131072|8006|--max-num-seqs 64 --quantization ascend --enable-expert-parallel --enforce-eager"
52|    
53|    # DeepSeek-V4 Flash (大模型, 需要 MTP)
54|    ["deepseek-v4-flash"]="openai-cpu-poc|deepseek-ai/DeepSeek-V4|16|1|1048576|8006|--max-num-seqs 32 --quantization ascend --enable-expert-parallel --enforce-eager --language-model-only --no-disable-hybrid-kv-cache-manager"
55|    
56|    # ==================== GLM 系列 ====================
57|    ["glm-4-9b"]="openai-cpu-poc|ZhipuAI/glm-4-9b|2|1|32768|8006|--max-num-seqs 256 --block-size 128 --gpu-memory-utilization 0.9"
58|    ["glm-4-9b-chat"]="openai-cpu-poc|ZhipuAI/glm-4-9b-chat|2|1|32768|8006|--max-num-seqs 256 --block-size 128 --gpu-memory-utilization 0.9"
59|    ["chatglm3-6b"]="openai-cpu-poc|ZhipuAI/chatglm3-6b|2|1|32768|8006|--max-num-seqs 256 --block-size 128 --gpu-memory-utilization 0.9"
60|    
61|    # GLM-5 (大模型, 需要 MTP)
62|    ["glm-5"]="openai-cpu-poc|ZhipuAI/GLM-5|8|1|202752|8006|--max-num-seqs 64 --quantization ascend --enable-expert-parallel --enforce-eager --async-scheduling"
63|    
64|    # GLM-5.1
65|    ["glm-5.1"]="openai-cpu-poc|ZhipuAI/GLM-5.1|8|1|202752|8006|--max-num-seqs 64 --quantization ascend --enable-expert-parallel --enforce-eager --async-scheduling"
66|    
67|    # ==================== LLaMA 系列 ====================
68|    ["llama3-70b"]="openai-cpu-poc|LLM-Research/Meta-Llama-3-70B|8|1|32768|8006|--max-num-seqs 128 --block-size 128 --gpu-memory-utilization 0.9"
69|    ["llama3-8b"]="openai-cpu-poc|LLM-Research/Meta-Llama-3-8B|2|1|8192|8006|--max-num-seqs 256 --block-size 128 --gpu-memory-utilization 0.9"
70|    ["llama3.1-70b"]="openai-cpu-poc|LLM-Research/Meta-Llama-3.1-70B|8|1|131072|8006|--max-num-seqs 128 --block-size 128 --gpu-memory-utilization 0.9"
71|    
72|    # ==================== Kimi 系列 ====================
73|    ["kimi-k2"]="openai-cpu-poc|moonshotai/Kimi-K2|8|1|131072|8006|--max-num-seqs 64 --quantization ascend --enable-expert-parallel --enforce-eager"
74|    ["kimi-k2.5"]="openai-cpu-poc|moonshotai/Kimi-K2.5|8|1|262144|8006|--max-num-seqs 64 --quantization ascend --enable-expert-parallel --enforce-eager --mm-processor-cache-type shm"
75|    
76|    # ==================== MiniMax 系列 ====================
77|    ["minimax-m2.5"]="openai-cpu-poc|MiniMax/MiniMax-M2.5|8|1|131072|8006|--max-num-seqs 64 --quantization ascend --enable-expert-parallel --enforce-eager"
78|    
79|    # ==================== Wan2.2 视频生成 ====================
80|    ["wan2.2-i2v"]="openai-cpu-poc|Wan-AI/Wan2.2-I2V-14B|8|1|32768|8006|--max-num-seqs 16"
81|)
82|
83|# ============================================================
84|# 模型特定配置 (MTP, tokenizer 等)
85|# ============================================================
86|
87|declare -A MODEL_SPECULATIVE_CONFIG=(
88|    # DeepSeek 系列使用 deepseek_mtp
89|    ["deepseek-v3"]='--speculative-config {"num_speculative_tokens":1, "method": "deepseek_mtp"}'
90|    ["deepseek-r1"]='--speculative-config {"num_speculative_tokens":1, "method": "deepseek_mtp"}'
91|    ["deepseek-v4-flash"]='--speculative-config {"num_speculative_tokens":1, "method": "mtp", "enforce_eager": true}'
92|    
93|    # Qwen3.5 使用 qwen3_5_mtp
94|    ["qwen3.5-397b"]='--speculative-config {"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'
95|    ["qwen3.5-122b"]='--speculative-config {"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'
96|    
97|    # GLM-5 使用 deepseek_mtp
98|    ["glm-5"]='--speculative-config {"num_speculative_tokens": 2, "method": "deepseek_mtp"}'
99|    ["glm-5.1"]='--speculative-config {"num_speculative_tokens": 2, "method": "deepseek_mtp"}'
100|)
101|
102|declare -A MODEL_TOKENIZER_CONFIG=(
103|    # DeepSeek V4 特定 tokenizer
104|    ["deepseek-v4-flash"]='--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4'
105|    
106|    # GLM-5 特定配置
107|    ["glm-5"]='--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice'
108|    ["glm-5.1"]='--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice'
109|)
110|
111|declare -A MODEL_ADDITIONAL_CONFIG=(
112|    # Qwen3 Dense 优化
113|    ["qwen3-32b"]='--additional-config {"ascend_scheduler_config":{"enabled":false,"enable_chunked_prefill":true}}'
114|    
115|    # Qwen3.5 MoE 优化
116|    ["qwen3.5-397b"]='--additional-config {"enable_cpu_binding":true, "multistream_overlap_shared_expert": false}'
117|    ["qwen3.5-122b"]='--additional-config {"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
118|    
119|    # DeepSeek V4 优化
120|    ["deepseek-v4-flash"]='--additional-config {"enable_cpu_binding": "True", "enable_flashcomm1": true, "multistream_dsa_preprocess": true}'
121|    
122|    # GLM-5 优化
123|    ["glm-5"]='--additional-config {"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}'
124|    ["glm-5.1"]='--additional-config {"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true}'
125|)
126|
127|# ============================================================
128|# 辅助函数
129|# ============================================================
130|
131|get_model_config() {
132|    local model_name=$1
133|    if [[ -n "${MODEL_CONFIGS[$model_name]}" ]]; then
134|        echo "${MODEL_CONFIGS[$model_name]}"
135|        return 0
136|    fi
137|    return 1
138|}
139|
140|get_speculative_config() {
141|    local model_name=$1
142|    if [[ -n "${MODEL_SPECULATIVE_CONFIG[$model_name]}" ]]; then
143|        echo "${MODEL_SPECULATIVE_CONFIG[$model_name]}"
144|        return 0
145|    fi
146|    return 1
147|}
148|
149|get_tokenizer_config() {
150|    local model_name=$1
151|    if [[ -n "${MODEL_TOKENIZER_CONFIG[$model_name]}" ]]; then
152|        echo "${MODEL_TOKENIZER_CONFIG[$model_name]}"
153|        return 0
154|    fi
155|    return 1
156|}
157|
158|get_additional_config() {
159|    local model_name=$1
160|    if [[ -n "${MODEL_ADDITIONAL_CONFIG[$model_name]}" ]]; then
161|        echo "${MODEL_ADDITIONAL_CONFIG[$model_name]}"
162|        return 0
163|    fi
164|    return 1
165|}
166|
167|# 列出所有支持的模型
168|list_models() {
169|    echo "预定义模型列表:"
170|    echo "========================================================================================================"
171|    printf "%-18s %-35s %-4s %-4s %-8s %-6s\n" "别名" "ModelScope ID" "TP" "DP" "max_len" "port"
172|    echo "--------------------------------------------------------------------------------------------------------"
173|    for model in "${!MODEL_CONFIGS[@]}"; do
174|        IFS='|' read -r _ modelscope_id tp dp max_len port _ _ <<< "${MODEL_CONFIGS[$model]}"
175|        printf "%-18s %-35s %-4s %-4s %-8s %-6s\n" "$model" "$modelscope_id" "$tp" "$dp" "$max_len" "$port"
176|    done | sort
177|    echo ""
178|    echo "提示:"
179|    echo "  - 使用 ./quick_deploy.sh <别名> 一键部署"
180|    echo "  - ModelScope 模型 ID 需要与用户确认具体版本！"
181|    echo "  - 查询地址: https://www.modelscope.cn/models"
182|}
183|
184|# 按类别列出模型
185|list_models_by_category() {
186|    echo "=== Qwen 系列 ==="
187|    for m in qwen3-32b qwen3-30b-a3b qwen3-235b-a22b qwen2.5-72b qwen2.5-32b qwen2.5-14b qwen2.5-7b qwen2.5-vl-72b qwen3.5-397b qwen3.5-122b qwq-32b; do
188|        if [[ -n "${MODEL_CONFIGS[$m]}" ]]; then
189|            IFS='|' read -r _ mid tp dp _ _ _ _ <<< "${MODEL_CONFIGS[$m]}"
190|            printf "  %-18s %s (TP=%s, DP=%s)\n" "$m" "$mid" "$tp" "$dp"
191|        fi
192|    done
193|    
194|    echo ""
195|    echo "=== DeepSeek 系列 ==="
196|    for m in deepseek-v3 deepseek-r1 deepseek-v4-flash; do
197|        if [[ -n "${MODEL_CONFIGS[$m]}" ]]; then
198|            IFS='|' read -r _ mid tp dp _ _ _ _ <<< "${MODEL_CONFIGS[$m]}"
199|            printf "  %-18s %s (TP=%s, DP=%s)\n" "$m" "$mid" "$tp" "$dp"
200|        fi
201|    done
202|    
203|    echo ""
204|    echo "=== GLM 系列 ==="
205|    for m in glm-4-9b glm-4-9b-chat chatglm3-6b glm-5 glm-5.1; do
206|        if [[ -n "${MODEL_CONFIGS[$m]}" ]]; then
207|            IFS='|' read -r _ mid tp dp _ _ _ _ <<< "${MODEL_CONFIGS[$m]}"
208|            printf "  %-18s %s (TP=%s, DP=%s)\n" "$m" "$mid" "$tp" "$dp"
209|        fi
210|    done
211|    
212|    echo ""
213|    echo "=== 其他模型 ==="
214|    for m in llama3-70b llama3-8b llama3.1-70b kimi-k2 kimi-k2.5 minimax-m2.5 wan2.2-i2v; do
215|        if [[ -n "${MODEL_CONFIGS[$m]}" ]]; then
216|            IFS='|' read -r _ mid tp dp _ _ _ _ <<< "${MODEL_CONFIGS[$m]}"
217|            printf "  %-18s %s (TP=%s, DP=%s)\n" "$m" "$mid" "$tp" "$dp"
218|        fi
219|    done
220|}
221|