1|# 预定义模型注册表
2|
3|## 重要提示
4|
5|**模型名称需与用户确认具体版本！**
6|- 同一模型有多个版本（w8a8、w4a8 量化版等）
7|- ModelScope ID 可能与本地命名不同
8|- 查询地址: https://www.modelscope.cn/models
9|
10|---
11|
12|## 配置来源
13|
14|- `$VLLM_TOOLS_SCRIPT_ROOT/`
15|- `$VLLM_TOOLS_ROOT/`
16|- https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/tutorials/models/
17|
18|---
19|
20|## Qwen 系列
21|
22|| 别名 | ModelScope ID | TP | DP | max_len | 特殊配置 |
23||------|---------------|----|----|---------|----------|
24|| qwen3-32b | Qwen/Qwen3-32B | 8 | 1 | 32768 | Dense 优化, FlashComm |
25|| qwen3-30b-a3b | Qwen/Qwen3-30B-A3B | 4 | 1 | 32768 | MoE, 量化 |
26|| qwen3-235b-a22b | Qwen/Qwen3-235B-A22B | 16 | 1 | 262144 | MoE, async-scheduling |
27|| qwen2.5-72b | Qwen/Qwen2.5-72B-Instruct | 8 | 1 | 32768 | - |
28|| qwen2.5-32b | Qwen/Qwen2.5-32B-Instruct | 8 | 1 | 32768 | - |
29|| qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct | 4 | 1 | 32768 | - |
30|| qwen2.5-7b | Qwen/Qwen2.5-7B-Instruct | 2 | 1 | 32768 | - |
31|| qwen2.5-vl-72b | Qwen/Qwen2.5-VL-72B-Instruct | 8 | 1 | 32768 | 多模态, mm-processor-cache |
32|| qwen3.5-397b | Qwen/Qwen3.5-397B-A17B | 16 | 1 | 262144 | MoE, MTP (qwen3_5_mtp), async-scheduling |
33|| qwen3.5-122b | Qwen/Qwen3.5-122B-A10B | 8 | 1 | 131072 | MoE, MTP (qwen3_5_mtp) |
34|| qwq-32b | Qwen/QwQ-32B | 8 | 1 | 32768 | 推理模型 |
35|
36|---
37|
38|## DeepSeek 系列
39|
40|| 别名 | ModelScope ID | TP | DP | max_len | 特殊配置 |
41||------|---------------|----|----|---------|----------|
42|| deepseek-v3 | deepseek-ai/DeepSeek-V3 | 8 | 2 | 65536 | MoE, MTP, 量化 |
43|| deepseek-r1 | deepseek-ai/DeepSeek-R1 | 8 | 1 | 131072 | 推理模型, MoE, MTP |
44|| deepseek-v4-flash | deepseek-ai/DeepSeek-V4 | 16 | 1 | 1048576 | 大模型, MTP, V4 专用 tokenizer, FlashComm1 |
45|
46|**DeepSeek V4 特殊配置:**
47|```bash
48|--tokenizer-mode deepseek_v4
49|--tool-call-parser deepseek_v4
50|--reasoning-parser deepseek_v4
51|--language-model-only
52|--no-disable-hybrid-kv-cache-manager
53|```
54|
55|---
56|
57|## GLM 系列
58|
59|| 别名 | ModelScope ID | TP | DP | max_len | 特殊配置 |
60||------|---------------|----|----|---------|----------|
61|| glm-4-9b | ZhipuAI/glm-4-9b | 2 | 1 | 32768 | - |
62|| glm-4-9b-chat | ZhipuAI/glm-4-9b-chat | 2 | 1 | 32768 | - |
63|| chatglm3-6b | ZhipuAI/chatglm3-6b | 2 | 1 | 32768 | - |
64|| glm-5 | ZhipuAI/GLM-5 | 8 | 1 | 202752 | 大模型, MTP, async-scheduling |
65|| glm-5.1 | ZhipuAI/GLM-5.1 | 8 | 1 | 202752 | 大模型, MTP, async-scheduling |
66|
67|**GLM-5 特殊配置:**
68|```bash
69|--tool-call-parser glm47
70|--reasoning-parser glm45
71|--enable-auto-tool-choice
72|--additional-config {"fuse_muls_add": true, "multistream_overlap_shared_expert": true}
73|```
74|
75|---
76|
77|## LLaMA 系列
78|
79|| 别名 | ModelScope ID | TP | DP | max_len |
80||------|---------------|----|----|---------|
81|| llama3-70b | LLM-Research/Meta-Llama-3-70B | 8 | 1 | 32768 |
82|| llama3-8b | LLM-Research/Meta-Llama-3-8B | 2 | 1 | 8192 |
83|| llama3.1-70b | LLM-Research/Meta-Llama-3.1-70B | 8 | 1 | 131072 |
84|
85|---
86|
87|## Kimi 系列
88|
89|| 别名 | ModelScope ID | TP | DP | max_len | 特殊配置 |
90||------|---------------|----|----|---------|----------|
91|| kimi-k2 | moonshotai/Kimi-K2 | 8 | 1 | 131072 | MoE, 量化 |
92|| kimi-k2.5 | moonshotai/Kimi-K2.5 | 8 | 1 | 262144 | MoE, 量化, mm-processor-cache |
93|
94|---
95|
96|## MiniMax 系列
97|
98|| 别名 | ModelScope ID | TP | DP | max_len | 特殊配置 |
99||------|---------------|----|----|---------|----------|
100|| minimax-m2.5 | MiniMax/MiniMax-M2.5 | 8 | 1 | 131072 | MoE, 量化 |
101|
102|---
103|
104|## Wan 系列 (视频生成)
105|
106|| 别名 | ModelScope ID | TP | DP | max_len |
107||------|---------------|----|----|---------|
108|| wan2.2-i2v | Wan-AI/Wan2.2-I2V-14B | 8 | 1 | 32768 |
109|
110|---
111|
112|## MTP (Multi-Token Prediction) 配置
113|
114|需要 MTP 的模型:
115|
116|| 模型 | MTP 方法 | num_speculative_tokens |
117||------|----------|------------------------|
118|| DeepSeek V3/R1/V4 | deepseek_mtp / mtp | 1 |
119|| Qwen3.5 MoE | qwen3_5_mtp | 3 |
120|| GLM-5/5.1 | deepseek_mtp | 2 |
121|
122|---
123|
124|## 环境变量说明
125|
126|### 通用环境变量
127|```bash
128|export VLLM_USE_V1=1
129|export VLLM_VERSION=0.18.0
130|export TASK_QUEUE_ENABLE=1
131|export HCCL_OP_EXPANSION_MODE=AIV
132|export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
133|```
134|
135|### 模型特定环境变量
136|```bash
137|# DeepSeek V4
138|export VLLM_ASCEND_APPLY_DSV4_PATCH=1
139|export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
140|
141|# Qwen3 Dense
142|export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
143|export VLLM_ASCEND_ENABLE_FLASHCOMM=1
144|
145|# Qwen3.5 / GLM-5 MoE
146|export VLLM_ASCEND_ENABLE_NZ=1
147|export VLLM_ASCEND_ENABLE_FUSED_MC2=1
148|```
149|
150|---
151|
152|## 添加新模型
153|
154|编辑 `scripts/model_registry.sh`，添加配置：
155|
156|```bash
157|# 基础配置
158|["别名"]="镜像tag|ModelScope模型ID|tp|dp|max_len|port|vllm_args"
159|
160|# MTP 配置 (如需要)
161|MODEL_SPECULATIVE_CONFIG["别名"]='--speculative-config {...}'
162|
163|# Tokenizer 配置 (如需要)
164|MODEL_TOKENIZER_CONFIG["别名"]='--tokenizer-mode xxx'
165|
166|# 额外配置 (如需要)
167|MODEL_ADDITIONAL_CONFIG["别名"]='--additional-config {...}'
168|```
169|