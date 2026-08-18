1|# PD 部署模式说明
2|
3|PD = Prefill (预填充) + Decode (解码)
4|
5|## 两种部署模式
6|
7|### 1. mix-pd (混合部署)
8|
9|Prefill 和 Decode 在**同一个进程/节点**中运行。
10|
11|```
12|┌─────────────────────────────────┐
13|│         单一 vLLM 进程          │
14|│  ┌─────────┐    ┌─────────┐    │
15|│  │ Prefill │───▶│ Decode  │    │
16|│  └─────────┘    └─────────┘    │
17|└─────────────────────────────────┘
18|```
19|
20|**特点:**
21|- `TASK_QUEUE_ENABLE=0` 或 `1`
22|- 不需要 MooncakeConnector
23|- 配置简单，适合单机部署
24|- Prefill 和 Decode 共享 GPU/NPU 资源
25|
26|**启动命令示例:**
27|```bash
28|vllm serve /model \
29|    --tensor-parallel-size 8 \
30|    --data-parallel-size 2 \
31|    --port 8006 \
32|    --max-model-len 131072
33|```
34|
35|---
36|
37|### 2. full-pd (分离部署)
38|
39|Prefill 和 Decode 分别运行在**不同进程/节点**，通过 MooncakeConnector 传输 KV Cache。
40|
41|```
42|┌──────────────────┐         ┌──────────────────┐
43|│   Prefill 节点    │         │   Decode 节点    │
44|│  ┌─────────────┐ │  KV     │ ┌─────────────┐  │
45|│  │  Prefill    │─│────────▶│ │  Decode     │  │
46|│  │ (Producer)  │ │ Cache   │ │ (Consumer)  │  │
47|│  └─────────────┘ │         │ └─────────────┘  │
48|└──────────────────┘         └──────────────────┘
49|        │                              │
50|     port: 8100                    port: 8200
51|```
52|
53|**特点:**
54|- Prefill: `TASK_QUEUE_ENABLE` 可配置 (常见值: 1, 2, 6, 10 等)
55|- Decode: `TASK_QUEUE_ENABLE=0`
56|- 需要 MooncakeConnector 配置 `--kv-transfer-config`
57|- Prefill 和 Decode 可以有不同的 TP/DP 配置
58|- 适合大规模部署，可独立扩展 Prefill/Decode
59|
60|**TASK_QUEUE_ENABLE 说明:**
61|
62|| 值 | 含义 |
63||----|------|
64|| 0 | 禁用任务队列 |
65|| 1 | 启用任务队列 (常见) |
66|| 2 | Prefill 模式 (full-pd) |
67|| 6, 10, ... | 根据具体场景配置 |
68|
69|**环境变量差异:**
70|
71|| 变量 | mix-pd | full-pd Prefill | full-pd Decode |
72||------|--------|-----------------|----------------|
73|| TASK_QUEUE_ENABLE | 0/1 | 可配置 (1,2,6,10...) | 0 |
74|| kv_role | - | kv_producer | kv_consumer |
75|| VLLM_BASE_PORT | - | 9100 | 9200 |
76|
77|**Prefill 启动示例:**
78|```bash
79|# TASK_QUEUE_ENABLE 根据场景配置
80|export TASK_QUEUE_ENABLE=2  # 或 1, 6, 10 等
81|
82|vllm serve /model \
83|    --port 8100 \
84|    --tensor-parallel-size 4 \
85|    --data-parallel-size 1 \
86|    --enforce-eager \
87|    --kv-transfer-config '{
88|        "kv_connector": "MooncakeConnector",
89|        "kv_role": "kv_producer",
90|        "kv_port": "20002",
91|        "engine_id": "prefill-0",
92|        "kv_rank": 0,
93|        "kv_connector_extra_config": {
94|            "prefill": {"dp_size": 1, "tp_size": 4},
95|            "decode": {"dp_size": 8, "tp_size": 4}
96|        }
97|    }'
98|```
99|
100|**Decode 启动示例:**
101|```bash
102|export TASK_QUEUE_ENABLE=0
103|
104|vllm serve /model \
105|    --port 8200 \
106|    --tensor-parallel-size 4 \
107|    --data-parallel-size 8 \
108|    --kv-transfer-config '{
109|        "kv_connector": "MooncakeConnector",
110|        "kv_role": "kv_consumer",
111|        "kv_port": "20002",
112|        "engine_id": "decode-0",
113|        "kv_rank": 1,
114|        "kv_connector_extra_config": {
115|            "prefill": {"dp_size": 1, "tp_size": 4},
116|            "decode": {"dp_size": 8, "tp_size": 4}
117|        }
118|    }'
119|```
120|
121|---
122|
123|## 通信环境变量
124|
125|两种模式都需要配置 HCCL 通信:
126|
127|```bash
128|# 网卡名称
129|export NET_CARD_NAME="bond0"  # A2 环境
130|export NET_CARD_NAME="eth0"   # A3 环境
131|
132|# 通信接口
133|export GLOO_SOCKET_IFNAME=${NET_CARD_NAME}
134|export TP_SOCKET_IFNAME=${NET_CARD_NAME}
135|export HCCL_SOCKET_IFNAME=${NET_CARD_NAME}
136|
137|# 层级通信 (A2 环境)
138|export HCCL_INTRA_PCIE_ENABLE=1
139|export HCCL_INTRA_ROCE_ENABLE=0
140|
141|# 层级通信 (A3 环境)
142|export HCCL_INTRA_PCIE_ENABLE=0
143|export HCCL_INTRA_ROCE_ENABLE=1
144|
145|# HCCL 优化
146|export HCCL_OP_EXPANSION_MODE=AIV
147|export HCCL_BUFFSIZE=512
148|```
149|
150|---
151|
152|## 选择建议
153|
154|| 场景 | 推荐模式 |
155||------|----------|
156|| 单机部署 | mix-pd |
157|| 小规模测试 | mix-pd |
158|| 大规模生产 | full-pd |
159|| Prefill/Decode 独立扩展 | full-pd |
160|| 低延迟要求 | full-pd (更多 Decode 节点) |
161|| 高吞吐要求 | full-pd (更多 Prefill 节点) |
162|
163|---
164|
165|## 参考脚本位置
166|
167|```
168|$VLLM_TOOLS_SCRIPT_ROOT/
169|├── deepseek/
170|│   ├── mix-pd/start_mix.sh      # DeepSeek mix-pd
171|│   └── full-pd/
172|│       ├── start_prefill.sh
173|│       └── start_decode.sh
174|├── glm5.0/
175|│   └── mix-pd/start_mix.sh
176|├── qwen3.5/
177|│   ├── full-pd/
178|│   └── start_qwen35_397b_A3.sh
179|└── kimi-k25/
180|    └── A3-full-pd/
181|        ├── start_prefill.sh
182|        └── start_decode.sh
183|```
184|