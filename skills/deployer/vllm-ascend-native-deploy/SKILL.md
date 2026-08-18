---
name: vllm-ascend-native-deploy
description: >
  原生支持（无需代码修改）的 HuggingFace 模型在 vLLM-Ascend 上的快速部署与 Smoke 验证 Skill。
  涵盖环境变量配置、vllm serve 启动命令模板、关键参数说明、API 端点 Smoke 测试、常见问题排查。
  适用于 Gemma、Qwen、Llama 等已被 vLLM-Ascend 原生支持的模型架构。
  当用户提到在昇腾 NPU 上部署 vLLM 模型、启动推理服务、Smoke 验证时触发。
metadata:
  short-description: vLLM-Ascend 原生模型快速部署与 Smoke 验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, vllm, deployment, inference, smoke-test, native-support]
---

# vLLM-Ascend 原生模型快速部署与 Smoke 验证 Skill

本 Skill 提供在华为昇腾 NPU 上通过 vLLM-Ascend 快速部署原生支持模型（无需代码修改）的标准化流程。
以 `google/gemma-3-270m-it` 在 `vLLM-Ascend 0.18.0rc1` 上的验证为参考案例。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 910 系列（至少 1 卡） |
| CANN | >= 8.0（推荐 8.3+） |
| Python | 3.9 – 3.13 |
| vllm-ascend | >= 0.18.0rc1 |
| 模型 | 本地路径或 HuggingFace/ModelScope 模型 ID |

## 流程总览

```
0. 环境检查
→ 1. 环境变量配置
→ 2. 端口检查
→ 3. 启动 vLLM 服务
→ 4. Smoke 验证（API 端点测试）
→ 5. 验收确认
```

---

## 0. 环境检查

```bash
# 确认 NPU 可用
npu-smi info

# 确认 vllm-ascend 已安装
python -c "import vllm_ascend; print(vllm_ascend.__version__)"
```

**通过标准**：至少 1 张 NPU 显示 `Health: OK`，vllm-ascend 版本符合预期。

---

## 1. 环境变量配置

在启动服务前，按需设置以下环境变量：

```bash
# 使用 ModelScope 下载（国内推荐）
export VLLM_USE_MODELSCOPE=true

# NPU 内存分配策略：启用 expandable_segments 减少 OOM
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# HCCL 缓冲区大小（多卡时需要，单卡可省略）
export HCCL_BUFFSIZE=512

# OpenMP 绑定与线程数（避免 CPU 资源争抢）
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

# 启用 TaskQueue 优化
export TASK_QUEUE_ENABLE=1

# 日志目录（如遇权限问题，设置为可写路径）
export ASCEND_LOG_PATH=/tmp/ascend/log
```

> **注意**：`ASCEND_RT_VISIBLE_DEVICES` 通常无需手动设置，vLLM-Ascend 会自动发现可用 NPU。

---

## 2. 端口检查

```bash
ss -lntp | grep ':8000 ' || true
```

若 8000 端口已被占用，启动时换用其他端口（如 `--port 8001`）。

---

## 3. 启动 vLLM 服务

### 3.1 单卡启动命令模板

```bash
vllm serve <model_path_or_id> \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --seed 1024 \
  --served-model-name <model_alias> \
  --max-num-seqs 32 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true}'
```

### 3.2 关键参数说明

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `--tensor-parallel-size` | 1 | Tensor 并行大小，小模型单卡即可 |
| `--data-parallel-size` | 1 | Data 并行大小 |
| `--max-model-len` | 32768 | 最大上下文长度 |
| `--max-num-seqs` | 32 | 每 DP 组最大并发请求数 |
| `--max-num-batched-tokens` | 4096 | 单步最大处理 tokens 数 |
| `--gpu-memory-utilization` | 0.90 | HBM 利用率上限 |
| `--compilation-config` | `FULL_DECODE_ONLY` | ACL Graph 编译模式 |
| `--trust-remote-code` | — | 允许加载自定义模型架构 |

### 3.3 后台运行（nohup）

```bash
nohup vllm serve <model_path> \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --served-model-name <alias> \
  --trust-remote-code \
  > serve.log 2>&1 &
```

---

## 4. Smoke 验证

服务启动后（首次启动需等待 ACL Graph 编译完成，约 30 秒），执行以下验证：

### 4.1 模型列表端点

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

**通过标准**：返回 HTTP 200，JSON 中包含模型信息。

### 4.2 聊天补全端点

```bash
curl -sf http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_alias>",
    "messages": [
      {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ],
    "temperature": 0,
    "max_tokens": 16
  }'
```

**通过标准**：返回 HTTP 200，JSON 中 `choices[0].message.content` 包含有效文本。

### 4.3 离线推理脚本（可选）

若无需长期运行服务，可直接用离线模式加载：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="<model_path>",
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=32768,
    gpu_memory_utilization=0.90,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
messages = [{"role": "user", "content": "Hello, how are you?"}]

from vllm.entrypoints.chat_utils import apply_chat_template
tokenizer = llm.get_tokenizer()
prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)

outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## 5. 验收确认

- [ ] `npu-smi info` 显示至少 1 卡 Health: OK
- [ ] `vllm serve` 启动无报错，日志中显示模型加载成功
- [ ] 首次请求后 ACL Graph 编译完成，后续请求延迟稳定
- [ ] `/v1/models` 返回 200
- [ ] `/v1/chat/completions` 返回 200 且生成有效文本

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 启动报错 `can not create directory, directory: /home/xxx/ascend/log` | 默认日志目录无写权限 | `export ASCEND_LOG_PATH=/tmp/ascend/log` |
| 首次请求延迟极高（>30s） | ACL Graph 首次编译 | 正常现象，编译完成后速度稳定 |
| 端口 8000 被占用 | 已有其他服务监听 | 启动时换 `--port 8001` |
| `No module named 'vllm_ascend'` | vllm-ascend 未安装 | `pip install vllm-ascend` |
| OOM | 模型过大或并发过高 | 降低 `--max-num-seqs` 或 `--gpu-memory-utilization` |
| 模型下载失败 | 网络问题 | 设置 `VLLM_USE_MODELSCOPE=true` 或手动下载 |

---

## 参考文档

- vLLM-Ascend 官方文档：<https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/>
- ACL Graph 设计文档：<https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/developer_guide/Design_Documents/ACL_Graph.html>
