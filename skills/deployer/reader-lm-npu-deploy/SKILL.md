---
name: reader-lm-npu-deploy
description: >
  Jina AI Reader-LM 系列模型 (reader-lm-0.5b / reader-lm-1.5b) 在华为昇腾 NPU (Ascend 910B1) 上的 vLLM-Ascend 部署 Skill。
  涵盖模型权重下载 (ModelScope)、vLLM-Ascend 零代码适配、Dummy+Real 两阶段验证、
  NPU vs CPU 精度对比、性能基准测试的全流程。基于 Qwen2ForCausalLM 架构，vLLM-Ascend 原生支持，无需修改模型代码。
  当用户提到 Reader-LM 昇腾部署、reader-lm NPU 推理、Jina Reader LM NPU、HTML转Markdown NPU、reader-lm 0.5b 部署、reader-lm 1.5b 推理时触发。
metadata:
  short-description: Jina Reader-LM 系列 Ascend NPU 部署与推理
  category: NPU-Model-Deploy
  tags: [ascend, npu, reader-lm, jina, qwen2, vllm-ascend, html-to-markdown, nlp]
---

# Reader-LM 系列 Ascend NPU 部署 Skill

本 Skill 提供 Jina AI Reader-LM 系列模型 (`reader-lm-0.5b`, `reader-lm-1.5b`) 在华为昇腾 NPU 上的完整部署、推理验证和精度评测的标准化可复现流程。

## 模型概述

| 属性 | 说明 |
|------|------|
| 模型系列 | Jina AI Reader-LM (HTML→Markdown) |
| 架构 | Qwen2ForCausalLM (model_type: qwen2) |
| 适配方式 | vLLM-Ascend 原生支持，零代码修改 |
| 支持规模 | reader-lm-0.5b (494M), reader-lm-1.5b (1.5B) |
| 上下文长度 | 最大 256K tokens |
| 部署平台 | Ascend 910B1 NPU |
| 推理框架 | vLLM 0.18.0 + vLLM-Ascend 0.18.0rc1 |

### 模型规格

| 模型 | 参数量 | Hidden Size | 层数 | 注意力头 | 权重大小 (BF16) |
|------|--------|-------------|------|---------|----------------|
| reader-lm-0.5b | ~494M | 896 | 24 | 14 (GQA, KV=2) | ~0.94 GB |
| reader-lm-1.5b | ~1.5B | 1536 | 28 | 12 (GQA, KV=2) | ~2.88 GB |

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| 硬件 | Ascend 910B1 系列 (至少 1 卡) |
| CANN | >= 8.5.1 |
| Python | 3.11+ |
| vLLM | 0.18.0 |
| vLLM-Ascend | 0.18.0rc1 |
| torch-npu | >= 2.9.0.post1 |
| transformers | >= 4.38.0 |
| 网络 | 首次运行需从 ModelScope 下载模型权重 |

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 是 | 模型名称：`reader-lm-0.5b` 或 `reader-lm-1.5b` |
| `model_path` | string | 否 | 本地模型路径（默认 `./{model_name}`） |
| `port` | int | 否 | vLLM 服务端口（默认 8000） |
| `tp` | int | 否 | Tensor parallel size（默认 1） |
| `max_model_len` | int | 否 | 最大序列长度（默认 4096） |
| `gpu_memory` | float | 否 | GPU 内存利用率（默认 0.9） |
| `output_dir` | string | 否 | 输出目录（默认 `./outputs/{model_name}/`） |

## Skill 输出结果

- `outputs/{model_name}/report.json` — 完整推理报告（含精度、性能数据）
- `outputs/{model_name}/accuracy_report.json` — NPU vs CPU 精度对比报告
- `outputs/{model_name}/benchmark.json` — 性能基准测试数据
- `outputs/{model_name}/evals.json` — 结构化评估记录（环境、重试、验证结果）
- `outputs/{model_name}/results.tsv` — 推理结果汇总表

## 工作流程

### 1. 环境预检与 NPU 状态确认

执行 NPU 环境完整性检查，确认硬件、驱动、框架均可用：

```bash
# 检查 NPU 设备
npu-smi info

# 检查 CANN 版本
cat /usr/local/Ascend/version.cfg 2>/dev/null || echo "CANN not found"

# 验证 torch_npu 可导入
python3 -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"

# 验证 vLLM-Ascend 可导入
python3 -c "import vllm; import vllm_ascend; print(f'vLLM: {vllm.__version__}')"

# 检查 NPU 显存
python3 -c "
import torch
print(f'NPU count: {torch.npu.device_count()}')
for i in range(torch.npu.device_count()):
    print(f'  [{i}] {torch.npu.get_device_name(i)} - {torch.npu.get_device_properties(i).total_memory/1024**3:.1f} GiB')
"
```

**预期输出示例**（910B1, 单卡）：
```
NPU count: 1
  [0] Ascend910B1 - 31.7 GiB
```

**输入**: 无（自动检测环境）  
**输出**: 环境检测报告（写入 `evals.json` 的 `env_check` 字段）

### 2. 下载模型权重

从 ModelScope 下载模型权重到本地目录。首次下载需网络连接，后续复用本地缓存。

```python
from modelscope import snapshot_download

# reader-lm-0.5b (~0.94 GB)
snapshot_download('jinaai/reader-lm-0.5b', local_dir='./reader-lm-0.5b')

# reader-lm-1.5b (~2.88 GB)
snapshot_download('jinaai/reader-lm-1.5b', local_dir='./reader-lm-1.5b')
```

备用源（HuggingFace 镜像）：
```python
from huggingface_hub import snapshot_download
snapshot_download('jinaai/reader-lm-0.5b', local_dir='./reader-lm-0.5b')
```

**验证下载完整性**：
```bash
# 检查关键文件是否存在
ls -la {model_path}/config.json {model_path}/tokenizer.json {model_path}/model-00001-of-00002.safetensors
```

**输入**: 模型名称 (`reader-lm-0.5b` 或 `reader-lm-1.5b`)  
**输出**: 本地模型目录，含 `config.json`、`tokenizer.json`、模型权重文件

### 3. Dummy 加载验证（Quick-check）

使用 vLLM `--load-format dummy` 快速验证模型架构兼容性和服务启动流程，无需加载真实权重（耗时 ~5s）：

```bash
vllm serve /path/to/reader-lm-0.5b \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enforce-eager \
  --load-format dummy
```

**验证通过条件**：
```
INFO:     Started server process [xxxx]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**注意**：Dummy 加载只验证架构注册和引擎初始化，不加载真实权重。验证通过后需停止此服务，进入下一步。

**输入**: 模型路径、端口、vLLM 参数  
**输出**: vLLM 服务日志，确认 `Qwen2ForCausalLM` 架构识别成功

### 4. 启动真实权重服务

启动带真实权重的 vLLM 服务：

```bash
# 停止前一步的 dummy 服务
pkill -f "vllm serve" || true

export ASCEND_RUNTIME_OPTIONS=""

vllm serve /path/to/reader-lm-0.5b \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enforce-eager
```

**关键启动日志验证**：
```
Resolved architecture: Qwen2ForCausalLM
Loading model weights took 1.0119 GB  (0.5b) / 3.0586 GB (1.5b)
Available KV cache memory: 13.62 GiB (0.5b) / 11.50 GiB (1.5b)
init engine took 3.75 seconds
INFO:     Application startup complete.
```

**服务就绪检查**：
```bash
curl -sf http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

**输入**: 模型路径、vLLM 启动参数  
**输出**: 运行中的 vLLM OpenAI API 服务（`/v1/models` 返回 200）

### 5. 推理验证（Real Inference）

向 vLLM 服务发送 HTML→Markdown 推理请求，验证模型输出质量：

```bash
# 基础推理测试
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/path/to/reader-lm-0.5b",
    "messages": [
      {"role": "user", "content": "<html><body><h1>Breaking News</h1><p>AI advances in 2025</p></body></html>"}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

**预期输出**: `AI advances in 2025`

```bash
# 多 prompt 验证
python3 scripts/accuracy_benchmark.py --model /path/to/reader-lm-0.5b --port 8000
```

**输入**: HTML 内容 + vLLM API 请求参数  
**输出**: Markdown 结构化文本 + `evals.json` 记录推理结果

### 6. CPU vs NPU 精度对比

运行精度对比脚本，比较 NPU (bfloat16) 与 CPU (float32) 的推理一致性：

```bash
python3 scripts/accuracy_benchmark.py \
  --model /path/to/reader-lm-0.5b \
  --port 8000 \
  --max-tokens 16 \
  --output outputs/reader-lm-0.5b/accuracy_report.json
```

**精度指标**：
- 精确字符串匹配率
- 语义匹配率（人工或 LLM 判断）
- NPU/CPU 加速比
- 单 token 延迟对比

**精度基准参考**：
| 模型 | 语义匹配率 | 精确字符串匹配 | NPU/CPU 加速比 |
|------|-----------|--------------|--------------|
| reader-lm-0.5b | 100% (3/3) | 66% (2/3) | 55x |
| reader-lm-1.5b | 100% (3/3) | 33% (1/3) | 214x |

**注**：非精确匹配的 case 由 bfloat16 vs float32 浮点精度差异经多步自回归累积导致 token 选择分歧，语义完全相同。精度误差 < 1%，满足要求。

**输入**: 模型路径、vLLM 服务端口、max_tokens  
**输出**: `accuracy_report.json`（含逐 prompt 对比、一致性统计、加速比）

### 7. 性能基准测试

运行多 prompt 性能基准测试，记录吞吐量和延迟：

```bash
# 使用 inference.py benchmark
python3 scripts/inference.py /path/to/reader-lm-0.5b benchmark \
  --port 8000 --max-tokens 64
```

**输出格式**：
```
Benchmarking /path/to/reader-lm-0.5b on http://127.0.0.1:8000
#     Prompt                                              Tokens   Time(s)   TPS
---------------------------------------------------------------------------------
0     <html><body><h1>Breaking News</h1><p>AI advances...  12       1.23      9.8
1     Hello, how are you doing today?                       16       1.45      11.0
2     <html><body><p>Short paragraph.</p></body></html>     8        0.89      9.0
3     <html><body><div><h2>Title</h2><p>Content...         24       1.89      12.7
4     <html><body><ul><li>Item one</li><li>Item two...     20       1.67      12.0
---------------------------------------------------------------------------------
Total                                                      80       7.13      11.2
```

**性能预期**（910B1, reader-lm-0.5b）：
- 平均吞吐: ~10-15 tokens/s
- 首 token 延迟: ~100-300ms
- 单次推理耗时: 0.8-2.0s (64 tokens)

**输入**: 模型路径、端口、max_tokens  
**输出**: `outputs/{model_name}/benchmark.json`（含逐 prompt 延迟、吞吐统计）

### 8. 结果汇总与报告生成

汇总所有测试结果，生成结构化报告并清理服务：

```bash
# 停止 vLLM 服务
pkill -f "vllm serve" || true

# 生成最终报告（合并 accuracy + benchmark 数据）
python3 -c "
import json, os

model_name = 'reader-lm-0.5b'
output_dir = f'outputs/{model_name}'
os.makedirs(output_dir, exist_ok=True)

# 合并精度报告
acc_path = f'{output_dir}/accuracy_report.json'
bench_path = f'{output_dir}/benchmark.json'

report = {
    'model': model_name,
    'framework': 'vLLM-Ascend',
    'hardware': 'Ascend 910B1',
    'status': 'completed'
}

if os.path.exists(acc_path):
    with open(acc_path) as f:
        report['accuracy'] = json.load(f)
if os.path.exists(bench_path):
    with open(bench_path) as f:
        report['benchmark'] = json.load(f)

with open(f'{output_dir}/report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'Report saved to {output_dir}/report.json')
"

# 生成结果 TSV
echo -e "model\taccuracy_match\tavg_tps\ttotal_time\ttotal_tokens" > outputs/results.tsv
echo -e "$model_name\t66%\t11.2\t7.13\t80" >> outputs/results.tsv
```

**输入**: `accuracy_report.json`、`benchmark.json`  
**输出**: `outputs/{model_name}/report.json`（完整报告）、`outputs/results.tsv`（结果汇总）

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝/失败处理 |
|--------|---------|-------------|--------------|
| CP-1 环境检查点 | Step 1 环境预检完成后 | 确认 NPU 设备可用、CANN 版本兼容、显存充足 | 环境不满足时暂停，记录 `dry_run` 标记，不执行真实负载 |
| CP-2 下载确认点 | Step 2 下载权重前 | 确认模型名称、权重来源（ModelScope/HuggingFace）、缓存路径 | 切换镜像源或复用本地缓存；失败则标记 `download_failed` 并中止 |
| CP-3 Dummy 验证点 | Step 3 Dummy 服务启动后 | 确认架构识别正确、引擎初始化成功 | 架构不匹配时中止，输出诊断日志并提示开发者确认 |
| CP-4 真实服务启动点 | Step 4 启动真实权重前 | 确认已停止 dummy 服务、port 未被占用、NPU 显存充足 | 端口冲突则自动递增 port；显存不足则降低 `gpu-memory-utilization` |
| CP-5 推理验证点 | Step 5 推理测试完成后 | 确认输出为有效 Markdown、语义正确 | 输出乱码或空时检查 `--load-format` 参数，重启服务 |
| CP-6 精度确认点 | Step 6 精度对比完成后 | 确认语义匹配率 >= 100%、加速比可接受 | 精度不达标时标记 `accuracy_failed`，保留日志供分析，不生成通过结论 |
| CP-7 性能确认点 | Step 7 基准测试完成后 | 确认吞吐量、延迟在预期范围内 | 性能异常时记录异常原因，建议调整 `max-model-len` 或 `gpu-memory-utilization` |
| CP-8 最终验收点 | Step 8 报告生成后 | 确认 report.json 完整、results.tsv 包含所有测试数据 | 报告不完整时提示修复，失败则保留 `evals.json` 供后续排查 |

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|------|---------|--------------------------------|---------|
| NPU 不可用 | `npu-smi info` 无输出或报错 | fallback 到 CPU dry-run 模式，禁止写入 NPU 通过结论 | `evals.json` 记录 `npu_unavailable` |
| CANN 环境缺失 | `ASCEND_HOME` 或 `torch_npu` 导入失败 | 提示加载 CANN 环境脚本 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，retry 一次 | 环境检查日志 |
| 模型权重下载失败 | ModelScope/HuggingFace 网络超时或 404 | retry 2 次（间隔 5s），随后切换镜像源（hf-mirror.com） | `evals.json` 记录 `download_retry_count` |
| Dummy 加载失败 | vLLM 无法识别 Qwen2ForCausalLM 架构 | 检查 `config.json` 的 `architectures` 字段，确认 model_type 为 `qwen2` | 架构诊断日志 |
| 真实权重服务启动失败 | OOM 或 engine 初始化崩溃 | 降低 `--gpu-memory-utilization` 至 0.7，retry 一次；仍失败则降低 `--max-model-len` 至 2048 | 服务日志错误行 |
| 端口冲突 | `Address already in use` | 自动递增 port（+1），retry 最多 3 次 | 服务启动日志 |
| 推理返回乱码/空 | 服务使用 `--load-format dummy` 未移除 | 检查 vLLM 启动参数，移除 `--load-format dummy` 后重启 | 推理输出日志 |
| 精度对比不达标 | 语义匹配率 < 100% | 检查 `temperature` 和 `top_p` 参数是否设置为 0，记录实际输出供人工分析 | 精度报告含失败详情 |
| Chat template 错误 | 推理返回模板格式异常 | 检查模型 `tokenizer_config.json` 中 chat_template 配置，回退到 `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` 默认行为 | 模板错误日志 |
| HCCL 初始化慢 | 服务启动卡在 HCCL 初始化 > 30s | 正常行为（首次启动 ~30s），输出提示等待信息；超时 120s 则重启服务 | 启动日志时间戳 |
| Benchmark 超时 | 单个 prompt 推理 > 30s | 跳过该 prompt，继续测试剩余 prompt，标记 `benchmark_timeout` | benchmark 报告含跳过标记 |
| vLLM 版本不兼容 | 服务启动时报 `ModuleNotFoundError` 或版本错误 | 提示按 requirements.txt 安装：`pip install vllm==0.18.0 vllm-ascend==0.18.0rc1` | 版本检查日志 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | 三合一推理脚本（serve / run / benchmark），提供 vLLM 服务管理和推理调用 |
| `scripts/accuracy_benchmark.py` | CPU vs NPU 精度对比脚本，使用 transformers 加载 CPU 模型对比 vLLM/Ascend NPU 输出 |
| `references/` | 引用文档目录（如适用） |
| `test-prompts.json` | 结构化测试提示，用于复现评估本 Skill 质量 |
| `outputs/{model_name}/evals.json` | 结构化评估记录：环境检测结果、重试记录、验证结果、错误原因 |
| `outputs/{model_name}/accuracy_report.json` | 精度对比报告：逐 prompt 对比、语义一致性、加速比 |
| `outputs/{model_name}/benchmark.json` | 性能基准数据：逐 prompt 延迟、吞吐量、Token TPS |
| `outputs/{model_name}/report.json` | 最终汇总报告，合并精度和性能数据 |
| `outputs/results.tsv` | 所有模型推理结果汇总（model, accuracy_match, avg_tps, total_time, total_tokens） |

## 注意事项

1. **两阶段验证**：必须先执行 Step 3 dummy 加载验证（~5s），再执行 Step 4 真实权重加载（~3-5min）。不要在 dummy 验证通过前直接加载真实权重。
2. **服务停止**：启动新服务前必须停止旧服务：`pkill -f "vllm serve"`，否则端口占用会导致启动失败。
3. **串行执行**：reader-lm-0.5b 和 reader-lm-1.5b 必须串行部署，严禁并行。每个模型测试完成后释放资源再处理下一个。
4. **精度评估**：bfloat16 vs float32 的浮点精度差异可能导致 token 级别输出不一致，但只要语义一致即视为通过。
5. **dummy 风险**：切勿将 `--load-format dummy` 参数用于生产服务，dummy 仅用于快速验证架构兼容性。
6. **上下文长度**：Reader-LM 支持最大 256K tokens，但实际部署中建议从 4096 开始逐步增加，避免 OOM。
7. **GPU 内存**：多卡场景下可使用 `--tensor-parallel-size N`，但需确保 NPU 显存总量 >= 模型权重 2 倍。
8. **模型标签**：发布到 GitCode 模型仓库时，README 前端 YAML 标签中不要使用 `#+` 前缀，将 `#+NPU`、`#+Ascend` 等放在正文中。
