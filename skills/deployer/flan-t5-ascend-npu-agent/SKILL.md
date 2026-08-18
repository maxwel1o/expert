---
name: flan-t5-ascend-npu-agent
description: Flan-T5 模型在华为昇腾 800I A2 NPU 上的适配与推理 Agent Skill。基于 transformers + torch_npu 原生推理，覆盖环境检查、权重加载、NPU 推理、精度验证、性能基准测试全流程。
keywords:
  - flan-t5
  - t5
  - ascend
  - ascend-910
  - npu
  - torch_npu
  - transformers
  - inference
  - 800I-A2
  - encoder-decoder
  - text-generation
tags:
  - nlp
  - text-generation
  - Flan-T5
  - T5
  - ascend
  - ascend-910
  - npu
  - 800I-A2
  - model-agent-tagged
datasets:
  - glue
  - superglue
pipeline_tag: text2text-generation
inference:
  - device: npu
    backend: torch_npu
    precision: fp32
license: apache-2.0
---

# Flan-T5 on Ascend NPU - Agent Skill

## 概述

本 Skill 提供 Flan-T5 系列模型在昇腾 NPU（Ascend910 系列）上的完整推理适配方案，支持环境验证、模型加载、NPU 推理执行、CPU vs NPU 精度对比以及性能基准测试。

**适用模型**：google/flan-t5-base / large / xl / xxl (0.3B ~ 11B)

**适用场景**：
- 在昇腾 NPU 上部署 Flan-T5 系列模型
- 验证 T5 encoder-decoder 架构在 NPU 上的兼容性
- 执行 CPU vs NPU 精度对比验证
- 采集 NPU 单卡/批量推理性能基线

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64） |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.8 - 3.11 |
| PyTorch | 与 CANN 版本匹配 |
| torch_npu | 与 PyTorch 版本一致 |
| transformers | >= 4.30 |

## 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定可见 NPU
export ASCEND_RT_VISIBLE_DEVICES=0

# 验证 torch_npu
python3 -c "import torch; import torch_npu; a = torch.randn(3,4).npu(); print('OK:', a.device)"
```

## 模型准备

### 权重下载

```bash
# HuggingFace 官方
huggingface-cli download google/flan-t5-base --local-dir ./flan-t5-base

# GitCode 镜像（需认证）
export HF_ENDPOINT=https://ai.gitcode.com/hf_mirrors
huggingface-cli download google/flan-t5-base --local-dir ./flan-t5-base
```

### 配置文件（无权重快速验证）

仅需要以下文件即可进行架构兼容性测试：
- `config.json`
- `tokenizer_config.json`
- `spiece.model`

## 核心文件

| 文件 | 说明 |
|------|------|
| `inference.py` | NPU 推理脚本 |
| `benchmark.py` | 精度/性能评测脚本 |
| `profiling.py` | L0/L1/L2 Profiling 采集脚本 |
| `readme.md` | 部署文档 |
| `inference_verification.md` | 单图推理验证文档 |
| `logs/benchmark_run.log` | 评测运行日志 |
| `logs/benchmark_result.json` | 结构化评测结果 |
| `screenshots/*.png` | 运行截图 |

## 推理脚本使用

### 单条推理

```bash
python3 inference.py /path/to/flan-t5-base \
  --prompt "translate English to German: Hello world"
```

### 批量推理

```bash
python3 inference.py /path/to/flan-t5-base --batch
```

### 测试模式（随机权重，验证架构兼容性）

```bash
python3 inference.py /path/to/flan-t5-base --test-mode --batch
```

## 精度验证

```bash
python3 benchmark.py /path/to/flan-t5-base
```

验证维度：
1. **Encoder hidden_states**：对比 encoder 输出张量的相对误差
2. **Decoder logits**：对比 decoder 首步输出的相对误差
3. **Generate token IDs**：对比 greedy generate 的 token 序列一致性

通过标准：
- mean_rel_diff < 1%
- generate token IDs 完全匹配

## 性能基准

```bash
python3 benchmark.py /path/to/flan-t5-base
```

输出指标：
- 单条推理延迟（greedy / beam search）
- 批量推理延迟与吞吐（batch_size=1,2,4,8）

## 关键适配点

### 1. 设备选择

```python
import torch
import torch_npu

device = torch.device("npu:0") if torch_npu.npu.is_available() else torch.device("cpu")
model = model.to(device)
```

### 2. 无算子替换需求

T5 模型完全由标准 PyTorch 算子组成（Linear、LayerNorm、Softmax、Embedding 等），`torch_npu` 已全覆盖，无需手动替换算子。

### 3. 图编译 warmup

首次推理会触发 CANN 算子编译，建议 warmup：

```python
for _ in range(3):
    with torch.no_grad():
        _ = model.generate(**inputs, max_length=64)
```

### 4. 不使用 vLLM

Flan-T5 为 encoder-decoder 架构，vLLM 当前（0.18.0）对 encoder-decoder 支持有限（仅 Whisper）。推荐使用 `transformers` 原生推理。

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `cannot create directory /home/atomgit/ascend/log` | 只读文件系统 | 可忽略，不影响功能 |
| `Permission mismatch` 警告 | CANN 目录权限 | 可忽略，不影响推理 |
| 首次推理延迟高 | CANN 算子编译 | warmup 3 次后再测 |
| 模型无法下载 | 网络受限 | 离线下载后上传 |
| Tokenizer legacy 警告 | transformers 版本差异 | 可忽略 |

## 赛道信息

- **赛道**：赛道二：性能优化（800I A2）
- **模型地址**：https://ai.gitcode.com/hf_mirrors/google/flan-t5-base

## 性能数据

| 场景 | CPU 延迟 | NPU 延迟 | 加速比 |
|------|---------|---------|--------|
| 单条推理（greedy） | ~8453 ms | ~1001 ms | **8.4×** |
| 单条推理（beam=4） | ~9066 ms | ~1135 ms | **8.0×** |
| 批量推理（bs=4） | ~9058 ms | ~1006 ms | **9.0×** |
| 批量推理（bs=8） | ~10598 ms | ~1009 ms | **10.5×** |

## 精度数据

| 指标 | 结果 | 通过标准 |
|------|------|----------|
| Encoder mean_rel_diff | **0.00129%** | < 1% ✅ |
| Decoder mean_rel_diff | **0.00155%** | < 1% ✅ |
| Token match rate | **100%** | 100% ✅ |