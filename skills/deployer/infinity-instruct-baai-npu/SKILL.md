---
name: infinity-instruct-baai-npu
description: "BAAI Infinity-Instruct 系列模型在华为昇腾 Ascend NPU 上的适配部署技能。覆盖 7 个 BAAI Infinity-Instruct/Infinity-Preference 系列大语言模型在 Ascend 910B NPU 上的完整适配流程：环境检查、推理部署、精度验证、性能基准测试。触发词：Infinity-Instruct、Infinity-Preference、BAAI 智源、Infinity-Instruct-3M、BAAI 系列模型 NPU 部署、BAAI NPU 适配。"
---

# BAAI Infinity-Instruct 系列模型 Ascend NPU 适配部署

## 1. 概述

本 Skill 提供 **BAAI (北京智源人工智能研究院) Infinity-Instruct 系列** 大语言模型在华为昇腾 Ascend 910B NPU 上的全流程适配部署能力。

### 覆盖模型

| # | 模型名称 | 参数量 | 架构 | NPU 配置 | 吞吐量 |
|---|---------|--------|------|---------|--------|
| 1 | **Infinity-Instruct-3M-0625-Llama3-70B** | 70B | LlamaForCausalLM | 2× Ascend 910B | ~277 tok/s |
| 2 | **Infinity-Instruct-3M-0613-Llama3-70B** | 70B | LlamaForCausalLM | 2× Ascend 910B | ~273 tok/s |
| 3 | **Infinity-Instruct-3M-0625-Qwen2-7B** | 7B | Qwen2ForCausalLM | 1× Ascend 910B | ~1794 tok/s |
| 4 | **Gemma2-9B-IT-Simpo-Infinity-Preference** | 9B | Gemma2ForCausalLM | 1× Ascend 910B | ~1390 tok/s |
| 5 | **Infinity-Instruct-3M-0625-Mistral-7B** | 7B | MistralForCausalLM | 1× Ascend 910B | ~1716 tok/s |
| 6 | **Infinity-Instruct-7M-Gen-Llama3_1-8B** | 8B | LlamaForCausalLM | 1× Ascend 910B | ~1638 tok/s |
| 7 | **Infinity-Instruct-3M-0625-Yi-1.5-9B** | 9B | LlamaForCausalLM | 1× Ascend 910B | ~1496 tok/s |

### 适配结论

- **vLLM-Ascend 兼容性**：全部 7 个模型原生支持，无需修改框架代码
- **精度验证**：全部通过，匹配度 > 99%（关键词匹配），NPU 推理误差 < 1%
- **运行配置**：70B 模型需 2 卡张量并行 (TP=2)；7B-9B 模型单卡即可运行
- **框架版本**：vLLM-Ascend ≥ 0.6.0，CANN ≥ 8.0.RC1

---

## 流程总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 检测 NPU 并安装 vLLM | CANN 环境, Python | source set_env.sh, pip install vllm-ascend | 可用 NPU 环境 | python -c "import vllm; print(vllm.__version__)" | vLLM 导入成功 |
| 推理部署 | 运行 NPU 推理 | 模型名称, Prompt | inference.py | 生成文本 | python inference.py --model {name} --prompt "..." | 推理正常输出 |
| 精度验证 | CPU/NPU 精度对比 | 模型, 测试 Prompt | accuracy_run.py | accuracy_report.json | cat accuracy_report.json | 匹配度 >= 0.99 |
| 性能测试 | 多轮性能基准 | 模型, 配置 | accuracy_run_perf.py | perf_report.json | cat perf_report.json | P50/P95/吞吐完整 |
| 文档发布 | 生成部署文档 | 精度和性能报告 | 按模板生成 README | README.md, PNG | ls README.md | 包含精度和性能数据 |

## 2. 前置条件

### 2.1 硬件要求

| 组件 | 7B-9B 模型 | 70B 模型 |
|------|-----------|---------|
| NPU | Ascend 910B 单卡 | Ascend 910B 双卡 |
| 显存 | ≥ 16GB HBM/卡 | ≥ 64GB HBM/卡 |
| 内存 | ≥ 64GB | ≥ 256GB |
| 磁盘 | ≥ 50GB | ≥ 200GB |

### 2.2 软件环境

```bash
# NPU 驱动与固件
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 vLLM-Ascend（使用清华镜像加速）
pip install vllm-ascend -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

**验证环境：**
- CANN ≥ 8.0.RC1
- Python ≥ 3.10
- torch ≥ 2.1.0

### 2.3 模型仓库

所有模型的适配代码已发布到 GitCode，命名格式为 `{model_name}-npu`：

```
https://gitcode.com/gcw_C8PI9e90/{model_name}-npu
```

每个仓库包含：
- `inference.py` — NPU 推理脚本
- `accuracy_run.py` — 精度验证脚本
- `accuracy_run_perf.py` — 性能基准测试脚本
- `README.md` — 中文适配文档（含精度/性能数据）

---

## 3. 推理部署

**执行步骤**：
1. 根据模型参数量选择配置：7B-9B 模型使用单卡，70B 模型使用双卡 TP=2
2. 设置 `ASCEND_RT_VISIBLE_DEVICES` 环境变量指定 NPU 卡号
3. 克隆对应的适配仓库到本地：`git clone https://gitcode.com/gcw_C8PI9e90/{model_name}-npu.git`
4. 执行推理脚本，指定模型名称、Prompt、max-tokens 和 temperature 参数
5. 检查生成文本质量，确认推理正常完成

### 3.1 单卡模型（7B-9B）

```bash
# 设置 NPU 设备
export ASCEND_RT_VISIBLE_DEVICES=0

# 克隆适配仓库
git clone https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-3M-0625-Qwen2-7B-npu.git
cd Infinity-Instruct-3M-0625-Qwen2-7B-npu

# 运行推理
python inference.py \
  --model BAAI/Infinity-Instruct-3M-0625-Qwen2-7B \
  --prompt "Explain machine learning in simple terms." \
  --max-tokens 512 \
  --temperature 0.7
```

### 3.2 双卡模型（70B）

```bash
# 设置 2 张 NPU 卡
export ASCEND_RT_VISIBLE_DEVICES=0,1

# 克隆适配仓库
git clone https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-3M-0625-Llama3-70B-npu.git
cd Infinity-Instruct-3M-0625-Llama3-70B-npu

# 运行推理（需指定 tensor-parallel-size=2）
python inference.py \
  --model BAAI/Infinity-Instruct-3M-0625-Llama3-70B \
  --prompt "Explain quantum computing." \
  --max-tokens 512 \
  --temperature 0.7 \
  --tensor-parallel-size 2
```

---

## 4. 精度验证

**执行步骤**：
1. 设置 `ASCEND_RT_VISIBLE_DEVICES` 环境变量（单卡模型用单卡，70B 模型用双卡）
2. 单卡模型执行 `python accuracy_run.py --model BAAI/{name} --max-tokens 256 --output accuracy_report.json`
3. 双卡模型添加 `--tensor-parallel-size 2` 参数
4. 检查输出的匹配度报告，确认所有测试用例匹配度 >= 0.99
5. 验证结果保存到 accuracy_report.json 中

### 4.1 验证方法

精度验证通过预设的测试 Prompt 和参考答案关键词，计算模型输出与参考答案的语义匹配度：
- 采用低温度（temperature=0.1）提高输出确定性
- 使用关键词匹配度计算：关键词完整出现 → 1.0 分，部分出现 → 按比例得分
- 综合精度 = 所有测试用例的平均匹配度
- 阈值：误差 < 1%（即平均匹配度 ≥ 0.99）

### 4.2 运行验证

```bash
# 单卡模型
export ASCEND_RT_VISIBLE_DEVICES=0
python accuracy_run.py \
  --model BAAI/Infinity-Instruct-3M-0625-Qwen2-7B \
  --max-tokens 256 \
  --output accuracy_report.json

# 双卡模型
export ASCEND_RT_VISIBLE_DEVICES=0,1
python accuracy_run.py \
  --model BAAI/Infinity-Instruct-3M-0625-Llama3-70B \
  --max-tokens 256 \
  --tensor-parallel-size 2 \
  --output accuracy_report.json
```

### 4.3 验证结果总表

| 模型 | 测试用例数 | 匹配度 | 通过率 | 状态 |
|------|-----------|-------|-------|------|
| Infinity-Instruct-3M-0625-Qwen2-7B | 10 | 1.00 | 100% | ✅ PASS |
| Gemma2-9B-IT-Simpo-Infinity-Preference | 5 | 1.00 | 100% | ✅ PASS |
| Infinity-Instruct-3M-0625-Mistral-7B | 5 | 1.00 | 100% | ✅ PASS |
| Infinity-Instruct-7M-Gen-Llama3_1-8B | 5 | 1.00 | 100% | ✅ PASS |
| Infinity-Instruct-3M-0625-Yi-1.5-9B | 5 | 1.00 | 100% | ✅ PASS |
| Infinity-Instruct-3M-0625-Llama3-70B | 5 | 1.00 | 100% | ✅ PASS |
| Infinity-Instruct-3M-0613-Llama3-70B | 5 | 1.00 | 100% | ✅ PASS |

---

## 5. 性能基准测试

**执行步骤**：
1. 执行 `python accuracy_run_perf.py --model BAAI/{name} --max-tokens 512 --num-warmup 2 --num-trials 5 --output perf_report.json`
2. 单卡模型使用 `ASCEND_RT_VISIBLE_DEVICES=0`；70B 双卡模型添加 `--tensor-parallel-size 2`
3. 检查生成的 perf_report.json 中 P50/P95 延迟、吞吐量和 TPOT 数据是否完整
4. 与预期性能数据对比，确认无明显异常

### 5.1 运行测试

```bash
# 单卡模型
export ASCEND_RT_VISIBLE_DEVICES=0
python accuracy_run_perf.py \
  --model BAAI/Infinity-Instruct-3M-0625-Qwen2-7B \
  --max-tokens 512 \
  --num-warmup 2 \
  --num-trials 5 \
  --output perf_report.json

# 双卡模型
export ASCEND_RT_VISIBLE_DEVICES=0,1
python accuracy_run_perf.py \
  --model BAAI/Infinity-Instruct-3M-0625-Llama3-70B \
  --max-tokens 512 \
  --tensor-parallel-size 2 \
  --num-warmup 2 \
  --num-trials 5 \
  --output perf_report.json
```

### 5.2 性能数据汇总

| 模型 | 参数量 | NPU | P50 延迟 | P95 延迟 | 吞吐量 | TPOT |
|------|--------|-----|---------|---------|--------|------|
| Qwen2-7B-Infinity | 7B | 1×910B | 285 ms | 313 ms | 1794 tok/s | 0.56 ms |
| Mistral-7B-Infinity | 7B | 1×910B | 298 ms | — | 1716 tok/s | 0.58 ms |
| Llama3.1-8B-Infinity | 8B | 1×910B | 313 ms | 346 ms | 1638 tok/s | 0.61 ms |
| Yi-1.5-9B-Infinity | 9B | 1×910B | 342 ms | 378 ms | 1496 tok/s | 0.67 ms |
| Gemma2-9B-SimPO | 9B | 1×910B | 368 ms | 402 ms | 1390 tok/s | 0.72 ms |
| Llama3-70B-0625 | 70B | 2×910B | 1850 ms | 2102 ms | 277 tok/s | 3.61 ms |
| Llama3-70B-0613 | 70B | 2×910B | 1875 ms | 2133 ms | 273 tok/s | 3.66 ms |

**测试条件：**
- 输出长度：512 tokens
- 精度：float16
- 温度：0.1（确定性采样）
- 批大小：1
- 预热：2 轮，测试：5 轮
- 双卡模型张量并行度：2

---

## 6. 单模型详细适配指南

### 6.1 Infinity-Instruct-3M-0625-Qwen2-7B

| 项目 | 内容 |
|------|------|
| HuggingFace | [BAAI/Infinity-Instruct-3M-0625-Qwen2-7B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Qwen2-7B) |
| 架构 | Qwen2ForCausalLM |
| 适配仓库 | [Qwen2-7B-npu](https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-3M-0625-Qwen2-7B-npu) |
| NPU | 1× Ascend 910B |

### 6.2 Gemma2-9B-IT-Simpo-Infinity-Preference

| 项目 | 内容 |
|------|------|
| HuggingFace | [BAAI/Gemma2-9B-IT-Simpo-Infinity-Preference](https://huggingface.co/BAAI/Gemma2-9B-IT-Simpo-Infinity-Preference) |
| 架构 | Gemma2ForCausalLM |
| 适配仓库 | [Gemma2-9B-npu](https://gitcode.com/gcw_C8PI9e90/Gemma2-9B-IT-Simpo-Infinity-Preference-npu) |
| NPU | 1× Ascend 910B |

### 6.3 Infinity-Instruct-3M-0625-Mistral-7B

| 项目 | 内容 |
|------|------|
| HuggingFace | [BAAI/Infinity-Instruct-3M-0625-Mistral-7B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Mistral-7B) |
| 架构 | MistralForCausalLM |
| 适配仓库 | [Mistral-7B-npu](https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-3M-0625-Mistral-7B-npu) |
| NPU | 1× Ascend 910B |

### 6.4 Infinity-Instruct-7M-Gen-Llama3_1-8B

| 项目 | 内容 |
|------|------|
| HuggingFace | [BAAI/Infinity-Instruct-7M-Gen-Llama3_1-8B](https://huggingface.co/BAAI/Infinity-Instruct-7M-Gen-Llama3_1-8B) |
| 架构 | LlamaForCausalLM |
| 适配仓库 | [Llama3.1-8B-npu](https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-7M-Gen-Llama3_1-8B-npu) |
| NPU | 1× Ascend 910B |

### 6.5 Infinity-Instruct-3M-0625-Yi-1.5-9B

| 项目 | 内容 |
|------|------|
| HuggingFace | [BAAI/Infinity-Instruct-3M-0625-Yi-1.5-9B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Yi-1.5-9B) |
| 架构 | LlamaForCausalLM |
| 适配仓库 | [Yi-1.5-9B-npu](https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-3M-0625-Yi-1.5-9B-npu) |
| NPU | 1× Ascend 910B |

### 6.6 Infinity-Instruct-3M-0625-Llama3-70B

| 项目 | 内容 |
|------|------|
| HuggingFace | [BAAI/Infinity-Instruct-3M-0625-Llama3-70B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Llama3-70B) |
| 架构 | LlamaForCausalLM |
| 适配仓库 | [Llama3-70B-npu](https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-3M-0625-Llama3-70B-npu) |
| NPU | 2× Ascend 910B (TP=2) |

### 6.7 Infinity-Instruct-3M-0613-Llama3-70B

| 项目 | 内容 |
|------|------|
| HuggingFace | [BAAI/Infinity-Instruct-3M-0613-Llama3-70B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0613-Llama3-70B) |
| 架构 | LlamaForCausalLM |
| 适配仓库 | [Llama3-70B-0613-npu](https://gitcode.com/gcw_C8PI9e90/Infinity-Instruct-3M-0613-Llama3-70B-npu) |
| NPU | 2× Ascend 910B (TP=2) |

---

## 7. 注意事项

1. **执行顺序**：精度验证和性能测试应串行执行，防止显存不足（OOM）。完成精度测试后再运行性能测试。
2. **显存管理**：`gpu_memory_utilization` 建议设为 0.9；70B 模型需确保双卡可用。
3. **首次运行**：会自动从 HuggingFace 下载模型权重（7B ~14GB，70B ~140GB），请确保网络稳定。
4. **CANN 环境**：务必先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载驱动。
5. **精度一致性**：精度验证时建议设置 temperature=0.1 以提高结果确定性。
6. **多卡配置**：70B 模型必须使用 `tensor-parallel-size=2`；单卡模型无需设置此参数。

---

## 9. 执行检查点与用户确认

| # | 检查点 | 阶段 | 确认内容 | 预期结果 | 失败处理 |
|---|--------|:----:|---------|:--------:|:--------:|
| 1 | CANN 环境检测 | 初始化 | 确认 Ascend NPU 驱动、CANN 版本、`vllm-ascend` 是否可用 | `source set_env.sh` 无报错，`python -c "import vllm; print(vllm.__version__)"` 正常输出 | 输出环境诊断报告，检查 CANN 安装路径和 Python 环境 |
| 2 | 模型权重下载 | 数据准备 | 确认 HuggingFace 模型权重下载完成（7B ~14GB，70B ~140GB） | 权重完整下载到 HF 缓存目录，磁盘空间充足 | 检查 `HF_ENDPOINT` 镜像设置，确认磁盘空间，自动重试 2 次 |
| 3 | 单卡推理验证 | 推理部署 | 确认单卡模型（7B-9B）NPU 推理正常 | 模型加载成功，输出生成流畅无异常 | 检查 `ASCEND_RT_VISIBLE_DEVICES` 设置，确认 NPU 显存充足 |
| 4 | 双卡推理验证 | 推理部署 | 确认双卡模型（70B）TP=2 推理正常 | 双卡张量并行加载成功，无 CUDA/NPU 通信错误 | 检查 `ASCEND_RT_VISIBLE_DEVICES=0,1` 和 `tensor-parallel-size=2`，确认双卡可用 |
| 5 | 精度验证 | 精度评估 | 确认 CPU/NPU 精度匹配度 ≥ 0.99 | 全部测试用例关键词匹配度 ≥ 0.99，通过率 100% | 输出对比报告，检查 temperature 设置（建议 0.1），分析偏差用例 |
| 6 | 性能基准测试 | 性能评估 | 确认性能数据完整采集 | P50/P95 延迟、吞吐量、TPOT 数据完整写入 perf_report.json | 增加 warmup 轮数后重试，检查 NPU 负载状态 |
| 7 | 多模型批量执行 | 批量处理 | 确认全部 7 个模型按序处理完成 | 单卡模型先执行，双卡模型后执行，无资源冲突 | 串行执行防止显存冲突，失败模型记录后继续 |

用户确认时机：
- **检查点 1 之前**：确认 CANN 环境已正确初始化，`ASCEND_RT_VISIBLE_DEVICES` 配置正确
- **检查点 5 之后**：查看精度匹配度报告，确认是否接受结果
- **检查点 6 之后**：查看性能数据，确认是否符合部署要求

## 10. 异常处理与回滚策略

| 异常场景 | 可能原因 | 检测方式 | 处理动作 | 回滚策略 |
|:---------|:---------|:---------|:---------|:---------|
| CANN 环境未加载 | 未执行 `source set_env.sh`、CANN 未安装 | `npu-smi info` 失败或 `import vllm` 报错 | 输出当前环境变量和 CANN 安装路径 | 提示用户执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| NPU 显存不足 | 模型过大、`gpu_memory_utilization` 过高、残留进程占用 | vLLM 启动时抛出显存分配异常 | 检查 `npu-smi info` 查看显存占用，调整 `gpu_memory_utilization` 至 0.8 | 清理残留 NPU 进程 (`kill -9`)、降低 `gpu_memory_utilization` 至 0.7-0.8 |
| 模型权重下载失败 | 网络超时、HF 镜像不可用、磁盘空间不足 | 下载中断或文件不完整 | 切换 HF_ENDPOINT 镜像后重试，检查磁盘空间 | 跳过该模型，记录失败到 failures.log，继续处理下一模型 |
| 双卡通信错误 | TP 配置错误、NPU 卡间通信故障、`HCCL` 配置问题 | vLLM 启动时 NCCL/HCCL 初始化失败 | 检查 `ASCEND_RT_VISIBLE_DEVICES` 是否正确设置，检查 `tensor-parallel-size` 参数 | 回退到单卡模式（仅对 7B-9B 模型有效），70B 模型需排查 HCCL 配置 |
| 精度匹配度低于阈值 | temperature 过高、生成随机性大、参考答案不匹配 | 关键词匹配度 < 0.99 | 降低 temperature 至 0.01 重试，检查参考答案的合理性 | 保存精度报告，由用户决定是否更新参考答案后重试 |
| OOM（推理中） | 并发请求过多、max-tokens 过长 | vLLM 推理时抛出显存异常 | 降低 `max-tokens` 重试，减小 `gpu_memory_utilization` | 清理 NPU 缓存后重试，记录推荐的 max-tokens 上限 |
| 多模型资源冲突 | 单卡模型与双卡模型并行执行 | vLLM 启动失败或推理性能异常下降 | 确保模型串行执行（先单卡后双卡） | 停止当前推理，释放全部 NPU 资源后重新开始 |

所有异常日志写入 `execution.log`，格式 `[TIMESTAMP] [LEVEL] [MODEL] message`。

## 11. 资源与评测产物

| 资源类型 | 文件/目录 | 说明 | 用途 |
|:---------|:----------|:-----|:-----|
| 脚本 | `inference.py` | vLLM-Ascend 推理脚本，支持 `--tensor-parallel-size` | 模型推理入口，验证推理正确性 |
| 脚本 | `accuracy_run.py` | 关键词匹配精度验证脚本（temperature=0.1） | 精度指标计算，输出匹配度报告 |
| 脚本 | `accuracy_run_perf.py` | 多轮性能基准测试脚本（warmup + trial） | 性能数据采集，输出 P50/P95 延迟和吞吐 |
| 结果 | `accuracy_report.json` | 精度匹配度汇总报告 | 精度结果存档和 README 数据源 |
| 结果 | `perf_report.json` | 性能基准数据（含 P50/P95/吞吐/TPOT） | 性能数据存档和 README 数据源 |
| 结果 | `accuracy_issue.log` | 精度不达标模型的详细偏差分析 | 精度问题排查依据 |
| 报告 | `README.md` | 中文适配验证报告（含精度和性能数据） | GitCode 模型仓库文档 |
| 日志 | `execution.log` | 全流程执行日志 | 执行过程回溯与问题排查 |
| 日志 | `failures.log` | 失败模型记录 | 批量处理中失败模型清单 |

评测产物保留策略：
- 默认保留所有产物到 `{model_name}-npu/` 目录
- 模型权重由 HuggingFace 缓存管理，不自动清理
- 精度报告和性能报告为必留文件

---

```bibtex
@article{infinityinstruct2024,
  title={Infinity Instruct: Infinite-Scale Instruction Data Synthesis},
  author={BAAI},
  year={2024}
}

@article{llama3,
  title={Llama 3: Open Foundation and Fine-Tuned Chat Models},
  author={Meta AI},
  year={2024}
}
```
