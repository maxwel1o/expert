---
name: sequencer2d-npu
description: "Sequencer2D 系列 3 个模型 (sequencer2d_s/m/l) 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：Sequencer2D NPU 部署、sequencer2d 昇腾推理、精度对比、npu-smi。"
---

# Sequencer2D NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 Sequencer2D 系列图像分类模型，完成推理、精度验证，并发布模型仓库。

## 概述

本 Skill 提供 Sequencer2D 系列 3 个模型在昇腾 NPU 上的自动化部署、推理测试、CPU/NPU 精度对比、README 生成和模型仓库发布能力。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend 910B/910) |
| 模型数量 | 3 个（sequencer2d_s.in1k、sequencer2d_m.in1k、sequencer2d_l.in1k） |
| 框架版本 | PyTorch 2.1+, torch_npu, timm 1.0+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| CANN 版本 | 8.5.1+ |

### 支持的模型

| 模型名称 | 参数量 | GitCode 仓库 |
|----------|--------|-------------|
| sequencer2d_s.in1k | 24M | [repo](https://gitcode.com/m0_74196153/sequencer2d_s.in1k-npu) |
| sequencer2d_m.in1k | 55M | [repo](https://gitcode.com/m0_74196153/sequencer2d_m.in1k-npu) |
| sequencer2d_l.in1k | 113M | [repo](https://gitcode.com/m0_74196153/sequencer2d_l.in1k-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.10+ 环境，昇腾 NPU 驱动，CANN 8.5.1+。

**动作**:
1. 检查 Python 版本：

```bash
python3 --version
```

2. 检测 NPU 状态：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch>=2.1.0 torchvision>=0.16.0 timm>=1.0.0 Pillow>=10.0.0 torch_npu modelscope>=1.0.0
```

**输出**: NPU 可用状态，依赖已安装。

### Step 2: 模型选择与权重下载

**输入**: 模型名称。

**动作**:
4. 校验模型名是否在支持列表内（sequencer2d_s.in1k / sequencer2d_m.in1k / sequencer2d_l.in1k 或 all）。
5. 从 ModelScope 下载模型权重文件（model.safetensors）。

```bash
python3 scripts/batch_runner.py --model sequencer2d_s.in1k
```

**输出**: 模型名和权重已就绪。

### Step 3: CPU 基线推理

**输入**: 模型名、设备类型 `cpu`。

**动作**:
6. 执行 CPU 推理并保存 logits：

```bash
python3 scripts/inference.py --model sequencer2d_s.in1k --weights /path/to/model.safetensors --device cpu --dump logits_cpu.npy
```

7. 验证推理输出包含 top-5 预测结果。

**输出**: CPU 推理 logits（`logits_cpu.npy`）和 top-5 预测。

### Step 4: NPU 推理

**输入**: 模型名、设备类型 `npu`。

**动作**:
8. 检查 NPU 可用性，不可用时自动回退 CPU。
9. 执行 NPU 推理：

```bash
python3 scripts/inference.py --model sequencer2d_s.in1k --weights /path/to/model.safetensors --device npu --dump logits_npu.npy
```

10. 处理 OOM：释放缓存后重试。

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

**输出**: NPU 推理 logits（`logits_npu.npy`），标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU、NPU 推理 logits 文件。

**动作**:
11. 执行精度对比脚本：

```bash
python3 scripts/compare_cpu_npu.py --cpu logits_cpu.npy --npu logits_npu.npy --model sequencer2d_s
```

12. 计算逐类精度误差：误差率 = max(|CPU_logits - NPU_logits|) / (max(CPU_logits) - min(CPU_logits))。若 `max_error < 1%` 标记通过，否则标记失败。

**输出**: 精度对比报告（误差指标表格）。

### Step 6: README 与终端截图生成

**输入**: 精度通过的结果。

**动作**:
13. 根据实际测试数据自动生成中文 README，包含模型介绍、NPU 适配说明、CPU/NPU 对比表格、精度数据和性能数据。
14. 生成模拟终端输出截图：

```bash
python3 /opt/atomgit/terminal_screenshot.py --text "推理结果文本" --output terminal.png
```

**输出**: 每个模型的 README.md 和 terminal_*.png 截图。

### Step 7: 串行批量执行

**输入**: 剩余模型列表。

**动作**:
15. 使用 batch_runner.py 自动串行执行所有剩余模型：

```bash
python3 scripts/batch_runner.py
```

16. 每完成一个模型释放资源。

**输出**: 所有模型的推理结果和精度报告。

### Step 8: 文档生成与 GitCode 仓库发布

**输入**: 精度通过的模型结果。

**动作**:
17. 生成终端截图。
18. 发布模型仓库到 GitCode：

```bash
# 创建仓库
curl -X POST -H "Authorization: Bearer ${ATOMGIT_USER_TOKEN}" \
  -d '{"name":"sequencer2d_s.in1k-npu","repository_type":"model"}' \
  https://api.gitcode.com/api/v5/user/repos

# 推送代码
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/sequencer2d_s.in1k-npu.git
git push -u origin main
```

**输出**: 模型 README、截图、GitCode 模型仓库。

## 执行检查点与用户确认

每个关键步骤设有检查点 checkpoint，用户必须暂停确认后才能继续。以下 7 个检查点覆盖从环境准备到仓库发布的完整流程：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测后 | NPU 设备是否可用，CANN 版本是否正确 | 暂停并提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 模型确认检查点 | 模型选择后 | 选定的模型名称是否正确 | 暂停并返回重新选择 |
| 3 | CP-3: CPU 基线检查点 | CPU 推理后 | CPU 推理 top-5 是否合理 | 暂停并检查模型加载 |
| 4 | CP-4: NPU 推理确认 checkpoint | NPU 推理后 | NPU 推理是否成功，logits 是否保存 | 暂停并检查 NPU 显存 |
| 5 | CP-5: 精度确认检查点 | 精度对比后 | 误差是否 < 1%，对比指标是否达标 | 暂停并检查推理脚本一致性后重试 |
| 6 | CP-6: 截图检查点 | 截图生成后 | 截图内容是否准确 | 暂停并修改后重新生成截图 |
| 7 | CP-7: 发布前审批检查点 | 推送仓库前 | 仓库名和 README 是否正确 | 暂停并修改后重新申请用户确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 回退 CPU 推理，标记 fallback | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 OOM 错误 | 释放缓存后重试，最多 3 次 | 调整 batch 或释放其他进程 |
| 模型加载异常 | timm.create_model 抛出异常 | 打印错误堆栈，跳过当前模型 | 修正模型名或检查权重文件 |
| 网络超时 | pip/curl 下载失败 | 重试最多 3 次，每次间隔 5 秒 | 切换镜像源或检查网络连接 |
| 精度超标 | 误差 >= 1% | 记录偏差明细，中止发布 | 检查推理脚本和数据类型一致性 |
| API 调用失败 | GitCode API 返回非 200 | 打印错误状态码和响应体 | 检查 token 权限和仓库名 |
| 权重文件损坏 | safetensors 加载异常 | 重新下载权重文件 | 删除缓存后重新下载 |
| 用户确认超时 | 检查点等待用户确认超时 | 暂停流程并通知用户，保留当前进度 | 用户恢复后从当前 checkpoint 继续 |
| 截图生成失败 | terminal_screenshot.py 异常 | 跳过截图步骤，继续文档生成 | 手动运行截图命令 |
| Python 版本不兼容 | Python 版本 < 3.10 | 提示升级 Python | 安装 Python 3.10+ |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/batch_runner.py` | 串行批量执行所有模型的入口脚本 |
| `scripts/inference.py` | CPU/NPU 推理脚本，保存 logits 为 .npy 文件 |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本，输出误差指标 |
| `skill.json` | 技能元数据（模型列表、依赖版本、硬件要求） |
| `test-prompts.json` | 本技能测试提示词 |
| `cpu_logits.npy` | CPU 推理结果（运行后生成） |
| `npu_logits.npy` | NPU 推理结果（运行后生成） |
| `compare_results.json` | CPU/NPU 精度对比结果 evals.json（运行后生成） |
| `references/` | 精度验证参考指标和 benchmark 数据（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 否 | all | 模型名称（sequencer2d_s.in1k / sequencer2d_m.in1k / sequencer2d_l.in1k 或 all） |
| `skip_inference` | boolean | 否 | false | 跳过推理步骤，只生成文档和提交仓库 |
| `skip_push` | boolean | 否 | false | 跳过 GitCode 仓库推送 |

## 使用约束

1. 使用 ModelScope 下载模型权重，优先使用国内镜像加速。
2. 精度验证通过前不提交模型仓库。
3. 串行执行避免 NPU 显存溢出，每个模型完成后主动释放资源。
4. CPU/NPU 对比需使用相同输入图片和预处理流程。
