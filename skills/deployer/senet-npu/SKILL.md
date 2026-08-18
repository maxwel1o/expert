---
name: senet-npu
description: "SE-Net 系列 9 个模型 (SE-ResNet/SENet154/SE-HaLO-Net/SE-BotNet) 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：SE-Net NPU 部署、senet 昇腾推理、精度对比、npu-smi。"
---

# SE-Net NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 SE-Net 系列图像分类模型，完成推理、精度验证，并发布模型仓库。

## 概述

本 Skill 提供 SE-Net 系列 9 个模型在昇腾 NPU 上的自动化部署、推理测试、CPU/NPU 精度对比、README 生成和模型仓库发布能力。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend 910B/910) |
| 模型数量 | 9 个（SE-ResNet50/33ts/152d、SENet154、SE-HaLO-Net、SE-BotNet） |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 0.9+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| CANN 版本 | 8.5.1+ |

### 支持的模型

| 模型名称 | 架构 | 参数量 | GitCode 仓库 |
|----------|------|--------|-------------|
| seresnet50.a1_in1k | SE-ResNet50 | 28M | [repo](https://gitcode.com/m0_74196153/seresnet50.a1_in1k-npu) |
| seresnet50.ra2_in1k | SE-ResNet50 | 28M | [repo](https://gitcode.com/m0_74196153/seresnet50.ra2_in1k-npu) |
| seresnet50.a3_in1k | SE-ResNet50 | 28M | [repo](https://gitcode.com/m0_74196153/seresnet50.a3_in1k-npu) |
| seresnet50.a2_in1k | SE-ResNet50 | 28M | [repo](https://gitcode.com/m0_74196153/seresnet50.a2_in1k-npu) |
| seresnet33ts.ra2_in1k | SE-ResNet33ts | 16M | [repo](https://gitcode.com/m0_74196153/seresnet33ts.ra2_in1k-npu) |
| seresnet152d.ra2_in1k | SE-ResNet152d | 67M | [repo](https://gitcode.com/m0_74196153/seresnet152d.ra2_in1k-npu) |
| senet154.gluon_in1k | SENet154 | 115M | [repo](https://gitcode.com/m0_74196153/senet154.gluon_in1k-npu) |
| sehalonet33ts.ra2_in1k | SE-HaLO-Net33ts | 22M | [repo](https://gitcode.com/m0_74196153/sehalonet33ts.ra2_in1k-npu) |
| sebotnet33ts_256.a1h_in1k | SE-BotNet33ts | 21M | [repo](https://gitcode.com/m0_74196153/sebotnet33ts_256.a1h_in1k-npu) |

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
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch>=2.0.0 torchvision>=0.15.0 timm>=0.9.0 Pillow>=10.0.0 torch_npu
```

**输出**: NPU 可用状态，依赖已安装。

### Step 2: 模型选择与数据准备

**输入**: 模型名称。

**动作**:
4. 校验模型名是否在支持列表内。
5. 检查测试图片，缺失则下载。

**输出**: 模型名和测试图片已就绪。

### Step 3: CPU 基线推理

**输入**: 模型名、设备类型 `cpu`。

**动作**:
6. 执行 CPU 推理：

```bash
python3 examples/example.py --model seresnet50.a1_in1k --device cpu
```

7. 验证推理输出包含 top-5 预测结果。

**输出**: CPU 推理结果（top-5 类别和概率）。

### Step 4: NPU 推理

**输入**: 模型名、设备类型 `npu`。

**动作**:
8. 检查 NPU 可用性，不可用时自动回退 CPU。
9. 执行 NPU 推理：

```bash
python3 examples/example.py --model seresnet50.a1_in1k --device npu
```

10. 处理 OOM：释放缓存后重试。

**输出**: NPU 推理结果（top-5 类别和概率），标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU、NPU 推理结果。

**动作**:
11. 计算逐类精度误差（置信度误差百分比）。
12. 若 `max_error < 1%` 标记通过，否则标记失败。

```bash
bash scripts/inference.sh
```

**输出**: 精度对比报告（误差指标表格）。

### Step 6: 串行批量执行

**输入**: 剩余模型列表。

**动作**:
13. 对每个模型串行执行 Step 3-5。
14. 每完成一个模型释放资源：

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

**输出**: 每个模型的推理结果和精度报告。

### Step 7: 文档生成与仓库发布

**输入**: 精度通过的模型结果。

**动作**:
15. 生成每个模型的 README.md 和终端截图。
16. 发布模型仓库到 GitCode。

**输出**: 模型 README、截图、GitCode 仓库。

## 执行检查点与用户确认

每个关键步骤设有检查点 checkpoint，用户必须暂停确认后才能继续。以下 6 个检查点覆盖从环境准备到仓库发布的完整流程：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测后 | NPU 设备是否可用，CANN 版本是否正确 | 提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 模型确认检查点 | 模型选择后 | 选定的模型名称是否正确 | 返回重新选择 |
| 3 | CP-3: CPU 基线检查点 | CPU 推理后 | CPU 推理 top-5 是否合理 | 检查模型加载 |
| 4 | CP-4: NPU 推理检查点 | NPU 推理后 | NPU 推理是否成功 | 检查 NPU 显存 |
| 5 | CP-5: 精度确认检查点 | 精度对比后 | 误差是否 < 1% | 检查精度原因后重试 |
| 6 | CP-6: 发布前审批检查点 | 推送仓库前 | 仓库名和 README 是否正确 | 修改后重新申请用户确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 回退 CPU 推理，标记 fallback | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 OOM | 释放缓存后重试 | 调整配置或释放进程 |
| 模型加载异常 | timm.create_model 异常 | 打印错误堆栈，跳过 | 修正模型名 |
| 网络超时 | pip/curl 下载失败 | 重试最多 3 次 | 切换镜像源 |
| 精度超标 | 误差 >= 1% | 记录偏差明细，中止发布 | 检查推理脚本一致性 |
| API 失败 | GitCode API 返回非 200 | 打印错误状态码 | 检查 token 权限 |
| 用户确认超时 | 检查点 checkpoint 等待超时 | 暂停流程并通知用户，保留当前进度 | 用户恢复后从当前检查点继续 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `examples/example.py` | CPU/NPU 推理示例脚本，含 top-5 输出和 JSON 结果持久化 |
| `scripts/inference.sh` | 推理和精度对比 shell 脚本，输出 evals.json |
| `skill.json` | 技能元数据（模型列表、依赖版本、硬件要求） |
| `test-prompts.json` | 本技能测试提示词（含 NPU 回退场景） |
| `cpu_results.json` | CPU 推理结果 evals.json（运行后生成） |
| `npu_results.json` | NPU 推理结果 evals.json（运行后生成） |
| `compare_results.json` | CPU/NPU 精度对比结果（运行后生成） |
| `references/` | 精度验证参考指标和 benchmark（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 是 | seresnet50.a1_in1k | 模型名称（支持列表中的 9 个模型） |
| `device` | string | 否 | npu | 推理设备: cpu 或 npu |
| `image` | string | 否 | test_image.jpg | 输入图像路径 |

## 使用约束

1. 使用 ModelScope 或 hf-mirror.com 下载模型权重。
2. 精度验证通过前不提交模型仓库。
3. 串行执行避免 NPU 显存溢出。
