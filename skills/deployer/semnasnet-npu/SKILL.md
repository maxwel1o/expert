---
name: semnasnet-npu
description: "SEMNASNet (semnasnet_100/semnasnet_075) 图像分类模型在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：SEMNASNet NPU 部署、semnasnet 昇腾推理、SEMNASNet 精度对比、npu-smi。"
---

# SEMNASNet NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 SEMNASNet 系列图像分类模型，完成推理、精度验证，并发布模型仓库。

## 概述

本 Skill 用于在华为昇腾 NPU 上自动完成 SEMNASNet（Squeeze-and-Excitation MobileNets）图像分类系列模型的部署、推理、CPU/NPU 精度对比、README 生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910) |
| 模型数量 | 2 个（semnasnet_100 / semnasnet_075） |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 1.0+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| 执行方式 | 逐模型串行执行 |

## 支持的模型

| 模型名称 | 原始地址 | 适配仓库 |
|---------|---------|---------|
| `semnasnet_100.rmsp_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/semnasnet_100.rmsp_in1k) | [GitCode](https://gitcode.com/m0_74196153/semnasnet_100.rmsp_in1k-npu) |
| `semnasnet_075.rmsp_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/semnasnet_075.rmsp_in1k) | [GitCode](https://gitcode.com/m0_74196153/semnasnet_075.rmsp_in1k-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.11+ 环境，昇腾 NPU 驱动。

**动作**:
1. 检查 Python 版本：

```bash
python3 --version
```

2. 检测 NPU 状态并验证环境：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm torch_npu pillow numpy modelscope
```

**输出**: NPU 可用状态，依赖已安装。

### Step 2: 模型选择

**输入**: 模型名称（semnasnet_100.rmsp_in1k / semnasnet_075.rmsp_in1k）。

**动作**:
4. 校验模型名是否在支持列表内。
5. 从 ModelScope 下载模型权重：

```python
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download(f"timm/{model_name}")
```

**输出**: 模型权重已下载到本地。

### Step 3: CPU 基线推理

**输入**: 模型名、设备类型 `cpu`。

**动作**:
6. 执行 CPU 推理：

```bash
python3 examples/inference_example.py --model semnasnet_100.rmsp_in1k --device cpu
```

7. 验证推理输出包含 top-5 预测类别和概率。

**输出**: CPU 推理结果（top-5 类别和概率）。

### Step 4: NPU 推理

**输入**: 模型名、设备类型 `npu`。

**动作**:
8. 检查 NPU 可用性，不可用时自动回退 CPU。
9. 执行 NPU 推理：

```bash
python3 examples/inference_example.py --model semnasnet_100.rmsp_in1k --device npu
```

10. 处理 OOM：释放缓存后重试。

**输出**: NPU 推理结果（top-5 类别和概率），标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU、NPU 推理结果。

**动作**:
11. 计算精度指标：Logits 误差、概率误差、余弦相似度、Top-1 一致性。
12. 若各指标误差 < 1% 则标记通过。

```bash
python3 scripts/batch_run.py --model semnasnet_100.rmsp_in1k
```

**输出**: `compare_results.json`，包含 Logits 误差、概率误差、余弦相似度。

### Step 6: 串行批量执行

**输入**: 剩余模型列表。

**动作**:
13. 对每个模型串行执行 Step 3-5。
14. 每完成一个模型释放 NPU 显存：

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

```bash
bash scripts/run.sh
```

**输出**: 每个模型的推理结果和精度对比报告。

### Step 7: 生成文档与截图

**输入**: 精度对比结果。

**动作**:
15. 生成每个模型的 README.md。
16. 生成终端截图。

**输出**: README.md 和终端截图。

### Step 8: 模型仓库发布

**输入**: 精度验证通过的模型名、GitCode token。

**动作**:
17. 确认 `PRECISION_PASS=true`。
18. 通过 GitCode API 创建仓库并推送。

**输出**: 模型仓库推送至 GitCode。

## 执行检查点与用户确认

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 模型确认检查点 | 模型选择后 | 选定的模型名称是否正确 | 返回重新选择模型 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理 top-5 是否合理 | 检查模型加载和输入数据 |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功，耗时是否合理 | 检查 NPU 显存和驱动 |
| 5 | CP-5: 精度确认检查点 | 精度对比完成后 | 误差是否 < 1% | 检查精度不达标原因后重试 |
| 6 | CP-6: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、README 是否正确 | 修改配置后重新申请确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 回退 CPU 推理，标记 fallback | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 OOM | 释放缓存后重试 | 调整配置或释放进程 |
| 模型下载失败 | snapshot_download 异常 | 重试最多 3 次，切换镜像 | 检查网络连接和模型名 |
| 模型加载异常 | timm.create_model 异常 | 打印错误堆栈，跳过 | 修正模型名 |
| 精度超标 | 误差 >= 1% | 记录偏差明细，中止发布 | 检查推理脚本一致性 |
| API 失败 | GitCode API 返回非 200 | 打印错误状态码 | 检查 token 权限 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `examples/inference_example.py` | CPU/NPU 推理示例脚本，含 top-5 输出和 JSON 结果 |
| `scripts/batch_run.py` | 串行批量执行和精度对比，输出 evals.json |
| `scripts/run.sh` | 完整流水线 shell 脚本（推理→对比→截图→发布） |
| `skill.json` | 技能元数据（模型列表、参数定义、输入输出） |
| `test-prompts.json` | 本技能测试提示词（含 NPU 回退场景） |
| `cpu_results.json` | CPU 推理结果（运行后生成） |
| `npu_results.json` | NPU 推理结果（运行后生成） |
| `compare_results.json` | CPU/NPU 精度对比 evals.json（运行后生成） |
| `references/` | 精度验证参考指标（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 是 | — | semnasnet_100.rmsp_in1k 或 semnasnet_075.rmsp_in1k |
| `device` | string | 否 | npu | 推理设备: cpu 或 npu |
| `image_path` | string | 否 | — | 输入图像路径（默认合成测试图像） |

## 使用约束

1. 使用 ModelScope 下载模型权重（HF 官方可能无法直接访问）。
2. 精度验证通过前不提交模型仓库。
3. 串行执行避免 NPU 显存爆炸。
