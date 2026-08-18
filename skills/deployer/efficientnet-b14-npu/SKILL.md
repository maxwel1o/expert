---
name: efficientnet-b14-npu
description: "EfficientNet 系列 68 个模型 (EfficientNetV2/Lite/B0-B8/CondConv/NoisyStudent/Test) 在昇腾 NPU 上的批量部署、推理、CPU/NPU 精度验证与 GitCode 模型仓库发布。适用于：昇腾部署、NPU 批量推理、精度验证、模型仓库发布。触发词：EfficientNet NPU 部署、EfficientNet 昇腾推理、EfficientNet 精度对比、npu-smi、batch 部署。"
---

# EfficientNet Batch 14 昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上批量部署 EfficientNet 系列 68 个图像分类模型，完成批量推理、精度验证，并逐个发布模型仓库。

## 概述

本 Skill 用于在华为昇腾 Ascend910 NPU 上完成 **EfficientNet** 系列 68 个图像分类模型的部署、推理、CPU/NPU 精度对比、README 文档生成、终端截图生成和 GitCode 模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910) |
| 模型数量 | 68 个（EfficientNetV2/Lite/B0-B8/EdgeTPU/CondConv/NoisyStudent/Test） |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 1.0+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| 执行方式 | 串行逐模型执行，支持起始/结束索引分批 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |

### 模型子系列

| 子系列 | 数量 | 说明 |
|--------|:---:|------|
| EfficientNetV2 (XL/S/M/L/B3/B2/B1/B0) | 17 | 改进的训练效率架构 |
| EfficientNet Lite (Lite4/3/2/1/0) | 5 | 轻量级移动端变体 |
| EfficientNet B0-B8 | 9 | 原始 EfficientNet 各规模版本 |
| EfficientNet-EdgeTPU/CondConv | 3 | 条件卷积变体 |
| EfficientNet-NoisyStudent | 2 | 半监督训练变体 |
| Test EfficientNet | 32 | 实验性架构变体（LN、GN、Evos） |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.11+ 环境，昇腾 NPU 驱动。

**动作**:
1. 检查 Python 版本，确认 >= 3.11：

```bash
python3 --version
```

2. 运行 NPU 检测：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm torch_npu pillow numpy modelscope
```

**输出**: NPU 可用状态，依赖已安装。

### Step 2: 模型列表与参数配置

**输入**: 目标模型名称、起始索引、结束索引。

**动作**:
4. 从 `skill.json` 或 README 加载模型列表（共 68 个）。
5. 根据 `start_index` 和 `end_index` 切片子列表。
6. 若指定单个 `model_name`，只处理该模型。

**输出**: 本次要处理的模型子列表。

### Step 3: 下载测试图片

**输入**: 测试图片 URL。

**动作**:
7. 检查测试图片是否存在：

```bash
if [ ! -f test_image.jpg ]; then
    curl -sL -o test_image.jpg https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
fi
```

**输出**: `test_image.jpg`。

### Step 4: CPU 基线推理

**输入**: 模型名、测试图片路径、设备类型 `cpu`。

**动作**:
8. 执行 CPU 推理：

```bash
python3 scripts/inference.py tf_efficientnetv2_xl.in21k_ft_in1k --device cpu --output cpu_results.json
```

9. 验证输出 JSON 包含 top-5 类别和置信度。

**输出**: `cpu_results.json`（格式：`{"model": "...", "device": "cpu", "top5": [...], "latency_ms": ...}`）。

### Step 5: NPU 推理

**输入**: 模型名、测试图片路径、设备类型 `npu`。

**动作**:
10. 检查 NPU 可用性，不可用时自动回退 CPU。
11. 执行 NPU 推理：

```bash
python3 scripts/inference.py tf_efficientnetv2_xl.in21k_ft_in1k --device npu --output npu_results.json
```

12. 处理 OOM：释放缓存后重试，仍失败则跳过。

**输出**: `npu_results.json`，标注 fallback 状态。

### Step 6: CPU/NPU 精度对比

**输入**: CPU、NPU 推理结果 JSON。

**动作**:
13. 计算逐类精度误差：

```
error_pct = |confidence_npu - confidence_cpu| / confidence_cpu * 100
```

14. 若 `max_error < 1%` 标记通过，否则标记失败并记录明细。
15. 执行精度对比脚本：

```bash
python3 scripts/compare_cpu_npu.py --model tf_efficientnetv2_xl.in21k_ft_in1k --output compare_results.json
```

**输出**: `compare_results.json`，包含最大/平均误差和通过状态。

### Step 7: 串行批量执行

**输入**: 模型子列表。

**动作**:
16. 对每个模型串行执行 Step 4-6。
17. 每完成一个模型释放资源：

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

18. 调用 batch runner：

```bash
python3 scripts/batch_runner.py --start 0 --end 68
```

**输出**: 每个模型的推理和精度结果。

### Step 8: 文档生成与仓库发布

**输入**: 精度验证通过的模型结果。

**动作**:
19. 生成每个模型的 README.md。
20. 生成终端截图：

```bash
python3 scripts/terminal_screenshot.py --input terminal_output.txt --output terminal_screenshot.png
```

21. 调用推送脚本发布到 GitCode：

```bash
python3 scripts/push_repos.py --token ${ATOMGIT_USER_TOKEN} --models-passed results.json
```

**输出**: 各模型 README 和截图，GitCode 仓库成功推送。

## 执行检查点与用户确认

以下需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 模型确认检查点 | 模型列表配置后 | 模型列表和索引范围是否正确 | 返回重新配置 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理输出和 top-5 类别是否合理 | 检查测试图片和模型加载 |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功，耗时是否合理 | 检查 NPU 显存和驱动状态 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | 精度误差是否 < 1%，日志是否完整 | 检查精度不达标原因后重试 |
| 6 | CP-6: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、README 内容是否正确 | 修改配置后重新申请确认 |
| 7 | CP-7: 全量完成检查点 | 全部模型处理完毕后 | 所有模型精度是否达标，报告是否完整 | 返回未通过模型重新验证 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退 CPU 推理，标记 fallback | CP-1 | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 OOM | 释放缓存后重试，仍失败则跳过该模型 | CP-4 | 减小 batch 或释放其他进程 |
| 模型加载异常 | timm.create_model 抛出异常 | 打印错误堆栈，跳过该模型 | CP-3 | 修正模型名或检查网络 |
| 下载超时 | pip/curl 无响应 | 重试最多 3 次，仍失败则跳过 | CP-1 | 切换镜像源或离线安装 |
| 精度超标 | 误差 >= 1% | 记录偏差明细，中止该模型发布 | CP-5 | 检查推理脚本和数据一致性 |
| 磁盘空间不足 | 权重下载失败 | 提示清理磁盘后重试 | CP-1 | 释放磁盘后重试 |
| API 调用失败 | GitCode API 返回非 200 | 打印错误状态码和响应体 | CP-6 | 检查 token 权限 |
| 端口冲突 | 推理端口被占用 | 提示更换端口或结束占用进程 | CP-1 | `lsof -i :port` 处理 |
| 截图生成失败 | 截图模块异常 | 跳过截图记录 warning | CP-7 | 安装截图依赖后重试 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理执行入口 |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本 |
| `scripts/batch_runner.py` | 串行批量执行全部模型的流水线 |
| `scripts/push_repos.py` | GitCode 模型仓库推送脚本 |
| `scripts/terminal_screenshot.py` | 终端截图生成脚本 |
| `examples/example.md` | 使用示例和参数说明 |
| `skill.json` | 技能元数据（模型列表、参数定义） |
| `test-prompts.json` | 本技能测试提示词 |
| `cpu_results.json` | CPU 推理结果（运行后生成） |
| `npu_results.json` | NPU 推理结果（运行后生成） |
| `compare_results.json` | CPU/NPU 精度对比结果（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 否 | all | 单个模型名，不指定则执行全部 68 个 |
| `start_index` | integer | 否 | 0 | 起始索引（从 0 开始），用于分批 |
| `end_index` | integer | 否 | 68 | 结束索引（不包含），用于分批 |
| `skip_inference` | boolean | 否 | false | 跳过推理，只生成文档和提交 |
| `skip_push` | boolean | 否 | false | 跳过 GitCode 推送 |

## 使用约束

1. 使用 ModelScope 或 hf-mirror.com 下载模型权重（HF 官方可能无法直接访问）。
2. 精度验证通过前不提交模型仓库（必须有 `PRECISION_PASS=true` 标记）。
3. 每个模型独立提交仓库，不混合多个模型到同一仓库。
4. 模型仓库使用 `main` 分支。
5. 串行执行避免 NPU 显存爆炸，每完成一个模型必须释放资源。
