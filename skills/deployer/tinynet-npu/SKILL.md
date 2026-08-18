---
name: tinynet-npu
description: "TinyNet (tinynet_e/d/c/b/a) 图像分类模型在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：TinyNet NPU 部署、tinynet 昇腾推理、TinyNet 精度对比、npu-smi。"
---

# TinyNet NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 TinyNet 系列图像分类模型，完成推理、精度验证，并发布模型仓库。

## 概述

本 Skill 用于在华为昇腾 NPU 上自动完成 TinyNet 系列（tinynet_e / tinynet_d / tinynet_c / tinynet_b / tinynet_a）图像分类模型的部署、推理、CPU/NPU 精度对比、终端截图生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910) |
| 模型数量 | 5 个（tinynet_e / tinynet_d / tinynet_c / tinynet_b / tinynet_a） |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 0.9.0+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1%，余弦相似度 > 0.999 |
| 执行方式 | 逐模型串行执行，每完成一个释放 NPU 显存 |

## 支持的模型

| 模型名称 | 原始地址 | 适配仓库 |
|---------|---------|---------|
| `tinynet_e.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tinynet_e.in1k) | [GitCode](https://gitcode.com/m0_74196153/tinynet_e-npu) |
| `tinynet_d.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tinynet_d.in1k) | [GitCode](https://gitcode.com/m0_74196153/tinynet_d-npu) |
| `tinynet_c.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tinynet_c.in1k) | [GitCode](https://gitcode.com/m0_74196153/tinynet_c-npu) |
| `tinynet_b.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tinynet_b.in1k) | [GitCode](https://gitcode.com/m0_74196153/tinynet_b-npu) |
| `tinynet_a.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tinynet_a.in1k) | [GitCode](https://gitcode.com/m0_74196153/tinynet_a-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.10+ 环境，昇腾 NPU 驱动。

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
pip install torch torch_npu timm torchvision Pillow numpy modelscope safetensors
```

**输出**: NPU 可用状态，依赖已安装。

### Step 2: 模型选择与权重下载

**输入**: 模型名称（tinynet_e / tinynet_d / tinynet_c / tinynet_b / tinynet_a）。

**动作**:
4. 校验模型名是否在支持列表内。
5. 从 ModelScope 下载模型权重：

```python
from modelscope import snapshot_download
model_path = snapshot_download(f"timm/{model_name}.in1k")
```

6. 确认权重文件（pytorch_model.bin 或 model.safetensors）已下载到本地。

**输出**: 模型权重已下载到本地。

### Step 3: CPU 基线推理

**输入**: 模型名、设备类型 `cpu`。

**动作**:
7. 创建模型架构（timm.create_model），加载本地权重，移至 CPU。
8. 执行 CPU 推理：

```bash
cd scripts/
python inference.py --model tinynet_e --device cpu
```

9. 验证推理输出包含 top-5 预测类别和概率。

**输出**: CPU 推理结果（top-5 类别和概率），保存为 `{model}_cpu_result.json`。

### Step 4: NPU 推理

**输入**: 模型名、设备类型 `npu`。

**动作**:
10. 检查 NPU 可用性，不可用时自动回退 CPU。
11. 创建模型架构，加载权重，移至 `npu:0`。
12. 执行 NPU 推理：

```bash
cd scripts/
python inference.py --model tinynet_e --device npu
```

13. 处理 OOM：释放缓存后重试。

**输出**: NPU 推理结果（top-5 类别和概率），保存为 `{model}_npu_0_result.json`，标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU、NPU 推理结果文件。

**动作**:
14. 计算精度指标：余弦相似度、最大绝对误差、平均绝对误差、RMSE、平均相对误差、Top-1 一致性、Top-5 重叠数。
15. 若各指标误差 < 1% 则标记通过。

```bash
cd scripts/
python compare_cpu_npu.py --model tinynet_e
```

**输出**: `{model}_compare_result.json`，包含完整精度对比指标。

### Step 6: 串行批量执行

**输入**: 剩余模型列表（tinynet_d / tinynet_c / tinynet_b / tinynet_a）。

**动作**:
16. 对每个模型串行执行 Step 3-5。
17. 每完成一个模型释放 NPU 显存：

```bash
python -c "import gc; gc.collect(); import torch; torch.npu.empty_cache()"
```

18. 全量自动执行：

```bash
bash examples/run_all.sh
```

**输出**: 每个模型的推理结果和精度对比报告。

### Step 7: 生成文档与截图

**输入**: 精度对比结果。

**动作**:
19. 生成每个模型的终端输出文本：

```bash
cd scripts/
python generate_screenshot.py
```

20. 验证生成的 `{model}_terminal_output.txt` 和 `{model}_terminal_output.html` 文件完整性。

**输出**: 终端输出文本和 HTML 截图文件。

### Step 8: 模型仓库发布

**输入**: 精度验证通过的模型名、GitCode token。

**动作**:
21. 确认所有模型 `PRECISION_PASS=true`。
22. 通过 GitCode API 创建仓库并推送：

```python
POST https://api.gitcode.com/api/v5/user/repos
{
    "name": "tinynet_e-npu",
    "repository_type": "model",
    "visibility": "public"
}
```

23. 推送完成后验证仓库可访问。

**输出**: 模型仓库推送至 GitCode，输出仓库 URL 列表。

## 执行检查点与用户确认

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 模型确认检查点 | 模型选择后 | 选定的模型名称是否正确（tinynet_e/d/c/b/a） | 返回重新选择模型 |
| 3 | CP-3: 权重下载检查点 | 权重下载完成后 | 模型权重文件是否成功下载 | 检查网络连接和 modelscope 状态 |
| 4 | CP-4: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理 top-5 是否合理 | 检查模型加载和输入数据 |
| 5 | CP-5: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功，耗时是否合理 | 检查 NPU 显存和驱动 |
| 6 | CP-6: 精度确认检查点 | 精度对比完成后 | 误差是否 < 1%，余弦相似度 > 0.999 | 检查精度不达标原因后重试 |
| 7 | CP-7: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、README 是否正确 | 修改配置后重新申请确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 回退 CPU 推理，标记 fallback | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 OOM | `torch.npu.empty_cache()` 后重试 | 调整配置或释放进程 |
| 模型下载失败 | snapshot_download 异常 | 重试最多 3 次，切换镜像源 | 检查网络连接和模型名称 |
| 模型加载异常 | timm.create_model 异常或 load_state_dict 失败 | 打印错误堆栈，跳过当前模型 | 检查模型名与权重文件对应关系 |
| 权重文件格式异常 | safetensors 或 bin 文件加载失败 | 自动尝试另一种格式 | 检查权重文件完整性 |
| 图片下载失败 | requests.get 返回非 200 | 使用随机图片替代 | 检查网络连接或手动提供图片 |
| 精度超标 | 误差 >= 1% 或余弦相似度 < 0.999 | 记录偏差明细，中止发布 | 检查推理脚本与模型权重一致性 |
| GPU/NPU 结果不一致 | Top-1 不一致 | 打印 Top-5 明细对比图 | 检查模型加载和设备驱动 |
| CPU 推理过慢 | 推理耗时异常高（> 100 ms） | 打印警告，继续执行 | 检查 CPU 负载和进程资源 |
| API 调用失败 | GitCode API 返回非 200 | 打印错误状态码和响应体 | 检查 token 权限和网络连接 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理脚本，含权重下载、数据预处理、top-5 输出和 JSON 结果保存 |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本，计算余弦相似度、误差指标和 Top-1/5 一致性 |
| `scripts/generate_screenshot.py` | 终端输出文本和 HTML 截图生成脚本 |
| `examples/run_all.sh` | 全量模型串行执行流水线脚本（推理 -> 对比 -> 截图） |
| `skill.json` | 技能元数据（模型列表、参数定义、输入输出） |
| `test-prompts.json` | 本技能测试提示词（含 NPU 回退场景） |
| `{model}_cpu_result.json` | CPU 推理结果（运行后生成） |
| `{model}_npu_0_result.json` | NPU 推理结果（运行后生成） |
| `{model}_compare_result.json` | CPU/NPU 精度对比 evals.json（运行后生成） |
| `{model}_terminal_output.txt` | 终端输出文本（运行后生成） |
| `{model}_terminal_output.html` | 终端输出 HTML 截图（运行后生成） |

## 精度测试指标

| 指标 | 说明 | 阈值 |
|:---|:---|:---:|
| 余弦相似度 (Cosine Similarity) | 衡量输出向量的方向一致性 | > 0.999 |
| 最大绝对误差 (Max Absolute Error) | 所有元素中的最大差异 | < 1.0 |
| 平均绝对误差 (Mean Absolute Error) | 所有元素的平均差异 | < 0.1 |
| 均方根误差 (RMSE) | 输出差异的均方根 | < 1.0 |
| 平均相对误差 (Relative Error) | 相对差异百分比 | < 1% |
| Top-1 一致性 | 最高概率类别是否一致 | 一致 |
| Top-5 重叠数 | 前 5 预测类别的重叠数 | >= 4/5 |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model` | string | 是 | — | 模型名称：tinynet_e / tinynet_d / tinynet_c / tinynet_b / tinynet_a |
| `device` | string | 否 | npu | 推理设备：cpu 或 npu |

## 使用约束

1. 使用 ModelScope 下载模型权重（HF 官方可能无法直接访问），需配置 modelscope 缓存路径。
2. 精度验证通过前不提交模型仓库。
3. 串行执行避免 NPU 显存爆炸，每完成一个模型释放缓存。
4. 模型执行顺序建议：tinynet_e -> tinynet_d -> tinynet_c -> tinynet_b -> tinynet_a（从轻到重）。
