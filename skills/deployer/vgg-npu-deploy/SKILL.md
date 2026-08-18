---
name: vgg-npu-deploy
description: "VGG 系列 7 个模型 (vgg19/vgg16/vgg13/vgg11 及 BN 变体) 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：VGG NPU 部署、vgg 昇腾推理、精度对比、npu-smi、timm。"
---

# VGG 系列图像分类模型 昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 VGG 系列 7 个图像分类模型，完成推理、精度验证，并发布模型仓库。

## 概述

本 Skill 提供 VGG（Visual Geometry Group）系列图像分类模型在华为昇腾 NPU 上的完整部署、推理验证和 CPU/NPU 精度对比的标准化可复现流程。涵盖 timm 框架下 vgg19/vgg16/vgg13/vgg11 及其 BN 变体在 NPU 上的环境准备、推理验证、精度对比、README 生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend 910B/910) |
| 模型数量 | 7 个（vgg19/vgg16/vgg13/vgg11 及 BN 变体） |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 0.9+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| CANN 版本 | 8.0+ |

### 支持的模型

| 模型名称 | 深度 | BN | 输入尺寸 | 参数量 | GitCode 仓库 |
|---------|------|----|---------|--------|-------------|
| vgg19.tv_in1k | 19 | 否 | 224x224 | ~144M | [repo](https://gitcode.com/m0_74196153/vgg19.tv_in1k-npu) |
| vgg16_bn.tv_in1k | 16 | 是 | 224x224 | ~138M | [repo](https://gitcode.com/m0_74196153/vgg16_bn.tv_in1k-npu) |
| vgg16.tv_in1k | 16 | 否 | 224x224 | ~138M | [repo](https://gitcode.com/m0_74196153/vgg16.tv_in1k-npu) |
| vgg13_bn.tv_in1k | 13 | 是 | 224x224 | ~133M | [repo](https://gitcode.com/m0_74196153/vgg13_bn.tv_in1k-npu) |
| vgg13.tv_in1k | 13 | 否 | 224x224 | ~133M | [repo](https://gitcode.com/m0_74196153/vgg13.tv_in1k-npu) |
| vgg11_bn.tv_in1k | 11 | 是 | 224x224 | ~133M | [repo](https://gitcode.com/m0_74196153/vgg11_bn.tv_in1k-npu) |
| vgg11.tv_in1k | 11 | 否 | 224x224 | ~133M | [repo](https://gitcode.com/m0_74196153/vgg11.tv_in1k-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.9+ 环境，昇腾 NPU 驱动，CANN 8.0+。

**动作**:
1. 选择空闲 NPU 卡：

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

2. 检查 Python 版本：

```bash
python3 --version
```

3. 安装全部依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torch_npu timm Pillow numpy safetensors modelscope
```

4. 验证 NPU 可用性：

```python
import torch
import torch_npu
print('torch:', torch.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('Device name:', torch.npu.get_device_name(0))
```

**输出**: NPU 可用状态，依赖已安装。通过标准：`NPU available: True` 且无报错。

### Step 2: 模型选择与权重下载

**输入**: 目标模型名称（从 7 个支持模型中选择）。

**动作**:
5. 从 ModelScope 下载模型权重：

```python
from modelscope import snapshot_download
model_dir = snapshot_download('timm/vgg19.tv_in1k')
```

6. 确认权重文件已下载到本地缓存，检查 safetensors 或 bin 文件是否存在：

```bash
ls /opt/atomgit/.cache/modelscope/hub/models/timm/vgg19___tv_in1k/
```

**输出**: 模型权重文件就绪（`model.safetensors` 或 `pytorch_model.bin`）。

### Step 3: CPU 基线推理

**输入**: 模型名 `vgg19.tv_in1k`、设备类型 `cpu`。

**动作**:
7. 执行 CPU 推理（使用 scripts/inference.py）：

```bash
cd /path/to/working/dir
python3 scripts/inference.py --model vgg19.tv_in1k --device cpu --num_runs 10
```

8. 验证输出文件 `results_cpu.json` 包含 top-5 预测类别和置信度。

**输出**: `results_cpu.json`，包含 top-5 分类结果和平均推理耗时。

### Step 4: NPU 推理

**输入**: 模型名 `vgg19.tv_in1k`、设备类型 `npu`。

**动作**:
9. 检查 NPU 可用性，不可用时自动回退 CPU 并标记 fallback。
10. 执行 NPU 推理：

```bash
python3 scripts/inference.py --model vgg19.tv_in1k --device npu --num_runs 10
```

11. 处理 OOM：捕获内存不足异常，释放缓存后重试。

**输出**: `results_npu.json`，标注 fallback 状态（如有）。

### Step 5: CPU/NPU 精度对比

**输入**: CPU 和 NPU 推理结果 JSON 文件。

**动作**:
12. 执行精度对比脚本：

```bash
python3 scripts/compare_cpu_npu.py --model vgg19.tv_in1k --num_runs 10
```

13. 检查精度指标：
    - Logits 最大/平均绝对误差
    - Logits 余弦相似度
    - Softmax 概率最大/平均差异
    - Top-1 一致性
    - Top-5 重合率

14. 判定标准：若 `max_prob_error < 1%` 则标记通过，否则标记失败。

**输出**: `compare_results.json`，包含完整精度对比指标和结论。

### Step 6: 串行批量执行全部模型

**输入**: 7 个 VGG 模型列表。

**动作**:
15. 对每个模型串行执行 Step 3-5，防止 NPU OOM：

```bash
for model_name in vgg19.tv_in1k vgg16_bn.tv_in1k vgg16.tv_in1k vgg13_bn.tv_in1k vgg13.tv_in1k vgg11_bn.tv_in1k vgg11.tv_in1k; do
    mkdir -p output/${model_name}
    cd output/${model_name}
    python3 scripts/inference.py ${model_name} --device cpu --num_runs 10
    python3 scripts/inference.py ${model_name} --device npu --num_runs 10
    python3 scripts/compare_cpu_npu.py ${model_name} --num_runs 10

    # 释放资源
    python3 -c "
import gc, torch
gc.collect()
if hasattr(torch, 'npu'):
    torch.npu.empty_cache()
"
    cd /path/to/working/dir
done
```

**输出**: 每个模型的推理结果和精度对比报告。

### Step 7: 生成截图与文档

**输入**: 精度验证通过的模型结果。

**动作**:
16. 生成终端截图：

```bash
python3 scripts/generate_screenshot.py <日志文件路径> <输出HTML路径>
```

17. 记录实测精度汇总表：

| 模型 | CPU (ms) | NPU (ms) | 加速比 | 最高 Logits 误差 | Top-1 | Top-5 | 精度结论 |
|------|---------|---------|-------|----------------|-------|-------|---------|
| vgg19.tv_in1k | 1046.38 | 5.53 | 189x | 3.70e-03 | ✓ | 5/5 | 通过(<1%) |
| vgg16_bn.tv_in1k | 850.92 | 5.53 | 154x | 6.43e-03 | ✓ | 5/5 | 通过(<1%) |
| vgg16.tv_in1k | 867.32 | 5.17 | 168x | 3.95e-03 | ✓ | 5/5 | 通过(<1%) |
| vgg13_bn.tv_in1k | 712.24 | 5.17 | 138x | 9.21e-03 | ✓ | 5/5 | 通过(<1%) |
| vgg13.tv_in1k | 731.80 | 5.02 | 146x | 7.31e-03 | ✓ | 5/5 | 通过(<1%) |
| vgg11_bn.tv_in1k | 562.68 | 5.02 | 112x | 6.11e-03 | ✓ | 5/5 | 通过(<1%) |
| vgg11.tv_in1k | 586.19 | 4.97 | 118x | 7.27e-03 | ✓ | 5/5 | 通过(<1%) |

**输出**: HTML 终端截图、精度汇总报告。

### Step 8: 模型标签与仓库发布

**输入**: 全部精度通过的结果。

**动作**:
18. 为模型仓库添加标签：
    - `#+NPU`、`#+CV`、`#+昇腾`、`#+图像分类`、`#+timm`、`#+VGG`
19. 发布模型仓库到 GitCode（包含 README、截图、精度报告）。

**输出**: GitCode 上的 VGG 系列 NPU 模型仓库（7 个）。

## 执行检查点与用户确认

每个关键步骤设有检查点 checkpoint，用户必须暂停确认后才能继续。以下 7 个检查点覆盖从环境准备到仓库发布的完整流程：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，CANN 版本是否正确 | 提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 模型确认检查点 | 模型权重下载后 | 模型名称与权重文件是否正确 | 返回重新选择模型 |
| 3 | CP-3: CPU 基线检查点 | CPU 推理完成后 | CPU 推理 top-5 结果是否合理 | 检查测试图片和模型加载 |
| 4 | CP-4: NPU 推理检查点 | NPU 推理完成后 | NPU 推理是否成功，耗时是否合理 | 检查 NPU 显存和驱动状态 |
| 5 | CP-5: 精度确认检查点 | 精度对比完成后 | 误差是否 < 1%，日志是否完整 | 检查精度原因后重试 |
| 6 | CP-6: 批量完成检查点 | 全部模型处理完毕后 | 所有模型精度是否达标 | 返回未通过模型重新验证 |
| 7 | CP-7: 发布前审批检查点 | 推送模型仓库前 | 仓库名、README、截图内容是否正确 | 修改后重新申请用户确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 回退 CPU 推理，标记 fallback | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 OOM 异常 | 释放缓存后重试，仍失败则跳过 | 调整 batch_size=1，串行执行 |
| 模型加载失败 | timm.create_model 异常 | 打印错误堆栈，跳过该模型 | 检查模型名和缓存路径 |
| 权重文件缺失 | safetensors 或 bin 文件不存在 | 重新从 ModelScope 下载 | 检查网络和缓存目录 |
| 下载超时 | pip/curl 无响应 | 重试最多 3 次 | 切换镜像源或离线安装 |
| 精度超标 | 误差 >= 1% | 记录偏差明细，中止发布 | 检查推理脚本和数据一致性 |
| 网络连接失败 | HuggingFace 不可达 | 回退到 ModelScope 下载 | 使用国内镜像加速 |
| git push 超时 | gitcode.com 不可达 | 使用 API 上传文件替代 | 检查网络和 token 权限 |
| Python 版本不兼容 | Python 版本 < 3.9 | 提示升级 Python | 安装 Python 3.9+ |
| 用户确认超时 | 检查点等待用户确认超时 | 暂停流程并通知用户，保留当前进度 | 用户恢复后从当前检查点继续 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理执行入口，支持单模型运行 |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本，输出误差指标 |
| `examples/vgg_inference_example.py` | VGG 单模型 NPU 推理参考示例 |
| `skill.json` | 技能元数据（模型列表、依赖版本、硬件要求） |
| `test-prompts.json` | 本技能测试提示词 |
| `results_cpu.json` | CPU 推理结果 evals.json（运行后生成） |
| `results_npu.json` | NPU 推理结果 evals.json（运行后生成） |
| `compare_results.json` | CPU/NPU 精度对比结果（运行后生成） |
| `cpu_output.pt` / `npu_output.pt` | 推理 logits 快照（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 是 | vgg19.tv_in1k | 模型名称（支持列表中的 7 个模型） |
| `device` | string | 否 | npu | 推理设备: cpu 或 npu |
| `image` | string | 否 | test_image.jpg | 输入图像路径 |
| `num_runs` | integer | 否 | 10 | 推理重复次数，用于计算平均延迟 |

## 使用约束

1. 使用 ModelScope 下载模型权重（优先国内镜像，避免 HuggingFace 连接超时）。
2. 精度验证通过前不提交模型仓库。
3. 串行执行避免 NPU 显存溢出，每完成一个模型必须释放资源。
4. CPU/NPU 对比需使用相同输入图片和预处理流程。
5. 模型权重缓存路径为 `/opt/atomgit/.cache/modelscope/hub/models/timm/`。
