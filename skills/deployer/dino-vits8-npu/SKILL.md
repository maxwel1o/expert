---
name: dino-vits8-npu
description: "DINO ViT-S/8 自监督视觉 Transformer 模型在昇腾 NPU 上的完整推理部署与 CPU/NPU 精度验证 Skill。涵盖环境初始化、权重下载、transfer_to_npu 自动迁移、单图推理验证、随机/真实图像精度对比验收的完整可复现流程。可在任意 Ascend910 系列服务器上一键复现。适用于：DINO 昇腾部署、ViT-S/8 NPU 推理、DINOv1 适配验证、自监督视觉模型 NPU 部署、精度对比。触发词：DINO NPU 部署、dino-vits8 昇腾推理、ViT-S/8 精度对比、dino_deitsmall8 NPU、DINO 自监督模型 昇腾。"
---

# DINO ViT-S/8 昇腾 NPU 部署与推理验证 Skill

> 在昇腾 NPU 和 CPU 上自动部署 DINO ViT-S/8（dino_vits8）自监督视觉 Transformer 模型，串行完成环境检查、权重下载、transfer_to_npu 自动迁移、单图推理验证和 CPU/NPU 精度对比。执行流程分 8 步：先环境初始化和代码获取，再完成 NPU 验证、单图推理、精度对比，最后进行验收确认。

## 概述

本 Skill 用于自动完成 **DINO ViT-S/8（dino_vits8）自监督视觉 Transformer 模型** 在昇腾 NPU 上的部署、推理验证、CPU/NPU 精度对比验收的完整可复现流程。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910) |
| 框架版本 | PyTorch 2.0+, torch_npu, torchvision |
| 模型架构 | ViT-S/8 (embed_dim=384, depth=12, num_heads=6) |
| 输出特征 | 384 维特征向量（CLS token） |
| 精度目标 | CPU 与 NPU 推理结果 L2 相对误差 < 1% |
| 执行方式 | 串行逐流程执行，完成后释放资源 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.8-3.11 环境，昇腾 NPU 驱动，CANN >= 8.0（推荐 8.2+）。

**动作**:
1. 加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 检查 NPU 可用性：

```bash
npu-smi info
```

3. 选择空闲 NPU 卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

4. 验证 PyTorch 和 torch_npu：

```bash
python3 -c "import torch; import torch_npu; print(torch.randn(3,4).npu() + torch.randn(3,4).npu())"
```

5. 确认 PyTorch + NPU：

```python
import torch
print(f"NPU available: {torch.npu.is_available()}")
if torch.npu.is_available():
    print(f"NPU device count: {torch.npu.device_count()}")
```

**输出**: NPU 可用状态确认，CANN 环境已加载，空闲卡已指定。

### Step 2: 获取代码与权重

**输入**: 可用的网络连接，约 200MB 磁盘空间。

**动作**:
6. 克隆 DINO 官方仓库：

```bash
git clone https://github.com/facebookresearch/dino.git
cd dino
```

7. 创建权重目录并下载预训练权重：

```bash
mkdir -p weights
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth \
  -O weights/dino_deitsmall8_pretrain.pth
```

8. 验证权重完整性（文件约 83MB）：

```bash
ls -lh weights/dino_deitsmall8_pretrain.pth
```

9. 复制本 Skill 目录下的脚本：

```bash
cp <skill-path>/scripts/*.py ./
```

**输出**: DINO 仓库已克隆，预训练权重已下载验证，Skill 脚本已就位。

### Step 3: NPU 自动迁移验证

**输入**: DINO 代码仓库和预训练权重就绪。

**动作**:
10. 通过 `transfer_to_npu` 自动迁移加载模型到 NPU：

```bash
python3 -c "
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

import vision_transformer as vits
model = vits.vit_small(patch_size=8, num_classes=0)
print('Model loaded on NPU:', next(model.parameters()).device)
"
```

11. 确认输出设备为 `npu:0`。
12. 处理 `torch.jit.script` 警告：忽略即可，推理不依赖 JIT。

**输出**: 模型在 NPU 上加载成功验证。

### Step 4: 单图推理验证

**输入**: 模型权重，一张 224x224 的测试图像。

**动作**:
13. 准备测试图像：

```bash
# 可从 https://picsum.photos/224/224 下载一张
curl -s -o test_image.jpg https://picsum.photos/224/224
```

14. 执行 NPU 单图推理：

```bash
python3 inference.py \
  --pretrained_weights weights/dino_deitsmall8_pretrain.pth \
  --image test_image.jpg \
  --device npu
```

15. 验证输出 shape 和 device：

```text
Input shape:  torch.Size([1, 3, 224, 224])
Output shape: torch.Size([1, 384])
Device:       npu:0
First 10 features: [ 0.30 4.80 0.60 -0.52 ...]
```

**输出**: 单张测试图片的 384 维特征向量（NPU 推理结果）。

### Step 5: 随机输入 CPU/NPU 精度对比

**输入**: 预训练权重，随机种子。

**动作**:
16. 以 CPU 为基准，用随机输入对比 NPU 推理精度：

```bash
python3 eval_accuracy.py \
  --pretrained_weights weights/dino_deitsmall8_pretrain.pth \
  --seed 42 \
  --warmup 5 \
  --runs 20
```

17. 预期输出应符合精度标准（L2 相对误差 < 1%）：

```text
============================================================
Accuracy Comparison
============================================================
Max absolute difference:     5.41e-02
Mean absolute difference:    1.08e-02
L2 relative error:           3.69e-03 (0.37%)

Precision check (L2 relative < 0.01): PASS
```

**输出**: 随机输入上的 CPU/NPU 精度对比结果报告。

### Step 6: 真实图像 CPU/NPU 精度对比

**输入**: 预训练权重，真实测试图像。

**动作**:
18. 用真实图像进行 CPU/NPU 精度对比：

```bash
python3 eval_accuracy.py \
  --pretrained_weights weights/dino_deitsmall8_pretrain.pth \
  --image test_image.jpg \
  --warmup 5 \
  --runs 20
```

19. 检查 L2 相对误差是否 < 1%。

**输出**: 真实图像上的 CPU/NPU 精度对比结果报告。

### Step 7: 多模型串行执行推理

**输入**: 需要推理的模型配置列表。

**动作**:
20. 可在同一环境中依次执行多个 DINO 系列模型的推理和精度验证：

```bash
for model_config in "vit_small-8" "vit_small-16" "vit_base-8" "vit_base-16"; do
    echo "=== Processing $model_config ==="
    # 下载对应权重
    # 执行推理和精度对比
    # 释放资源
    python3 -c "import gc, torch; gc.collect(); torch.npu.empty_cache()"
    sleep 2
done
```

**输出**: 各模型推理精度对比结果。

### Step 8: 验收确认与结果记录

**输入**: 所有步骤完成后的输出日志和结果文件。

**动作**:
21. 按以下清单逐项确认：

```
□ Step 1: CANN 环境加载成功，npu-smi info 可正常查看卡状态
□ Step 2: DINO 仓库克隆成功，权重下载完整（83MB）
□ Step 3: NPU 验证通过，模型可正常加载到 NPU
□ Step 4: 单图推理成功，输出 shape 为 [1, 384]
□ Step 5: 随机输入精度对比 PASS（L2 相对误差 < 1%）
□ Step 6: 真实图像精度对比 PASS（可选，L2 相对误差 < 1%）
```

22. 记录推理性能数据：

| 指标 | CPU | NPU |
|:---|:---:|:---:|
| 推理时间（平均） | XX ms | XX ms |
| 加速比 | 1x | CPU/NPU 倍 |
| 精度检查 | - | PASS/FAIL |

**输出**: 完整验收日志，性能对比数据，精度确认记录。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | Step 1 完成后 | NPU 设备是否可用，CANN 版本是否正确 | 暂停，检查 NPU 驱动或 CANN 安装 |
| 2 | CP-2: 权重准备检查点 | Step 2 完成后 | 权重文件大小 83MB，下载是否完整 | 重新下载权重 |
| 3 | CP-3: NPU 迁移检查点 | Step 3 完成后 | 模型是否成功加载到 npu:0 | 检查 transfer_to_npu 兼容性 |
| 4 | CP-4: 单图推理检查点 | Step 4 完成后 | 输出 shape 是否为 [1, 384]，device 为 npu:0 | 检查输入图像和模型加载路径 |
| 5 | CP-5: 随机输入精度检查点 | Step 5 完成后 | L2 相对误差是否 < 1% | 检查预处理流程一致性后重试 |
| 6 | CP-6: 真实图像精度检查点 | Step 6 完成后 | 真实图像精度误差是否满足标准 | 检查图像预处理差异 |
| 7 | CP-7: 串行执行检查点 | Step 7 每个模型完成后 | 资源释放是否成功 | 手动释放 NPU 缓存后继续 |
| 8 | CP-8: 验收确认检查点 | Step 8 完成后 | 所有步骤是否通过，日志是否完整 | 返回未通过步骤重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 异常 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `npu-smi info` 无输出或错误 | 降级为 CPU 推理，结果标记 fallback | CP-1 | 安装 torch_npu 或检查 CANN 驱动 |
| NPU 显存 OOM | 推理时报 `RuntimeError: CUDA out of memory` | 释放缓存，减小 batch_size，切到 CPU | CP-4 | 释放其他 NPU 进程或减小输入分辨率 |
| 权重下载失败 | wget 链接超时或校验和不匹配 | 检查网络连接，重新下载，最多重试 3 次 | CP-2 | 使用代理或手动下载后本地加载 |
| 模型加载失败 | `state_dict` key 不匹配 | 打印 mismatch 日志，检查权重版本 | CP-3 | 重新下载正确版本权重 |
| `torch.jit.script` 警告 | `transfer_to_npu` 与 JIT 不兼容 | 忽略警告，推理不依赖 JIT | CP-3 | 无需处理 |
| 精度误差超标 | L2 相对误差 >= 1% | 输出偏差明细，标记 PRECISION_FAIL | CP-5 | 检查 CPU/NPU 预处理一致性 |
| 图像加载失败 | PIL 无法打开图片 | 提示重新下载或生成随机图像 | CP-4 | 重新下载测试图片或使用内置随机输入 |
| 脚本复制失败 | cp 命令路径错误 | 提示 `<skill-path>` 占位符替换 | CP-2 | 手动复制 scripts/ 到工作目录 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | NPU/CPU 单图推理入口脚本，支持 --arch、--patch_size、--device 参数 |
| `scripts/eval_accuracy.py` | CPU/NPU 精度对比验证脚本，支持随机输入和真实图像两种模式 |
| `scripts/vision_transformer.py` | DINO ViT 模型定义（vit_tiny/vit_small/vit_base），从 DINO 官方仓库复制 |
| `scripts/utils.py` | 工具函数：权重加载、精度计算、分布式训练辅助等 |
| `weights/dino_deitsmall8_pretrain.pth` | 预训练权重（运行后生成，约 83MB） |

## 设备与精度指标

### 推理性能对比

| 设备 | 预期耗时（单次推理） | 说明 |
|:---|:---:|:---|
| CPU | ~50-200 ms | 取决于 CPU 型号和负载 |
| NPU (Ascend910) | ~3-15 ms | 基于 transfer_to_npu 自动迁移 |

### 精度验收标准

| 指标 | 标准 | 说明 |
|:---|:---:|:---|
| Max absolute difference | < 0.1 | CPU/NPU 输出最大绝对误差 |
| Mean absolute difference | < 0.05 | CPU/NPU 输出平均绝对误差 |
| L2 relative error | < 0.01 (1%) | CPU/NPU 输出 L2 相对误差 |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--pretrained_weights` | string | 是 | - | 预训练权重路径 |
| `--image` | string | 否 | - | 测试图像路径（不指定则使用随机输入） |
| `--arch` | string | 否 | vit_small | 模型架构名称 |
| `--patch_size` | int | 否 | 8 | ViT patch 大小 |
| `--device` | string | 否 | npu | 推理设备（cpu/cuda/npu） |
| `--seed` | int | 否 | 42 | 随机种子（仅随机输入模式） |
| `--warmup` | int | 否 | 3 | 预热推理次数 |
| `--runs` | int | 否 | 10 | 正式推理次数 |
| `--img_size` | int | 否 | 224 | 输入图像尺寸 |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `inference.py` 输出 | stdout | 输入/输出 shape、推理设备、前 10 个特征值 |
| `eval_accuracy.py` 输出 | stdout | CPU/NPU 推理耗时、精度对比指标、PASS/FAIL 判定 |

## 使用约束

1. 使用 Facebook Research 官方 DINO 权重（`dino_deitsmall8_pretrain.pth`，约 83MB）。
2. CANN 环境必须通过 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载。
3. `transfer_to_npu` 自动迁移方式会触发 `torch.jit.script` 警告，不影响推理结果。
4. 精度验证必须使用 CPU 作为基准（NPU 精度相对于 CPU 的偏差）。
5. 权重加载时自动去除 `module.` 和 `backbone.` 前缀。
6. 测试前确认 Ascend910 驱动和 CANN 环境已正确安装。
