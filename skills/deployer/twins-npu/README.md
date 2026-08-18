# Twins NPU 部署 Skill

## 概述

本 Skill 用于在华为昇腾 NPU（Ascend 910）上自动完成 Twins 系列视觉 Transformer 模型的推理部署、CPU/NPU 精度对比、README 生成和模型仓库发布。

支持以下 6 个模型：

| 模型名称 | 模型仓库地址 |
| --- | --- |
| twins_svt_small.in1k | [twins_svt_small.in1k-npu](https://gitcode.com/m0_74196153/twins_svt_small.in1k-npu) |
| twins_svt_large.in1k | [twins_svt_large.in1k-npu](https://gitcode.com/m0_74196153/twins_svt_large.in1k-npu) |
| twins_svt_base.in1k | [twins_svt_base.in1k-npu](https://gitcode.com/m0_74196153/twins_svt_base.in1k-npu) |
| twins_pcpvt_small.in1k | [twins_pcpvt_small.in1k-npu](https://gitcode.com/m0_74196153/twins_pcpvt_small.in1k-npu) |
| twins_pcpvt_large.in1k | [twins_pcpvt_large.in1k-npu](https://gitcode.com/m0_74196153/twins_pcpvt_large.in1k-npu) |
| twins_pcpvt_base.in1k | [twins_pcpvt_base.in1k-npu](https://gitcode.com/m0_74196153/twins_pcpvt_base.in1k-npu) |

## 任务类型

图像分类（ImageNet-1K）

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- torch_npu >= 2.0.0
- Ascend NPU（如 Ascend 910）
- CANN >= 8.0

## 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model_name` | string | 是 | — | 模型名称（如 `twins_svt_small.in1k`） |
| `device` | string | 否 | `npu` | 推理设备（`cpu` 或 `npu`） |
| `num_runs` | integer | 否 | 5 | 推理轮次 |

## 使用方法

### 1. 安装依赖

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch>=2.0.0 \
  torch_npu>=2.0.0 \
  timm>=0.9.0 \
  torchvision>=0.15.0 \
  Pillow>=10.0.0 \
  numpy>=1.22.0 \
  modelscope>=1.0.0 \
  safetensors>=0.4.0
```

### 2. 下载模型权重

脚本自动从 ModelScope 下载权重，无需手动操作。

### 3. 执行 NPU 推理

```bash
cd scripts/
python3 inference.py --model-name twins_svt_small.in1k --device npu --num-runs 5
```

### 4. 执行 CPU/NPU 精度对比

```bash
cd scripts/
python3 compare_cpu_npu.py --model-name twins_svt_small.in1k
```

### 5. 串行执行多个模型

```bash
cd scripts/
python3 batch_run.py
```

此脚本会串行执行全部 6 个模型的 CPU/NPU 对比，避免显存爆炸。

## 输出结果

- **NPU 推理结果**：Top-5 分类标签及概率
- **精度对比结果**：Max Absolute Error、Max Probability Difference、Cosine Similarity、Top-1/Top-5 一致性
- **性能数据**：CPU/NPU 推理耗时及加速比
- **终端截图**：模拟终端输出截图文件

## 精度要求

NPU 与 CPU 推理结果 Max Probability Difference < 1%。

## 已知结果

| 模型 | MaxProbDiff | CosineSim | Speedup |
| --- | ---:| ---:| ---:|
| twins_svt_small.in1k | 0.000200 | 0.99999547 | 0.63x |
| twins_svt_large.in1k | 0.000280 | 0.99999636 | 2.66x |
| twins_svt_base.in1k | 0.000356 | 0.99999672 | 1.60x |
| twins_pcpvt_small.in1k | 0.000327 | 0.99999624 | 0.86x |
| twins_pcpvt_large.in1k | 0.000461 | 0.99999022 | 1.88x |
| twins_pcpvt_base.in1k | 0.000181 | 0.99999499 | 1.40x |

所有模型精度验证均通过（Max Probability Difference < 0.05%）。

## 资源释放

```python
import gc
import torch

del model, input_tensor
gc.collect()
torch.npu.empty_cache()
```

## 参考仓库

- Model-Agent 项目：[Ascend/model-agent](https://gitcode.com/Ascend/model-agent)
