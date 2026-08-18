# Swin Transformer Batch NPU Deployment Skill

## 概述

本 Skill 用于在昇腾 NPU (Ascend910) 上一站式完成 Swin Transformer 系列图像分类模型的适配、推理验证和模型仓库发布。支持 19 个 Swin Transformer 变体（tiny/small/base/large，in1k/in22k/384分辨率）。

## 支持的模型列表

| # | 模型名称 | 类别数 | 输入尺寸 |
|---|---------|-------:|:-------:|
| 1 | swin_tiny_patch4_window7_224.ms_in1k | 1000 | 3×224×224 |
| 2 | swin_tiny_patch4_window7_224.ms_in22k_ft_in1k | 1000 | 3×224×224 |
| 3 | swin_tiny_patch4_window7_224.ms_in22k | 21841 | 3×224×224 |
| 4 | swin_small_patch4_window7_224.ms_in1k | 1000 | 3×224×224 |
| 5 | swin_small_patch4_window7_224.ms_in22k_ft_in1k | 1000 | 3×224×224 |
| 6 | swin_small_patch4_window7_224.ms_in22k | 21841 | 3×224×224 |
| 7 | swin_s3_tiny_224.ms_in1k | 1000 | 3×224×224 |
| 8 | swin_s3_small_224.ms_in1k | 1000 | 3×224×224 |
| 9 | swin_s3_base_224.ms_in1k | 1000 | 3×224×224 |
| 10 | swin_large_patch4_window7_224.ms_in22k_ft_in1k | 1000 | 3×224×224 |
| 11 | swin_large_patch4_window7_224.ms_in22k | 21841 | 3×224×224 |
| 12 | swin_large_patch4_window12_384.ms_in22k_ft_in1k | 1000 | 3×384×384 |
| 13 | swin_large_patch4_window12_384.ms_in22k | 21841 | 3×384×384 |
| 14 | swin_base_patch4_window7_224.ms_in22k | 21841 | 3×224×224 |
| 15 | swin_base_patch4_window7_224.ms_in22k_ft_in1k | 1000 | 3×224×224 |
| 16 | swin_base_patch4_window7_224.ms_in1k | 1000 | 3×224×224 |
| 17 | swin_base_patch4_window12_384.ms_in22k_ft_in1k | 1000 | 3×384×384 |
| 18 | swin_base_patch4_window12_384.ms_in22k | 21841 | 3×384×384 |
| 19 | swin_base_patch4_window12_384.ms_in1k | 1000 | 3×384×384 |

## Skill 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | string | "all" | 指定单个模型名（如 `swin_tiny_patch4_window7_224.ms_in1k`）或 "all" 执行全部 19 个 |
| skip_inference | boolean | false | 跳过推理，只生成文档和截图 |
| skip_push | boolean | false | 跳过 GitCode 仓库提交 |

## Skill 输出结果

每个模型输出以下交付件：

- `inference.py` — NPU/CPU 推理脚本
- `compare_cpu_npu.py` — CPU/NPU 精度对比脚本
- `requirements.txt` — 依赖清单
- `README.md` — 详细中文文档（含精度数据）
- `screenshots/*.png` — 模拟终端输出截图
- GitCode 模型仓库（自动创建并推送）

## 环境要求

- 昇腾 NPU 环境（Ascend910）
- Python 3.10+
- PyTorch 2.x + torch_npu
- timm 1.0+
- modelscope

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision timm pillow numpy modelscope
```

## 使用方法

### 1. 下载所有模型权重

```bash
python3 scripts/download_all.py
```

### 2. 执行单个模型推理

```bash
python3 scripts/inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device cpu
python3 scripts/inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device npu
```

### 3. CPU/NPU 精度对比

```bash
python3 scripts/compare_cpu_npu.py --model swin_tiny_patch4_window7_224.ms_in1k
```

### 4. 执行全部模型（串行，自动清理显存）

```bash
python3 scripts/batch_runner.py
```

## 串行执行策略

为防止 NPU 显存爆炸，所有模型以串行方式执行。每个模型测试完成后自动：

1. 删除模型对象释放 CPU 内存
2. 调用 `gc.collect()` 回收 Python GC
3. 调用 `torch.npu.empty_cache()` 释放 NPU 显存

## 精度测试方法

1. 对同一随机输入张量，分别在 CPU 和 NPU 上运行推理
2. 计算逐元素最大绝对误差（Max Absolute Error）
3. 计算余弦相似度（Cosine Similarity）
4. 对比 Top-5 预测结果

## 精度通过标准

- **绝对误差 < 1%**（即 max_abs_error < 0.01）

## 验证结果摘要

所有 19 个模型在昇腾 NPU 上的推理结果与 CPU 对比：

- 最大绝对误差范围：0.13% ~ 0.75%（全部 < 1%）
- 余弦相似度范围：0.99989 ~ 0.99999
- Top-5 一致率：大部分模型 5/5
- NPU 加速比：18x ~ 208x

## 已发布的模型仓库

| 模型 | GitCode 仓库 |
|------|-------------|
| swin_tiny in1k | https://gitcode.com/m0_74196153/swin_tiny_patch4_window7_224.ms_in1k-npu |
| swin_tiny in22k ft in1k | https://gitcode.com/m0_74196153/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_tiny in22k | https://gitcode.com/m0_74196153/swin_tiny_patch4_window7_224.ms_in22k-npu |
| swin_small in1k | https://gitcode.com/m0_74196153/swin_small_patch4_window7_224.ms_in1k-npu |
| swin_small in22k ft in1k | https://gitcode.com/m0_74196153/swin_small_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_small in22k | https://gitcode.com/m0_74196153/swin_small_patch4_window7_224.ms_in22k-npu |
| swin_s3 tiny | https://gitcode.com/m0_74196153/swin_s3_tiny_224.ms_in1k-npu |
| swin_s3 small | https://gitcode.com/m0_74196153/swin_s3_small_224.ms_in1k-npu |
| swin_s3 base | https://gitcode.com/m0_74196153/swin_s3_base_224.ms_in1k-npu |
| swin_large in22k ft in1k | https://gitcode.com/m0_74196153/swin_large_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_large in22k | https://gitcode.com/m0_74196153/swin_large_patch4_window7_224.ms_in22k-npu |
| swin_large 384 in22k ft in1k | https://gitcode.com/m0_74196153/swin_large_patch4_window12_384.ms_in22k_ft_in1k-npu |
| swin_large 384 in22k | https://gitcode.com/m0_74196153/swin_large_patch4_window12_384.ms_in22k-npu |
| swin_base in22k | https://gitcode.com/m0_74196153/swin_base_patch4_window7_224.ms_in22k-npu |
| swin_base in22k ft in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window7_224.ms_in22k_ft_in1k-npu |
| swin_base in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window7_224.ms_in1k-npu |
| swin_base 384 in22k ft in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window12_384.ms_in22k_ft_in1k-npu |
| swin_base 384 in22k | https://gitcode.com/m0_74196153/swin_base_patch4_window12_384.ms_in22k-npu |
| swin_base 384 in1k | https://gitcode.com/m0_74196153/swin_base_patch4_window12_384.ms_in1k-npu |

## 注意事项

1. **模型权重下载**：优先使用 ModelScope 下载，必要时使用 hf-mirror 镜像
2. **串行执行**：必须串行测试每个模型，防止 NPU OOM
3. **num_classes 自动检测**：in1k 模型使用 1000 类，in22k 模型使用 21841 类，脚本自动检测
4. **HF_HUB_OFFLINE**：设置此环境变量可避免 HuggingFace 连接超时
