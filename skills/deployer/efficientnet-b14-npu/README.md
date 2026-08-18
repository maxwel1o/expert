# EfficientNet Batch 14 NPU Deployment Skill

## 概述

本 Skill 用于在华为昇腾 Ascend910 NPU 上完成 **EfficientNet** 系列 68 个图像分类模型的部署、推理、CPU/NPU 精度对比、README 文档生成、终端截图生成和 GitCode 模型仓库发布。

包含以下子系列：
- **EfficientNetV2** (XL/S/M/L/B3/B2/B1/B0)：改进的训练效率架构
- **EfficientNet Lite** (Lite4/3/2/1/0)：轻量级移动端变体
- **EfficientNet B0-B8**：原始 EfficientNet 各规模版本
- **EfficientNet-EdgeTPU/CondConv** (cc_b0/b1)：条件卷积变体
- **EfficientNet-NoisyStudent** (ns_jft)：使用 NoisyStudent 半监督训练的变体
- **Test EfficientNet**：实验性架构变体（LN、GN、Evos）

## 支持的模型列表

| # | 模型名称 | 输入尺寸 | GitCode 仓库地址 |
|---|---------|---------|-----------------|
| 1 | tf_efficientnetv2_xl.in21k_ft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_xl.in21k_ft_in1k-npu) |
| 2 | tf_efficientnetv2_xl.in21k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_xl.in21k-npu) |
| 3 | tf_efficientnetv2_s.in21k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_s.in21k-npu) |
| 4 | tf_efficientnetv2_s.in21k_ft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_s.in21k_ft_in1k-npu) |
| 5 | tf_efficientnetv2_s.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_s.in1k-npu) |
| 6 | tf_efficientnetv2_m.in21k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_m.in21k-npu) |
| 7 | tf_efficientnetv2_m.in21k_ft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_m.in21k_ft_in1k-npu) |
| 8 | tf_efficientnetv2_m.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_m.in1k-npu) |
| 9 | tf_efficientnetv2_l.in21k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_l.in21k-npu) |
| 10 | tf_efficientnetv2_l.in21k_ft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_l.in21k_ft_in1k-npu) |
| 11 | tf_efficientnetv2_l.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_l.in1k-npu) |
| 12 | tf_efficientnetv2_b3.in21k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_b3.in21k-npu) |
| 13 | tf_efficientnetv2_b3.in21k_ft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_b3.in21k_ft_in1k-npu) |
| 14 | tf_efficientnetv2_b3.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_b3.in1k-npu) |
| 15 | tf_efficientnetv2_b2.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_b2.in1k-npu) |
| 16 | tf_efficientnetv2_b1.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_b1.in1k-npu) |
| 17 | tf_efficientnetv2_b0.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnetv2_b0.in1k-npu) |
| 18 | tf_efficientnet_lite4.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_lite4.in1k-npu) |
| 19 | tf_efficientnet_lite3.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_lite3.in1k-npu) |
| 20 | tf_efficientnet_lite2.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_lite2.in1k-npu) |
| 21 | tf_efficientnet_lite1.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_lite1.in1k-npu) |
| 22 | tf_efficientnet_lite0.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_lite0.in1k-npu) |
| 23 | tf_efficientnet_l2.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_l2.ns_jft_in1k-npu) |
| 24 | tf_efficientnet_l2.ns_jft_in1k_475 | 475x475 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_l2.ns_jft_in1k_475-npu) |
| 25 | tf_efficientnet_es.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_es.in1k-npu) |
| 26 | tf_efficientnet_em.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_em.in1k-npu) |
| 27 | tf_efficientnet_el.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_el.in1k-npu) |
| 28 | tf_efficientnet_cc_b1_8e.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_cc_b1_8e.in1k-npu) |
| 29 | tf_efficientnet_cc_b0_8e.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_cc_b0_8e.in1k-npu) |
| 30 | tf_efficientnet_cc_b0_4e.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_cc_b0_4e.in1k-npu) |
| 31 | tf_efficientnet_b8.ra_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b8.ra_in1k-npu) |
| 32 | tf_efficientnet_b8.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b8.ap_in1k-npu) |
| 33 | tf_efficientnet_b7.ra_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b7.ra_in1k-npu) |
| 34 | tf_efficientnet_b7.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b7.ns_jft_in1k-npu) |
| 35 | tf_efficientnet_b7.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b7.aa_in1k-npu) |
| 36 | tf_efficientnet_b7.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b7.ap_in1k-npu) |
| 37 | tf_efficientnet_b6.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b6.ns_jft_in1k-npu) |
| 38 | tf_efficientnet_b6.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b6.ap_in1k-npu) |
| 39 | tf_efficientnet_b6.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b6.aa_in1k-npu) |
| 40 | tf_efficientnet_b5.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b5.ns_jft_in1k-npu) |
| 41 | tf_efficientnet_b5.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b5.ap_in1k-npu) |
| 42 | tf_efficientnet_b5.ra_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b5.ra_in1k-npu) |
| 43 | tf_efficientnet_b5.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b5.aa_in1k-npu) |
| 44 | tf_efficientnet_b5.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b5.in1k-npu) |
| 45 | tf_efficientnet_b4.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b4.ap_in1k-npu) |
| 46 | tf_efficientnet_b4.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b4.aa_in1k-npu) |
| 47 | tf_efficientnet_b4.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b4.ns_jft_in1k-npu) |
| 48 | tf_efficientnet_b4.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b4.in1k-npu) |
| 49 | tf_efficientnet_b3.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b3.ap_in1k-npu) |
| 50 | tf_efficientnet_b3.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b3.ns_jft_in1k-npu) |
| 51 | tf_efficientnet_b3.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b3.aa_in1k-npu) |
| 52 | tf_efficientnet_b3.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b3.in1k-npu) |
| 53 | tf_efficientnet_b2.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b2.ns_jft_in1k-npu) |
| 54 | tf_efficientnet_b2.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b2.ap_in1k-npu) |
| 55 | tf_efficientnet_b2.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b2.in1k-npu) |
| 56 | tf_efficientnet_b2.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b2.aa_in1k-npu) |
| 57 | tf_efficientnet_b1.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b1.in1k-npu) |
| 58 | tf_efficientnet_b1.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b1.aa_in1k-npu) |
| 59 | tf_efficientnet_b1.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b1.ap_in1k-npu) |
| 60 | tf_efficientnet_b1.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b1.ns_jft_in1k-npu) |
| 61 | tf_efficientnet_b0.ns_jft_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b0.ns_jft_in1k-npu) |
| 62 | tf_efficientnet_b0.ap_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b0.ap_in1k-npu) |
| 63 | tf_efficientnet_b0.in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b0.in1k-npu) |
| 64 | tf_efficientnet_b0.aa_in1k | 224x224 | [repo](https://gitcode.com/m0_74196153/tf_efficientnet_b0.aa_in1k-npu) |
| 65 | test_efficientnet_ln.r160_in1k | 160x160 | [repo](https://gitcode.com/m0_74196153/test_efficientnet_ln.r160_in1k-npu) |
| 66 | test_efficientnet_gn.r160_in1k | 160x160 | [repo](https://gitcode.com/m0_74196153/test_efficientnet_gn.r160_in1k-npu) |
| 67 | test_efficientnet_evos.r160_in1k | 160x160 | [repo](https://gitcode.com/m0_74196153/test_efficientnet_evos.r160_in1k-npu) |
| 68 | test_efficientnet.r160_in1k | 160x160 | [repo](https://gitcode.com/m0_74196153/test_efficientnet.r160_in1k-npu) |

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- torch_npu（昇腾 NPU 支持）
- timm >= 1.0.0
- modelscope（模型权重下载）
- 昇腾 CANN 及 NPU 驱动
- Ascend910 NPU（测试通过）

## 输入参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model_name` | string | 否 | `all` | 指定单个模型名称，为 `all` 时执行全部 68 个模型 |
| `start_index` | integer | 否 | 0 | 起始索引，用于分批处理 |
| `end_index` | integer | 否 | 68 | 结束索引，用于分批处理 |
| `skip_inference` | boolean | 否 | `false` | 跳过推理步骤，只生成文档和提交仓库 |
| `skip_push` | boolean | 否 | `false` | 跳过 GitCode 仓库推送 |

## 输出结果

- 每个模型独立的推理结果（output_cpu.pt、output_npu.pt）
- CPU/NPU 精度对比报告
- README.md 文档
- 模拟终端输出截图 screenshot.png
- 自动创建的 GitCode 模型仓库

## 使用方法

### 安装依赖

```bash
pip install torch torchvision torch_npu timm modelscope safetensors pillow
```

### 执行单个模型推理

```bash
python3 scripts/inference.py tf_efficientnetv2_s.in1k --device cpu
python3 scripts/inference.py tf_efficientnetv2_s.in1k --device npu
```

### CPU/NPU 精度对比

```bash
python3 scripts/compare_cpu_npu.py tf_efficientnetv2_s.in1k
```

### 批量执行全部模型

```bash
# 全部 68 个模型
python3 scripts/batch_runner.py

# 指定范围
python3 scripts/batch_runner.py --start 0 --end 10
```

### 生成终端截图

```bash
python3 /opt/atomgit/terminal_screenshot.py --file scripts/inference.py --model tf_efficientnetv2_s.in1k
```

### 发布到 GitCode

```bash
python3 scripts/push_repos.py
```

## 精度测试结果

所有模型通过精度测试的统计：

- **余弦相似度**: 绝大多数 > 0.99999
- **相对误差**: 全部 < 1% (通过测试的模型)
- **Top-1 一致率**: 100%
- **Top-5 重叠率**: 100%

> 注意: 少数模型（如 tf_efficientnet_em.in1k、tf_efficientnet_cc_b0_4e.in1k）在 NPU 上存在精度误差超过 1% 的情况，原因可能是 torch_npu 的特定算子兼容性问题。这类模型已记录在各自 README 的 FAIL 结论中。

详细精度数据见每个模型的 README.md。

## 串行执行说明

为避免 NPU 显存爆炸，所有模型串行执行：

1. 先执行 CPU 推理，保存 CPU 输出
2. 释放内存（gc.collect()）
3. 执行 NPU 推理，保存 NPU 输出
4. CPU/NPU 精度对比（MAE、余弦相似度、Top-1/5 一致率）
5. 生成截图和文档
6. 释放 NPU 显存（torch.npu.empty_cache()）
7. 处理下一个模型

## GitCode 仓库发布

使用 GitCode API 自动创建模型仓库并推送代码：

```bash
# 手动发布示例
git init
git branch -M main
git add inference.py compare_cpu_npu.py requirements.txt README.md terminal_screenshot.png
git commit -m "Add model NPU adaptation"
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/model-name-npu.git
git push -u origin main
```

## 模型仓库结构

每个模型仓库包含以下文件：

```
model-name-npu/
├── inference.py          # CPU/NPU 推理脚本
├── compare_cpu_npu.py    # 精度对比脚本
├── requirements.txt      # 依赖清单
├── README.md             # 中文文档（含真实测试数据）
├── terminal_screenshot.png # 模拟终端截图
├── inference_cpu.log     # CPU 推理日志
├── inference_npu.log     # NPU 推理日志
├── compare.log           # 对比日志
├── results_cpu.json      # CPU 结果
├── results_npu.json      # NPU 结果
└── compare_results.json  # 精度对比结果
```
