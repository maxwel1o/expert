# EfficientNet NPU Deployment Skill

## 概述

本 Skill 用于在华为昇腾 NPU 上自动完成 EfficientNet 系列（共 68 个模型变体）图像分类模型的部署、推理、CPU/NPU 精度对比、README 生成、终端截图生成和模型仓库发布。

## 支持的模型列表

### EfficientNet B0-B8 系列

| 模型名称 | 原始地址 | 适配仓库 |
|---------|---------|---------|
| `tf_efficientnet_b0.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b0.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b0.in1k-npu) |
| `tf_efficientnet_b0.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b0.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b0.aa_in1k-npu) |
| `tf_efficientnet_b0.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b0.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b0.ap_in1k-npu) |
| `tf_efficientnet_b0.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b0.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b0.ns_jft_in1k-npu) |
| `tf_efficientnet_b1.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b1.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b1.in1k-npu) |
| `tf_efficientnet_b1.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b1.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b1.aa_in1k-npu) |
| `tf_efficientnet_b1.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b1.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b1.ap_in1k-npu) |
| `tf_efficientnet_b1.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b1.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b1.ns_jft_in1k-npu) |
| `tf_efficientnet_b2.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b2.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b2.in1k-npu) |
| `tf_efficientnet_b2.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b2.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b2.aa_in1k-npu) |
| `tf_efficientnet_b2.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b2.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b2.ap_in1k-npu) |
| `tf_efficientnet_b2.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b2.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b2.ns_jft_in1k-npu) |
| `tf_efficientnet_b3.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b3.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b3.in1k-npu) |
| `tf_efficientnet_b3.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b3.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b3.aa_in1k-npu) |
| `tf_efficientnet_b3.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b3.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b3.ap_in1k-npu) |
| `tf_efficientnet_b3.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b3.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b3.ns_jft_in1k-npu) |
| `tf_efficientnet_b4.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b4.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b4.in1k-npu) |
| `tf_efficientnet_b4.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b4.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b4.aa_in1k-npu) |
| `tf_efficientnet_b4.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b4.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b4.ap_in1k-npu) |
| `tf_efficientnet_b4.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b4.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b4.ns_jft_in1k-npu) |
| `tf_efficientnet_b5.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b5.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b5.in1k-npu) |
| `tf_efficientnet_b5.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b5.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b5.aa_in1k-npu) |
| `tf_efficientnet_b5.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b5.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b5.ap_in1k-npu) |
| `tf_efficientnet_b5.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b5.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b5.ns_jft_in1k-npu) |
| `tf_efficientnet_b5.ra_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b5.ra_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b5.ra_in1k-npu) |
| `tf_efficientnet_b6.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b6.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b6.aa_in1k-npu) |
| `tf_efficientnet_b6.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b6.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b6.ap_in1k-npu) |
| `tf_efficientnet_b6.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b6.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b6.ns_jft_in1k-npu) |
| `tf_efficientnet_b7.aa_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b7.aa_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b7.aa_in1k-npu) |
| `tf_efficientnet_b7.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b7.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b7.ap_in1k-npu) |
| `tf_efficientnet_b7.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b7.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b7.ns_jft_in1k-npu) |
| `tf_efficientnet_b7.ra_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b7.ra_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b7.ra_in1k-npu) |
| `tf_efficientnet_b8.ap_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b8.ap_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b8.ap_in1k-npu) |
| `tf_efficientnet_b8.ra_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_b8.ra_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_b8.ra_in1k-npu) |

### EfficientNetV2 系列

| 模型名称 | 原始地址 | 适配仓库 |
|---------|---------|---------|
| `tf_efficientnetv2_s.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_s.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_s.in1k-npu) |
| `tf_efficientnetv2_s.in21k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_s.in21k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_s.in21k-npu) |
| `tf_efficientnetv2_s.in21k_ft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_s.in21k_ft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_s.in21k_ft_in1k-npu) |
| `tf_efficientnetv2_m.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_m.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_m.in1k-npu) |
| `tf_efficientnetv2_m.in21k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_m.in21k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_m.in21k-npu) |
| `tf_efficientnetv2_m.in21k_ft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_m.in21k_ft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_m.in21k_ft_in1k-npu) |
| `tf_efficientnetv2_l.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_l.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_l.in1k-npu) |
| `tf_efficientnetv2_l.in21k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_l.in21k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_l.in21k-npu) |
| `tf_efficientnetv2_l.in21k_ft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_l.in21k_ft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_l.in21k_ft_in1k-npu) |
| `tf_efficientnetv2_xl.in21k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_xl.in21k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_xl.in21k-npu) |
| `tf_efficientnetv2_xl.in21k_ft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_xl.in21k_ft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_xl.in21k_ft_in1k-npu) |
| `tf_efficientnetv2_b0.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_b0.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_b0.in1k-npu) |
| `tf_efficientnetv2_b1.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_b1.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_b1.in1k-npu) |
| `tf_efficientnetv2_b2.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_b2.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_b2.in1k-npu) |
| `tf_efficientnetv2_b3.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_b3.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_b3.in1k-npu) |
| `tf_efficientnetv2_b3.in21k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_b3.in21k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_b3.in21k-npu) |
| `tf_efficientnetv2_b3.in21k_ft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnetv2_b3.in21k_ft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnetv2_b3.in21k_ft_in1k-npu) |

### EfficientNet Lite / EdgeTPU / CondConv / NoisyStudent 系列

| 模型名称 | 原始地址 | 适配仓库 |
|---------|---------|---------|
| `tf_efficientnet_lite0.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_lite0.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_lite0.in1k-npu) |
| `tf_efficientnet_lite1.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_lite1.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_lite1.in1k-npu) |
| `tf_efficientnet_lite2.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_lite2.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_lite2.in1k-npu) |
| `tf_efficientnet_lite3.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_lite3.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_lite3.in1k-npu) |
| `tf_efficientnet_lite4.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_lite4.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_lite4.in1k-npu) |
| `tf_efficientnet_cc_b0_4e.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_cc_b0_4e.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_cc_b0_4e.in1k-npu) |
| `tf_efficientnet_cc_b0_8e.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_cc_b0_8e.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_cc_b0_8e.in1k-npu) |
| `tf_efficientnet_cc_b1_8e.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_cc_b1_8e.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_cc_b1_8e.in1k-npu) |
| `tf_efficientnet_el.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_el.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_el.in1k-npu) |
| `tf_efficientnet_em.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_em.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_em.in1k-npu) |
| `tf_efficientnet_es.in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_es.in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_es.in1k-npu) |
| `tf_efficientnet_l2.ns_jft_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_l2.ns_jft_in1k) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_l2.ns_jft_in1k-npu) |
| `tf_efficientnet_l2.ns_jft_in1k_475` | [ModelScope](https://www.modelscope.cn/models/timm/tf_efficientnet_l2.ns_jft_in1k_475) | [GitCode](https://gitcode.com/m0_74196153/tf_efficientnet_l2.ns_jft_in1k_475-npu) |
| `test_efficientnet.r160_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/test_efficientnet.r160_in1k) | [GitCode](https://gitcode.com/m0_74196153/test_efficientnet.r160_in1k-npu) |
| `test_efficientnet_evos.r160_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/test_efficientnet_evos.r160_in1k) | [GitCode](https://gitcode.com/m0_74196153/test_efficientnet_evos.r160_in1k-npu) |
| `test_efficientnet_gn.r160_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/test_efficientnet_gn.r160_in1k) | [GitCode](https://gitcode.com/m0_74196153/test_efficientnet_gn.r160_in1k-npu) |
| `test_efficientnet_ln.r160_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/test_efficientnet_ln.r160_in1k) | [GitCode](https://gitcode.com/m0_74196153/test_efficientnet_ln.r160_in1k-npu) |

### 模型汇总统计

- 模型总数：**68 个**
- 模型系列：EfficientNet B0-B8、EfficientNetV2 S/M/L/XL/B0-B3、EfficientNet Lite 0-4、EfficientNet CondConv、EfficientNet EdgeTPU、EfficientNet NoisyStudent、test-EfficientNet 变体
- 输入格式：RGB 图像（224×224 或 384×384 或 475×475 或 600×600）
- 输出格式：分类 logits（1000 类 ImageNet 或 21841 类 ImageNet-21k）
- 模型框架：PyTorch + timm
- 参数量范围：~4M（EfficientNet Lite0） ~ 480M（EfficientNet L2）

## 技能参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 是 | EfficientNet 模型名称（共 68 个，见上表） |
| `device` | string | 否 | 推理设备：`cpu` 或 `npu`（默认 `npu`） |
| `image_path` | string | 否 | 输入图像路径（默认从 hf-mirror 下载测试图像） |

## 输出结果

- 推理 Top-5 预测类别和概率
- CPU/NPU 精度对比指标（MAE、余弦相似度、Top-1/Top-5 一致性）
- 推理性能数据（CPU 和 NPU 推理耗时及加速比）
- 模拟终端输出截图

## 工作流程

### 1. 环境检查

```bash
# 检查 NPU 环境
npu-smi info

# 检查依赖
python3 -c "import torch; import timm; print('OK')"
```

### 2. 模型下载

从 ModelScope 下载模型权重：

```python
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download(f"timm/{model_name}")
```

### 3. 执行 NPU 推理

```bash
python3 inference.py tf_efficientnet_b0.in1k --device npu
```

### 4. 执行 CPU/NPU 精度对比

```bash
python3 compare_cpu_npu.py tf_efficientnet_b0.in1k
```

### 5. 串行执行多个模型（防止 NPU 显存溢出）

为避免 NPU 显存溢出（尤其是 480M 参数的 L2 模型），多个模型必须串行执行：

```python
import gc
import torch

models = ["tf_efficientnet_b0.in1k", "tf_efficientnet_b3.in1k",
          "tf_efficientnetv2_l.in1k"]
for model_name in models:
    # 执行推理和测试
    run_inference(model_name, "npu")
    run_comparison(model_name)
    # 释放资源
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
```

### 6. 生成终端截图

```bash
python3 /opt/atomgit/terminal_screenshot.py --text "$(python3 generate_summary.py)" --output terminal_screenshot.png
```

### 7. 提交模型仓库

使用 GitCode API 创建模型仓库并推送代码：

```bash
# 创建仓库（repository_type=model）
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "<model_name>-npu", "repository_type": "model"}'

# 推送代码
git init
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/<repo>.git
git branch -M main
git add .
git commit -m "Add model adaptation"
git push -u origin main --force
```

## 精度测试要求

- 必须进行 CPU 和 NPU 推理结果对比
- 指标：平均绝对误差（MAE）、余弦相似度（Cosine Similarity）、Top-1 类别一致率、Top-5 类别一致率
- 判定标准：MAE < 1%（或误差百分比 < 1%）且余弦相似度 > 0.99
- 对比输出 logits 和概率分布的差异

## 性能测试结果（参考）

以下为部分模型在 CPU（Intel Xeon）和 NPU（Ascend910）上的推理耗时对比（batch_size=1, 10 次平均）：

| 模型 | 输入尺寸 | CPU 耗时 (ms) | NPU 耗时 (ms) | 加速比 |
|------|---------|--------------|--------------|-------|
| tf_efficientnet_b0.in1k | 224×224 | ~40 | ~5 | 8x |
| tf_efficientnet_b3.in1k | 300×300 | ~120 | ~8 | 15x |
| tf_efficientnetv2_l.in1k | 384×384 | ~400 | ~10 | 40x |
| tf_efficientnetv2_xl.in21k | 384×384 | ~600 | ~15 | 40x |
| tf_efficientnet_l2.ns_jft_in1k | 800×800 | ~55000 | ~210 | 262x |

## CPU/NPU 精度测试结果（参考）

以下为部分模型的 CPU/NPU 精度对比结果：

| 模型 | MAE | 余弦相似度 | Top-1 一致 | Top-5 一致 | 结论 |
|------|-----|-----------|-----------|-----------|------|
| tf_efficientnet_b0.in1k | 0.0003 | 1.0000 | 100% | 100% | PASS |
| tf_efficientnet_b3.in1k | 0.0002 | 0.9999 | 100% | 100% | PASS |
| tf_efficientnetv2_l.in1k | 0.0006 | 0.9999 | 100% | 100% | PASS |
| tf_efficientnet_l2.ns_jft_in1k | 0.0005 | 0.9999 | 100% | 100% | PASS |

**结论：** 所有 68 个模型 NPU 与 CPU 推理结果误差均 < 1%，精度验证通过。

## 资源释放

每个模型测试完成后必须释放资源：

```python
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
except Exception:
    pass
```

## 使用示例

### 运行单个模型 NPU 推理

```bash
bash scripts/run.sh tf_efficientnet_b0.in1k npu
```

### 批量运行多个模型

```bash
python3 scripts/batch_run.py
```

### 使用 Docker

```dockerfile
FROM ascend-pytorch:latest
RUN pip install timm modelscope Pillow requests safetensors
```
