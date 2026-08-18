# SEMNASNet NPU Deployment Skill

## 概述

本 Skill 用于在华为昇腾 NPU 上自动完成 SEMNASNet（Squeeze-and-Excitation MobileNets）图像分类系列模型的部署、推理、CPU/NPU 精度对比、README 生成和模型仓库发布。

## 支持的模型列表

| 模型名称 | 原始地址 | 适配仓库 |
|---------|---------|---------|
| `semnasnet_100.rmsp_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/semnasnet_100.rmsp_in1k) | [GitCode](https://gitcode.com/m0_74196153/semnasnet_100.rmsp_in1k-npu) |
| `semnasnet_075.rmsp_in1k` | [ModelScope](https://www.modelscope.cn/models/timm/semnasnet_075.rmsp_in1k) | [GitCode](https://gitcode.com/m0_74196153/semnasnet_075.rmsp_in1k-npu) |

## 技能参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 是 | 模型名称：`semnasnet_100.rmsp_in1k` 或 `semnasnet_075.rmsp_in1k` |
| `device` | string | 否 | 推理设备：`cpu` 或 `npu`（默认 `npu`） |
| `image_path` | string | 否 | 输入图像路径（默认使用合成测试图像） |

## 输出结果

- 推理 Top-5 预测类别和概率
- CPU/NPU 精度对比指标（Logits 误差、概率误差、余弦相似度、Top-1 一致性）
- 推理性能数据（推理耗时）
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
```bash
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download(f"timm/{model_name}")
```

### 3. 执行 NPU 推理

```bash
python3 inference.py --model-name <model_name> --device npu
```

### 4. 执行 CPU/NPU 精度对比

```bash
python3 compare_cpu_npu.py --model-name <model_name>
```

### 5. 串行执行多个模型

为避免 NPU 显存溢出，多个模型必须串行执行：

```python
import gc
import torch

models = ["semnasnet_100.rmsp_in1k", "semnasnet_075.rmsp_in1k"]
for model_name in models:
    # 执行推理和测试
    run_inference(model_name)
    run_comparison(model_name)
    # 释放资源
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
```

### 6. 生成终端截图

```bash
python3 terminal_screenshot.py --input output.txt --output terminal_output.png
```

### 7. 提交模型仓库

使用 GitCode API 创建模型仓库并推送代码：

```bash
# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "<model_name>-npu", "repository_type": "model"}'

# 推送代码
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/<repo>.git
git push -u origin main
```

## 精度测试要求

- 必须进行 CPU 和 NPU 推理结果对比
- 概率最大绝对误差必须 < 0.01（1%）
- 使用多张测试图像（不同颜色/内容）进行综合评估
- 记录 Logits 误差、概率误差、余弦相似度、Top-1 类别一致性

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
