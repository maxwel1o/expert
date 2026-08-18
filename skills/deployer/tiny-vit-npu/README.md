# TinyViT NPU Deployment Skill

## 概述

本 Skill 提供 TinyViT (Tiny Vision Transformer) 系列模型在昇腾 Ascend910 NPU 上的自动化部署、推理和精度验证能力。

## 支持的模型列表

| 模型名称 | 参数量 | 输入尺寸 | 类别数 | 数据集 |
|---------|-------|---------|-------|-------|
| tiny_vit_5m_224.dist_in22k | 12.1M | 224x224 | 21841 | ImageNet-22K |
| tiny_vit_5m_224.in1k | 5.4M | 224x224 | 1000 | ImageNet-1K |
| tiny_vit_5m_224.dist_in22k_ft_in1k | 5.4M | 224x224 | 1000 | ImageNet-22K+1K |
| tiny_vit_21m_512.dist_in22k_ft_in1k | 21.3M | 512x512 | 1000 | ImageNet-22K+1K |
| tiny_vit_21m_384.dist_in22k_ft_in1k | 21.2M | 384x384 | 1000 | ImageNet-22K+1K |
| tiny_vit_21m_224.dist_in22k_ft_in1k | 21.2M | 224x224 | 1000 | ImageNet-22K+1K |
| tiny_vit_21m_224.in1k | 21.2M | 224x224 | 1000 | ImageNet-1K |
| tiny_vit_21m_224.dist_in22k | 33.2M | 224x224 | 21841 | ImageNet-22K |
| tiny_vit_11m_224.dist_in22k | 20.4M | 224x224 | 21841 | ImageNet-22K |
| tiny_vit_11m_224.in1k | 11.0M | 224x224 | 1000 | ImageNet-1K |
| tiny_vit_11m_224.dist_in22k_ft_in1k | 11.0M | 224x224 | 1000 | ImageNet-22K+1K |

## 已发布的模型仓库

- [tiny_vit_5m_224_dist_in22k_npu](https://gitcode.com/m0_74196153/tiny_vit_5m_224_dist_in22k_npu)
- [tiny_vit_5m_224_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_5m_224_in1k_npu)
- [tiny_vit_5m_224_dist_in22k_ft_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_5m_224_dist_in22k_ft_in1k_npu)
- [tiny_vit_21m_512_dist_in22k_ft_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_21m_512_dist_in22k_ft_in1k_npu)
- [tiny_vit_21m_384_dist_in22k_ft_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_21m_384_dist_in22k_ft_in1k_npu)
- [tiny_vit_21m_224_dist_in22k_ft_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_21m_224_dist_in22k_ft_in1k_npu)
- [tiny_vit_21m_224_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_21m_224_in1k_npu)
- [tiny_vit_21m_224_dist_in22k_npu](https://gitcode.com/m0_74196153/tiny_vit_21m_224_dist_in22k_npu)
- [tiny_vit_11m_224_dist_in22k_npu](https://gitcode.com/m0_74196153/tiny_vit_11m_224_dist_in22k_npu)
- [tiny_vit_11m_224_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_11m_224_in1k_npu)
- [tiny_vit_11m_224_dist_in22k_ft_in1k_npu](https://gitcode.com/m0_74196153/tiny_vit_11m_224_dist_in22k_ft_in1k_npu)

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model_name | string | 是 | 模型名称 (如 tiny_vit_5m_224.in1k) |
| device | string | 否 | 推理设备: cpu 或 npu, 默认 npu |
| image_size | int | 否 | 输入图像尺寸, 根据模型自动选择 |

## Skill 输出结果

- CPU/NPU 推理结果和性能数据
- CPU/NPU 精度对比结果 (概率相对误差、余弦相似度、Top-1 匹配率)
- 推断结论: 是否通过精度对齐 (误差 < 1%)

## 环境要求

- Python >= 3.9
- PyTorch >= 2.0.0
- torch_npu >= 2.0.0
- timm >= 1.0.0
- modelscope >= 1.0.0
- Ascend910 NPU

## 使用方法

### 单个模型 NPU 推理

```bash
python scripts/inference.py --model-name tiny_vit_5m_224.in1k --device npu
```

### 单个模型 CPU/NPU 精度对比

```bash
python scripts/compare_cpu_npu.py --model-name tiny_vit_5m_224.in1k
```

### 批量处理所有模型

```bash
bash scripts/run_all.sh
```

或者使用 Python:

```python
from scripts.inference import run_inference

# NPU 推理
result = run_inference("tiny_vit_5m_224.in1k", device="npu")
print(result)
```

## 精度验证标准

使用概率相对误差 (Prob Relative Error) 作为主要指标:

- Prob Relative Error = ||P_cpu - P_npu|| / ||P_cpu||
- 要求误差 < 1%
- 辅助指标: Cosine Similarity > 0.999, Top-1 Class Match = True

## 串行执行策略

为避免 NPU 显存爆炸, 多个模型使用串行执行。每个模型推理完成后:
1. 删除模型对象和输入张量
2. 调用 `gc.collect()` 回收内存
3. 调用 `torch.npu.empty_cache()` 清空 NPU 缓存
4. 间隔 2 秒再进行下一个模型

## README 生成

每个模型仓库包含:
- inference.py: 独立推理脚本
- compare_cpu_npu.py: CPU/NPU 精度对比脚本
- requirements.txt: 依赖清单
- readme.md: 中文文档 (含真实数据和截图)
- terminal_output.png: 模拟终端输出截图

## 终端截图生成

使用 `/opt/atomgit/terminal_screenshot.py` 生成模拟终端输出截图:

```bash
python /opt/atomgit/terminal_screenshot.py --input output.txt --output terminal_output.png
```

## 模型仓库发布

使用 GitCode API 创建和推送模型仓库:

```bash
# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"model_name_npu","repository_type":"model"}'

# 推送代码
git remote add origin https://auth:${TOKEN}@gitcode.com/username/repo.git
git push -u origin main
```

## 标签

- #+NPU #+CV #+图像分类 #+昇腾 #+TinyViT #+timm
