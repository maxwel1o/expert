---
name: wide-resnet-npu-deploy
description: >
  Wide ResNet 系列模型在昇腾 NPU 上的完整部署与推理验证 Skill。
  涵盖 timm 框架下 wide_resnet50_2 和 wide_resnet101_2 系列的
  racm/tv/tv2 三种训练配方变体在昇腾 NPU 上的环境准备、推理验证、
  CPU/NPU 精度对比、README 生成、终端截图生成和模型仓库发布
  的完整流程。可在任意 Ascend910 系列服务器上一键复现。
metadata:
  short-description: Wide ResNet 系列图像分类模型昇腾 NPU 部署
  category: NPU-Model-Deploy
  tags: [ascend, npu, wide-resnet, timm, image-classification, pytorch, cv, resnet]
---

# Wide ResNet 系列图像分类模型 昇腾 NPU 部署 Skill

本 Skill 提供 Wide ResNet（Wide Residual Networks）系列图像分类模型在华为昇腾 NPU 上的完整部署、推理验证和精度对比的标准化可复现流程。

## 支持的模型

| 模型名称 | 深度 | 宽度因子 | 训练配方 | 输入尺寸 | 任务类型 |
|---------|------|---------|---------|---------|---------|
| wide_resnet50_2.racm_in1k | 50 | 2× | racm | 224×224 | 图像分类 |
| wide_resnet50_2.tv2_in1k | 50 | 2× | TorchVision V2 | 224×224 | 图像分类 |
| wide_resnet50_2.tv_in1k | 50 | 2× | TorchVision | 224×224 | 图像分类 |
| wide_resnet101_2.tv_in1k | 101 | 2× | TorchVision | 224×224 | 图像分类 |
| wide_resnet101_2.tv2_in1k | 101 | 2× | TorchVision V2 | 224×224 | 图像分类 |

**已适配并交付的模型仓库：**
- [wide_resnet50_2.racm_in1k-npu](https://gitcode.com/m0_74196153/wide_resnet50_2.racm_in1k-npu)
- [wide_resnet50_2.tv2_in1k-npu](https://gitcode.com/m0_74196153/wide_resnet50_2.tv2_in1k-npu)
- [wide_resnet50_2.tv_in1k-npu](https://gitcode.com/m0_74196153/wide_resnet50_2.tv_in1k-npu)
- [wide_resnet101_2.tv_in1k-npu](https://gitcode.com/m0_74196153/wide_resnet101_2.tv_in1k-npu)
- [wide_resnet101_2.tv2_in1k-npu](https://gitcode.com/m0_74196153/wide_resnet101_2.tv2_in1k-npu)

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0 |
| Python | 3.9 – 3.11 |
| 网络 | 需联网下载模型权重，或从 ModelScope 本地加载 |

## 流程总览

```
0. 环境初始化
→ 1. 安装依赖（torch_npu + timm + modelscope）
→ 2. NPU 验证
→ 3. 下载模型权重（ModelScope）
→ 4. 推理运行（CPU + NPU）
→ 5. CPU/NPU 精度对比
→ 6. 生成终端截图
→ 7. 生成 README 和提交模型仓库
```

按以下各节顺序执行，每步完成后再进入下一步。

---

## 0. 环境初始化

```bash
# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 配置安装源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1. 安装依赖

```bash
pip install torch torch_npu timm Pillow numpy safetensors modelscope \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

详细依赖见 `scripts/requirements.txt`。

## 2. NPU 基础验证

```python
import torch
import torch_npu
print('torch:', torch.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('Device name:', torch.npu.get_device_name(0))
```

**通过标准**：`NPU available: True` 且无报错。

## 3. 下载模型权重

从 ModelScope 下载模型权重：

```python
from modelscope import snapshot_download
model_dir = snapshot_download('timm/wide_resnet50_2.racm_in1k')
```

本系列所有模型均可通过替换模型名称从 ModelScope 下载：
- `timm/wide_resnet50_2.racm_in1k`
- `timm/wide_resnet50_2.tv2_in1k`
- `timm/wide_resnet50_2.tv_in1k`
- `timm/wide_resnet101_2.tv_in1k`
- `timm/wide_resnet101_2.tv2_in1k`

## 4. 推理运行

### 4.1 单模型推理

```bash
cd /path/to/working/dir
python3 scripts/inference.py <模型名称> [--device cpu|npu]
```

脚本说明：
- 从本地 modelscope 缓存加载权重（避免访问 HuggingFace）
- 可选 CPU 或 NPU 推理
- 输出 Top-5 分类结果和推理耗时
- 结果保存到 `results_{device}.json`

### 4.2 串行执行多个模型（防止 OOM）

为防止 NPU 显存和系统内存爆炸，多个模型必须串行执行：

```bash
for model_name in wide_resnet50_2.racm_in1k wide_resnet50_2.tv2_in1k wide_resnet50_2.tv_in1k wide_resnet101_2.tv_in1k wide_resnet101_2.tv2_in1k; do
    mkdir -p output/${model_name}
    cd output/${model_name}
    python3 scripts/inference.py ${model_name} --device cpu --num_runs 10
    python3 scripts/inference.py ${model_name} --device npu --num_runs 10

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

## 5. CPU/NPU 精度对比

```bash
python3 scripts/compare_cpu_npu.py <模型名称> --num_runs 10
```

精度对比脚本计算以下指标：
- Logits 最大/平均绝对误差
- Logits 余弦相似度
- Softmax 概率最大/平均差异
- Top-1 一致性
- Top-5 重合率

结果保存到 `compare_results.json`。

### 精度要求

NPU 与 CPU 推理结果误差 < 1%（以最大概率差异计）。

**实测结果汇总**：

| 模型 | Max Prob Error | Top-1 一致 | Top-5 一致 | NPU 加速比 |
|------|---------------|-----------|-----------|-----------|
| wide_resnet50_2.racm_in1k | 0.018% | ✓ | 5/5 | 90.51× |
| wide_resnet50_2.tv2_in1k | 0.008% | ✓ | 5/5 | 93.52× |
| wide_resnet50_2.tv_in1k | 0.000024% | ✓ | 5/5 | 140.11× |
| wide_resnet101_2.tv_in1k | 0.0015% | ✓ | 5/5 | 126.05× |
| wide_resnet101_2.tv2_in1k | 0.00045% | ✓ | 5/5 | 85.59× |

## 6. 生成终端截图

```bash
python3 scripts/generate_screenshot.py <日志文件路径> <输出HTML路径>
```

生成 HTML 格式的模拟终端输出截图，可直接在浏览器中打开查看。

## 7. 生成 README 和提交模型仓库

### 7.1 README 要求

README 必须包含：
- 模型介绍
- 原始模型地址
- 任务类型
- 依赖环境
- NPU 适配说明
- 推理命令
- 推理结果（含真实数据）
- CPU/NPU 精度测试结果（表格展示）
- 明确结论：NPU 与 CPU 推理误差 < 1%
- 性能测试结果
- 终端截图
- 模型标签（#+NPU, #+CV, #+昇腾 等）

### 7.2 创建模型仓库

```bash
# 使用 GitCode API 创建 model 类型仓库
curl -X POST \
  --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "name": "{model_name}-npu",
    "private": false,
    "repository_type": "model"
  }' \
  "https://api.gitcode.com/api/v5/user/repos"
```

### 7.3 推送代码

```bash
git init
git checkout -b main
git add inference.py compare_cpu_npu.py requirements.txt readme.md screenshot.html compare_results.json
git commit -m "Add {model_name} NPU adaptation"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/{model_name}-npu.git"
git push -u origin main
```

## 精度测试方法

所有测试使用以下方法：
1. 加载同一张测试图像（随机 RGB 图像或 ImageNet 样本）
2. 使用 timm 的 `create_transform` 进行预处理（归一化、resize）
3. 分别在 CPU 和 NPU 上运行模型，确保输入完全相同
4. 对比两个设备的输出 logits 和 softmax 概率分布
5. 计算误差指标和分类一致性

## 输入参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 模型名称 | timm 模型名称 | wide_resnet50_2.racm_in1k |
| 测试图片 | 任意 RGB 图像路径 | 自动生成/下载 |
| 推理设备 | cpu / npu | cpu + npu 对比 |
| num_runs | 推理重复次数 | 10 |

## 输出结果

| 输出 | 格式 | 说明 |
|------|------|------|
| results_{device}.json | JSON | CPU/NPU 推理结果 |
| compare_results.json | JSON | 精度对比数据 |
| screenshot.html | HTML | 模拟终端截图 |
| 模型仓库 | GitCode repo | 包含所有交付件 |

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

## 模型标签

发布仓库时建议包含以下标签：
- `#+NPU`：在昇腾 NPU 上完成适配验证
- `#+CV`：计算机视觉模型
- `#+昇腾`：华为昇腾平台
- `#+图像分类`：图像分类任务
- `#+timm`：基于 timm 框架
- `#+WideResNet`：Wide ResNet 架构

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| HuggingFace 连接超时 | 网络限制 | 从 ModelScope 下载或使用 `pretrained=False` + 本地加载 |
| 模型加载失败 | 权重路径错误 | 检查 modelscope 缓存路径，使用本地 `.safetensors` 加载 |
| NPU OOM | 模型过大或 batch 过大 | batch_size=1，串行执行多个模型 |
| git push 超时 | gitcode.com 不可达 | 使用 API 上传文件替代 |
