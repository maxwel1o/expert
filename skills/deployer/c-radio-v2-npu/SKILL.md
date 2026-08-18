---
name: c-radio-v2-npu
description: >
  NVIDIA C-RADIOv2 系列视觉特征提取模型（B/L/H/g）在昇腾 NPU 上的完整部署与精度验证 Skill。
  涵盖环境准备、模型下载、NPU 推理验证、精度对比验证全流程。
  支持 C-RADIOv2-B（90M）、C-RADIOv2-L（320M）、C-RADIOv2-H（653M）和
  C-RADIOv2-g（1.1B）四个版本。
  当用户提到 C-RADIOv2 昇腾部署、C-RADIOv2 NPU 推理、NVIDIA RADIO NPU、
  C-RADIOv2-B、C-RADIOv2-L、C-RADIOv2-H、C-RADIOv2-g 时触发。
metadata:
  short-description: C-RADIOv2 昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, c-radio, radio, nvidia, vision-transformer, feature-extraction, pytorch, inference]
---

# C-RADIOv2 昇腾 NPU 部署与精度验证 Skill

本 Skill 提供 NVIDIA C-RADIOv2 系列视觉特征提取模型在华为昇腾 NPU 上的
完整部署、推理验证和精度对比的标准化可复现流程。

C-RADIOv2 基于 Vision Transformer (ViT) 架构，通过聚合多个视觉教师模型的知识
进行训练，输出 `(summary, spatial_features)` 特征元组。该系列包含四个版本：

| 模型 | 参数量 | Summary 维度 | Spatial 维度 | 权重大小 |
|------|--------|-------------|-------------|---------|
| C-RADIOv2-B | 90M | 2304 | 729×768 | 0.37 GB |
| C-RADIOv2-L | 320M | 3072 | 729×1024 | 1.19 GB |
| C-RADIOv2-H | 653M | 3840 | 729×1280 | 2.43 GB |
| C-RADIOv2-g | 1.1B | 3072 | 196×1536 | 6.05 GB |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | Linux (aarch64) |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.9 – 3.13 |
| 依赖 | torch, torch_npu, transformers >= 4.47, timm >= 0.9.0, Pillow, numpy, huggingface_hub |
| 网络 | 首次运行需联网下载模型权重（建议使用 hf-mirror.com 镜像） |

## 下载交付物

四个模型的适配交付件已发布到 AtomGit：

- [C-RADIOv2-B-npu](https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-B-npu)
- [C-RADIOv2-L-npu](https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-L-npu)
- [C-RADIOv2-H-npu](https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-H-npu)
- [C-RADIOv2-g-npu](https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-g-npu)

可直接 clone 使用（权重需另行下载）：

```bash
git clone https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-B-npu.git
git clone https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-L-npu.git
git clone https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-H-npu.git
git clone https://gitcode.com/gcw_C8PI9e90/C-RADIOv2-g-npu.git
```

## 流程总览

```
0. 环境初始化与 NPU 预检
→ 1. 安装依赖
→ 2. 模型下载
→ 3. NPU 推理验证
→ 4. 精度对比验证（NPU vs CPU）
→ 5. 验收确认
```

按以下各节顺序执行。四个模型流程相同，建议串行验证防止显存冲突。

---

## 0. 环境初始化与 NPU 预检

| 项目 | 内容 |
|------|------|
| **输入** | 昇腾 NPU 服务器，root/sudo 权限 |
| **操作** | 加载 CANN 环境、检查 NPU 状态、选择空闲卡 |
| **输出** | NPU 环境就绪，设备状态 OK |
| **异常** | CANN 路径不存在 → 检查安装路径；NPU 卡全满 → 等待资源释放 |

### 0.1 加载 CANN 环境

1. 确认 CANN 安装路径
2. 加载环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**如果加载失败**，请检查 `/usr/local/Ascend/ascend-toolkit/` 是否存在。

### 0.2 NPU 状态检查

1. 运行 npu-smi 查看设备状态
2. 确认至少 1 张卡可用

```bash
npu-smi info
```

**用户确认**：确保 `npu-smi info` 输出显示至少 1 张卡状态为 `OK`，且显存占用 < 80%。

### 0.3 选择空闲 NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

### 0.4 设置国内镜像

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
```

---

## 1. 安装依赖

| 项目 | 内容 |
|------|------|
| **输入** | Python 3.9–3.13，昇腾 CANN 已加载 |
| **操作** | 安装 PyTorch、torch_npu、transformers、timm 等依赖 |
| **输出** | 依赖安装完成，NPU 环境验证通过 |
| **异常** | torch_npu 报错 → 回退到 0.1 重新加载 CANN 环境 |

1. 安装基础 Python 依赖包
2. 额外安装 C-RADIOv2-g 需要的 open_clip_torch
3. 验证 NPU 环境

```bash
pip install torch torch_npu transformers timm Pillow numpy huggingface_hub
```

注意：C-RADIOv2-g 需要额外安装 open_clip_torch：

```bash
pip install open_clip_torch
```

### 1.1 验证 NPU 环境

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor，且无报错。

**如果报错 `No module named 'torch_npu'`**，请回滚到步骤 0.1 重新加载环境后执行 `pip install torch_npu`。

---

## 2. 模型下载

| 项目 | 内容 |
|------|------|
| **输入** | 模型名称（C-RADIOv2-B/L/H/g）、网络连接、HF_ENDPOINT 镜像设置 |
| **操作** | 使用 huggingface_hub SDK 从镜像站下载模型 |
| **输出** | config.json、model.safetensors、自定义 Python 代码到本地目录 |
| **异常** | 下载中断 → 重新运行（resume_download=True 支持续传）；HF_ENDPOINT 未设置 → 检查环境变量 |

使用国内 HF 镜像下载模型配置文件和权重（以 C-RADIOv2-B 为例，其他模型替换名称即可）：

```bash
export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME="C-RADIOv2-B"  # 可选: C-RADIOv2-B/L/H/g
HF_REPO="nvidia/C-RADIOv2-B"
LOCAL_DIR="./${MODEL_NAME}-npu"
mkdir -p $LOCAL_DIR

# 下载全部文件（配置 + 权重）
python3 << 'PY'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download, snapshot_download

model_id = "nvidia/C-RADIOv2-B"
local_dir = "./C-RADIOv2-B-npu"

snapshot_download(model_id, allow_patterns=["*.json", "*.py", "*.md"], local_dir=local_dir)
path = hf_hub_download(model_id, "model.safetensors", cache_dir="./cache", resume_download=True)
import shutil
shutil.copy(path, os.path.join(local_dir, "model.safetensors"))
print("Download complete!")
PY
```

> 注意：C-RADIOv2-g 约 6GB，下载时间较长（镜像站约 30MB/s，2-3 分钟）。

**输入/输出定义**：
- 输入：模型名称、网络连接
- 输出：`config.json`、`model.safetensors`、自定义 Python 代码
- 异常：如果下载中断，请重新运行下载脚本；如果 huggingface_hub 报错，请检查 `HF_ENDPOINT` 是否已导出。

---

## 3. NPU 推理验证

| 项目 | 内容 |
|------|------|
| **输入** | 模型权重文件、模型目录 |
| **操作** | 加载模型到 NPU，运行推理，验证输出形状 |
| **输出** | summary 和 spatial_features 张量 |
| **异常** | 输出形状不匹配 → 检查权重文件是否完整；OOM → 降低输入分辨率 |

### 3.1 推理脚本

每个模型目录自带 `scripts/inference.py`，也可参考以下内容创建：

```python
import argparse, os, time
import torch, torch_npu
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# 预填充 transformers 缓存（解决自定义代码相对导入问题）
from model_loader import prepare_cache
prepare_cache(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("npu:0")
processor = AutoImageProcessor.from_pretrained("./", trust_remote_code=True)
model = AutoModel.from_pretrained("./", trust_remote_code=True).to(device)
model.eval()

image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

with torch.no_grad():
    outputs = model(pixel_values)

if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
    summary, spatial_features = outputs[0], outputs[1]
    print(f"Summary shape:     {tuple(summary.shape)}")
    print(f"Spatial shape:     {tuple(spatial_features.shape)}")
```

### 3.2 运行推理

```bash
cd ./C-RADIOv2-B-npu
HF_MODULES_CACHE=/opt/atomgit/.cache/huggingface/modules \
HF_ENDPOINT=https://hf-mirror.com \
python3 scripts/inference.py
```

### 3.3 预期推理结果

| 模型 | Summary 维度 | Spatial 维度 | 推理耗时 |
|------|-------------|-------------|---------|
| C-RADIOv2-B | (1, 2304) | (1, 729, 768) | ~0.33 s |
| C-RADIOv2-L | (1, 3072) | (1, 729, 1024) | ~0.34 s |
| C-RADIOv2-H | (1, 3840) | (1, 729, 1280) | ~0.37 s |
| C-RADIOv2-g | (1, 3072) | (1, 196, 1536) | ~0.41 s |

> 推理耗时取决于 NPU 型号和当前负载，以上为单次推理结果。

**输入/输出定义**：
- 输入：图像（默认随机生成 224×224）
- 输出：`summary` 张量、`spatial_features` 张量
- 异常：如果输出形状与上表不符，请检查模型权重是否完整下载。

---

## 4. 精度对比验证

### 4.1 精度测试脚本

每个模型目录自带 `scripts/accuracy_test.py`，核心逻辑：

1. CPU 基线：模型加载到 CPU，对随机输入推理，记录输出
2. NPU 推理：模型加载到 NPU，相同输入推理
3. 对比指标：Cosine Similarity、MSE、Max Absolute Error、Relative Error

### 4.2 运行

```bash
cd ./C-RADIOv2-B-npu
HF_MODULES_CACHE=/opt/atomgit/.cache/huggingface/modules \
HF_ENDPOINT=https://hf-mirror.com \
python3 scripts/accuracy_test.py --num-runs 3 --output results/eval.json
```

### 4.3 预期精度结果

| 模型 | 指标 | Summary | Spatial | 要求 |
|------|------|---------|---------|------|
| C-RADIOv2-B | Cosine Similarity | 0.999975 | 0.999975 | > 0.99 |
| C-RADIOv2-B | MSE | 3.68e-06 | 8.66e-06 | < 1e-4 |
| C-RADIOv2-L | Cosine Similarity | 0.999974 | 0.999974 | > 0.99 |
| C-RADIOv2-L | MSE | 3.68e-06 | 8.66e-06 | < 1e-4 |
| C-RADIOv2-H | Cosine Similarity | 0.999998 | 1.000001 | > 0.99 |
| C-RADIOv2-H | MSE | 1.62e-07 | 6.74e-07 | < 1e-4 |
| C-RADIOv2-g | Cosine Similarity | 0.999998 | 0.999997 | > 0.99 |
| C-RADIOv2-g | MSE | 5.25e-06 | 4.00e-05 | < 1e-4 |

**结论：全部 PASS** — NPU 输出与 CPU 输出一致性极高（余弦相似度 > 0.9999），满足精度要求。

---

## 5. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| `FileNotFoundError: .../dual_hybrid_vit.py` | 自定义代码未正确复制到 transformers 缓存 | 运行 `model_loader.prepare_cache()` | 或设置 `HF_MODULES_CACHE` 环境变量 |
| `KeyError: 'RADIOModel'` | 自定义代码未正确加载 | 暂停，检查配置 | 确认 `trust_remote_code=True` 且所有 `.py` 文件完整 |
| `No module named 'torch_npu'` | CANN 未加载或 torch_npu 未安装 | 回滚到步骤 0.1 | `source set_env.sh` 后重装 torch_npu |
| `size mismatch for bias` | 权重文件与架构不匹配 | 失败，需重新下载 | 重新从 HF 下载完整的 `model.safetensors` |
| OOM | C-RADIOv2-g 模型较大 | 回滚，降低输入 | 降低输入分辨率或使用单张推理 |
| 下载速度慢 | 直连 huggingface.co 速度慢 | 重试，切换镜像 | 设置 `export HF_ENDPOINT=https://hf-mirror.com` |
| 首次推理慢 | 图编译开销 | 正常行为 | 后续推理性能稳定 |

---

## 6. 检查点与验收确认

完成以下检查清单即为部署成功。每步完成后请**用户确认**再打勾：

- [ ] **Checkpoint 1**：`npu-smi info` 显示设备正常
- [ ] **Checkpoint 2**：`import torch_npu` 无报错
- [ ] **Checkpoint 3**：模型加载成功，Summary 和 Spatial 输出形状正确
- [ ] **Checkpoint 4**：`scripts/accuracy_test.py` 精度对比通过，Cosine Similarity > 0.99
- [ ] **Checkpoint 5**：B/L 模型单次推理 < 0.5s，H/g 模型 < 0.6s

---

## 7. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| 推理脚本 | `scripts/inference.py` |
| 精度测试脚本 | `scripts/accuracy_test.py` |
| 精度报告 | `results/eval.json` |
| 模型缓存 | `/opt/atomgit/.cache/huggingface/modules` |
| 参考文档 | https://www.hiascend.com/document/ |
| HuggingFace 镜像 | https://hf-mirror.com |
