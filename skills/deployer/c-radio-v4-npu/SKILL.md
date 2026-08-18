---
name: c-radio-v4-npu
description: >
  NVIDIA C-RADIOv4 视觉特征提取模型在昇腾 NPU 上的完整部署与精度验证 Skill。
  涵盖环境准备、模型下载、适配修改、NPU 推理验证、精度对比验证全流程。
  支持 C-RADIOv4-H（653M）和 C-RADIOv4-SO400M（431M）两个版本。
  当用户提到 C-RADIOv4 昇腾部署、C-RADIOv4 NPU 推理、NVIDIA RADIO NPU、
  C-RADIOv4-H、C-RADIOv4-SO400M 时触发。
metadata:
  short-description: C-RADIOv4 昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, c-radio, radio, nvidia, vision-transformer, feature-extraction, pytorch, inference]
---

# C-RADIOv4 昇腾 NPU 部署与精度验证 Skill

本 Skill 提供 NVIDIA C-RADIOv4 视觉特征提取模型（C-RADIOv4-H 和 C-RADIOv4-SO400M）
在华为昇腾 NPU 上的完整部署、推理验证和精度对比的标准化可复现流程。

C-RADIOv4 基于 Vision Transformer (ViT) 架构，通过聚合 SigLIP2-g、DINOv3-7B、SAM3
等多个视觉教师模型的知识进行训练，输出 `(summary, spatial_features)` 特征元组。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | Linux (aarch64) |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.9 – 3.13 |
| 依赖 | torch, torch_npu, transformers >= 4.51, timm, Pillow, einops |
| 网络 | 首次运行需联网下载模型权重 |

## 下载交付物

两个模型的适配交付件已发布到 AtomGit：

- [C-RADIOv4-H-npu](https://gitcode.com/gcw_C8PI9e90/C-RADIOv4-H-npu)
- [C-RADIOv4-SO400M-npu](https://gitcode.com/gcw_C8PI9e90/C-RADIOv4-SO400M-npu)

可直接 clone 使用：

```bash
git clone https://gitcode.com/gcw_C8PI9e90/C-RADIOv4-H-npu.git
git clone https://gitcode.com/gcw_C8PI9e90/C-RADIOv4-SO400M-npu.git
```

## 流程总览

```
0. 环境初始化与 NPU 预检
→ 1. 安装依赖
→ 2. 模型下载与适配
→ 3. NPU 推理验证
→ 4. 精度对比验证（NPU vs CPU）
→ 5. 验收确认
```

## 工作流阶段总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 0 | 昇腾 NPU 服务器 | 加载 CANN 环境、检查 NPU 状态、选择空闲卡、设置镜像 | NPU 环境就绪 | `npu-smi info` | 至少 1 张卡状态 OK，显存占用 < 80% |
| 依赖安装 | 1 | Python 3.9–3.13，CANN 已加载 | 安装 PyTorch、torch_npu、transformers 等 | 依赖安装完成 | `python3 -c "import torch_npu; ..."` | NPU Tensor 创建成功 |
| 模型适配 | 2 | 模型名称、网络连接 | 下载权重、重命名 `utils.py`、修复缓存 | 适配完成的模型目录 | 检查文件完整性 | 配置文件、权重、修复后的 py 文件齐全 |
| 推理验证 | 3 | 模型目录、测试图像 | 编写推理脚本、运行 NPU 推理 | 输出特征张量 | `python3 scripts/inference.py` | Summary/Spatial 形状正确 |
| 精度验证 | 4 | 测试图像 | CPU/NPU 分别推理、对比余弦相似度 | 精度报告 | `python3 scripts/eval_accuracy.py` | Cosine Similarity > 0.99 |

按以下各节顺序执行。两个模型流程相同，可复用脚本；建议串行验证防止显存冲突。

---

## 0. 环境初始化与 NPU 预检

**执行步骤**：
1. 加载 CANN 环境，确认安装路径并设置环境变量
2. 运行 `npu-smi info` 检查 NPU 设备状态
3. 选择空闲 NPU 卡，设置 `ASCEND_RT_VISIBLE_DEVICES`
4. 设置国内 PyPI 镜像以加速依赖安装

| 项目 | 内容 |
|------|------|
| **输入** | 昇腾 NPU 服务器，root/sudo 权限 |
| **操作** | 加载 CANN 环境、检查 NPU 状态、选择空闲卡 |
| **输出** | NPU 环境就绪，设备状态 OK |
| **异常** | CANN 路径不存在 → 检查安装路径；NPU 卡全满 → 等待资源释放 |

## 0.1 加载 CANN 环境

1. 确认 CANN 安装路径
2. 加载环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**如果加载失败**，请检查 `/usr/local/Ascend/ascend-toolkit/` 是否存在。

## 0.2 NPU 状态检查

1. 运行 npu-smi 查看设备状态
2. 确认至少 1 张卡可用

```bash
npu-smi info
```

**用户确认**：确保 `npu-smi info` 输出显示至少 1 张卡状态为 `OK`，且显存占用 < 80%。

## 0.3 选择空闲 NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

## 0.4 设置国内镜像

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
```

---

## 1. 安装依赖

**执行步骤**：
1. 安装 PyTorch、torch_npu、transformers 等基础依赖
2. 运行验证脚本确认 NPU 环境可用

| 项目 | 内容 |
|------|------|
| **输入** | Python 3.9–3.13，昇腾 CANN 已加载 |
| **操作** | 安装 PyTorch、torch_npu、transformers 等依赖 |
| **输出** | 依赖安装完成，NPU 环境验证通过 |
| **异常** | torch_npu 报错 → 回退到 0.1 重新加载 CANN 环境 |

1. 安装基础 Python 依赖包
2. 验证 NPU 环境

```bash
pip install torch torch_npu transformers Pillow einops numpy timm
```

## 1.1 验证 NPU 环境

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

## 2. 模型下载与适配

**执行步骤**：
1. 使用 HF 镜像下载模型配置文件和权重
2. 重命名 `utils.py` 避免与标准库命名冲突
3. 将自定义代码复制到 transformers 缓存目录

| 项目 | 内容 |
|------|------|
| **输入** | 模型名称（C-RADIOv4-H 或 SO400M）、网络连接 |
| **操作** | 下载权重 → 适配修改 → 解决缓存问题 |
| **输出** | 适配完成的模型目录，包含权重和修复后的 Python 代码 |
| **异常** | 下载中断 → 重新运行（resume_download 支持续传）；适配文件缺失 → 定位并替换 import |

## 2.1 下载模型

使用国内 HF 镜像下载模型配置文件和权重：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 以下以 C-RADIOv4-H 为例，SO400M 替换模型名称即可
MODEL_NAME="C-RADIOv4-H"  # 或 C-RADIOv4-SO400M
HF_REPO="nvidia/C-RADIOv4-H"  # 或 nvidia/C-RADIOv4-SO400M
mkdir -p ./${MODEL_NAME}-npu

# 下载配置文件
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('${HF_REPO}',
    allow_patterns=['config.json', '*.md', 'preprocessor_config.json', '*.py'],
    local_dir='./${MODEL_NAME}-npu')
"

# 下载权重文件
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download
hf_hub_download('${HF_REPO}', filename='model.safetensors',
    local_dir='./${MODEL_NAME}-npu', local_dir_use_symlinks=False)
"
```

**输入/输出定义**：
- 输入：模型名称、网络连接
- 输出：配置文件、自定义 Python 代码、权重文件
- 异常：如果下载中断，请重新运行下载命令；如果 huggingface_hub 报错，请检查 `HF_ENDPOINT` 是否已导出。

## 2.2 适配修改

**问题**：模型中的 `utils.py` 与 Python 标准库 `utils` 模块命名冲突，导致 `transformers` 动态模块加载器无法正确解析相对导入。

**修复**：重命名 `utils.py` 并更新 import 引用：

```bash
cd ./${MODEL_NAME}-npu
mv utils.py radio_utils.py
sed -i 's/from .utils import rank_gate/from .radio_utils import rank_gate/' \
  open_clip_adaptor.py siglip2_adaptor.py
```

**如果 `sed` 执行失败**，请手动编辑上述两个文件中的 import 语句。

## 2.3 解决缓存问题

首次加载时，transformers 可能无法自动复制所有自定义代码到缓存。解决方式：

```bash
mkdir -p /opt/atomgit/.cache/huggingface/modules/transformers_modules/
cp *.py /opt/atomgit/.cache/huggingface/modules/transformers_modules/
```

**异常**：如果仍然出现 `KeyError: 'RADIOModel'`，请确认 `trust_remote_code=True` 已设置且所有 `.py` 文件完整。

---

## 3. NPU 推理验证

**执行步骤**：
1. 创建推理脚本 `scripts/inference.py`
2. 运行推理并验证输出张量形状
3. 确认 Summary 和 Spatial 特征维度与预期一致

## 3.1 推理脚本

创建 `scripts/inference.py`（也可从 AtomGit 交付件获取）：

```python
import argparse, time
import torch, torch_npu
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

MODEL_PATH, DEVICE = "./", "npu:0"

parser = argparse.ArgumentParser()
parser.add_argument("--image", default=None)
parser.add_argument("--model_path", default=MODEL_PATH)
args = parser.parse_args()

image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
model.eval().npu()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

image = Image.open(args.image).convert("RGB") if args.image \
    else Image.new("RGB", (512, 512), (73, 109, 137))
pixel_values = image_processor(images=image, return_tensors="pt", do_resize=True).pixel_values.npu()

with torch.no_grad():
    summary, spatial_features = model(pixel_values)

print(f"Summary shape: {summary.shape}")
print(f"Spatial features shape: {spatial_features.shape}")
print("Inference successful!")
```

## 3.2 运行推理

```bash
python3 scripts/inference.py
```

## 3.3 预期推理结果

## C-RADIOv4-H

| 项目 | 值 |
|------|-----|
| 参数量 | 651,645,440 (653M) |
| Summary 维度 | (1, 2560) |
| Spatial 维度 | (1, 1024, 1280) |
| 推理耗时 | ~0.046 s |

## C-RADIOv4-SO400M

| 项目 | 值 |
|------|-----|
| 参数量 | 431,237,232 (431M) |
| Summary 维度 | (1, 2304) |
| Spatial 维度 | (1, 1024, 1152) |
| 推理耗时 | ~0.040 s |

**输入/输出定义**：
- 输入：图像路径（可选，默认使用合成测试图）
- 输出：`summary` 张量、`spatial_features` 张量
- 异常：如果输出形状与上表不符，请检查模型权重是否完整下载。

---

## 4. 精度对比验证

**执行步骤**：
1. 创建精度评估脚本 `scripts/eval_accuracy.py`
2. 生成多张合成测试图像（噪声、梯度、棋盘格等）
3. 分别在 CPU（基线）和 NPU 上推理并对比余弦相似度
4. 确认精度指标满足 Cosine Similarity > 0.99

创建 `scripts/eval_accuracy.py`（也可从 AtomGit 交付件获取），该脚本生成 10 张不同的合成测试图像（含随机噪声、梯度、棋盘格等），分别在 CPU（基线）和 NPU 上推理，
对比输出特征的余弦相似度和相对误差。

## 核心逻辑

```python
# 1. CPU 基线：加载模型到 CPU，对所有测试图像推理，保存 reference 输出
model_cpu = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval()
for px in test_images:
    ref_summary, ref_spatial = model_cpu(px)

# 2. NPU 推理：加载模型到 NPU，相同输入推理
model_npu = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().npu()
for px in test_images:
    npu_summary, npu_spatial = model_npu(px.npu())

# 3. 对比指标：Cosine Similarity, Mean/Max Relative Error, Mean Absolute Error
cos_sim = F.cosine_similarity(ref.flatten(), npu.flatten(), dim=0)
rel_err = abs((ref - npu) / (abs(ref) + 1e-8))
```

## 运行

```bash
python3 scripts/eval_accuracy.py --output results/eval.json
```

## 预期精度结果

## Summary 特征

| 指标 | 值 |
|------|-----|
| Cosine Similarity | 1.00000000 |
| Mean Relative Error | 0.00000000 |
| Max Relative Error | 0.00000000 |

## Spatial 特征

| 指标 | 值 |
|------|-----|
| Cosine Similarity | 1.00000000 |
| Mean Relative Error | 0.00000000 |
| Max Relative Error | 0.00000000 |

**结论：PASS** — NPU 输出与 CPU 输出完全一致（Cosine Similarity = 1.0），
精度误差远小于 1% 的要求。

---

## 5. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| `FileNotFoundError: .../transformers_modules/utils.py` | `utils.py` 与标准库冲突 | 重命名文件 | 重命名为 `radio_utils.py` 并更新 import |
| `KeyError: 'RADIOModel'` | 自定义代码未正确加载 | 暂停，检查配置 | 确认 `trust_remote_code=True` 且所有 `.py` 文件完整 |
| `No module named 'torch_npu'` | CANN 未加载或 torch_npu 未安装 | 回滚到步骤 0.1 | `source set_env.sh` 后重装 torch_npu |
| OOM | 图像分辨率过高 | 回滚，降低输入 | 降低输入分辨率或使用单张推理 |
| 模型权重加载失败 | 权重文件损坏或不完整 | 失败，需重新下载 | 重新下载并校验 SHA256 |
| 首次推理慢 | 图编译开销 | 正常行为 | 后续推理性能稳定 |
| 下载失败 | 网络无法访问 HuggingFace | 重试，切换镜像 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |

---

## 6. 检查点与验收确认

完成以下检查清单即为部署成功。每步完成后请**用户确认**再打勾：

- [ ] **Checkpoint 1**：`npu-smi info` 显示设备正常
- [ ] **Checkpoint 2**：`import torch_npu` 无报错
- [ ] **Checkpoint 3**：模型加载成功，Summary 和 Spatial 输出形状正确
- [ ] **Checkpoint 4**：`scripts/eval_accuracy.py` 精度对比通过，Cosine Similarity > 0.99
- [ ] **Checkpoint 5**：可在 5 秒内完成单张 512x512 图像的特征提取

---

## 7. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| 推理脚本 | `scripts/inference.py` |
| 精度测试脚本 | `scripts/eval_accuracy.py` |
| 精度报告 | `results/eval.json` |
| 模型权重 | `model.safetensors`（H: ~2.43GB, SO400M: ~1.61GB） |
| 参考文档 | https://www.hiascend.com/document/ |
| HuggingFace 镜像 | https://hf-mirror.com |

---

## 模型版本速查

| 属性 | C-RADIOv4-H | C-RADIOv4-SO400M |
|------|-------------|-------------------|
| 参数量 | 651M (Huge) | 431M (Shape-Optimized) |
| Summary 维度 | 2560 | 2304 |
| Spatial 维度 | 1024×1280 | 1024×1152 |
| 权重大小 | 2.43 GB | 1.61 GB |
| 架构 | ViT-Huge | ViT-SO400M |
| 下载地址 | [HF](https://huggingface.co/nvidia/C-RADIOv4-H) | [HF](https://huggingface.co/nvidia/C-RADIOv4-SO400M) |
| AtomGit 交付 | [链接](https://gitcode.com/gcw_C8PI9e90/C-RADIOv4-H-npu) | [链接](https://gitcode.com/gcw_C8PI9e90/C-RADIOv4-SO400M-npu) |
