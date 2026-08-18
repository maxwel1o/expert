---
name: git-npu-deploy
description: >
  微软 GIT（GenerativeImage2Text）系列模型在昇腾 NPU 上的完整部署与
  推理验证 Skill。涵盖环境准备、依赖安装、transfer_to_npu 自动迁移、
  单图推理、精度评测（NPU vs CPU）、性能基准测试的全流程。
  支持 5 个 GIT 变体：git-base / git-base-coco / git-large /
  git-large-coco / git-large-textcaps。可在任意 Ascend910 系列
  服务器上一键复现。当用户提到 GIT NPU、GIT 昇腾、图像描述 NPU 时触发。
metadata:
  short-description: GIT 系列图像描述模型昇腾 NPU 部署与验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, git, image-captioning, transformers, pytorch, inference]
---

# GIT 系列图像描述模型昇腾 NPU 部署与推理验证 Skill

本 Skill 提供微软 GIT（GenerativeImage2Text）系列图像描述模型在华为昇腾 NPU 上的完整部署、推理验证和精度评测的标准化可复现流程。

## 支持模型

| 模型 | 参数量 | 微调数据集 | 描述 |
|------|--------|-----------|------|
| `git-base` | ~177M | - | 基础版本，通用图像描述 |
| `git-base-coco` | ~177M | COCO Captions | COCO 数据集微调，自然图像 |
| `git-large` | ~677M | - | 大版本，通用图像描述 |
| `git-large-coco` | ~677M | COCO Captions | COCO 数据集微调，自然图像 |
| `git-large-textcaps` | ~677M | TextCaps | TextCaps 数据集微调，文本感知 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（~0.7-2.8GB） |

## 流程总览

```
0. 环境初始化与 NPU 预检
→ 1. 安装依赖（torch_npu + transformers + Pillow）
→ 2. NPU 基础验证
→ 3. 模型下载
→ 4. 单图推理验证
→ 5. 精度评测（NPU vs CPU 对比）
→ 6. 性能基准测试
→ 7. 验收确认
```

按以下各节顺序执行，每步完成后再进入下一步。

---

## 0. 环境初始化与 NPU 预检

| 项目 | 内容 |
|------|------|
| **输入** | 昇腾 NPU 服务器，CANN 已安装 |
| **操作** | 加载 CANN 环境、检查 NPU 状态、选择空闲卡、设置国内镜像 |
| **输出** | NPU 环境就绪，设备状态 OK，国内镜像已配置 |
| **异常** | CANN 路径不存在 → 检查安装路径；NPU 卡全满 → 等待资源释放 |

### 0.1 加载 CANN 环境

1. 确认 CANN 安装路径
2. 加载环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**如果加载失败**，请检查 CANN 安装路径是否正确。

### 0.2 NPU 状态检查

1. 运行 npu-smi 查看设备状态
2. 确认至少 1 张卡可用

```bash
npu-smi info
```

**用户确认**：确保 `npu-smi info` 输出显示至少 1 张卡状态为 `OK`。

### 0.3 选择空闲 NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

### 0.4 设置国内镜像（可选）

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 1. 安装依赖

| 项目 | 内容 |
|------|------|
| **输入** | Python 3.9–3.13，昇腾 CANN 已加载 |
| **操作** | 安装 torch、torch_npu、transformers、Pillow |
| **输出** | 依赖安装完成，版本验证通过 |
| **异常** | torch_npu 报错 → 回退到 0.1 重新加载 CANN 环境 |

1. 安装基础 Python 依赖包
2. 验证安装版本

```bash
pip install torch torch_npu transformers pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 1.1 验证安装版本

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```

**如果报错 `No module named 'torch_npu'`**，说明 CANN 环境未加载或 torch_npu 未安装，请回滚到步骤 0.1 重新加载环境后执行 `pip install torch_npu`。

**已验证环境**：

| 组件 | 版本 |
|------|------|
| `torch` | `2.9.0+cpu` |
| `torch-npu` | `2.9.0.post1+gitee7ba04` |
| `transformers` | `4.57.6` |
| `Pillow` | `12.2.0` |
| CANN | `8.5.1` |
| NPU | `Ascend910B4` |

---

## 2. NPU 基础验证

在依赖安装完成后，运行以下 Python 代码确认 NPU 环境可用：

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
print('NPU available:', torch.npu.is_available())
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错，`NPU available: True`。

**输入/输出定义**：
- 输入：已安装 torch_npu 的 Python 环境
- 输出：NPU 设备信息和张量运算结果
- 异常：如果 `torch.npu.is_available()` 返回 `False`，请检查 CANN 驱动和固件版本。

---

## 3. 模型下载

选择一个目标模型，使用 HuggingFace 国内镜像下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 以下命令以下载 git-large 为例，替换 MODEL_ID 为对应模型名
# 支持的模型: microsoft/git-base, microsoft/git-base-coco,
#            microsoft/git-large, microsoft/git-large-coco,
#            microsoft/git-large-textcaps

MODEL_ID="microsoft/git-large"
LOCAL_DIR="./git-large-npu"

# 下载配置文件
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}',
    allow_patterns=['config.json', '*.md', '*tokenizer*', '*.json',
                    'vocab.*', 'special_tokens*', '*.txt',
                    'generation*', 'preprocessor*'],
    local_dir='${LOCAL_DIR}')
"

# 下载权重文件（pytorch_model.bin 或 model.safetensors 视模型而定）
wget -c "https://hf-mirror.com/${MODEL_ID}/resolve/main/pytorch_model.bin" \
  -P "${LOCAL_DIR}" 2>/dev/null || \
wget -c "https://hf-mirror.com/${MODEL_ID}/resolve/main/model.safetensors" \
  -P "${LOCAL_DIR}"
```

**注意**：`git-base-coco`、`git-large`、`git-large-textcaps` 使用 `pytorch_model.bin`；`git-large-coco` 使用 `model.safetensors`。

**输入/输出定义**：
- 输入：模型 ID、网络连接
- 输出：配置文件、tokenizer 文件、权重文件
- 异常：如果下载超时，请检查 `HF_ENDPOINT` 是否已设置为镜像地址，或尝试重试 `wget -c`。

---

## 4. 单图推理验证

使用本 Skill 提供的 `scripts/inference.py` 脚本进行单图推理：

```bash
# NPU 推理
python3 scripts/inference.py \
  --model_path /path/to/git-model-npu \
  --image_path /path/to/image.jpg \
  --device npu:0

# CPU 推理（对比用）
python3 scripts/inference.py \
  --model_path /path/to/git-model-npu \
  --image_path /path/to/image.jpg \
  --device cpu

# Benchmark 模式（带 warmup）
python3 scripts/inference.py \
  --model_path /path/to/git-model-npu \
  --image_path /path/to/image.jpg \
  --device npu:0 --benchmark
```

### 推理流程

1. `AutoProcessor` 加载图像并预处理（resize 至 224×224）
2. 将 `pixel_values` 移至 NPU 设备
3. 调用 `model.generate()` 自回归生成文本 token
4. `processor.batch_decode()` 将 token 序列解码为文本描述

### 代码示例

```python
import torch
import torch_npu
from PIL import Image
from transformers import AutoProcessor, GitForCausalLM

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
model = GitForCausalLM.from_pretrained("microsoft/git-large-coco")
model.to("npu:0")
model.eval()

image = Image.open("example.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to("npu:0")

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50)

caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(caption)
```

**通过标准**：
- 输出合理的图像描述文本
- 无 `torch_npu` 相关报错
- Benchmark 模式输出推理耗时

**输入/输出定义**：
- 输入：图像路径（JPG/PNG）、模型路径、设备类型
- 输出：图像描述文本字符串
- 异常：如果输出乱码，请检查 tokenizer 配置是否完整下载。

---

## 5. 精度评测

使用本 Skill 提供的 `scripts/accuracy.py` 脚本进行精度评测，对比 NPU 与 CPU 推理结果：

```bash
python3 scripts/accuracy.py \
  --model_path /path/to/git-model-npu \
  --device npu:0
```

### 评测方法

1. **Encoder 数值对比**：对比 Vision Encoder（`model.git.image_encoder`）在 NPU 和 CPU 上的输出差异
2. **NPU vs CPU 文本一致性**：NPU 与 CPU 生成的图像描述是否完全一致
3. **NPU 自一致性**：NPU 两次推理是否产生相同结果

### 实测精度数据

在 Ascend910B4 (CANN 8.5.1) 上各模型的实测精度：

| 模型 | Encoder 最大差异 | Encoder 平均差异 | 文本一致率 |
|------|------------------|------------------|-----------|
| `git-base` | 0.6964 | 0.00094 | 100% (5/5) |
| `git-base-coco` | < 1.0 | < 0.01 | 100% (5/5) |
| `git-large` | 0.6839 | 0.00169 | 100% (5/5) |
| `git-large-coco` | 0.2082 | 0.0020 | 100% (5/5) |
| `git-large-textcaps` | 0.6117 | 0.0045 | 100% (5/5) |

**通过标准**：文本一致率 100%，Encoder 余弦/数值差异在浮点误差范围内。

---

## 6. 性能基准测试

使用本 Skill 提供的 `scripts/benchmark.py` 脚本进行性能测试：

```bash
# NPU 性能测试（10 次运行）
python3 scripts/benchmark.py \
  --model_path /path/to/git-model-npu \
  --device npu:0 --runs 10

# CPU 性能测试（对比）
python3 scripts/benchmark.py \
  --model_path /path/to/git-model-npu \
  --device cpu --runs 5
```

### 实测性能数据

在 Ascend910B4 上各模型的性能参考：

| 模型 | NPU 推理耗时 | CPU 推理耗时 | 加速比 |
|------|-------------|-------------|--------|
| `git-base` | ~0.5-1.5s | ~2-5s | ~3-4x |
| `git-base-coco` | ~0.5-1.5s | ~2-5s | ~3-4x |
| `git-large` | ~1-3s | ~4-8s | ~3x |
| `git-large-coco` | ~1-3s | ~4-8s | ~3x |
| `git-large-textcaps` | ~1-3s | ~4-8s | ~3x |

> 注：具体性能数据取决于实际运行环境和输入图像大小。上述数据为单次 warm 推理耗时，max_new_tokens=50。

---

## 7. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| `No module named 'torch_npu'` | CANN 未加载或 torch_npu 未安装 | 暂停执行，回滚到步骤 0.1 | `source set_env.sh` 后重装 torch_npu |
| NPU 不可用 | 卡被占用或 CANN 未初始化 | 失败，需用户确认 | `npu-smi info` 检查，换空闲卡 |
| 模型下载失败 | 网络问题 | 重试，切换镜像 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| OOM | 序列过长或 batch 过大 | 回滚，减小输入 | 单图推理无需 batch |
| 多卡抢占冲突 | 默认都用 0 号卡 | 失败，需用户确认 | `npu-smi info` 选空闲卡 |
| `model.git` 报错 | NPU 上 embedding 层不支持 float 输入 | 使用替代 API | 使用 `model.git.image_encoder(...)` 获取视觉特征 |
| 首次推理慢 | 图编译开销 | 正常行为 | 后续推理性能稳定 |

---

## 8. 检查点与验收确认

完成以下检查清单即为部署成功。每步完成后请**用户确认**再打勾：

- [ ] **Checkpoint 1**：`npu-smi info` 显示设备正常
- [ ] **Checkpoint 2**：`import torch_npu` 无报错，`torch.npu.is_available()` 返回 True
- [ ] **Checkpoint 3**：`scripts/inference.py` 输出合理的图像描述
- [ ] **Checkpoint 4**：`scripts/accuracy.py` 文本一致率达到 100%（5/5）
- [ ] **Checkpoint 5**：`scripts/benchmark.py` 能正常完成性能测试
- [ ] **Checkpoint 6**：NPU 和 CPU 推理结果完全一致

---

## 9. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| 推理脚本 | `scripts/inference.py` |
| 精度评测脚本 | `scripts/accuracy.py` |
| 性能测试脚本 | `scripts/benchmark.py` |
| 精度报告 | `results/eval.json` |
| 性能报告 | `results/perf_report.json` |
| 参考文档 | https://www.hiascend.com/document/ |
| HuggingFace 镜像 | https://hf-mirror.com |

---

## 10. 多模型切换说明

所有脚本均通过 `--model_path` 参数指定不同模型，无需修改代码。不同模型的区别：

| 差异点 | 说明 |
|--------|------|
| 模型大小 | Base (~177M) vs Large (~677M) 影响推理速度 |
| 微调数据集 | COCO / TextCaps 影响生成结果的风格和领域 |
| 权重格式 | `pytorch_model.bin` 或 `model.safetensors`，均由 `from_pretrained` 自动处理 |
| 生成行为 | 可通过 `model.generate()` 的 `max_new_tokens`、`num_beams` 等参数调节 |

---

## 附录：GIT 模型 NPU 适配要点

| 特征 | GIT 实现 | NPU 适配说明 |
|------|---------|-------------|
| Vision Encoder | ViT (GitVisionModel) | 直接调用 `model.git.image_encoder` |
| Text Decoder | CausalLM (GitForCausalLM) | `model.generate()` 自动在 NPU 上运行 |
| 图像预处理 | AutoProcessor (224×224) | 与 CPU 行为一致，无差异 |
| LayerNorm | `nn.LayerNorm` | 无需替换，原生支持 |
| 激活函数 | GELU | 原生支持 |
| 推理精度 | FP32 | 与 CPU FP32 精度一致 |
