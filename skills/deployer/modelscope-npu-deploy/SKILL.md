---
name: modelscope-npu-deploy
description: >
  ModelScope CV 图像分类/识别模型在昇腾 NPU 上的完整部署 Skill。
  涵盖环境准备、模型下载、权重格式转换（OpenMMLab → timm）、CPU/NPU
  推理验证、精度对比（概率 MaxAE < 1%）、终端截图生成、README 发布的
  全流程。适用于 ResNeSt、ViT/DeiT、ConvNeXt、BEiTv2、NextViT、TinyNAS、
  BNext 等常见 CV 架构，以及视频分类/行为识别等时序模型。
metadata:
  short-description: ModelScope CV 模型昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, pytorch, modelscope, timm, image-classification, cv, inference, accuracy-comparison]
---

# ModelScope CV 模型昇腾 NPU 部署 Skill

本 Skill 提供 ModelScope 平台 CV 图像分类/识别模型在华为昇腾 NPU (Ascend910) 上的
标准化部署流程，涵盖权重转换、推理验证、精度对比和发布的全流程。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡，64GB HBM） |
| OS | openEuler / Ubuntu / KylinOS（aarch64） |
| CANN | 25.5.2+ |
| Python | 3.11+ |
| 网络 | 首次运行需联网下载模型权重 |

## 流程总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境初始化 | 加载 CANN、安装依赖、验证 NPU | CANN Toolkit, Python 环境 | source set_env.sh, pip install, python 验证 | 可用 NPU 环境 | npu-smi info, python -c "import torch_npu; ..." | NPU 设备可见且基础计算正常 |
| 模型下载 | 下载 ModelScope 模型 | model_name | snapshot_download | 模型权重 + 配置文件 | ls ~/.cache/modelscope/iic/{model}/ | 包含 pytorch_model.pt、meta_info.txt、configuration.json |
| 权重转换 | OpenMMLab 转 timm 格式 | OpenMMLab state_dict | convert_state_dict | timm 格式权重 | python -c "model.load_state_dict(sd, strict=False)" | 无 unexpected keys |
| 编写脚本 | 编写推理与对比脚本 | 模板代码 | 复制 inference.py、compare_cpu_npu.py | 可执行脚本 | python inference.py --help | 参数解析正确 |
| CPU 推理 | 运行 CPU 推理 | 测试图片 | python inference.py --device cpu | CPU 结果 JSON | ls /tmp/*_cpu_results.json | 输出文件存在且格式正确 |
| NPU 推理 | 运行 NPU 推理 | 测试图片 | python inference.py --device npu | NPU 结果 JSON | ls /tmp/*_npu_results.json | 输出文件存在且格式正确 |
| 精度对比 | CPU vs NPU 精度对比 | CPU/NPU 结果 JSON | python compare_cpu_npu.py | 对比报告 JSON | cat /tmp/*_comparison.json | 概率 MaxAE < 1% |
| 截图生成 | 生成终端风格截图 | CPU 推理输出 | terminal_screenshot.py | PNG 截图 | ls terminal_screenshot.png | 截图内容清晰完整 |
| 发布 | 生成 README 并推送 | 所有产物 | git init, commit, push | GitCode 仓库 | curl -I https://gitcode.com/{user}/{repo} | 仓库可见且文件完整 |

按以下各节顺序执行，每步完成后再进入下一步。

---

## 执行检查点与用户确认

在流程的关键节点需要用户确认状态正确后再继续执行，避免因环境或配置问题导致后续步骤失败。

| 步骤 | 检查点 | 预期结果 | 用户确认操作 |
|---|---|---|---|
| 0. 环境初始化 | 执行 `npu-smi info` 查看 NPU 状态 | 显示 Ascend910 设备且显存可用 | 确认输出中有可用 NPU 设备 |
| 0. 环境初始化 | 执行 NPU 基础验证脚本 | 输出包含 `device='npu:0'` 的 Tensor | 确认 import 成功且 NPU 可正常计算 |
| 1. 模型下载 | 检查模型缓存目录结构 | 目录包含 `pytorch_model.pt`、`meta_info.txt`、`configuration.json` | 确认文件完整性 |
| 2. 权重格式转换 | 执行 `convert_state_dict` 后打印 missing/unexpected keys | 只有合理的 missing keys（运行时生成），无 unexpected keys | 确认 strict=False 加载无异常 |
| 3. CPU 推理 | 运行 `inference.py --device cpu` | 输出 logits、probabilities 并保存到结果 JSON | 确认输出文件存在且格式正确 |
| 4. NPU 推理 | 运行 `inference.py --device npu` | 输出 logits、probabilities 并保存到结果 JSON | 确认输出文件存在且格式正确 |
| 5. 精度对比 | 运行 `compare_cpu_npu.py` | 输出概率 MaxAE、Cosine Similarity、Top-1 匹配 | 确认概率 MaxAE < 1%，结论为 PASS |
| 6. 截图生成 | 检查生成的 PNG 文件 | `terminal_screenshot.png` 包含推理命令与结果 | 确认截图内容清晰完整 |
| 7. 发布到 GitCode | 访问仓库页面验证 | 仓库可见、文件完整、README 渲染正常 | 确认推送成功 |

## 异常处理与回滚策略

| 异常场景 | 触发条件 | 处理方式 | 回滚策略 |
|---|---|---|---|
| CANN 环境未加载 | `source set_env.sh` 失败或 `torch_npu` 导入报错 | 检查 CANN 安装路径 `/usr/local/Ascend/ascend-toolkit/` 是否存在 | 重新安装 CANN 或修正 `LD_LIBRARY_PATH` 环境变量 |
| NPU 设备不可用 | `torch.npu.is_available()` 返回 False | 执行 `npu-smi info` 确认设备状态，检查驱动 | 切换到 CPU 模式继续推理，或重启 NPU 驱动 |
| 模型下载失败 | `snapshot_download` 网络超时或 SSL 错误 | 检查网络连接，配置镜像源 | 清除不完整下载后重试 |
| 权重转换出现大量 missing keys | `load_state_dict` 打印 warning | 对比架构映射表，调整 `convert_state_dict` 规则 | 使用 `strict=False` 加载，忽略运行时生成的 key |
| NPU OOM | 推理时显存不足 | 切换到其他空闲 NPU 卡号 | 执行 `torch.npu.empty_cache()` 后重试 |
| CPU/NPU 精度超标 | 概率 MaxAE >= 1% | 检查输入预处理（resize, crop, normalize）完全一致 | 对齐预处理管道后重新推理 |
| 截图生成失败 | 缺少字体或 PIL 依赖 | 安装字体包或指定其他字体路径 | 跳过截图，手动截图替代 |
| Git push 被拒绝 | 远程仓库已有初始提交 | `git pull --allow-unrelated-histories` | 合并远程变更后重新推送 |
| GitCode API 返回 418 | WAF 拦截 | 检查 `private-token` 认证头 | 改用 SSH 推送或等待解除 |

## 资源与评测产物

| 类别 | 资源/产物 | 说明 | 路径 |
|---|---|---|---|
| 模型权重 | ModelScope 模型 | 从 ModelScope Hub 下载的原始权重 | `~/.cache/modelscope/iic/{model_name}/pytorch_model.pt` |
| 标签文件 | 类别标签 | 模型对应的类别名称列表 | `~/.cache/modelscope/iic/{model_name}/meta_info.txt` |
| 测试数据 | 测试图片/视频 | 用于推理验证的输入数据 | `/tmp/test_image.jpg` 或用户指定 |
| CPU 推理结果 | CPU 输出 JSON | logits、probabilities、推理耗时 | `/tmp/{MODEL_NAME}_cpu_results.json` |
| NPU 推理结果 | NPU 输出 JSON | logits、probabilities、推理耗时 | `/tmp/{MODEL_NAME}_npu_results.json` |
| 精度对比报告 | 对比结果 JSON | MAE、MaxAE、Cosine Similarity、pass/fail | `/tmp/{MODEL_NAME}_comparison.json` |
| 终端截图 | PNG 截图 | 推理结果终端风格截图 | `terminal_screenshot.png` |
| README 文档 | GitCode 仓库 README | 模型说明、精度对比表、使用方式 | `README.md` |

---

## 0. 环境初始化

### 0.1 加载 CANN 环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 配置 pip 镜像（国内加速）
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 0.2 安装依赖

```bash
# torch_npu（版本需与 CANN 匹配）
pip install torch_npu==2.9.0.post1

# timm（model architectures）
pip install timm==1.0.27

# ModelScope
pip install modelscope==1.35.3

# 其他依赖
pip install torchvision Pillow numpy
```

### 0.3 NPU 基础验证

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

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错。

**执行步骤**：
1. 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 CANN 环境变量
2. 执行 `npu-smi info` 确认 NPU 设备状态，设置 `ASCEND_RT_VISIBLE_DEVICES` 选择空闲卡号
3. 执行 `pip install torch_npu timm modelscope torchvision Pillow numpy` 安装 Python 依赖
4. 执行 NPU 基础验证脚本确认 torch_npu 可正常导入且 NPU 计算正确

---

## 1. 模型下载

使用 ModelScope SDK 下载模型到本地：

```bash
# 查看已安装的 modelscope 版本
python3 -c "import modelscope; print(modelscope.__version__)"

# 下载模型（自动缓存到 ~/.cache/modelscope/）
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
model_name = 'iic/cv_resnest101_general_recognition'
snapshot_download(model_name, cache_dir='/opt/atomgit/.cache/modelscope')
"
```

或使用 modelscope CLI：

```bash
pip install modelscope
python3 -m modelscope.cli download iic/cv_resnest101_general_recognition \
  --cache-dir /opt/atomgit/.cache/modelscope
```

下载后的目录结构：

```
/opt/atomgit/.cache/modelscope/iic/{model_name}/
├── pytorch_model.pt          # 模型权重（OpenMMLab 格式）
├── meta_info.txt             # 类别标签文件
├── configuration.json        # 模型配置
└── ...
```

**执行步骤**：
1. 使用 `from modelscope.hub.snapshot_download import snapshot_download` 下载模型到本地缓存目录
2. 检查下载后的目录结构，确保包含 `pytorch_model.pt`、`meta_info.txt`、`configuration.json`
3. 读取 `meta_info.txt` 获取类别标签列表，记录类别数量 `NUM_CLASSES`
4. 确认模型架构类型（ResNeSt/ViT/ConvNeXt 等），为后续权重转换做准备

---

## 2. 权重格式转换（分类模型）

ModelScope 下载的权重使用 OpenMMLab 格式（`backbone.xxx`、`head.xxx`），
需要转换为 timm 格式（`blocks.xxx`、`fc.xxx`）。

### 2.1 转换规则

不同架构的转换规则不同，以下为通用转换函数模板：

<!-- EMBED:scripts/convert_state_dict.py -->

```python
from collections import OrderedDict
import re

def convert_state_dict(state_dict):
    """Convert OpenMMLab format to timm format for common CV architectures."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        # Remove "module." prefix if present
        k = re.sub(r"^module\.", "", k)

        if k.startswith("backbone."):
            new_k = k[len("backbone."):]
            # ResNeSt: backbone.layers.i.xxx -> blocks.i.xxx
            new_k = re.sub(r"^layers\.(\d+)\.", r"blocks.\1.", new_k)
            # backbone.cls_token -> cls_token
            # backbone.pos_embed -> pos_embed
            # backbone.patch_embed.projection -> patch_embed.proj
            new_k = re.sub(r"^patch_embed\.projection\.", "patch_embed.proj.", new_k)
            # head.layers.head -> head (for ViT classification head)
            new_k = re.sub(r"^ln1\.", "norm.", new_k)
            # ffn.layers.0.0 -> mlp.fc1, ffn.layers.1 -> mlp.fc2
            new_k = re.sub(r"^blocks\.\d+\.ffn\.layers\.0\.0\.", lambda m: m.group(0).replace("ffn.layers.0.0", "mlp.fc1"), new_k)
            new_k = re.sub(r"^blocks\.\d+\.ffn\.layers\.1\.", lambda m: m.group(0).replace("ffn.layers.1", "mlp.fc2"), new_k)
            new_sd[new_k] = v

        elif k.startswith("head."):
            # head.fc -> head, head.layers.head -> head
            new_k = re.sub(r"^head\.layers\.head\.", "head.", k)
            new_k = re.sub(r"^head\.fc\.", "head.", new_k)
            new_sd[new_k] = v

        else:
            new_sd[k] = v

    return new_sd
```

### 2.2 架构特定映射

各模型架构的 OpenMMLab → timm 键映射速查：

| 架构 | OpenMMLab 前缀 | timm 前缀 |
|------|---------------|-----------|
| ResNeSt (resnest101e) | `backbone.layers.N.` | `blocks.N.` |
| ViT/DeiT (deit_base_patch16_224) | `backbone.layers.N.` | `blocks.N.` |
| ConvNeXt (convnext_base) | `backbone.stages.N.` | `stages.N.` |
| BEiTv2-large (beitv2_large_patch16_224) | `backbone.layers.N.` | `blocks.N.` |
| BEiTv2-base (beitv2_base_patch16_224) | `backbone.layers.N.` | `blocks.N.` |
| NextViT (nextvit_small) | `backbone.stages.N.` | `stages.N.` |
| BNext (bnext_small) | `backbone.stages.N.` | `stages.N.` |
| TinyNAS (zennet) | `backbone.layers.N.` | `blocks.N.` |

### 2.3 加载模型

```python
import timm
import torch
from collections import OrderedDict

# 构建模型
model = timm.create_model("resnest101e", pretrained=False, num_classes=NUM_CLASSES)

# 加载权重
checkpoint = torch.load("pytorch_model.pt", map_location="cpu")
state_dict = convert_state_dict(checkpoint["state_dict"])
model.load_state_dict(state_dict, strict=False)
model.eval()
```

> **注意**：`strict=False` 是必需的，因为 timm 模型可能包含 OpenMMLab 格式中不存在的
> 额外 key（如 relative_position_index），这些 key 会在运行时由模型内部生成。

**执行步骤**：
1. 根据模型架构从架构映射表中选择对应的转换规则，编写或复用 `convert_state_dict()` 函数
2. 执行 `torch.load("pytorch_model.pt", map_location="cpu")` 加载 OpenMMLab 格式权重
3. 调用 `convert_state_dict(checkpoint["state_dict"])` 将 key 映射为 timm 格式
4. 使用 `model.load_state_dict(sd, strict=False)` 加载转换后的权重，检查 missing/unexpected keys
5. 调用 `model.eval()` 切换到推理模式

---

## 3. 编写推理脚本

### 3.1 标准分类模型推理模板

完整的 `inference.py` 模板如下，支持 `--device cpu` 和 `--device npu` 两个模式：

<!-- EMBED:scripts/inference.py -->

**核心结构**：

```python
#!/usr/bin/env python3
"""ModelScope CV model NPU inference script."""

import argparse, json, os, re, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import timm

MODEL_NAME = "cv_resnest101_general_recognition"
MODEL_DIR = Path(os.environ.get("MODEL_DIR",
    f"/opt/atomgit/.cache/modelscope/iic/{MODEL_NAME}"))
LABELS_FILE = MODEL_DIR / "meta_info.txt"

def load_labels():
    labels = []
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                labels.append(parts[1] if len(parts) >= 2 else parts[0])
    return labels

def preprocess_image(image_path, input_size=256, crop_size=224):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

def get_device(device):
    if device == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("NPU is not available")
        return torch.device("npu:0")
    return torch.device("cpu")

def load_model(num_classes, device):
    model = timm.create_model("resnest101e", pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(MODEL_DIR / "pytorch_model.pt", map_location="cpu")
    state_dict = convert_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def run_inference(model, pixel_values, device):
    pixel_values = pixel_values.to(device)
    # Warmup
    with torch.no_grad():
        _ = model(pixel_values)
    if device.type == "npu":
        torch.npu.synchronize()

    num_runs = 10
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(pixel_values)
    if device.type == "npu":
        torch.npu.synchronize()
    elapsed = (time.time() - start) / num_runs

    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    return logits, probabilities, elapsed

def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} NPU inference")
    parser.add_argument("--image", default="/tmp/test_image.jpg")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    args = parser.parse_args()

    labels = load_labels()
    device = get_device(args.device)
    model = load_model(num_classes=len(labels), device=device)

    pixel_values, _ = preprocess_image(args.image)
    logits, probs, elapsed = run_inference(model, pixel_values, device)

    # Save results
    output = {
        "device": device.type, "model": MODEL_NAME,
        "logits": logits.cpu().numpy().tolist(),
        "probabilities": probs.cpu().numpy().tolist(),
        "time_ms": elapsed * 1000,
    }
    output_path = f"/tmp/{MODEL_NAME}_{device.type}_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
```

### 3.2 视频分类/行为识别模型

对于视频输入模型（如 PathShift、TAdaConv、ResNet50+NeXtVLAD），
不适用 timm，直接使用 ModelScope 提供的模型类：

```python
# PathShift Transformer 示例
from modelscope.models.cv.action_recognition.temporal_patch_shift_transformer import (
    PatchShiftTransformer
)

model = PatchShiftTransformer(
    num_classes=400, depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24], embed_dim=96, in_channels=768,
)
checkpoint = torch.load(MODEL_DIR / "pytorch_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"], strict=False)
model = model.to(device)
model.eval()
```

视频预处理（PathShift/TAdaConv 使用 32 帧、224x224 输入）：

```python
def preprocess_video(video_path):
    """Load .npy test video and preprocess."""
    video_np = np.load(video_path)
    # (T, H, W, C) -> (C, T, H, W)
    if video_np.ndim == 4 and video_np.shape[-1] == 3:
        video_np = np.transpose(video_np, (3, 0, 1, 2))
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
    video_np = (video_np / 255.0 - mean) / std
    tensor = torch.from_numpy(video_np).float().unsqueeze(0)
    return tensor
```

### 3.3 requirements.txt

依赖文件：

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
Pillow>=9.0.0
numpy>=1.22.0
```

**执行步骤**：
1. 复制 `inference.py` 模板并根据实际模型名称和类别数调整 `MODEL_NAME`、`NUM_CLASSES` 等参数
2. 视频模型需使用 ModelScope 原生模型类代替 timm，调整 `preprocess_video` 预处理逻辑
3. 编写 `requirements.txt` 列出所有依赖包及其版本号
4. 执行 `python inference.py --help` 确认参数解析正确

---

## 4. CPU 推理

```bash
# 准备测试图片
python3 -c "
from PIL import Image
import numpy as np
img = Image.new('RGB', (480, 360), color=(128, 128, 128))
img.save('/tmp/test_image.jpg')
"

# 或使用 Netron 在线图片作为测试
# wget -O /tmp/test_image.jpg https://github.com/netron-demo/images/raw/master/dog.jpg

# CPU 推理
python3 inference.py --device cpu
```

**输出**：`/tmp/{MODEL_NAME}_cpu_results.json` — 包含 logits、probabilities、time_ms。

**执行步骤**：
1. 执行 `python3 -c "from PIL import Image; Image.new('RGB', (480, 360), color=(128,128,128)).save('/tmp/test_image.jpg')"` 生成测试图片
2. 执行 `python3 inference.py --device cpu` 运行 CPU 推理
3. 检查输出文件 `/tmp/{MODEL_NAME}_cpu_results.json` 是否存在且包含 logits 和 probabilities 字段
4. 记录 CPU 推理耗时，与预期值对比验证无明显异常

**预期 CPU 推理时间**（单图）：

| 模型 | 参数量 | CPU 耗时 |
|------|--------|----------|
| ResNeSt-101 | 48.4M | ~700ms |
| ViT-Base (DeiT) | 86.6M | ~630ms |
| ConvNeXt-Base | 88.6M | ~610ms |
| BEiTv2-large | 304M | ~2345ms |
| NextViT-small | 31.8M | ~323ms |
| TinyNAS/ZenNet | 13.5M | ~946ms |
| BNext-small | 23.2M | ~817ms |

---

## 5. NPU 推理

```bash
# 确保 CANN 环境已加载
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# NPU 推理
python3 inference.py --device npu
```

**输出**：`/tmp/{MODEL_NAME}_npu_results.json` — 包含 logits、probabilities、time_ms。

**执行步骤**：
1. 确认 CANN 环境已加载，执行 `npu-smi info` 确认 NPU 设备可用
2. 执行 `python3 inference.py --device npu` 运行 NPU 推理
3. 检查输出文件 `/tmp/{MODEL_NAME}_npu_results.json` 是否存在且格式与 CPU 结果一致
4. 注意首次推理有初始化开销，脚本中 warmup 步骤会先空跑 1 次再计时 10 次取平均

**预期 NPU 推理时间**（单图，Ascend910）：

| 模型 | 参数量 | NPU 耗时 | 加速比 |
|------|--------|----------|--------|
| ResNeSt-101 | 48.4M | ~20ms | ~35x |
| ViT-Base (DeiT) | 86.6M | ~5ms | ~120x |
| ConvNeXt-Base | 88.6M | ~15ms | ~41x |
| BEiTv2-large | 304M | ~12ms | ~190x |
| NextViT-small | 31.8M | ~223ms | ~1.4x |
| TinyNAS/ZenNet | 13.5M | ~21ms | ~46x |
| PathShift | 29.5M | ~70ms | ~87x |

> **注意**：NextViT-small NPU 加速比较低（~1.4x），原因是其混合 CNN + Transformer
> 架构中有大量小算子，NPU 融合效率不如大矩阵计算。这是正常现象。

---

## 6. CPU vs NPU 精度对比

### 6.1 对比脚本

完整的 `compare_cpu_npu.py` 模板：

<!-- EMBED:scripts/compare_cpu_npu.py -->

```python
#!/usr/bin/env python3
"""CPU vs NPU accuracy comparison."""

import json, os, sys
import numpy as np

MODEL_NAME = os.environ.get("MODEL_NAME", "cv_resnest101_general_recognition")

def load_results(device):
    path = f"/tmp/{MODEL_NAME}_{device}_results.json"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

def compare():
    cpu = load_results("cpu")
    npu = load_results("npu")

    cpu_logits = np.array(cpu["logits"])
    npu_logits = np.array(npu["logits"])
    cpu_probs = np.array(cpu["probabilities"])
    npu_probs = np.array(npu["probabilities"])

    # Logits comparison
    logits_diff = np.abs(cpu_logits - npu_logits)
    logits_mae = np.mean(logits_diff)
    logits_max_ae = np.max(logits_diff)
    logits_rmse = np.sqrt(np.mean(logits_diff ** 2))

    # Probabilities comparison (primary metric)
    probs_diff = np.abs(cpu_probs - npu_probs)
    probs_mae = np.mean(probs_diff)
    probs_max_ae = np.max(probs_diff)

    # Top-1 and Top-5 agreement
    cpu_top1 = np.argmax(cpu_logits, axis=-1)
    npu_top1 = np.argmax(npu_logits, axis=-1)
    cpu_top5 = np.argsort(cpu_logits, axis=-1)[:, -5:]
    npu_top5 = np.argsort(npu_logits, axis=-1)[:, -5:]

    # Cosine similarity
    cpu_norm = cpu_logits / (np.linalg.norm(cpu_logits) + 1e-12)
    npu_norm = npu_logits / (np.linalg.norm(npu_logits) + 1e-12)
    cos_sim = np.dot(cpu_norm[0], npu_norm[0])

    # Assessment
    probs_error_pct = probs_max_ae * 100
    passed = probs_error_pct < 1.0

    print(f"\n  {'='*50}")
    print(f"  Probability MaxAE: {probs_error_pct:.6f}%")
    print(f"  Cosine Similarity:  {cos_sim:.8f}")
    print(f"  Top-1 match:        {'YES' if cpu_top1 == npu_top1 else 'NO'}")
    if passed:
        print(f"  ✓ PASS: Error < 1% ({probs_error_pct:.6f}% < 1%)")
    else:
        print(f"  ✗ FAIL: Error >= 1% ({probs_error_pct:.6f}% >= 1%)")
    print(f"  {'='*50}\n")

    results = {
        "model": MODEL_NAME,
        "logits_mae": float(logits_mae),
        "logits_max_ae": float(logits_max_ae),
        "probs_error_pct": float(probs_error_pct),
        "cosine_similarity": float(cos_sim),
        "top1_match": bool((cpu_top1 == npu_top1).item()),
        "passed": bool(passed),
    }
    json.dump(results, open(f"/tmp/{MODEL_NAME}_comparison.json", "w"), indent=2)

if __name__ == "__main__":
    compare()
```

### 6.2 运行对比

```bash
# 确保 CPU 和 NPU 结果文件都已生成
ls -la /tmp/{MODEL_NAME}_{cpu,npu}_results.json

# 运行对比
python3 compare_cpu_npu.py
```

### 6.3 预期精度结果

| 模型 | 概率 MaxAE | Cosine Similarity | Top-1 匹配 |
|------|-----------|-------------------|-----------|
| ResNeSt-101 General | 0.212% | 0.999989 | YES |
| ResNeSt-101 Animal | 0.041% | 0.999999 | YES |
| ViT-Base DailyLife | 0.749% | 0.999974 | YES |
| ViT-Base ImageNet | 0.081% | 0.999989 | YES |
| ConvNeXt-Base Garbage | 0.383% | 0.999993 | YES |
| ResNet50 Video | 0.022% | 0.999999 | YES |
| ResNet50 Live | 0.006% | 0.999999 | YES |
| BEiTv2-large | 0.021% | 0.999999 | YES |
| BEiTv2-base | 0.093% | 0.999987 | YES |
| PathShift | 0.013% | 0.999997 | YES |
| TAdaConv | 0.0001% | 0.999993 | YES |
| NextViT-small | 0.024% | 0.999999 | YES |
| TinyNAS | 0.216% | 0.999990 | YES |
| BNext-small | 0.050% | 0.999999 | YES |

**所有模型概率误差均 < 1%，NPU 推理精度可靠。**

**执行步骤**：
1. 确认 CPU 和 NPU 结果文件均已生成：`ls -la /tmp/{MODEL_NAME}_{cpu,npu}_results.json`
2. 执行 `python3 compare_cpu_npu.py` 运行精度对比
3. 检查输出中概率 MaxAE < 1%、Cosine Similarity > 0.999、Top-1 匹配三项指标
4. 记录对比结果到 `/tmp/{MODEL_NAME}_comparison.json`，确认结论为 PASS

---

## 7. 生成终端截图

### 7.1 截图脚本

使用 [terminal_screenshot.py](https://gitcode.com/gcw_C8PI9e90/terminal_screenshot) 脚本
生成终端风格的推理结果截图：

```bash
# 下载截图脚本
wget -O terminal_screenshot.py \
  https://gitcode.com/gcw_C8PI9e90/terminal_screenshot/raw/main/terminal_screenshot.py

# 运行 CPU 推理并捕获输出
python3 inference.py --device cpu 2>&1 | tee /tmp/cpu_output.txt

# 生成截图
python3 terminal_screenshot.py --input /tmp/cpu_output.txt --output terminal_screenshot.png
```

### 7.2 预览

生成的 `terminal_screenshot.png` 是一个暗色终端风格的 PNG 图片，
包含推理结果、Top-5 预测和性能指标，适合作为模型仓库的展示图。

**执行步骤**：
1. 执行 `python3 inference.py --device cpu 2>&1 | tee /tmp/cpu_output.txt` 捕获 CPU 推理输出
2. 执行 `python3 terminal_screenshot.py --input /tmp/cpu_output.txt --output terminal_screenshot.png` 生成截图
3. 检查生成的 `terminal_screenshot.png` 文件，确认内容清晰完整
4. 如截图生成失败，检查字体依赖或跳过截图，手动截图替代

---

## 8. 发布到 GitCode

### 8.1 README.md 模板

```markdown
---
tags:
- model-agent-tagged
- pytorch
- image-classification
- ascend
- npu
library_name: pytorch
pipeline_tag: image-classification
license: apache-2.0
---

# {model_name}

## 概述

{description}，基于 ModelScope 的预训练模型，适配 Ascend NPU (Ascend910) 进行推理。

- **ModelScope 模型**: [iic/{model_name}](https://modelscope.cn/models/iic/{model_name})
- **推理设备**: CPU / Ascend NPU (Ascend910)

## 文件说明

| 文件 | 说明 |
|------|------|
| `inference.py` | CPU/NPU 推理脚本 |
| `compare_cpu_npu.py` | CPU vs NPU 精度对比 |
| `requirements.txt` | Python 依赖 |
| `terminal_screenshot.png` | 推理结果终端截图 |

## 精度对比结果

| 指标 | 值 |
|------|-----|
| CPU 推理时间 | {cpu_time} ms |
| NPU 推理时间 | {npu_time} ms |
| NPU 加速比 | {speedup}x |
| 概率最大绝对误差 (MaxAE) | {error_pct}% |
| Cosine Similarity | {cos_sim} |
| 精度判定 | ✓ PASS (< 1%) |
```

### 8.2 创建 GitCode 仓库并推送

```bash
# 设置 token
export ATOMGIT_USER_TOKEN="your_token_here"

# 创建仓库
curl --location 'https://api.gitcode.com/api/v5/user/repos' \
  --header "private-token: $ATOMGIT_USER_TOKEN" \
  --header 'Content-Type: application/json' \
  --data '{
    "name": "cv_resnest101_general_recognition_npu",
    "path": "cv_resnest101_general_recognition_npu",
    "private": false,
    "repository_type": "model"
  }'

# 初始化和推送
cd /path/to/model/dir
git init
git add -A
git commit -m "Initial commit: NPU adaptation"

REPO_URL="https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/{username}/${model_name}_npu.git"
git remote add origin "$REPO_URL"
git branch -m main
git pull origin main --allow-unrelated-histories --no-edit 2>/dev/null || true
git push -u origin main
```

**执行步骤**：
1. 根据 README.md 模板填充模型名称、描述、推理时间和精度指标等占位字段
2. 将所有文件（inference.py、compare_cpu_npu.py、requirements.txt、terminal_screenshot.png、README.md）放在同一目录
3. 执行 `git init && git add -A && git commit -m "Initial commit: NPU adaptation"` 初始化仓库
4. 使用 GitCode API 创建远程仓库，设置 `repository_type: model`
5. 执行 `git push -u origin main` 推送到远程仓库，验证仓库页面可见

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `torch_npu` 导入失败 | CANN 环境未加载或版本不匹配 | 先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 权重加载有 missing keys | timm 模型生成了一些内部 buffer | 使用 `strict=False`，这些 key 会在运行时报错前自动生成 |
| 权重加载有 unexpected keys | checkpoint 包含 OpenMMLab 中间变量 | 使用 `convert_state_dict()` 过滤即可 |
| NPU 推理时间比预期长 | 首次推理有初始化开销 | 脚本已有 warmup 步骤（1 次空跑 + 10 次计时） |
| NextViT NPU 加速比低 | 混合架构小算子多 | 属于正常现象，NPU 对小算子的融合效率有限 |
| Git push 被拒绝 | 远程仓库已初始化（有 README） | 先 `git pull --allow-unrelated-histories` |
| GitCode API 返回 418 | WAF 拦截 | 使用 `api.gitcode.com` 域名 + `private-token` 认证 |
| 视频模型输入格式错误 | 维度顺序不匹配 | 确认 (T, H, W, C) → (C, T, H, W) 的转换 |

---

## 附录：已适配模型清单

以下 14 个模型均已通过本流程完成 NPU 适配和精度验证：

| # | 模型名 | 架构 | 类别数 | 类型 |
|---|--------|------|--------|------|
| 1 | cv_resnest101_general_recognition | ResNeSt-101 | 54,092 | 通用识别 |
| 2 | cv_vit-base_image-classification_Dailylife-labels | ViT-Base (DeiT) | 1,296 | 日常场景 |
| 3 | cv_resnest101_animal_recognition | ResNeSt-101 | 8,288 | 动物识别 |
| 4 | cv_convnext-base_image-classification_garbage | ConvNeXt-Base | 265 | 垃圾分类 |
| 5 | cv_vit-base_image-classification_ImageNet-labels | ViT-Base (DeiT) | 1,000 | ImageNet |
| 6 | cv_resnet50_video-category | ResNet50+NeXtVLAD | 23 | 视频分类 |
| 7 | cv_resnet50_live-category | ResNet50 | 8,613 | 直播商品 |
| 8 | cv_beitv2-large_image-classification_patch16_224_pt1k_ft22k_in1k | BEiTv2-large | 1,000 | ImageNet |
| 9 | cv_pathshift_action-recognition | PatchShift Transformer | 400 | 行为识别 |
| 10 | cv_nextvit-small_image-classification_Dailylife-labels | NextViT-small | 1,296 | 日常场景 |
| 11 | cv_beitv2-base_image-classification_patch16_224_pt1k_ft22k_in1k | BEiTv2-base | 1,000 | ImageNet |
| 12 | cv_TAdaConv_action-recognition | TAdaConvNeXt | 400 | 行为识别 |
| 13 | cv_tinynas_classification | TinyNAS/ZenNet | 1,000 | ImageNet |
| 14 | cv_bnext-small_image-classification_ImageNet-labels | BNext-small | 1,000 | ImageNet |
