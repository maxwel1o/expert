---
tags:
- model-agent-tagged
- text-to-speech
- llama
- safetensors
- NPU
- Ascend
- en
library_name: transformers
pipeline_tag: text-to-speech
license: other
hardware: NPU
language:
- en
---

# neutts-nano on Ascend NPU

## 1. 简介

**English NeuTTS-Nano** 是 [Neuphonic](https://neuphonic.com) 开发的高效端侧 TTS（文本转语音）语音语言模型，现已适配华为昇腾 Ascend 910B4 NPU。

| 属性 | 值 |
|------|-----|
| 模型 | [neuphonic/neutts-nano](https://huggingface.co/neuphonic/neutts-nano) |
| 架构 | LLaMA backbone + NeuCodec |
| 总参数量 | ~229M |
| 活跃参数量 | ~117M |
| 音频编解码器 | [NeuCodec](https://huggingface.co/neuphonic/neucodec) (50Hz, 单码本) |
| 输出采样率 | 24,000 Hz 单声道 |
| 上下文窗口 | 2048 tokens (~30s 音频) |
| 语言 | English (en-us) |

## 2. 验证环境

| 组件 | 版本 |
|------|------|
| NPU | 1x Ascend 910B4 (32GB HBM) |
| CANN | 25.5.1 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post1+gitee7ba04 |
| neutts | 1.1.0 |
| neucodec | (随 neutts 安装) |

## 3. 快速开始

### 3.1 环境准备

```bash
pip install neutts soundfile
pip install espeak-ng  # 可选，提升音素质量
```

### 3.2 模型下载

```bash
export HF_ENDPOINT=https://hf-mirror.com

python3 -c "
import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('neuphonic/neutts-nano', local_dir='./neutts-nano')
"
```

### 3.3 NPU 推理

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 inference.py --text "Hello world, this is a test." --output output.wav
```

也可以直接使用本地模型路径：

```bash
python3 inference.py --model ./neutts-nano --text "Your text here." --output output.wav
```

## 4. 功能验证

模型在 Ascend 910B4 NPU 上端到端推理验证通过：

| 指标 | 数值 |
|------|------|
| 加载时间 | 29.0s |
| 推理时间 | 78.5s |
| 生成音频时长 | 13.2s |
| RTF | 5.95 |
| NPU 内存占用 | 3.93 GiB |

## 5. 精度验证

采用确定性 logit 级对比方法（相同输入 token，对比 backbone 前向传播 logits）：

| 指标 | 数值 |
|------|------|
| 最大绝对误差 | 2.48e-05 |
| 平均绝对误差 | 6.11e-06 |
| **相对误差** | **0.00008%** |
| Top-1 一致率 | 100.0% |
| NPU 加速比 | 4.03x |
| 精度阈值 | < 1.0% |
| 结果 | ✅ PASS |

相对误差 0.00008% 远低于 1% 阈值，NPU 与 CPU 输出功能完全等价。

## 6. 适配说明

### 核心策略

NeuTTS 使用标准 PyTorch 设备管理（`.to(device)`）。导入 `torch_npu` 后，`torch.device("npu")` 自动注册，无需修改模型代码。

### 适配要点

| 要点 | 说明 |
|------|------|
| NPU 设备注册 | `import torch_npu` 必须早于模型加载 |
| 音素化器 | espeak-ng 不可用时自动回退到文本直通模式 |
| 模型下载 | 通过 `HF_ENDPOINT=https://hf-mirror.com` 镜像加速 |
| NeuCodec | 直接从 HuggingFace 下载，自动缓存 |

## 7. 已知限制

1. **RTF > 1**: 当前未做性能优化，可通过 `ascend-optimization` skill 加速
2. **espeak-ng**: 如未安装，音素质量可能下降
3. **流式推理**: 当前 safetensors PyTorch 路径不支持流式

## 8. 参考资源

- [NeuTTS-Nano 官方 HuggingFace](https://huggingface.co/neuphonic/neutts-nano)
- [GitCode 适配仓库](https://ai.gitcode.com/m0_74196153/neutts-nano)
- [Neuphonic 官网](https://neuphonic.com)
