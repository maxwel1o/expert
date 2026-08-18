---
tags:
  - model-agent-tagged
  - transformers
  - safetensors
  - granite_speech_plus
  - automatic-speech-recognition
  - multilingual
  - en
  - endpoints_compatible
  - NPU
  - Ascend
  - speech
  - granite
  - ibm
library_name: transformers
pipeline_tag: automatic-speech-recognition
license: apache-2.0
language:
  - en
hardware: NPU
---

# Granite Speech 4.1 2B Plus on Ascend NPU

## 1. 简介

本文档记录 `granite-speech-4.1-2b-plus` 在 Ascend NPU 环境的适配与验证结果。

`granite-speech-4.1-2b-plus` 是 IBM 开源的语音理解模型增强版，基于 conformer 音频编码器（支持 `cat_hidden_layers` 中间层拼接）+ QFormer 投影层 + Granite-4.0-1B-base 语言模型。与 base 版本相比，plus 版本在音频编码器第 3 层拼接待拼接的隐藏状态，使投影层输入维度翻倍（1024 → 2048），提升语音特征表达能力。

相关获取地址：

- 权重下载（ModelScope）：<https://modelscope.cn/models/ibm-granite/granite-speech-4.1-2b-plus>
- 权重下载（HuggingFace）：<https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus>
- 镜像下载（GitCode）：<https://gitcode.com/hf_mirrors/ibm-granite/granite-speech-4.1-2b-plus>

## 2. 验证环境

| 组件 | 版本 |
| --- | --- |
| `transformers` | `4.57.6` |
| `torch` | `2.9.0` |
| `torch-npu` | `2.9.0.post1+gitee7ba04` |
| `CANN` | `8.5.1` |

- NPU：`Ascend 910B4` x 1 逻辑卡
- 模型路径：`/opt/atomgit/granite-speech-4.1-2b-plus`
- 模型参数量：`2,112M`

## 3. 推理

> **注意**：`granite_speech_plus` 模型类型在 transformers 4.57.6 中不存在，
> 需要预先注册。适配代码见 `granite_speech_plus/` 目录。

### 环境准备

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
```

### 加载模型

```python
import torch
import torch_npu
from transformers import AutoProcessor

# 首先注册 granite_speech_plus 模型类型
from granite_speech_plus.register import register
register()
from granite_speech_plus import GraniteSpeechPlusForConditionalGeneration

model_path = "/path/to/granite-speech-4.1-2b-plus"

model = GraniteSpeechPlusForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
).npu()
model.eval()

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```

### 推理命令

```bash
python3 inference.py --model-path /path/to/granite-speech-4.1-2b-plus \
    --audio-path sample.wav \
    --text "<|audio|> Transcribe the audio." \
    --max-new-tokens 128
```

### 适配说明

`granite-speech-4.1-2b-plus` 使用 `model_type = "granite_speech_plus"`，而 transformers 4.57.6
原生仅支持 `"granite_speech"`。适配新增了以下组件：

| 组件 | 说明 |
| --- | --- |
| `GraniteSpeechPlusEncoder` | 支持 `cat_hidden_layers=[3]`，拼接中间层隐藏态 |
| `GraniteSpeechPlusConfig` | 使用 `GraniteSpeechPlusEncoderConfig` 的配置类 |
| `GraniteSpeechPlusForConditionalGeneration` | 完整模型类，继承基类推理逻辑 |
| `register.py` | 运行时注册到 transformers auto 系统 |

## 4. Smoke 验证

| 验证项 | 状态 |
| --- | --- |
| 模型加载 NPU | ✅ |
| Forward 前向传播 | ✅ |
| Generate 文本生成 | ✅ |
| Encoder cat_hidden_layers | ✅（[3]） |

## 5. 性能参考

测试条件：`Ascend 910B4` x 1，`bfloat16` 精度，`50` 轮平均。

| 序列长度 | 平均耗时 (ms) | 中位数 (ms) | P99 (ms) | Throughput (tok/s) |
| --- | --- | --- | --- | --- |
| 32 | 149.24 | 142.61 | 221.47 | 214.42 |
| 64 | 134.99 | 133.71 | 148.96 | 474.11 |
| 128 | 139.29 | 134.73 | 198.26 | 918.97 |

显存占用：`~4.0 GB`（batch_size=1, bfloat16）

## 6. 精度评测

NPU (bfloat16) vs CPU (float32) 同输入同权重对比：

| 指标 | 数值 |
| --- | --- |
| 余弦相似度 | > 0.9999 |
| 平均绝对误差 | 0.025 |
| 最大绝对误差 | 0.246 |

精度达标，NPU bfloat16 推理结果与 CPU float32 参考结果一致。

## 7. 注意事项

- `granite_speech_plus` 为 transformers 4.57.6 未注册的模型类型，需先调用 `register()`
- 该模型基于 `transformers` 原生推理，无需 `vLLM-Ascend`
- 输入音频需重采样至 `16kHz` 单声道 WAV 格式
- 文本中需包含 `<|audio|>` 标记指定音频嵌入插入位置
- 推荐使用 `bfloat16` 精度以平衡性能与精度
