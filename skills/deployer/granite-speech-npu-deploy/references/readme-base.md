---
tags:
  - model-agent-tagged
  - transformers
  - safetensors
  - granite_speech
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

# Granite Speech 4.1 2B on Ascend NPU

## 1. 简介

本文档记录 `granite-speech-4.1-2b` 在 Ascend NPU 环境的适配与验证结果。

`granite-speech-4.1-2b` 是 IBM 开源的语音理解模型，基于 conformer 音频编码器 + QFormer 投影层 + Granite-4.0-1B-base 语言模型。该模型接收音频与文本输入，可执行语音识别、语音理解等任务。

相关获取地址：

- 权重下载（ModelScope）：<https://modelscope.cn/models/ibm-granite/granite-speech-4.1-2b>
- 权重下载（HuggingFace）：<https://huggingface.co/ibm-granite/granite-speech-4.1-2b>
- 镜像下载（GitCode）：<https://gitcode.com/hf_mirrors/ibm-granite/granite-speech-4.1-2b>

## 2. 验证环境

| 组件 | 版本 |
| --- | --- |
| `transformers` | `4.57.6` |
| `torch` | `2.9.0` |
| `torch-npu` | `2.9.0.post1+gitee7ba04` |
| `CANN` | `8.5.1` |

- NPU：`Ascend 910B4` x 1 逻辑卡
- 模型路径：`/opt/atomgit/granite-speech-4.1-2b`
- 模型参数量：`2,313M`

## 3. 推理

### 环境准备

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
```

### 加载模型

```python
import torch
import torch_npu
from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
from transformers import AutoProcessor

model_path = "/path/to/granite-speech-4.1-2b"

model = GraniteSpeechForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
).npu()
model.eval()

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```

### 推理命令

```bash
python3 inference.py --model-path /path/to/granite-speech-4.1-2b \
    --audio-path sample.wav \
    --text "<|audio|> Transcribe the audio." \
    --max-new-tokens 128
```

### 性能基准

```bash
python3 benchmark_granite_speech.py
```

## 4. Smoke 验证

| 验证项 | 状态 |
| --- | --- |
| 模型加载 NPU | ✅ |
| Forward 前向传播 | ✅ |
| Generate 文本生成 | ✅ |
| 权重加载正确 | ✅ |

## 5. 性能参考

测试条件：`Ascend 910B4` x 1，`bfloat16` 精度，`50` 轮平均。

| 序列长度 | 平均耗时 (ms) | 中位数 (ms) | P99 (ms) | Throughput (tok/s) |
| --- | --- | --- | --- | --- |
| 32 | 134.01 | 131.61 | 167.60 | 238.78 |
| 64 | 139.03 | 136.05 | 164.99 | 460.32 |
| 128 | 136.33 | 134.23 | 160.79 | 938.89 |

显存占用：`~4.4 GB`（batch_size=1, bfloat16）

## 6. 精度评测

NPU (bfloat16) vs CPU (float32) 同输入同权重对比：

| 指标 | 数值 |
| --- | --- |
| 余弦相似度 | > 0.9999 |
| 平均绝对误差 | 0.030 |
| 最大绝对误差 | 0.401 |

精度达标，NPU bfloat16 推理结果与 CPU float32 参考结果一致。

## 7. 注意事项

- 该模型基于 `transformers` 原生推理，无需 `vLLM-Ascend`
- 输入音频需重采样至 `16kHz` 单声道 WAV 格式
- 文本中需包含 `<|audio|>` 标记指定音频嵌入插入位置
- 推荐使用 `bfloat16` 精度以平衡性能与精度
