---
name: moss-tts-npu-deployment
description: >
  MOSS-TTS-Nano NPU 适配推理部署技能。在用户需要将 MOSS-TTS 语音合成模型部署到昇腾 Ascend NPU 时调用。
  覆盖 PyTorch 和 ONNX 两种推理后端。
  短触发词：MOSS-TTS NPU 部署、MOSS-TTS 昇腾适配、MOSS-TTS 语音合成部署
metadata:
  short-description: MOSS-TTS-Nano 昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, moss-tts, tts, text-to-speech, pytorch, onnx, inference]
---

# MOSS-TTS NPU 部署技能

## 模型概述

MOSS-TTS-Nano 是一个 0.1B 参数的开源多语言轻量级语音合成模型，支持 20 种语言，采用纯自回归 Audio Tokenizer + LLM 架构，支持流式推理和声音克隆。

NPU 适配版分为两个仓库：

| 版本 | 仓库地址 | 说明 |
|------|----------|------|
| PyTorch | `gcw_C8PI9e90/moss-tts-nano-100m-npu` | 核心 NPU 适配版（推荐 NPU 使用） |
| ONNX | `gcw_C8PI9e90/moss-tts-nano-100m-onnx-npu` | ONNX 格式（主要面向浏览器部署） |

## 前置条件

| 组件 | 版本 |
|------|------|
| Python | >= 3.10 |
| torch | >= 2.0.0 |
| torch-npu | >= 2.0.0 |
| transformers | >= 4.35.0 |
| sentencepiece | >= 0.1.99 |
| torchaudio | >= 2.0.0 |
| scipy | >= 1.7.0 |

NPU：Ascend 910B4（1 逻辑卡）

## 流程总览

```
0. 环境初始化与 NPU 预检
→ 1. 安装依赖
→ 2. 模型获取
→ 3. NPU 推理验证
→ 4. 精度评测
→ 5. 验收确认
```

---

## 0. 环境初始化与 NPU 预检

0. 加载 CANN 环境
1. 检查 NPU 状态
2. 选择空闲 NPU 卡

### 0.1 加载 CANN 环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**如果加载失败**，请检查 CANN 安装路径是否正确，或执行回滚重试。

### 0.2 NPU 状态检查

```bash
npu-smi info
```

**用户确认**：确保 `npu-smi info` 输出显示至少 1 张卡状态为 `OK`，且显存占用 < 80%。

**异常处理**：如果 NPU 设备不可见，请检查驱动是否加载完成，尝试重启驱动 `npu-smi set -t reset -d 0`。

### 0.3 选择空闲 NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

---

## 1. 安装依赖

0. 安装 Python 依赖包
1. 验证 NPU 环境
2. 安装 ONNX 扩展（可选）

```bash
# 安装依赖
pip install torch torch-npu transformers sentencepiece torchaudio scipy \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如需 ONNX 推理，额外安装
pip install onnxruntime onnxruntime-extensions \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.1 验证安装

```bash
python3 -c "import torch_npu; import transformers; print('All dependencies OK')"
```

**如果报错 `No module named 'torch_npu'`**，请回滚到步骤 0.1 重新加载环境后执行 `pip install torch_npu`。

### 1.2 验证 NPU 设备可用性

```bash
python3 -c "
import torch
import torch_npu
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('NPU device:', torch.npu.get_device_name(0))
"
```

**通过标准**：NPU available 为 True，且 device_count >= 1。

---

## 2. 模型获取

0. 克隆 PyTorch 适配仓库
1. 下载原始模型权重
2. 下载音频分词器
3. 克隆 ONNX 版（备选）

### 2.1 PyTorch 版（推荐）

```bash
# 从 GitCode 克隆
git clone https://gitcode.com/gcw_C8PI9e90/moss-tts-nano-100m-npu.git
cd moss-tts-nano-100m-npu

# 下载原始模型权重（NPU 适配版不含权重文件）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano --local-dir ./hf_weights

# 下载音频分词器
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano \
  --local-dir ./MOSS-Audio-Tokenizer-Nano
```

### 2.2 ONNX 版

```bash
# 从 GitCode 克隆
git clone https://gitcode.com/gcw_C8PI9e90/moss-tts-nano-100m-onnx-npu.git
cd moss-tts-nano-100m-onnx-npu

# ONNX 模型文件需从原仓库下载：
# https://gitcode.com/OpenMOSS/MOSS-TTS-Nano-100M-ONNX
```

### 2.3 验证模型完整性

```bash
# 检查模型文件结构
ls -lh moss-tts-nano-100m-npu/
ls -lh moss-tts-nano-100m-npu/hf_weights/
ls -lh moss-tts-nano-100m-npu/MOSS-Audio-Tokenizer-Nano/
```

**通过标准**：hf_weights 目录存在且包含模型权重文件，分词器目录存在。

**输入/输出定义**：
- 输入：网络连接、GitCode/HuggingFace 访问权限
- 输出：克隆的仓库、模型权重、音频分词器
- 异常：如果下载中断，请重新运行 `huggingface-cli download`；如果 `huggingface-cli` 不存在，请执行 `pip install huggingface_hub`。

---

## 3. NPU 推理验证

0. PyTorch 基本文本合成
1. 指定 NPU 设备推理
2. 声音克隆验证
3. ONNX 推理验证

### 3.1 PyTorch 推理

```bash
# 文本合成
cd moss-tts-nano-100m-npu
python scripts/inference.py \
  --text "欢迎使用MOSS语音合成系统。" \
  --output ./output.wav

# 指定 NPU 设备
python scripts/inference.py \
  --text "Hello, this is MOSS TTS running on Ascend NPU." \
  --device npu:0 \
  --output ./output.wav

# 声音克隆
python scripts/inference.py \
  --text "欢迎使用MOSS语音合成系统。" \
  --prompt-audio-path ./reference.wav \
  --output ./output.wav
```

**异常处理**：
- 如果输出音频无声或异常，检查参考音频是否为 48kHz WAV 格式
- 如果声音克隆失败，确认 `--prompt-audio-path` 参考音频路径有效
- 如果 OOM 错误，尝试减小输入文本长度或使用单条推理

### 3.2 ONNX 推理

```bash
cd moss-tts-nano-100m-onnx-npu
python scripts/inference.py \
  --text "欢迎使用MOSS语音合成系统。" \
  --device cpu \
  --output ./output.wav
```

**输入/输出定义**：
- 输入：文本字符串、设备类型、输出路径、可选参考音频路径
- 输出：`scripts/output.wav` 音频文件
- 异常：如果 ONNX 推理在 NPU 上失败，请确认已安装 `onnxruntime-ascend` 包且 CANN 版本匹配。

---

## 4. 精度评测

0. 运行精度评测脚本
1. 验证 MSE 指标
2. 验证 SNR 指标
3. 验证相对误差
4. 确认通过标准

对比 NPU 和 CPU 的输出差异，确保误差 < 1%：

```bash
cd moss-tts-nano-100m-npu
python scripts/eval_accuracy.py \
  --text "Hello, this is a test of the MOSS TTS system."
```

**异常处理**：如果 eval_accuracy.py 找不到，确认仓库克隆完整；如果脚本报错，检查依赖是否齐全。

预期结果：

| 指标 | 差异 |
|------|------|
| MSE | < 1e-6 |
| SNR | > 40 dB |
| 相对误差 | < 0.1% |

**通过标准**：MSE < 1e-6，SNR > 40 dB，相对误差 < 0.1%。

---

## 5. 核心修改说明

NPU 适配涉及的关键改动：

1. **注意力实现切换**：`config.json` 中 `attn_implementation` 从 `flash_attention_2` → `sdpa`
2. **设备兼容**：`torch.cuda.*` → `torch.npu.*`（`mem_get_info` 等）
3. **fallback 策略**：`_select_fallback_attention_implementation` 增加 `npu` 设备支持
4. **批量解码**：`_resolve_voice_clone_chunk_batch_size` 增加 NPU 内存查询

---

## 6. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| `No module named 'torch_npu'` | CANN 未加载或 torch_npu 未安装 | 暂停执行，回滚到步骤 0.1 | `source set_env.sh` 后重装 torch_npu |
| ONNX NPU 推理失败 | 缺少 `onnxruntime-ascend` | 暂停，安装依赖 | 安装匹配 CANN 版本的 `onnxruntime-ascend` |
| 声音克隆失败 | 参考音频格式不支持 | 失败，需用户确认 | 提供 48kHz WAV 格式参考音频 |
| OOM | 并发推理或 batch 过大 | 回滚，单条推理 | 降低 batch size 或输入长度 |
| 模型下载失败 | 网络问题 | 重试，切换镜像 | 确认 `HF_ENDPOINT=https://hf-mirror.com` |
| 输出音频异常 | 文本过长或模型不匹配 | 失败，需用户确认 | 检查输入文本长度和模型兼容性 |

---

## 7. 检查点与验收确认

完成以下检查清单即为部署成功。每步完成后请**用户确认**再打勾：

- [ ] **Checkpoint 1**：`npu-smi info` 显示设备正常
- [ ] **Checkpoint 2**：`import torch_npu` 无报错
- [ ] **Checkpoint 3**：模型权重和音频分词器下载完成
- [ ] **Checkpoint 4**：PyTorch NPU 推理成功输出音频
- [ ] **Checkpoint 5**：`eval_accuracy.py` 精度评测通过（MSE < 1e-6）

---

## 8. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| PyTorch 适配仓库 | `https://gitcode.com/gcw_C8PI9e90/moss-tts-nano-100m-npu.git` |
| ONNX 适配仓库 | `https://gitcode.com/gcw_C8PI9e90/moss-tts-nano-100m-onnx-npu.git` |
| 原始权重 | `OpenMOSS-Team/MOSS-TTS-Nano` (HuggingFace) |
| 音频分词器 | `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (HuggingFace) |
| 参考文档 | https://www.hiascend.com/document/ |
| HuggingFace 镜像 | https://hf-mirror.com |

---

## 注意事项

1. PyTorch 版推荐在 NPU 上使用（ONNX 主要用于浏览器场景）
2. ONNX 在 NPU 上需要 `onnxruntime-ascend` 包（需匹配 CANN 版本）
3. 声音克隆需提供 48kHz 参考音频
4. 推理和评测串行执行，避免显存冲突
5. 模型文件从 HuggingFace 下载（`OpenMOSS-Team/MOSS-TTS-Nano`）

---

## Tokens 约束

- 安装日志仅显示关键步骤和错误
- 推理输出仅包含最终音频路径和耗时
- 不输出完整矩阵或中间激活值
