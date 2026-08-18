---
name: kani-tts-400m-npu
description: >
  KaniTTS 语音合成模型在华为昇腾 Ascend NPU 上的部署、推理与精度验证。
  涵盖 10 个模型（7 个语言微调模型 + 3 个预训练/微调模型），
  支持 LFM2 语言模型在 NPU 推理 + NanoCodec 在 CPU 解码的混合部署架构。
  触发词：kani-tts NPU、KaniTTS 昇腾、TTS NPU 部署、kani-tts-400m、语音合成 NPU
metadata:
  short-description: KaniTTS 昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, kani-tts, tts, text-to-speech, lfm2, nanocodec, pytorch, inference]
---

# KaniTTS NPU 部署 Skill

在华为昇腾 Ascend NPU 上部署 KaniTTS 语音合成模型的完整指南。KaniTTS 基于 LFM2 因果语言模型和 NVIDIA NanoCodec 音频编解码器。

## 支持的模型

### 语言微调模型（7 个）

| 语言 | 模型名称 | 说话人 | AtomGit 仓库 |
|------|----------|--------|-------------|
| 英文 (en) | KaniTTS-400M-English | 多说话人: andrew, katie | [kani-tts-400m-en-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-en-npu) |
| 韩语 (ko) | KaniTTS-400M-Korean | 单说话人 | [kani-tts-400m-ko-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-ko-npu) |
| 阿拉伯语 (ar) | KaniTTS-400M-Arabic | 单说话人 | [kani-tts-400m-ar-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-ar-npu) |
| 吉尔吉斯语 (ky) | KaniTTS-400M-Kyrgyz | 多说话人: syimyk, elina | [kani-tts-400m-ky-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-ky-npu) |
| 德语 (de) | KaniTTS-400M-German | 多说话人: bert, thorsten | [kani-tts-400m-de-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-de-npu) |
| 西班牙语 (es) | KaniTTS-400M-Spanish | 多说话人: nova, ballad, ash | [kani-tts-400m-es-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-es-npu) |
| 中文 (zh) | KaniTTS-400M-Chinese | 多说话人: ming, mei | [kani-tts-400m-zh-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-zh-npu) |

### 预训练/微调模型（3 个）

| 模型 | 参数量 | 说明 | AtomGit 仓库 |
|------|--------|------|-------------|
| kani-tts-400m-0.3-pt | 400M | 多语言预训练（英/日/德/阿/中/西/韩/吉） | [kani-tts-400m-0.3-pt-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-400m-0.3-pt-npu) |
| kani-tts-450m-0.2-pt | 450M | 多语言预训练（英/德/法/中/韩/日/阿） | [kani-tts-450m-0.2-pt-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-450m-0.2-pt-npu) |
| kani-tts-450m-0.1-ft | 450M | 英语对话微调（Expresso 数据集） | [kani-tts-450m-0.1-ft-npu](https://gitcode.com/gcw_C8PI9e90/kani-tts-450m-0.1-ft-npu) |

## 前置条件

### 环境要求

| 组件 | 推荐版本 |
|------|----------|
| NPU | Ascend 910B4 (≥ 4GB 可用显存) |
| Python | 3.11+ |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post1 |
| transformers | ≥ 4.57.0 (需要 LFM2 支持) |

## 流程总览

```
0. 环境初始化与 NPU 预检
→ 1. 安装依赖
→ 2. 模型下载
→ 3. NPU 推理验证
→ 4. 精度验证
→ 5. 验收确认
```

---

## 0. 环境初始化与 NPU 预检

0. 加载 CANN 环境
1. 检查 NPU 状态
2. 选择空闲 NPU 卡
3. 设置国内镜像

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

**异常处理**：如果 NPU 设备不可见，请检查驱动是否加载完成，尝试 `npu-smi set -t reset -d 0` 重启驱动。

### 0.3 选择空闲 NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

### 0.4 设置国内镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 1. 安装依赖

0. 安装 kani-tts 包
1. 手动安装所需依赖
2. 验证安装

```bash
# 安装 PyPI 包（使用清华镜像）
pip install kani-tts -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或手动安装所需依赖
pip install torch torch_npu transformers soundfile scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 注意: NanoCodec 需要 nemo-toolkit，会自动作为 kani-tts 的依赖安装
```

### 1.1 验证安装

```bash
python3 -c "import torch_npu; import transformers; import soundfile; print('All dependencies OK')"
```

**如果报错 `No module named 'torch_npu'`**，请回滚到步骤 0.1 重新加载环境后执行 `pip install torch_npu`。

### 1.2 验证 NPU 设备

```bash
python3 -c "
import torch
import torch_npu
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
"
```

**通过标准**：NPU available 为 True。

---

## 2. 模型下载

0. 下载语言微调模型（7 个）
1. 下载预训练/微调模型（3 个）
2. 验证模型完整性

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 语言微调模型（替换 {lang} 为: en, ko, ar, ky, de, es, zh）
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nineninesix/kani-tts-400m-{lang}', local_dir='./kani-tts-400m-{lang}')
"

# 预训练/微调模型
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nineninesix/kani-tts-400m-0.3-pt', local_dir='./kani-tts-400m-0.3-pt')
snapshot_download(repo_id='nineninesix/kani-tts-450m-0.2-pt', local_dir='./kani-tts-450m-0.2-pt')
snapshot_download(repo_id='nineninesix/kani-tts-450m-0.1-ft', local_dir='./kani-tts-450m-0.1-ft')
"
```

### 2.1 验证下载完整性

```bash
# 检查各语言模型目录
for lang in en ko ar ky de es zh; do
    echo "=== kani-tts-400m-$lang ==="
    ls -lh kani-tts-400m-$lang/ | head -5
done

# 检查预训练模型目录
for model in kani-tts-400m-0.3-pt kani-tts-450m-0.2-pt kani-tts-450m-0.1-ft; do
    echo "=== $model ==="
    ls -lh $model/ | head -5
done
```

**通过标准**：每个模型目录存在且包含 config.json 和权重文件。

**输入/输出定义**：
- 输入：目标语言代码或模型名称、网络连接
- 输出：模型配置文件和权重
- 异常：如果下载中断，请重新运行下载脚本；如果 huggingface_hub 报错，请检查 `HF_ENDPOINT` 是否已导出。

---

## 3. NPU 推理验证

0. 准备推理环境
1. 测试基本文本合成
2. 测试说话人指定
3. 验证输出音频

推理引擎架构：LFM2 语言模型运行在 NPU 上，NanoCodec 音频解码器运行在 CPU 上。

```python
from scripts.kani_npu import KaniTTSNPU

# 初始化（自动加载 LFM2 到 NPU，NanoCodec 到 CPU）
tts = KaniTTSNPU('/path/to/model')

# 语音合成（语言微调模型支持 speaker_id）
audio = tts("Hello, world!")

# 保存音频
import soundfile as sf
sf.write("output.wav", audio, 22050)
```

### 3.1 运行推理脚本

| 文件 | 说明 |
|------|------|
| `scripts/inference.py` | NPU 推理脚本（支持文本文件和单句输入） |
| `scripts/eval_accuracy.py` | NPU vs CPU 精度评测脚本 |
| `scripts/kani_npu.py` | NPU 推理引擎模块（LFM2 + NanoCodec） |
| `scripts/kani_npu_eval.py` | 精度评测引擎模块 |
| `scripts/README.md` | 部署文档（中文） |

推理示例：

```bash
# 预训练模型
python scripts/inference.py --text "Hello, how are you?" --output output.wav

# 英文微调模型（指定说话人）
python scripts/inference.py --text "Hello, how are you?" --speaker andrew --output output.wav
```

**输入/输出定义**：
- 输入：文本字符串、模型路径、可选 speaker_id
- 输出：`scripts/output.wav`（22.05kHz）
- 异常：如果输出为静音或噪声，请检查模型权重是否完整下载。

---

## 4. 精度验证

0. 运行精度评测脚本
1. 检查 Token 匹配率
2. 检查余弦相似度
3. 检查加权相对误差

比较 NPU 和 CPU 前向传播的 logits 差异：

```bash
python scripts/eval_accuracy.py
```

**异常处理**：如果 eval_accuracy.py 运行失败，请确认模型权重下载完整且依赖安装正确。

关键指标（实测结果）：
- Token 匹配率: 100%
- 余弦相似度: > 0.999
- 加权相对误差: < 3%

**通过标准**：Token 匹配率 > 99%，余弦相似度 > 0.999。

---

## 5. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| `No module named 'torch_npu'` | CANN 未加载或 torch_npu 未安装 | 暂停执行，回滚到步骤 0.1 | `source set_env.sh` 后重装 torch_npu |
| `No module named 'nemo_toolkit'` | kani-tts 依赖未完整安装 | 暂停，安装依赖 | `pip install kani-tts` |
| OOM | 多个模型并行评测 | 回滚，串行执行 | 单模型评测，降低 batch size |
| 模型下载失败 | 网络问题 | 重试，切换镜像 | 确认 `HF_ENDPOINT=https://hf-mirror.com` |
| 输出音频异常 | 文本过长或模型不匹配 | 失败，需用户确认 | 单次输入不超过 200 字符 |
| 单说话人模型不支持 speaker_id | 预训练模型无说话人参数 | 失败，需用户确认 | 预训练模型（0.3-pt, 0.2-pt, 0.1-ft）不使用 speaker_id |

---

## 6. 检查点与验收确认

完成以下检查清单即为部署成功。每步完成后请**用户确认**再继续下一步：

| # | 检查项 | 验证方法 | 通过标准 | 操作说明 |
|---|--------|---------|---------|---------|
| 1 | NPU 设备状态 | `npu-smi info` | 至少 1 张卡状态 OK，显存占用 < 80% | 用户确认设备正常 |
| 2 | NPU 环境就绪 | `python3 -c "import torch_npu"` | 无报错 | 用户确认无 ImportError |
| 3 | 模型权重完整 | 检查各模型目录 | 所有 10 个模型均已下载 | 用户确认文件完整性 |
| 4 | NPU 推理验证 | `python scripts/inference.py` | 成功输出音频文件 | 用户确认推理结果 |
| 5 | 精度验证通过 | `python scripts/eval_accuracy.py` | Token 匹配率 > 99%，余弦相似度 > 0.999 | 用户确认评测报告 |

---

## 7. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| 推理脚本 | `scripts/inference.py` |
| 精度验证脚本 | `scripts/eval_accuracy.py` |
| NPU 推理引擎 | `scripts/kani_npu.py` |
| 精度评测引擎 | `scripts/kani_npu_eval.py` |
| 参考文档 | https://www.hiascend.com/document/ |
| HuggingFace 镜像 | https://hf-mirror.com |

---

## 注意事项

1. **显存管理**: 由于 Ascend 910B4 显存有限（31GB），多个模型评测需串行执行
2. **设备分工**: LFM2 在 NPU 生成 token，NanoCodec 在 CPU 解码为波形
3. **精度说明**: bfloat16 推理下 NPU 与 CPU 的 Token 匹配率 > 99%，余弦相似度 > 0.999
4. **模型权重**: 从 HuggingFace 镜像站 hf-mirror.com 下载
5. **文本长度**: 建议单次输入不超过 200 字符，过长的文本可能降低生成质量
6. **单说话人模型**: 预训练模型（0.3-pt, 0.2-pt, 0.1-ft）不支持 speaker_id 参数
