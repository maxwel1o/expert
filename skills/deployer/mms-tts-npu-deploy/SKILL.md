---
name: mms-tts-npu-deploy
description: >
  多语言 MMS-TTS (VITS) 模型在昇腾 NPU 上的部署与验证 Skill。
  覆盖印地语、阿拉伯语、印尼语、西班牙语、约鲁巴语、俄语、越南语、土耳其语
  共 8 种语言的 MMS-TTS 模型。提供 transfer_to_npu 自动迁移、推理验证、
  精度验证（梅尔频谱分布稳定性）、性能基准测试的全流程。
  当用户提到 MMS TTS NPU、多语言 TTS 昇腾、VITS NPU 推理、facebook/mms-tts 时触发。
metadata:
  short-description: 多语言 MMS-TTS 昇腾 NPU 部署与验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, mms, tts, vits, multilingual, pytorch, inference]
---

# 多语言 MMS-TTS 昇腾 NPU 部署与验证 Skill

本 Skill 提供 Facebook MMS (Massively Multilingual Speech) 项目中 8 种语言的
VITS 文本转语音模型在华为昇腾 NPU 上的完整部署、推理验证与性能基准测试流程。

支持的语言：

| 语言 | HuggingFace 模型 ID |
|------|---------------------|
| 印地语 | `facebook/mms-tts-hin` |
| 阿拉伯语 | `facebook/mms-tts-ara` |
| 印尼语 | `facebook/mms-tts-ind` |
| 西班牙语 | `facebook/mms-tts-spa` |
| 约鲁巴语 | `facebook/mms-tts-yor` |
| 俄语 | `facebook/mms-tts-rus` |
| 越南语 | `facebook/mms-tts-vie` |
| 土耳其语 | `facebook/mms-tts-tur` |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 910B4（1 卡，32GB HBM） |
| OS | Linux aarch64（openEuler / Ubuntu） |
| CANN | >= 8.5.1 |
| Python | 3.9 – 3.13 |
| 网络 | 首次需联网下载模型权重（~138MB / 模型） |

## 流程总览

## 工作流阶段总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 0 | 昇腾 NPU 服务器 | 加载 CANN 环境、检查 NPU 状态、选择空闲卡 | NPU 环境就绪 | `python3 -c "import torch_npu; print(torch.npu.is_available())"` | `torch.npu.is_available()` 返回 `True` |
| 依赖安装 | 1 | Python 3.9–3.13，CANN 已加载 | 安装 torch、transformers、scipy | 依赖安装完成 | `python3 -c "import torch_npu; import transformers; import scipy"` | 无 ImportError |
| 模型下载 | 2 | 目标语言、网络连接 | 下载配置文件和权重 | 模型权重文件 | 检查文件完整性 | `config.json`、`model.safetensors` 存在 |
| 推理验证 | 3 | 模型目录、测试文本 | 运行推理脚本合成语音 | WAV 音频文件 | `python3 scripts/inference.py --text "..." --output output.wav` | 输出有效非静音 WAV |
| 精度验证 | 4 | 测试文本 | 多轮推理、梅尔频谱统计分析 | 精度报告 | `python3 scripts/accuracy_run.py` | Mel Mean 方差 < 3.0，输出有效 |
| 性能测试 | 5 | 模型目录 | 多次推理测量延迟和 RTF | 性能报告 | `python3 scripts/accuracy_run_perf.py` | RTF < 1.0 |

按以下各节顺序执行，每步完成后再进入下一步。

---

## 0. 环境初始化与 NPU 预检

**执行步骤**：
1. 加载 CANN 环境变量并确认安装路径
2. 运行 `npu-smi info` 检查 NPU 设备状态
3. 选择空闲 NPU 卡设置 `ASCEND_RT_VISIBLE_DEVICES`
4. 验证 `torch.npu.is_available()` 返回 True

| 项目 | 内容 |
|------|------|
| **输入** | 昇腾 NPU 服务器（Ascend 910B4），CANN 已安装 |
| **操作** | 加载 CANN 环境、检查 NPU 状态、选择空闲卡、验证环境 |
| **输出** | NPU 环境就绪，torch.npu.is_available() 返回 True |
| **异常** | CANN 路径不存在 → 按官方文档重新安装 CANN；NPU 卡全满 → 等待资源释放 |

## 0.1 加载 CANN 环境

1. 确认 CANN 安装路径
2. 加载环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**如果加载失败**，请检查 `/usr/local/Ascend/ascend-toolkit/` 是否存在；若不存在，按官方文档重新安装 CANN。

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

## 0.4 基础环境验证

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('NPU available:', torch.npu.is_available())"
```

**通过标准**：`torch.npu.is_available()` 返回 `True`。

---

## 1. 安装依赖

**执行步骤**：
1. 安装 torch、transformers、scipy 等依赖
2. 验证所有依赖导入正常

| 项目 | 内容 |
|------|------|
| **输入** | Python 3.9–3.13，昇腾 CANN 已加载 |
| **操作** | 安装 torch、transformers、scipy |
| **输出** | 依赖安装完成，NPU 环境验证通过 |
| **异常** | torch_npu 报错 → 回退到 0.1 重新加载 CANN 环境 |

1. 安装基础 Python 依赖包
2. 验证依赖安装

```bash
pip install torch transformers scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1.1 验证安装

```bash
python3 -c "import torch_npu; import transformers; import scipy; print('All dependencies OK')"
```

**如果报错 `No module named 'torch_npu'`**，说明 CANN 环境未加载或 torch_npu 未安装，请回滚到步骤 0.1 重新加载环境后执行 `pip install torch_npu`。

> 确保 CANN 和 torch_npu 已正确安装
> 参考: https://www.hiascend.com/document/

---

## 2. 下载模型

**执行步骤**：
1. 设置 HuggingFace 国内镜像 `HF_ENDPOINT`
2. 选择目标语言模型 ID
3. 下载配置文件和权重文件

选择要部署的语言，下载对应模型权重：

```bash
export HF_ENDPOINT=https://hf-mirror.com
MODEL_ID="facebook/mms-tts-hin"  # 替换为目标语言模型 ID

# 下载配置文件
python3 - <<'PY'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download("facebook/mms-tts-hin", allow_patterns=["config.json", "*.md", "*tokenizer*", "*.json"], local_dir="./model")
PY

# 下载权重文件
wget -c "https://hf-mirror.com/${MODEL_ID}/resolve/main/model.safetensors" -P ./model
```

支持的模型 ID 列表：

- `facebook/mms-tts-hin`（印地语）
- `facebook/mms-tts-ara`（阿拉伯语）
- `facebook/mms-tts-ind`（印尼语）
- `facebook/mms-tts-spa`（西班牙语）
- `facebook/mms-tts-yor`（约鲁巴语）
- `facebook/mms-tts-rus`（俄语）
- `facebook/mms-tts-vie`（越南语）
- `facebook/mms-tts-tur`（土耳其语）

**输入/输出定义**：
- 输入：目标语言代码、网络连接
- 输出：`./model/config.json`、`./model/model.safetensors`、tokenizer 文件
- 异常：如果下载中断，请重新运行 `wget -c` 断点续传；如果 huggingface_hub 报错，请检查 `HF_ENDPOINT` 是否已导出。

---

## 3. 基础推理验证

**执行步骤**：
1. 使用单条文本合成语音并验证输出 WAV 文件
2. 检查音频文件为有效 PCM 16-bit 格式

选择对应语言的测试文本，运行推理脚本：

## 3.1 单条文本合成

```bash
python3 scripts/inference.py \
  --model_path ./model \
  --text "नमस्ते, वाक् संश्लेषण की दुनिया में आपका स्वागत है।" \
  --output output.wav
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型权重路径 | `./model` |
| `--text` | 输入文本（需匹配模型语言） | - |
| `--output` | 输出音频路径 | `output.wav` |
| `--speaking_rate` | 语速倍率 | `1.0` |
| `--benchmark` | 启用性能测试模式 | `False` |

## 3.2 各语言测试文本

| 语言 | 测试文本 |
|------|----------|
| 印地语 | `नमस्ते, वाक् संश्लेषण की दुनिया में आपका स्वागत है।` |
| 阿拉伯语 | `مرحبًا بك في عالم تركيب الكلام.` |
| 印尼语 | `Halo, selamat datang di dunia sintesis ucapan.` |
| 西班牙语 | `Hola, bienvenido al mundo de la síntesis de voz.` |
| 约鲁巴语 | `Ẹ káàbọ̀ sí agbára ọ̀rọ̀ àsọyé.` |
| 俄语 | `Здравствуйте, добро пожаловать в мир синтеза речи.` |
| 越南语 | `Xin chào, chào mừng bạn đến với thế giới tổng hợp giọng nói.` |
| 土耳其语 | `Merhaba, konuşma sentezi dünyasına hoş geldiniz.` |

## 3.3 核心代码逻辑

```python
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wavfile

model = VitsModel.from_pretrained("./model").to("npu")
tokenizer = AutoTokenizer.from_pretrained("./model")
inputs = tokenizer(text, return_tensors="pt").to("npu")

with torch.no_grad():
    output = model(**inputs).waveform

waveform = output[0].cpu().numpy()
wav_data = (waveform * 32767).astype("int16")
wavfile.write("output.wav", rate=model.config.sampling_rate, data=wav_data)
```

**输入/输出定义**：
- 输入：文本字符串、模型路径
- 输出：`output.wav`（PCM 16-bit，采样率由 `model.config.sampling_rate` 决定）
- 异常：如果输出为静音或噪声，请检查输入文本语言是否与模型语言匹配。

---

## 4. 精度验证

**执行步骤**：
1. 运行精度验证脚本进行多轮推理
2. 检查 NPU 自一致性：Mel Mean 方差 < 3.0
3. 检查 CPU-NPU 结构一致性：Mel Mean 差值 < 2.0

> **重要说明**：VITS 配置了 `use_stochastic_duration_prediction=true`（随机时长预测器）
> 和 `noise_scale=0.667`（噪声注入），是生成式模型而非确定性模型。
> 同一文本每次推理会生成不同长度、不同波形的音频（CPU 上同一文本跑两次，
> 波形余弦相似度接近 0，长度差异可达 20%）。
> 因此验证聚焦于输出有效性和频谱分布稳定性，而非逐点波形匹配。

验证维度：

- **NPU 自一致性**：同一文本在 NPU 上多次运行（3 次），梅尔频谱统计量保持稳定
- **CPU-NPU 结构一致性**：CPU 与 NPU 均能生成有效语音波形，频谱统计量差异在合理范围内

运行命令：

```bash
# 设置测试文本（可选，默认使用内置测试集）
export MMS_TEST_TEXTS="text1||text2||text3"

python3 scripts/accuracy_run.py ./model results/eval.json
```

## 4.1 NPU 自一致性标准

| 指标 | 阈值 |
|------|------|
| Mel Mean 方差 | < 3.0 |
| Mel Std 方差 | < 2.0 |
| 输出有效性 | 非零、有限值、范围合理 |

## 4.2 CPU-NPU 结构一致性标准

| 指标 | 阈值 |
|------|------|
| Mel Mean 差值 | < 2.0 |
| Mel Std 差值 | < 2.0 |
| 输出有效性 | CPU 和 NPU 均为有效语音波形 |

---

## 5. 性能基准测试

**执行步骤**：
1. 运行性能测试脚本进行多轮推理
2. 检查平均延迟、RTF、字符吞吐指标
3. 确认 RTF < 1.0 满足实时合成要求

```bash
python3 scripts/accuracy_run_perf.py ./model 10 results/perf_report.json
```

## 5.1 性能指标说明

| 指标 | 含义 |
|------|------|
| 平均延迟 | 模型推理平均耗时（ms） |
| P50 / P90 延迟 | 中位数 / 90 分位延迟 |
| RTF (Real-Time Factor) | 合成时间 / 音频时长，< 1 即满足实时 |
| 字符吞吐 | 每秒合成字符数 |

## 5.2 参考性能（Ascend 910B4 实测）

| 语言 | 平均延迟 | RTF | 字符吞吐 |
|------|----------|-----|----------|
| 印地语 | ~101 ms | 0.029 | ~404 chars/s |
| 阿拉伯语 | ~114 ms | 0.025 | ~285 chars/s |
| 印尼语 | ~107 ms | 0.028 | ~397 chars/s |
| 西班牙语 | ~105 ms | 0.027 | ~437 chars/s |
| 约鲁巴语 | ~104 ms | 0.029 | ~421 chars/s |

> 首次推理包含图编译开销，延迟约 42s，后续推理稳定在 ~100ms。
> NPU 推理显存占用约 500MB，适合资源受限环境部署。

---

## 6. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| `No module named 'torch_npu'` | CANN 未加载或 torch_npu 未安装 | 暂停执行，回滚到步骤 0.1 | `source set_env.sh` 后重装 torch_npu |
| `transfer_to_npu` 警告 | torch_npu 正常初始化行为 | 可忽略 | 无需操作 |
| 首次推理极慢（~42s） | 图编译开销 | 正常行为 | 第二次调用性能明显提升 |
| 不同次输出波形不同 | VITS 随机时长预测器 | 正常行为 | 听感应保持一致 |
| 音频听感异常 | 文本与模型语言不匹配 | 失败，需用户确认输入 | 确认输入文本与模型语言一致 |
| 下载失败 / 超时 | 网络无法访问 HuggingFace | 重试，切换镜像 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| OOM | 并发推理或 batch 过大 | 回滚，减小 batch | 单条推理，降低输入长度 |

---

## 7. 检查点与验收确认

完成以下检查清单即为部署成功。每步完成后请**用户确认**再打勾：

- [ ] **Checkpoint 1**：`npu-smi info` 显示设备正常
- [ ] **Checkpoint 2**：`import torch_npu` 无报错
- [ ] **Checkpoint 3**：`scripts/inference.py` 成功输出 WAV 文件，内容为有效语音
- [ ] **Checkpoint 4**：`scripts/accuracy_run.py` 输出 PASS，梅尔频谱统计量稳定
- [ ] **Checkpoint 5**：`scripts/accuracy_run_perf.py` 输出稳定延迟指标，RTF < 1.0

---

## 8. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| 推理脚本 | `scripts/inference.py` |
| 精度验证脚本 | `scripts/accuracy_run.py` |
| 性能测试脚本 | `scripts/accuracy_run_perf.py` |
| 精度报告 | `results/eval.json` |
| 性能报告 | `results/perf_report.json` |
| 模型权重 | `./model/model.safetensors`（~138MB） |
| 参考文档 | https://www.hiascend.com/document/ |
| HuggingFace 镜像 | https://hf-mirror.com |

---

## 附录：已适配模型列表

| 语言 | 代码 | HF 模型 ID | 精度验证 | 性能测试 |
|------|------|-------------|----------|----------|
| 印地语 | hin | `facebook/mms-tts-hin` | ✅ PASS | ✅ ~101ms |
| 阿拉伯语 | ara | `facebook/mms-tts-ara` | ✅ PASS | ✅ ~114ms |
| 印尼语 | ind | `facebook/mms-tts-ind` | ✅ PASS | ✅ ~107ms |
| 西班牙语 | spa | `facebook/mms-tts-spa` | ✅ PASS | ✅ ~105ms |
| 约鲁巴语 | yor | `facebook/mms-tts-yor` | ✅ PASS | ✅ ~104ms |
| 俄语 | rus | `facebook/mms-tts-rus` | ✅ PASS | ✅ ~107ms |
| 越南语 | vie | `facebook/mms-tts-vie` | ✅ PASS | ✅ ~115ms |
| 土耳其语 | tur | `facebook/mms-tts-tur` | ✅ PASS | ✅ ~110ms |
