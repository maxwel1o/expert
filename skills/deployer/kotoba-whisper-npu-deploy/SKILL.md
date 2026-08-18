---
name: kotoba-whisper-npu-deploy
description: >
  Kotoba-Whisper 系列（6 个模型）在昇腾 NPU 上的完整部署与验证 Skill。
  涵盖环境准备、权重下载、NPU 推理、精度对比（NPU vs CPU）、性能基准测试的全流程。
  支持 kotoba-whisper v1.0/v1.1/v2.0/v2.1/v2.2 及 bilingual-v1.0 共 6 个模型。
  当用户提到 kotoba-whisper 昇腾部署、Whisper NPU 推理、日语语音识别 NPU 时触发。
metadata:
  short-description: Kotoba-Whisper 昇腾 NPU 部署与推理验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, whisper, kotoba-whisper, asr, speech-recognition, japanese, pytorch, inference]
---

# Kotoba-Whisper 昇腾 NPU 部署 Skill

本 Skill 提供 **6 个 Kotoba-Whisper 模型**在华为昇腾 NPU（Ascend910B4）上的完整部署流程：

| # | 模型 | HuggingFace ID | 参数量 | 特点 |
|---|------|----------------|--------|------|
| 1 | kotoba-whisper-bilingual-v1.0 | `kotoba-tech/kotoba-whisper-bilingual-v1.0` | ~756M | 日英双语，bfloat16 |
| 2 | kotoba-whisper-v1.0 | `kotoba-tech/kotoba-whisper-v1.0` | ~756M | 日语 ASR，fp32 |
| 3 | kotoba-whisper-v1.1 | `kotoba-tech/kotoba-whisper-v1.1` | ~756M | v1.0 改进版，fp32 |
| 4 | kotoba-whisper-v2.0 | `kotoba-tech/kotoba-whisper-v2.0` | ~756M | v2 系列初版，bfloat16 |
| 5 | kotoba-whisper-v2.1 | `kotoba-tech/kotoba-whisper-v2.1` | ~756M | v2.0 改进版，fp32 |
| 6 | kotoba-whisper-v2.2 | `kotoba-tech/kotoba-whisper-v2.2` | ~756M | 最新版，fp32 |

所有模型均为 `WhisperForConditionalGeneration` 架构，编码器 32 层，解码器 2 层（Distil-Whisper 变体）。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910B4 或更高（至少 1 卡，30GB HBM） |
| CANN | >= 8.5.1 |
| Python | 3.10 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（每模型 ~1.5~3GB） |
| 磁盘 | 至少 20GB 可用空间（模型缓存 + 评测数据） |
| 内存 | 建议 >= 32GB（加载 756M 参数模型） |

## 工作流概览

```
1. 环境初始化 → 2. NPU 基础验证 → 3. 模型选择 → 4. NPU 推理
→ 5. 精度验证 → 6. 性能基准 → 7. 结果归档 → 8. 验收交付
```

按以下各节顺序执行，每步完成后确认检查点再进入下一步。

---

## 1. 环境初始化

### 1.1 加载 CANN 环境

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 国内 pip 镜像（可选，加速）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/

# HuggingFace 镜像（国内网络必需）
export HF_ENDPOINT=https://hf-mirror.com
```

### 1.2 选择空闲 NPU 卡

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

**验证方法**：`npu-smi info` 输出中目标卡的状态为 `Normal`。

### 1.3 安装依赖

```bash
pip install torch torch_npu transformers soundfile librosa accelerate
```

验证安装：

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```

**通过标准**：版本正常输出且无报错。

### 检查点

- [ ] `source set_env.sh` 执行无报错
- [ ] `npu-smi info` 显示设备状态 Normal
- [ ] `torch_npu` 导入成功，版本正常输出

---

## 2. NPU 基础验证

验证 NPU 设备可用性和基本运算：

```bash
python3 -c "
import torch
import torch_npu
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('NPU name:', torch.npu.get_device_name(0))
a = torch.randn(3, 4).npu()
print('Basic tensor on NPU:', a + a)
"
```

**通过标准**：
- 返回 `NPU available: True`
- tensor 在 NPU 上运算无报错

### 检查点

- [ ] `torch.npu.is_available()` 返回 True
- [ ] NPU 上创建 tensor 和基本运算正常

### 异常处理

| 异常场景 | 原因 | 解决方案 |
|---------|------|----------|
| `torch_npu` 导入失败 | CANN 环境未加载或版本不匹配 | 检查 `source set_env.sh`，确认 CANN >= 8.5.1 |
| NPU 不可用 | 驱动问题或无空闲设备 | 运行 `npu-smi info` 确认设备状态，尝试重启驱动 |
| `torch.npu.device_count()` 返回 0 | 驱动未识别 NPU | 检查 `npu-smi info`，确认 `ASCEND_RT_VISIBLE_DEVICES` 设置正确 |

---

## 3. 模型选择

本 Skill 支持 6 个模型，根据用户需求选择目标模型：

```bash
# 方法一：直接在命令行指定模型
MODEL_NAME="kotoba-whisper-v1.0"   # 替换为其他模型名

# 方法二：确认 HF 镜像可访问
curl -sI "https://hf-mirror.com/kotoba-tech/${MODEL_NAME}" | head -1
```

下载测试权重：

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = f"kotoba-tech/{MODEL_NAME}"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)
# 权重自动缓存至 ~/.cache/huggingface/hub/
```

**注意**：首次下载需联网，每个模型权重约 1.5~3GB，请确保网络稳定。

### 检查点

- [ ] 模型 ID 格式为 `kotoba-tech/{MODEL_NAME}`
- [ ] `from_pretrained` 成功，无下载超时或 LFS 错误
- [ ] HF_ENDPOINT 已正确设置（国内网络使用 `https://hf-mirror.com`）

### 异常处理

| 异常场景 | 原因 | 解决方案 |
|---------|------|----------|
| 权重下载失败/超时 | 国内网络限制 | 设置 `export HF_ENDPOINT=https://hf-mirror.com` |
| 下载到 LFS 指针文件 | git-lfs 未正确安装 | 使用 `snapshot_download` 而非 git clone |
| 404 Not Found | 模型 ID 错误 | 确认 HuggingFace ID 拼写无误 |

---

## 4. NPU 推理

使用 `scripts/inference.py` 进行 NPU 推理：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/inference.py \
  --model kotoba-tech/kotoba-whisper-v1.0 \
  --audio speech.wav \
  --language ja \
  --task transcribe \
  --device npu
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | HuggingFace 模型 ID | 必填 |
| `--audio` | 输入音频路径（.wav） | 必填 |
| `--language` | 语言代码 | `ja` |
| `--task` | 任务类型（transcribe/translate） | `transcribe` |
| `--device` | 推理设备（npu/cpu） | `npu` |

**无音频时自动生成**：脚本在 `--audio` 文件不存在时会自动合成 3 秒测试音频（16kHz 正弦扫频）。

### 推理示例输出

```
Model: kotoba-tech/kotoba-whisper-v1.0
Device: Ascend910B4
Audio duration:  3.0s
Process time:    18.5s
RTF:             6.17 (0.16x real-time)
```

### 关键实现细节

```python
# NPU 推理核心逻辑
def transcribe(audio_path, model, processor, device="npu", language="ja", task="transcribe"):
    import soundfile as sf
    import librosa

    # 1. 加载并重采样音频到 16kHz
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

    # 2. CPU 端特征提取（Mel 频谱）
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # 3. NPU 端生成
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    with torch.no_grad():
        generated_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # 4. 解码结果
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription
```

### 检查点

- [ ] 推理脚本无报错完成
- [ ] 输出包含日语转写文本
- [ ] RTF < 10（实时因子小于 10 倍）

### 异常处理

| 异常场景 | 原因 | 解决方案 |
|---------|------|----------|
| OOM（Out of Memory） | NPU 显存不足 | 减小 `max_new_tokens`，确保单条音频推理 |
| 音频加载失败 | 格式不支持 | 使用 `ffmpeg` 转换为 16kHz 单声道 WAV |
| `soundfile` 不支持格式 | 缺少编解码器 | 安装 `pip install pydub` 或使用 `librosa` 直接加载 |
| 推理结果为空 | 任务参数不匹配 | 检查 `language` 和 `task`，日语 ASR 确保 `language=ja` |
| GPU 内存不足回滚 | 设备占用过高 | 先 `npu-smi info` 确认空闲卡，或释放其他进程 |

---

## 5. 精度验证

使用 `scripts/eval_accuracy.py` 对比 NPU 与 CPU 的 logits 输出：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/eval_accuracy.py --model kotoba-tech/kotoba-whisper-v1.0
```

### 评测方法

1. 加载模型到 CPU，前向推理得到参考 logits
2. 加载模型到 NPU，前向推理得到 NPU logits
3. 计算余弦相似度、最大/平均绝对误差
4. 判定标准：余弦相似度 > 0.99 且相对误差 < 1%

### 精度结果（全部 6 个模型）

| 模型 | 文本匹配 | 余弦相似度 | 相对误差(范围) | 结论 |
|------|---------|-----------|---------------|------|
| bilingual-v1.0 | PASS | 1.000000 | 0.19% | **PASS** |
| v1.0 | PASS | 0.999995 | 0.13% | **PASS** |
| v1.1 | PASS | 0.999995 | 0.13% | **PASS** |
| v2.0 | PASS | 0.999999 | 0.20% | **PASS** |
| v2.1 | PASS | 0.999999 | 0.20% | **PASS** |
| v2.2 | PASS | 0.999999 | 0.20% | **PASS** |

**通过标准**：余弦相似度 > 0.99 且文本完全匹配。所有模型均通过。

### NPU vs CPU 生成文本一致性

每个模型在 NPU 和 CPU 上的 `generate()` 输出文本完全一致（日语转写结果相同）。

### 检查点

- [ ] 每个模型余弦相似度 > 0.99
- [ ] 相对误差 < 1%
- [ ] eval 报告 JSON 正确输出

### 异常处理

| 异常场景 | 原因 | 解决方案 |
|---------|------|----------|
| 编码器误差偏大 | 32 层累积浮点差异 | 这是 FP32 正常现象，余弦相似度仍 > 0.999 |
| CPU 精度对比耗时过长 | CPU 推理比 NPU 慢 6-7 倍 | 可复用缓存结果，仅验证关键模型 |
| 精度评测结果不一致 | 随机性（dropout 层） | 确保 `model.eval()` 已调用 |

---

## 6. 性能基准测试

使用 `scripts/eval_performance.py` 进行 CPU vs NPU 性能对比：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/eval_performance.py --model kotoba-tech/kotoba-whisper-v1.0
```

**测试条件**：3 秒合成音频（16kHz 正弦扫频），`max_new_tokens=128`，各 5 轮取均值。

### 性能结果

| 模型 | CPU 均值 | NPU 均值 | 加速比 |
|------|---------|---------|--------|
| bilingual-v1.0 | 123.1s | 17.1s | **7.2x** |
| v1.0 | 121.9s | 19.2s | **6.4x** |
| v1.1 | 124.4s | 17.7s | **7.0x** |
| v2.0 | 124.4s | 18.7s | **6.6x** |
| v2.1 | 125.0s | 18.3s | **6.8x** |
| v2.2 | 126.5s | 18.5s | **6.8x** |

**通过标准**：NPU 推理正常完成，加速比 > 5x。所有模型通过。

### 检查点

- [ ] 加速比 > 5x
- [ ] 5 轮推理时间标准差 < 20%
- [ ] 性能 JSON 报告输出到 `{model}_perf.json`

### 异常处理

| 异常场景 | 原因 | 解决方案 |
|---------|------|----------|
| 首次推理慢（含编译） | torch_npu 图编译预热 | 启用 warmup 轮次（脚本默认 2 轮），取后续稳定值 |
| NPU 性能波动大 | 设备温度/频率动态调整 | 增加测试轮数至 10 轮，取中位数 |
| CPU 推理被中断 | 内存不足或 OOM | 减小 `max_new_tokens`，或增加系统 swap |

---

## 7. 结果归档与报告生成

### 7.1 保存评测结果

```bash
# 精度评测结果
python3 scripts/eval_accuracy.py --model kotoba-tech/kotoba-whisper-v1.0 --output results/
mv kotoba-whisper-v1.0_eval.json results/

# 性能评测结果
python3 scripts/eval_performance.py --model kotoba-tech/kotoba-whisper-v1.0 --output results/
mv kotoba-whisper-v1.0_perf.json results/
```

### 7.2 生成汇总报告

```bash
# 收集所有模型结果
mkdir -p results/
for model in bilingual-v1.0 v1.0 v1.1 v2.0 v2.1 v2.2; do
  python3 scripts/eval_accuracy.py --model kotoba-tech/kotoba-whisper-${model}
  python3 scripts/eval_performance.py --model kotoba-tech/kotoba-whisper-${model}
done
# 合并评测结果
cat results/*_eval.json > results/all_results.json
```

### 检查点

- [ ] 所有 6 个模型的精度和性能数据已收集
- [ ] `results/all_results.json` 存在且包含完整数据
- [ ] 评测结果与文档一致

---

## 8. 验收与交付

### 8.1 最终检查清单

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] `inference.py` 在 NPU 上输出日语转写文本
- [ ] `eval_accuracy.py` 输出余弦相似度 > 0.99
- [ ] `eval_performance.py` 输出 NPU 加速比 > 5x
- [ ] 所有 6 个模型均完成精度和性能验证
- [ ] 评测结果已归档至 `results/` 目录
- [ ] 模型仓库在 `https://ai.gitcode.com/m0_74196153/{model_name}` 可访问

### 8.2 交付物

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 推理脚本 | `scripts/inference.py` | NPU 推理入口 |
| 精度验证脚本 | `scripts/eval_accuracy.py` | NPU vs CPU 精度对比 |
| 性能基准脚本 | `scripts/eval_performance.py` | CPU vs NPU 性能对比 |
| 评测结果 | `results/*_eval.json` | 各模型精度数据 |
| 性能报告 | `results/*_perf.json` | 各模型性能数据 |
| 汇总报告 | `results/all_results.json` | 全模型合并结果 |

---

## 检查点总表

| # | 阶段 | 检查项 | 通过标准 |
|---|------|--------|---------|
| 1 | 环境初始化 | CANN 环境加载 | `source set_env.sh` 无报错 |
| 2 | 环境初始化 | NPU 状态确认 | `npu-smi info` 显示 Normal + 卡号匹配 |
| 3 | 环境初始化 | 依赖安装验证 | `torch_npu` 导入成功，版本输出正常 |
| 4 | NPU 基础验证 | 设备可用性 | `torch.npu.is_available()` 返回 True |
| 5 | NPU 基础验证 | 基础运算 | NPU tensor 运算无报错 |
| 6 | 模型选择 | 权重下载 | `from_pretrained` 成功，无超时或 LFS 错误 |
| 7 | NPU 推理 | 推理成功率 | 脚本无报错完成，输出有效转录文本 |
| 8 | NPU 推理 | 实时因子 | RTF < 10 |
| 9 | 精度验证 | 余弦相似度 | > 0.99 |
| 10 | 精度验证 | 相对误差 | < 1% |
| 11 | 精度验证 | 文本一致性 | NPU vs CPU 输出文本完全匹配 |
| 12 | 性能基准 | 加速比 | > 5x |
| 13 | 性能基准 | 结果稳定性 | 5 轮标准差 < 20% |
| 14 | 结果归档 | 数据完整性 | 6 个模型数据全部收集 |
| 15 | 验收交付 | 检查清单 | 所有验收项通过 |

---

## 边界条件与异常处理

### 网络与下载

| 异常场景 | 触发条件 | 严重程度 | 恢复策略 |
|---------|---------|---------|---------|
| HF 镜像不可达 | `curl hf-mirror.com` 返回 5xx | 高 | 回滚至官方 HF 源 `export HF_ENDPOINT=https://huggingface.co` |
| 权重下载超时 | 大文件下载中断 | 中 | retry 3 次，每次间隔 5s，使用 `resume_download=True` |
| LFS 指针文件 | git clone 而非 snapshot | 中 | fallback 到 `snapshot_download` 方法 |
| pip 安装失败 | 镜像源不稳定 | 低 | 切换 PyPI 官方源 `-i https://pypi.org/simple/` |

### 硬件与资源

| 异常场景 | 触发条件 | 严重程度 | 恢复策略 |
|---------|---------|---------|---------|
| NPU 卡被占用 | `npu-smi info` 显示显存已满 | 高 | 切换到其他空闲卡（修改 `ASCEND_RT_VISIBLE_DEVICES`） |
| OOM（NPU） | batch 过大或模型过大 | 中 | 减少 `max_new_tokens=128`，单条推理 |
| OOM（CPU） | 内存不足 | 中 | 增加 swap 或分批处理 |
| CANN 版本不兼容 | 旧版本不支持当前 torch_npu | 高 | 升级 CANN >= 8.5.1 |

### 推理与精度

| 异常场景 | 触发条件 | 严重程度 | 恢复策略 |
|---------|---------|---------|---------|
| 推理结果乱码 | 音频质量差或语言参数错误 | 低 | 确认 `language=ja`，检查音频采样率是否为 16kHz |
| 精度对比失败 | 余弦相似度 < 0.99 | 中 | recover：检查 `model.eval()`，排除 dropout 随机性后重测 |
| 生成文本不一致 | NPU vs CPU 输出不同 | 中 | recover：确认 32 层编码器数值误差在合理范围 |
| 性能波动大 | 设备温控降频 | 低 | retry 10 轮取中位数，确认散热正常 |

---

## 资源清单

| 资源类型 | 路径/描述 | 说明 |
|---------|----------|------|
| scripts | `scripts/inference.py` | NPU 推理脚本，支持自动生成测试音频 |
| scripts | `scripts/eval_accuracy.py` | 精度验证脚本，输出 `{model}_eval.json` |
| scripts | `scripts/eval_performance.py` | 性能基准脚本，输出 `{model}_perf.json` |
| results | `results/{model}_eval.json` | 各模型精度评测结果 |
| results | `results/{model}_perf.json` | 各模型性能评测结果 |
| results | `results/all_results.json` | 全模型合并评测报告 |
| evals.json | 精度报告格式：cosine_similarity, max_abs_error, relative_error | 结构化评测数据 |
| references | `kotoba-tech/kotoba-whisper-bilingual-v1.0` | 模型原始 HuggingFace 仓库 |
| references | `https://ai.gitcode.com/m0_74196153/{model_name}` | 模型 NPU 适配仓库 |
| references | Whisper 官方文档：`https://huggingface.co/docs/transformers/model_doc/whisper` | 模型架构参考 |

---

## 适配要点

1. **特征提取在 CPU 端**：`WhisperProcessor` 的 Mel 频谱提取基于 NumPy，在 CPU 执行，不影响 NPU 推理精度
2. **采样率要求**：模型输入要求 16kHz 单声道，`inference.py` 自动重采样
3. **权重下载**：国内网络使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像
4. **bfloat16 模型**：bilingual-v1.0 和 v2.0 原生为 bfloat16，本 Skill 统一以 fp32 加载以保证精度对比公平性
5. **32 层编码器精度**：编码器层数较多，累积浮点误差在合理范围（余弦相似度仍 > 0.999）
6. **NPU 图编译预热**：首次推理包含 torch_npu 图编译，后续推理速度显著提升
