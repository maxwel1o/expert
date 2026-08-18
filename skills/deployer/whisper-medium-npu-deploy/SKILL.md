---
name: whisper-medium-npu-deploy
description: "whisper-medium（OpenAI ~769M 语音识别 ASR 模型）在昇腾 NPU 上的完整部署与推理验证。涵盖环境准备、模型下载、NPU 推理部署、CPU/NPU 精度对比验证、性能基准测试的全流程。适用于：NPU 部署、昇腾推理、ASR 精度验证、Whisper 性能评测。触发词：whisper-medium NPU 部署、whisper 昇腾推理、whisper-medium 精度对比、whisper ASR NPU 部署。"
---

# whisper-medium 昇腾 NPU 部署与推理验证 Skill

> 在昇腾 NPU 上自动部署 openai/whisper-medium（~769M 参数，ASR 模型），完成环境初始化、模型下载、CPU/NPU 精度对比验证、性能基准测试和验收确认的全流程。执行流程分 8 步：先环境检查和 NPU 检测，再安装依赖、模型下载、基础推理验证、精度对比、性能测试，最后验收确认。

## 概述

本 Skill 用于自动完成 **whisper-medium 语音识别模型** 在昇腾 NPU 上的部署、推理验证、CPU/NPU 精度对比、性能基准测试和验收确认。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910) |
| 框架版本 | PyTorch + transformers + torch_npu |
| 模型参数量 | ~769M (Encoder-Decoder Transformer, 24 layers, 1024 dim, 16 heads) |
| 精度目标 | CPU 与 NPU 推理 logits 最大概率差异 < 1% |
| 执行方式 | 串行 8 步流程，每步完成后再进入下一步 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 测试音频 | 合成 16kHz WAV（4 秒）或用户提供 |
| 性能指标 | 端到端延迟、Encoder 延迟、实时倍率 (RTF) |

## 前置条件

| 项目 | 要求 |
|:---|:---|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0 |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（~3GB） |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.9+ 环境，昇腾 NPU 驱动，CANN >= 8.0。

**动作**:
1. 加载 CANN 环境：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 检查 Python 版本，确认 >= 3.9：
```bash
python3 --version
```

3. 检查 NPU 状态并选择空闲设备：
```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

4. （可选）设置国内 pip 镜像加速：
```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

5. 确认 torch_npu 可用性：
```python
import torch
import torch_npu
print(f"NPU available: {torch.npu.is_available()}")
if torch.npu.is_available():
    print(f"NPU device: {torch.npu.get_device_name(0)}")
    print(f"NPU device count: {torch.npu.device_count()}")
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，CANN 环境已加载，空闲设备号已选择。

### Step 2: 安装依赖

**输入**: 上一步确认的 NPU 可用状态。

**动作**:
6. 安装 torch_npu（版本必须与已安装的 torch 匹配）：
```bash
pip install torch_npu
```

7. 验证 torch 与 torch_npu 版本一致：
```bash
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}, torch_npu: {torch_npu.__version__}')"
```

8. 安装其他依赖：
```bash
pip install transformers soundfile librosa numpy huggingface_hub
```

9. 如需使用 ModelScope 镜像下载模型，额外安装：
```bash
pip install modelscope
```

**输出**: 依赖包已安装完成，版本兼容性已确认。

### Step 3: 模型下载

**输入**: 目标模型名称 `openai/whisper-medium`。

**动作**:
10. 方式一（HuggingFace 镜像，国内推荐）：
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openai/whisper-medium --local-dir ./whisper-medium
```

11. 方式二（ModelScope，国内备选）：
```bash
python3 -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('openai/whisper-medium', cache_dir='./whisper-medium')"
```

12. 验证模型文件完整性：
```bash
ls -la ./whisper-medium/
# 应包含 config.json, model.safetensors, preprocessor_config.json 等
```

13. 设置模型路径环境变量：
```bash
export MODEL_PATH=./whisper-medium
export WHISPER_MODEL_PATH=./whisper-medium
```

**输出**: 模型权重已下载到本地目录，模型路径环境变量已设置。

### Step 4: 准备测试音频

**输入**: 模型路径。

**动作**:
14. 生成 16kHz 测试音频（4 秒合成语音）：
```bash
python3 -c "
import numpy as np
import soundfile as sf
sr = 16000
duration = 4
t = np.linspace(0, duration, int(sr*duration), endpoint=False)
f0, amp = 800, 0.5
audio = amp * np.sin(2*np.pi*f0*t) * (0.5+0.5*np.sin(2*np.pi*3*t))
audio = audio / np.max(np.abs(audio)) * 0.9
sf.write('test_audio.wav', audio, sr)
print('Created test_audio.wav (16kHz, 4s)')
"
```

15. （可选）使用用户提供的 WAV 音频：
```bash
# 将用户音频复制为 test_audio.wav，确保为 16kHz mono
```

16. 设置测试音频路径环境变量：
```bash
export TEST_AUDIO_PATH=./test_audio.wav
```

**输出**: `test_audio.wav` 就绪，路径环境变量已设置。

### Step 5: 基础 NPU 推理验证

**输入**: 模型路径 `$MODEL_PATH`、测试音频 `test_audio.wav`。

**动作**:
17. 复制推理脚本到工作目录：
```bash
cp scripts/whisper_npu_infer.py ./
```

18. 执行 NPU 推理：
```bash
python3 whisper_npu_infer.py test_audio.wav
```

19. 核心推理逻辑说明：
```python
import torch
import torch_npu
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = torch.device("npu:0")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.float32,
).to(device)
model.eval()

input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

with torch.no_grad():
    generated = model.generate(
        input_features, language="english", task="transcribe",
        return_timestamps=False, max_length=448,
    )

torch.npu.synchronize()
transcription = processor.batch_decode(generated, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")
```

20. 支持额外参数调整：
```bash
# 指定语言
python3 whisper_npu_infer.py test_audio.wav --language chinese

# 翻译任务（转写为英文）
python3 whisper_npu_infer.py test_audio.wav --task translate

# FP16 推理（提升吞吐）
python3 whisper_npu_infer.py test_audio.wav --dtype float16
```

**通过标准**:
- 输出转录文本（即使是无意义的合成语音）
- 无 NPU 相关报错
- 推理时间在合理范围内（4s 音频 < 500ms）

**输出**: 转录文本输出到终端。

### Step 6: CPU/NPU 精度对比验证

**输入**: 模型路径 `$MODEL_PATH`、测试音频 `test_audio.wav`。

**动作**:
21. 复制精度对比脚本到工作目录：
```bash
cp scripts/accuracy_run.py ./
```

22. 执行精度对比：
```bash
python3 accuracy_run.py
```

23. 验证原理说明 — 同时用 CPU 和 NPU 运行相同输入，对比输出 logits：
    - Logits 余弦相似度：衡量方向一致性
    - Logits 最大/平均绝对误差
    - 概率分布最大差异（Softmax 后）
    - Token 预测匹配率（argmax 一致性）

24. 精度验收标准：
| 指标 | 阈值 | 说明 |
|:---|:---:|:---|
| 最大概率差异 | < 1% | `max_prob_diff_pct < 1.0` |
| 余弦相似度 | > 0.999 | Logits 方向一致性 |
| Token 预测匹配 | 100% | CPU 与 NPU 生成 token 一致 |

25. 查看结果文件：
```bash
cat accuracy_results.json
```

**输出**: `accuracy_results.json`，包含完整精度对比指标和 PASS/FAIL 结论。

### Step 7: 性能基准测试

**输入**: 模型路径 `$MODEL_PATH`、测试音频 `test_audio.wav`。

**动作**:
26. 复制性能测试脚本到工作目录：
```bash
cp scripts/accuracy_run_perf.py ./
```

27. 执行性能基准测试（默认 20 轮 warmup + 20 轮正式测试）：
```bash
python3 accuracy_run_perf.py
```

28. 预期性能参考（Ascend910B4, FP32, 4s 音频输入）：
| 指标 | 参考值 |
|:---|:---:|
| Encoder 延迟 | ~37 ms |
| 端到端延迟（均值） | ~186 ms |
| 实时倍率 (RTF) | ~0.046 |
| 加速比（vs 实时） | ~21x |

29. 查看结果文件：
```bash
cat perf_results.json
```

**输出**: `perf_results.json`，包含延迟统计、RTF 和加速比。

### Step 8: 验收确认

**输入**: 以上各步输出结果。

**动作**:
30. 逐项验收检查清单：
```bash
echo "=== 验收检查清单 ==="
echo "[ ] NPU 设备正常: $(npu-smi info | grep -c Ascend) 卡可用"
echo "[ ] torch_npu 可导入: $(python3 -c 'import torch_npu; print("OK")' 2>/dev/null || echo "FAIL")"
echo "[ ] 模型下载完整: $(ls -1 ./whisper-medium/*.safetensors 2>/dev/null | wc -l) 个权重文件"
echo "[ ] NPU 推理通过: $(python3 whisper_npu_infer.py test_audio.wav 2>&1 | grep -c Transcription)" 
echo "[ ] 精度对比通过: $(python3 -c "import json; d=json.load(open('accuracy_results.json')); print('PASS' if d.get('token_match') and d.get('max_prob_diff_pct',100)<1 else 'FAIL')")"
echo "[ ] 性能测试完成: $(ls perf_results.json 2>/dev/null && echo "OK" || echo "MISSING")"
```

31. 汇总所有结果路径：
```bash
ls -la accuracy_results.json perf_results.json test_audio.wav
```

**输出**: 验收结论（全部通过 / 存在未通过项），完整的结果文件集。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | Step 1 完成后 | NPU 设备是否可用，CANN 版本是否 >= 8.0 | 暂停，提示安装 CANN 或昇腾驱动 |
| 2 | CP-2: 依赖安装检查点 | Step 2 完成后 | torch_npu 与 torch 版本是否一致 | 检查版本，使用 compatible 版本重装 |
| 3 | CP-3: 模型下载检查点 | Step 3 完成后 | 模型文件是否下载完整（含 config.json 和 safetensors） | 切换镜像源或手动下载后重试 |
| 4 | CP-4: 音频就绪检查点 | Step 4 完成后 | 测试音频是否正确（16kHz, mono） | 重新生成或提供正确格式的音频文件 |
| 5 | CP-5: 基础推理检查点 | Step 5 完成后 | 是否输出转录文本，推理耗时是否合理 | 检查模型加载和 NPU 驱动状态后重试 |
| 6 | CP-6: 精度验证检查点 | Step 6 完成后 | 精度误差是否 < 1%，余弦相似度是否 > 0.999 | 检查推理脚本和数据一致性后重试 |
| 7 | CP-7: 性能测试检查点 | Step 7 完成后 | 性能指标是否在合理范围，RTF 是否 < 0.1 | 检查 NPU 负载，释放其他进程后重试 |
| 8 | CP-8: 验收确认检查点 | Step 8 完成后 | 所有检查项是否全部通过 | 返回未通过项重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查 NPU 驱动 |
| CANN 环境未加载 | `source set_env.sh` 失败或找不到文件 | 提示检查 CANN 安装路径和版本 | CP-1 | 确认 CANN 安装后重试 |
| NPU 显存 OOM | 推理时报内存不足 | 释放缓存，切换 FP16，减小输入音频长度 | CP-5 | 释放其他进程或重启 NPU 设备 |
| 模型下载失败 | huggingface-cli 或 modelscope 返回错误 | 重试最多 3 次，每次间隔 5 秒 | CP-3 | 切换镜像源或手动下载权重到本地 |
| 模型加载异常 | `WhisperForConditionalGeneration.from_pretrained` 抛出异常 | 打印错误堆栈，提示模型路径是否正确 | CP-3 | 修正模型路径或重新下载 |
| 安装网络超时 | pip install 长时间无响应 | 重试最多 3 次，使用国内镜像源 | CP-2 | 切换 pip 镜像源或离线安装 |
| 音频格式错误 | soundfile 无法读取或采样率不匹配 | 自动尝试 librosa 重采样至 16kHz | CP-4 | 使用 librosa 自动重采样 |
| `ModuleNotFoundError: No module named 'soundfile'` | 缺少音频处理库 | 自动安装缺失模块 soundfile/librosa | CP-2 | `pip install soundfile librosa` |
| 精度超标异常 | max_prob_diff_pct >= 1% 或 token 不匹配 | 记录偏差明细，中止后续步骤，打印诊断信息 | CP-6 | 检查推理脚本和 NPU 驱动版本一致性 |
| 多卡抢占冲突 | 多个进程使用同一 NPU 卡导致 OOM 或性能下降 | 提示使用 `npu-smi info` 选择空闲卡 | CP-1 | 使用 `ASCEND_RT_VISIBLE_DEVICES` 指定空闲卡号 |
| 性能异常偏差 | 端到端延迟远超预期（> 正常值 3 倍） | 记录异常，提示检查 NPU 负载和频率 | CP-7 | 释放其他进程，清理 NPU 缓存后重试 |
| 验收未通过 | 检查清单中存在未通过项 | 列出所有未通过项，暂停流程等待用户处理 | CP-8 | 逐项排查并重新运行对应步骤 |
| 磁盘空间不足 | 模型权重下载失败或写入错误 | 提示检查磁盘空间后重试 | CP-3 | 释放磁盘空间后重试 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/whisper_npu_infer.py` | NPU 推理入口脚本，支持语言选择、FP16 开关、翻译任务 |
| `scripts/accuracy_run.py` | CPU/NPU 精度对比脚本，输出 logits 差异 + 概率差异 + token 匹配率 |
| `scripts/accuracy_run_perf.py` | 性能基准测试脚本，输出端到端延迟、Encoder 延迟、RTF 和加速比 |
| `whisper_npu_infer.py` | 推理脚本副本，复制到工作目录后直接运行 |
| `accuracy_run.py` | 精度对比脚本副本，复制到工作目录后直接运行 |
| `accuracy_run_perf.py` | 性能测试脚本副本，复制到工作目录后直接运行 |
| `test-prompts.json` | 结构评测用测试提示词 |
| `accuracy_results.json` | CPU/NPU 精度对比结果（运行后生成）：logits 差异 + 概率差异 + token 匹配 |
| `perf_results.json` | 性能基准测试结果（运行后生成）：延迟统计 + RTF + 加速比 |
| `test_audio.wav` | 测试音频文件（运行后生成）：16kHz 合成语音，4 秒时长 |

## 精度要求

- NPU 与 CPU 推理 logits 最大概率差异必须 < 1%
- 对比指标：余弦相似度、最大/平均绝对误差、概率分布差异、Token 预测匹配率
- 结论标记：`max_prob_diff_pct < 1.0` 且 `token_match == true` 时记为 `PRECISION_PASS`

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_path` | string | 否 | `./whisper-medium` | 模型权重目录路径 |
| `audio_path` | string | 否 | `test_audio.wav` | 测试音频文件路径（16kHz WAV） |
| `language` | string | 否 | `english` | ASR 语言（english, chinese, french 等） |
| `task` | string | 否 | `transcribe` | 任务类型：transcribe（转写）或 translate（翻译为英文） |
| `dtype` | string | 否 | `float32` | 推理精度：float32 或 float16 |
| `num_runs` | int | 否 | 20 | 性能测试推理重复次数 |
| `audio_duration` | int | 否 | 4 | 合成测试音频时长（秒） |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| 转录文本 | str | whisper-medium 对输入音频的自动语音识别结果 |
| `accuracy_results.json` | JSON | CPU/NPU 精度对比：logits 差异 + 概率差异 + 余弦相似度 + token 匹配 + PASS/FAIL |
| `perf_results.json` | JSON | 性能基准：端到端延迟（均/中/最值/标准差/P90/P99）+ Encoder 延迟 + RTF + 加速比 |
| `test_audio.wav` | WAV | 16kHz 单声道测试音频（运行后生成） |

## 使用约束

1. 使用 HuggingFace 镜像（hf-mirror.com）或 ModelScope 下载模型权重（HF 官方可能无法访问）。
2. 验证精度通过前不进行后续步骤（必须有 max_prob_diff_pct < 1% 标记）。
3. 首次运行需联网下载 ~3GB 模型权重，建议在网络稳定的环境下执行。
4. 多卡环境需使用 `ASCEND_RT_VISIBLE_DEVICES` 选择空闲 NPU 卡，避免抢占冲突。
5. 测试前确认 Ascend910 驱动和 CANN 环境已正确安装（>= 8.0）。
6. 性能基准测试建议在无其他 NPU 进程干扰的环境下执行。
