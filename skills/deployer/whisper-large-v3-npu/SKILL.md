---
name: whisper-large-v3-npu
description: "OpenAI Whisper Large V3 语音识别模型在昇腾 NPU (Ascend 910B4) 上的适配、推理部署与精度验证 Skill。涵盖环境准备、模型权重获取、NPU 推理部署、精度评测（<1% 误差）、性能基准测试的全流程。适用于将 whisper-large-v3 部署到华为昇腾 NPU 上进行自动语音识别 (ASR) 和语音翻译任务。触发词：whisper NPU 部署、whisper-large-v3 昇腾推理、ASR 模型 NPU 适配、whisper 昇腾精度评测。"
metadata:
  short-description: Whisper Large V3 昇腾 NPU 部署与推理适配
  category: NPU-Model-Deploy
  tags: [ascend, npu, whisper, asr, speech-recognition, pytorch, inference, audio, transformers, ascend910]
license: apache-2.0
---

# whisper-large-v3 昇腾 NPU 部署与推理 Skill

> 在昇腾 NPU 和 CPU 上自动部署 OpenAI Whisper Large V3 语音识别模型，完成 NPU 推理验证、CPU/NPU 精度对比（误差 < 1%）和性能基准测试。执行流程分 8 步：先环境检查和 NPU 检测，再模型权重获取、NPU 推理验证、CPU 基线推理、NPU 精度评测、性能基准测试、适配要点确认，最后验收确认。

## 概述

本 Skill 用于自动完成 **OpenAI Whisper Large V3 语音识别模型**（约 1.55B 参数）在昇腾 NPU 上的完整部署、推理验证、CPU/NPU 精度对比、性能基准测试的全流程。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend 910B4, 至少 1 卡) |
| 框架版本 | PyTorch 2.9.0+, torch_npu 2.9.0+, transformers 4.57.6+ |
| 精度目标 | CPU float32 与 NPU float16 推理 Logits RMS 相对误差 < 1% |
| 内存需求 | 模型权重约 3GB，NPU 显存约 3GB |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 依赖 | soundfile, librosa, transformers, torch, torch_npu |

## 执行工作流

### 1. 环境初始化与 NPU 检测

**输入**: Python 3.9-3.13 环境，昇腾 NPU 驱动 (CANN >= 8.5.1)。

**动作**:
1. 检查 Python 版本，确认符合要求：
```bash
python3 --version
```
2. 加载 CANN 环境并运行 NPU 检测：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```
3. 验证 torch_npu 可正常导入：
```bash
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}, torch_npu: {torch_npu.__version__}'); print(f'NPU available: {torch.npu.is_available()}'); a=torch.randn(3,4).npu(); print(a+a)"
```
4. 安装依赖（如缺失）：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision soundfile librosa transformers
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，依赖已安装完成。

### 2. 获取模型权重

**输入**: 模型名称 `openai/whisper-large-v3`。

**动作**:
5. 从 HuggingFace 镜像下载模型权重（~3GB，首次运行需要网络）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/whisper-large-v3', cache_dir='./whisper_cache')
"
```
6. 可选方式：从 GitCode 镜像克隆：
```bash
git clone https://gitcode.com/hf_mirrors/openai/whisper-large-v3.git
```
7. 验证模型文件完整性（确认包含 config.json 和 model.safetensors 或 pytorch_model.bin）：
```bash
ls -la /path/to/whisper-large-v3/
```
8. 设置环境变量 `export MODEL_PATH=/path/to/whisper-large-v3`。

**输出**: 模型权重下载完成，`MODEL_PATH` 环境变量已设置。

### 3. NPU 推理验证

**输入**: 模型路径、测试音频文件（wav/flac/mp3/ogg）。

**动作**:
9. 运行 NPU 推理脚本：
```bash
python3 scripts/inference.py --audio test.wav --language en --model ${MODEL_PATH}
```
10. 支持的任务模式：
    - 语音转录（transcribe）：`--task transcribe`
    - 语音翻译（translate）：`--task translate --language zh`
11. 检查推理输出包含文本结果和控制台性能数据。
12. 验证关键配置：
    - `attn_implementation="eager"`（NPU 必须使用 eager，SDPA/Flash Attention 存在兼容性问题）
    - `torch_dtype=torch.float16`
    - `model.to("npu")` 后调用 `model.eval()`

**输出**: 推理结果文本、耗时和速度（ms/token）。

### 4. CPU 基线推理

**输入**: 模型路径、测试音频文件。

**动作**:
13. 使用 CPU float32 精度作为基准：
```bash
python3 -c "
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
model = WhisperForConditionalGeneration.from_pretrained(
    '$MODEL_PATH', torch_dtype=torch.float32, attn_implementation='eager'
)
model.eval()
processor = WhisperProcessor.from_pretrained('$MODEL_PATH')
import soundfile as sf
audio, sr = sf.read('test.wav')
inputs = processor(audio, sampling_rate=16000, return_tensors='pt')
with torch.no_grad():
    out = model.generate(inputs.input_features, max_new_tokens=64, task='transcribe', language='en')
text = processor.decode(out[0], skip_special_tokens=True)
print(f'CPU baseline: {text}')
torch.save(out, 'cpu_output.pt')
"
```
14. 保存 CPU 推理结果（logits）作为精度对比基准。

**输出**: CPU 输出文本和 `cpu_output.pt` 基准文件。

### 5. CPU/NPU 精度评测

**输入**: `cpu_output.pt` 基准、NPU 推理结果。

**动作**:
15. 执行精度验证脚本：
```bash
python3 scripts/accuracy_run.py --model ${MODEL_PATH} --audio test.wav --tolerance 0.01
```
16. 计算以下精度指标：
    - **生成文本匹配率**：CPU/NPU 输出文本完全一致，目标 100%
    - **Logits RMS 相对误差** = `rms(cpu - npu) / rms(cpu) * 100%`，通过标准：< 1%
    - **Top-1 Token 匹配率**：logits argmax 一致性，目标 100%
    - **Encoder 余弦相似度**：编码器隐藏状态的空间一致性，目标 > 0.999
17. 若 Logits RMS 相对误差 < 1% 且生成文本匹配率 = 100%，标记 `PRECISION_PASS=true`。
18. 若精度不达标，输出偏差明细并中止发布流程。

**输出**: 精度评测结果 JSON，包含各项指标和整体通过状态。

### 6. 性能基准测试

**输入**: 验证通过的模型和音频样本（1s/3s/5s/10s/30s）。

**动作**:
19. 对多个音频长度分别运行推理，记录性能数据：
```bash
# 使用 npu_wrapper 测试不同时长音频
python3 -c "
from scripts.npu_wrapper import WhisperNPU
import time
model = WhisperNPU('$MODEL_PATH', device='npu')
durations = [1, 3, 5, 10, 30]
for dur in durations:
    import numpy as np, soundfile as sf, librosa
    audio = np.random.randn(dur * 16000).astype(np.float32)
    sf.write(f'test_{dur}s.wav', audio, 16000)
    t0 = time.time()
    result = model.transcribe(f'test_{dur}s.wav')
    elapsed = time.time() - t0
    print(f'{dur}s: {elapsed:.2f}s, tok={len(result[\"tokens\"])}, speed={1000*elapsed/max(len(result[\"tokens\"]),1):.1f}ms/tok')
"
```
20. 汇总性能数据并与 CPU 对比。

**输出**: 性能基准测试结果表（含输入时长、平均耗时、输出 Token 数、速度）。

### 7. 适配要点与算子兼容性确认

**输入**: 运行时观察到的算子行为和错误。

**动作**:
21. 确认全部算子 NPU 原生支持，无需手动适配：
    - 特征提取 Conv1D (`nn.Conv1d`, `F.gelu`) — 原生支持
    - 注意力投影 (`nn.Linear` Q/K/V/O) — 原生支持
    - 注意力计算 (Eager Attention) — 原生支持
    - 位置编码 (`nn.Embedding`) — 原生支持
    - 层归一化 (`nn.LayerNorm`) — 原生支持
22. 确认关键配置正确：
    - `attn_implementation="eager"` 而非 sdpa/flash
    - 推理使用 float16，精度对比使用 float32 参考
    - 特征提取 (Mel 频谱) 在 CPU 执行，不涉及 NPU
23. 检查实测精度结果是否全部 PASS：
    - 生成文本匹配率 3/3 (100%)，目标 100%
    - Logits RMS 相对误差 0.55%，目标 < 1%
    - Top-1 Token 匹配率 100%，目标 100%
    - Encoder 平均余弦相似度 0.999653，目标 > 0.999

**输出**: 适配要点确认清单完成，算子兼容性和关键配置已验证。

### 8. 验收确认

**输入**: 所有步骤的输出结果和日志。

**动作**:
24. 完成以下检查清单：
    - [ ] `npu-smi info` 显示设备正常
    - [ ] `import torch_npu` 无报错
    - [ ] 模型加载成功 (1-3 分钟)
    - [ ] `inference.py` 输出文本结果
    - [ ] 精度评测: 生成文本匹配率 100%
    - [ ] 精度评测: Logits RMS 误差 < 1%
    - [ ] 精度评测: Encoder 余弦相似度 > 0.999
    - [ ] 性能基准数据已记录
25. 若所有检查项通过，标记 `DEPLOY_SUCCESS=true`。
26. 将实测精度结果与基准 `references/benchmark_summary.json` 对比，确认一致。

**输出**: 验收结果和最终部署结论。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，torch_npu 版本是否正确 | 暂停，提示安装 torch_npu 或检查 CANN 驱动 |
| 2 | CP-2: 模型权重就绪检查点 | 模型权重下载完成后 | 模型目录是否包含完整文件（config.json + safetensors） | 切换下载方式（HF 镜像 / GitCode），重试下载 |
| 3 | CP-3: NPU 推理完成检查点 | NPU 推理验证完成后 | 推理文本是否合理，控制台无报错日志 | 检查音频文件格式和 `attn_implementation` 设置后重试 |
| 4 | CP-4: CPU 基线完成检查点 | CPU 基线推理完成后 | CPU 输出文本是否合理，基线保存成功 | 检查测试音频和模型加载路径 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | Logits RMS 误差是否 < 1%，所有精度指标是否 PASS | 检查精度不达标原因，调整推理精度或数据一致性后重试 |
| 6 | CP-6: 性能测试确认检查点 | 性能基准测试完成后 | 性能数据是否合理，与 benchmark_summary.json 是否一致 | 检查 NPU 显存占用和系统负载后重试 |
| 7 | CP-7: 最终验收检查点 | 全部步骤完成 | 所有检查项是否通过，部署结论是否确认 | 返回未通过步骤重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查 CANN 驱动 |
| NPU 显存 OOM | 推理时报显存溢出错误 (CUDA OOM) | 依次释放缓存 `torch.npu.empty_cache()`、`gc.collect()`、pkill 其他 NPU 进程 | CP-3 | 确保至少 1 卡空闲，释放其他进程后重试 |
| 模型权重下载失败 | snapshot_download 超时或连接中断 | 自动尝试 GitCode 镜像克隆替代，最多重试 2 次 | CP-2 | 切换下载镜像源或使用离线模型目录 |
| 音频文件格式不支持 | soundfile/librosa 加载异常 | 提示转换为 wav 格式，自动调用 ffmpeg 转码 | CP-3 | 使用 `ffmpeg -i input.mp3 output.wav` 转换 |
| `SafetensorError: header too large` | Git LFS 文件未拉取 | 提示运行 `git lfs pull`，或从 HF 镜像重新下载 | CP-2 | 使用 `git lfs pull` 或 HF 快照下载 |
| 首次推理过慢 | 算子首次编译触发 | 在性能表中记录首次耗时，正常现象 | CP-3 | 后续推理自动加速，无需干预 |
| CPU/NPU 精度超标 | Logits RMS 误差 >= 1% | 记录偏差明细，生成对比报告，标记 PRECISION_FAIL | CP-5 | 检查推理脚本和 `attn_implementation` 设置后重试 |
| 输入特征长度不符合 | `Whisper expects mel input features of length 3000` | 自动填充到 3000 帧，记录 warning | CP-3 | Whisper 固定 3000 帧，自动处理无需用户干预 |
| `Input type (Half) vs bias type (float)` | CPU 上运行 fp16 推理 | 检测设备类型自动切换精度，CPU 用 fp32 | CP-1 | CPU 使用 float32，NPU 使用 float16 自动配置 |
| `F.scaled_dot_product_attention` 结果异常 | NPU SDPA 实现差异 | 自动回退到 `attn_implementation="eager"` | CP-3 | 在模型加载时强制使用 eager 模式 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | NPU 推理执行入口（CLI），支持 transcribe/translate 模式，输出文本和性能数据 |
| `scripts/accuracy_run.py` | CPU 与 NPU 精度对比脚本，输出 Logits RMS 误差、Top-1 匹配率、Encoder 余弦相似度等精度指标 |
| `scripts/npu_wrapper.py` | WhisperNPU Python 封装模块，提供 `transcribe()` 和 `translate()` 方法便于代码集成 |
| `references/benchmark_summary.json` | 性能基准数据（实测结果）：各输入时长下的推理耗时、输出 Token 数、速度指标 |
| `eval_output/` | 评测日志与结果（运行后生成）：精度评测 JSON 和性能测试记录 |
| `cpu_output.pt` | CPU 基线推理结果（运行后生成）：形状 `[1, 序列长度]` 的 logits 张量 |
| `test_wav/` | 测试音频样本（运行前准备）：wav 格式，采样率 16000 |
| `results/` | 精度和性能汇总结果（运行后生成）：JSON 格式，含全部指标数据 |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--model` | string | 否 | /opt/atomgit/whisper-large-v3 | whisper-large-v3 模型目录路径 |
| `--audio` | string | 是 | — | 输入音频文件路径 (wav/flac/mp3/ogg) |
| `--task` | string | 否 | transcribe | 任务类型: transcribe(转录) 或 translate(翻译为英文) |
| `--language` | string | 否 | en | 源语言代码 (en/zh/ja/ko/...) |
| `--max_tokens` | int | 否 | 256 | 最大输出 token 数 |
| `--attn` | string | 否 | eager | 注意力实现方式 (NPU 推荐 eager) |
| `--dtype` | string | 否 | float16 | 推理数据类型 (float16/float32) |
| `--tolerance` | float | 否 | 0.01 | RMS 误差容忍度 (默认 1%) |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| 推理文本 | string | 语音转录或翻译后的文本结果 |
| `cpu_output.pt` | PyTorch Tensor | CPU 基线推理 logits |
| 精度评测报告 | JSON | 各精度指标和 PASS/FAIL 状态 |
| 性能基准表 | Markdown | 各时长下的推理耗时和速度统计 |
| 评测日志 | text | 控制台完整输出日志 |

## 使用约束

1. 使用 hf-mirror.com 镜像下载模型权重（HF 官方可能无法访问），优先使用 GitCode 镜像。
2. `attn_implementation` 必须设置为 `"eager"`，SDPA 和 Flash Attention 在 torch_npu 上存在兼容性问题。
3. 精度验证通过前不确认部署成功（必须有 `PRECISION_PASS=true` 标记）。
4. NPU 推理使用 float16，精度对比使用 float32 参考基准。
5. 特征提取（Mel 频谱）在 CPU 上执行，不涉及 NPU。
6. 模型权重约 3GB，确保磁盘和 NPU 显存充足。
7. 首次推理因算子编译较慢（1-3 分钟），后续推理正常。
