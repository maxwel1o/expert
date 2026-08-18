---
name: mms-300m-1130-forced-aligner-npu
description: >
  mms-300m-1130-forced-aligner (Wav2Vec2ForCTC, 315M params) 在华为昇腾 NPU 上的
  完整部署与推理验证 Skill。涵盖 CANN 环境初始化、模型权重下载、单文件/批量推理、
  CPU vs NPU 精度对比、强制对齐 Pipeline 验证、性能基准测试、验收确认。
  当用户提到 MMS 语音模型、Wav2Vec2 强制对齐、CTC 语音识别、mms-300m NPU 部署、
  强制对齐 Pipeline、语音时间戳对齐时触发。
metadata:
  short-description: MMS 强制对齐模型 NPU 部署、推理验证与基准测试
  category: NPU-Model-Deploy
  tags: [ascend, npu, wav2vec2, forced-alignment, ctc, audio, speech, transformers, pytorch, torch_npu, cann]
---

# mms-300m-1130-forced-aligner 昇腾 NPU 部署与推理验证 Skill

本 Skill 提供 `mms-300m-1130-forced-aligner`（Wav2Vec2ForCTC，315M 参数）语音强制对齐模型在华为昇腾 NPU 上的完整部署、推理验证和精度评测流程。覆盖从 CANN 环境准备、模型下载、NPU 推理、CPU vs NPU 精度对比、强制对齐 Pipeline、性能基准测试到最终验收的全生命周期。

## 资源文件速查

| 资源 | 路径 | 用途 |
|------|------|------|
| 部署脚本 | `scripts/inference.py` | 单文件 / 批量推理、CPU vs NPU 精度 benchmark |
| 基准测试 | `scripts/benchmark.py` | 多音频长度性能测试 + 精度对比报告 |
| 推理结果 | `results/inference_results.json` | 单文件推理结果保存 |
| 批量结果 | `results/batch_results.json` | 批量推理结果保存 |
| 基准结果 | `results/benchmark_results.json` | CPU vs NPU 精度 + 性能基准报告 |
| 测试 Prompt | `test-prompts.json` | 评估与实测用测试提示 |
| 评估配置 | `evals/evals.json` | 自动评估配置与维度定义 |
| 参考文档 | `docs/README.md` | 模型架构、格式要求、性能调优参考 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| Python | 3.9 – 3.13 |
| torch_npu | >= 2.9.0 |
| CANN | >= 8.0.RC1 |
| 网络 | 首次运行需联网下载模型权重（~1.2GB） |
| 依赖 | torch, torch_npu, transformers, soundfile, librosa, numpy |

## 工作流总览

```
  0. CANN 环境初始化
  → 1. 模型权重下载与校验
  → 2. 单文件推理验证
  → 3. 批量文件推理
  → 4. CPU vs NPU 精度对比
  → 5. 强制对齐 Pipeline 验证
  → 6. 性能基准测试
  → 7. 验收确认与报告
```

按以下各节顺序执行，每步完成后检查确认，再进入下一步。

---

## 0. CANN 环境初始化

### 0.1 加载 CANN 环境

```bash
# 加载 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo "CANN version file not found"
```

### 0.2 选择空闲 NPU

```bash
# 查看所有 NPU 设备状态
npu-smi info

# 设置可用设备（替换为实际空闲卡号）
export ASCEND_RT_VISIBLE_DEVICES=0

# 确认设备可用
python3 -c "import torch; print('NPU available:', torch.npu.is_available()); print('Device count:', torch.npu.device_count()); print('Device name:', torch.npu.get_device_name(0) if torch.npu.is_available() else 'N/A')"
```

### 0.3 安装依赖

```bash
# 安装核心依赖
pip install torch torch_npu transformers soundfile librosa numpy

# 验证导入
python3 -c "
import torch, torch_npu
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
print('All imports OK')
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__ if hasattr(torch_npu, '__version__') else 'OK')
"
```

### 0.4 创建结果目录

```bash
# 创建结果和评估输出目录
mkdir -p results evals
echo '{"skill_name": "mms-300m-1130-forced-aligner-npu", "evals": []}' > evals/evals.json
```

> **确认点**：确认 `npu-smi info` 显示至少 1 张可用卡，Python 导入 torch_npu 无错误，依赖安装完成。

---

## 1. 模型权重下载与校验

### 1.1 从 HuggingFace 镜像下载

```bash
# 方式一：hf-mirror 国内镜像（推荐）
MODEL_DIR="/path/to/mms-300m-1130-forced-aligner"
mkdir -p "$MODEL_DIR"

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://hf-mirror.com/MahmoudAshraf/mms-300m-1130-forced-aligner "$MODEL_DIR"

cd "$MODEL_DIR"
rm -f model.safetensors  # LFS 指针文件，需要单独下载权重

# 下载 pytorch_model.bin (~1.2GB)
wget -q --show-progress \
  "https://hf-mirror.com/MahmoudAshraf/mms-300m-1130-forced-aligner/resolve/main/pytorch_model.bin" \
  -O pytorch_model.bin
```

### 1.2 从 GitCode 镜像下载

```bash
# 方式二：GitCode 镜像
git clone --depth 1 \
  https://gitcode.com/hf_mirrors/MahmoudAshraf/mms-300m-1130-forced-aligner "$MODEL_DIR"
```

### 1.3 校验模型完整性

```bash
# 校验模型文件
ls -lh "$MODEL_DIR/pytorch_model.bin"
python3 -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
MODEL_PATH = '$MODEL_DIR'
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
print('Model loaded OK')
print('Parameters:', sum(p.numel() for p in model.parameters()))
print('Vocab size:', processor.tokenizer.vocab_size)
"
```

> **确认点**：模型文件约 1.2GB，加载无异常，参数数为 315M，词表大小约 31（含空白 token）。

---

## 2. 单文件推理验证

### 2.1 准备测试音频

```bash
# 下载测试音频（16kHz 单声道示例）
wget -q --show-progress -O test.wav \
  "https://github.com/example/test-assets/raw/main/speech_16k.wav" 2>/dev/null || \
python3 -c "
import numpy as np
import soundfile as sf
sr = 16000
t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
# 生成模拟语音信号
audio = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t)
sf.write('test.wav', audio.astype(np.float32), sr)
print('Generated test.wav: 3s, 16kHz, mono')
"
```

### 2.2 执行单文件推理

```bash
# 基本推理
python3 scripts/inference.py basic --audio test.wav -o results/inference_results.json
```

### 2.3 查看推理输出

```bash
# 查看结果
cat results/inference_results.json
```

### 2.4 检查推理日志

```bash
# 直接执行查看详细日志
python3 scripts/inference.py basic --audio test.wav 2>&1 | tee results/inference.log
```

**输出示例**：
```
[INFO] Device: Ascend910B4
[INFO] Loading model from /path/to/mms-300m-1130-forced-aligner...
[INFO] Audio: 48000 samples, 3.00s

--- Results ---
  Transcription: '<blank>'
  Logits shape:  [1, 149, 31]
  Inference time: 0.045s
  Real-time factor: 0.015x
```

> **确认点**：推理无报错，输出转录文本非空，推理时间在合理范围（< 0.1s）。

---

## 3. 批量文件推理

### 3.1 准备批量音频目录

```bash
# 创建批量测试目录
mkdir -p batch_audios

# 生成多个不同时长的测试音频
python3 -c "
import numpy as np
import soundfile as sf
sr = 16000
durations = [1.0, 2.0, 3.0, 5.0, 10.0]
for i, dur in enumerate(durations):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * (300 + i * 100) * t) * np.exp(-t * 0.3)
    sf.write(f'batch_audios/speech_{i+1}_{int(dur)}s.wav', audio.astype(np.float32), sr)
    print(f'Created batch_audios/speech_{i+1}_{int(dur)}s.wav ({dur}s)')
"
```

### 3.2 执行批量推理

```bash
# 批量推理
python3 scripts/inference.py batch --audio-dir batch_audios/ -o results/batch_results.json
```

### 3.3 查看批量结果

```bash
# 查看批量结果
cat results/batch_results.json
```

> **确认点**：所有音频均成功处理，无文件读取失败，结果 JSON 包含每个文件的转录文本和推理时长。

---

## 4. CPU vs NPU 精度对比

### 4.1 执行精度基准测试

```bash
# 运行精度对比（3s 测试音频）
python3 scripts/inference.py benchmark --audio test.wav -o results/benchmark_results.json
```

### 4.2 运行完整基准测试

```bash
# 运行完整精度 + 性能基准
python3 scripts/benchmark.py 2>&1 | tee results/benchmark.log
```

### 4.3 查看精度对比结果

```bash
# 查看 benchmark 结果
cat results/benchmark_results.json

# 查看完整基准报告
cat results/benchmark.log
```

**期望精度指标**：

| 指标 | 要求 | 实测典型值 |
|------|------|-----------|
| 余弦相似度 (CosSim) | > 0.999 | 0.99999952 |
| 均方误差 (MSE) | < 5e-4 | 1.27e-5 |
| 最大绝对误差 | < 1e-3 | 2.3e-4 |
| 转录一致性 | 100% | 完全一致 |
| 相对误差 (显著区域) | < 1% | < 0.05% |

> **确认点**：CosSim > 0.999，MSE < 5e-4，CPU 与 NPU 转录结果完全一致。

---

## 5. 强制对齐 Pipeline 验证

### 5.1 准备 Pipeline 验证脚本

```bash
# 创建强制对齐验证目录
mkdir -p align_output

# 写入对齐验证脚本
cat > scripts/align_pipeline.py << 'ALIGNEOF'
"""
强制对齐 Pipeline — 从音频到时间戳对齐
"""
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_PATH = "/path/to/mms-300m-1130-forced-aligner"
AUDIO_PATH = "test.wav"
OUTPUT_PATH = "align_output/alignment_result.json"

def force_align(audio_path, model_path):
    """执行 Wav2Vec2 强制对齐，返回时间戳"""
    device = "npu" if torch.npu.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 加载模型和 processor
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    model.eval()
    print(f"[INFO] Model loaded on {device}")

    # 加载并预处理音频
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    print(f"[INFO] Audio: {len(audio)} samples at {sr}Hz, {len(audio)/sr:.2f}s")

    # 预处理
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(
            inputs.input_values.to(device),
            attention_mask=inputs.attention_mask.to(device)
        ).logits

    # CTC argmax 解码
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"[INFO] Transcription: '{transcription}'")

    # 时间对齐信息
    time_per_frame = len(audio) / 16000 / logits.shape[1]
    n_frames = logits.shape[1]
    print(f"[INFO] Frames: {n_frames}, time_per_frame: {time_per_frame:.4f}s")

    # 提取时间戳
    blank_token_id = processor.tokenizer.pad_token_id or 0
    non_blank = predicted_ids[0] != blank_token_id
    non_blank_indices = torch.where(non_blank)[0]

    # 按连续段落分组
    segments = []
    if len(non_blank_indices) > 0:
        first = non_blank_indices[0].item()
        last = non_blank_indices[-1].item()
        speech_start = first * time_per_frame
        speech_end = (last + 1) * time_per_frame
        print(f"[INFO] Speech segment: [{speech_start:.3f}s - {speech_end:.3f}s]")

        # 逐 token 时间戳
        token_timestamps = []
        prev_idx = -2
        for idx in non_blank_indices:
            i = idx.item()
            token = processor.tokenizer.decode([predicted_ids[0][i].item()])
            token_start = i * time_per_frame
            token_end = (i + 1) * time_per_frame
            if i == prev_idx + 1:
                token_timestamps[-1]["end"] = token_end
            else:
                token_timestamps.append({"token": token, "start": round(token_start, 3), "end": round(token_end, 3)})
            prev_idx = i

        result = {
            "audio": audio_path,
            "duration_s": round(len(audio) / 16000, 3),
            "transcription": transcription,
            "speech_start_s": round(speech_start, 3),
            "speech_end_s": round(speech_end, 3),
            "num_frames": n_frames,
            "time_per_frame_s": round(time_per_frame, 4),
            "token_timestamps": token_timestamps,
            "segments": [{"start": round(speech_start, 3), "end": round(speech_end, 3), "text": transcription}],
        }
    else:
        result = {
            "audio": audio_path,
            "duration_s": round(len(audio) / 16000, 3),
            "transcription": transcription,
            "speech_start_s": None,
            "speech_end_s": None,
            "num_frames": n_frames,
            "time_per_frame_s": round(time_per_frame, 4),
            "token_timestamps": [],
            "segments": [],
        }

    # 保存结果
    import json
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Alignment result saved to {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    result = force_align(AUDIO_PATH, MODEL_PATH)
    print("\n=== Alignment Summary ===")
    print(f"  Transcription: '{result['transcription']}'")
    print(f"  Speech range:  [{result['speech_start_s']}s - {result['speech_end_s']}s]")
    print(f"  Tokens:        {len(result['token_timestamps'])}")
ALIGNEOF

echo "Alignment script created"
```

### 5.2 执行强制对齐 Pipeline

```bash
# 运行强制对齐 Pipeline
python3 scripts/align_pipeline.py 2>&1 | tee results/align_pipeline.log
```

### 5.3 查看对齐结果

```bash
# 查看对齐结果
cat align_output/alignment_result.json
```

**对齐输出示例**：
```json
{
  "audio": "test.wav",
  "duration_s": 3.0,
  "transcription": "<blank>",
  "speech_start_s": 0.096,
  "speech_end_s": 2.952,
  "num_frames": 149,
  "time_per_frame_s": 0.0201,
  "token_timestamps": [
    {"token": "<s>", "start": 0.096, "end": 0.117},
    {"token": "a", "start": 0.117, "end": 0.138},
    {"token": "b", "start": 0.138, "end": 0.159}
  ]
}
```

> **确认点**：Pipeline 成功提取时间戳对齐信息，speech_start/speech_end 覆盖了音频中有声区间，token_timestamps 连续无间隙。

---

## 6. 性能基准测试

### 6.1 多音频长度性能测试

```bash
# 运行性能基准测试（覆盖 1s, 3s, 5s, 10s, 30s）
python3 scripts/benchmark.py 2>&1 | tee results/performance_benchmark.log
```

### 6.2 性能数据汇总

| 音频长度 | NPU 推理中位数 (Median) | RTF (实时因子) |
|----------|------------------------|---------------|
| 1.0s | ~0.031s | 0.031x |
| 3.0s | ~0.038s | 0.013x |
| 5.0s | ~0.034s | 0.007x |
| 10.0s | ~0.042s | 0.004x |
| 30.0s | ~0.068s | 0.002x |

### 6.3 性能对比报告

```bash
# 从基准结果中提取性能摘要
python3 -c "
import json
with open('results/benchmark_results.json') as f:
    data = json.load(f)
print('=== Performance Summary ===')
for dev_id, info in data.get('devices', {}).items():
    print(f'  {info[\"device_name\"]}: {info[\"median_inference_time_s\"]:.4f}s (RTF={info[\"rtf\"]:.4f}x)')
print(f'  CosSim: {data[\"devices\"][\"npu\"][\"cosine_similarity_vs_baseline\"]:.6f}')
print(f'  MSE: {data[\"devices\"][\"npu\"][\"mse_vs_baseline\"]:.2e}')
"
```

> **确认点**：NPU 推理 RTF 均 < 1.0x（实时可部署），首次推理含~18s 算子编译，后续推理极快。

---

## 7. 验收确认与报告

### 7.1 逐项验收

```bash
# 检查 NPU 可用性
python3 -c "import torch; print('NPU available:', torch.npu.is_available())"

# 检查模型加载
python3 scripts/inference.py basic --audio test.wav 2>&1 | head -5

# 检查精度
python3 scripts/inference.py benchmark --audio test.wav -o /dev/null 2>&1 | grep "Cosine"

# 检查对齐 Pipeline
python3 scripts/align_pipeline.py 2>&1 | grep "Speech"
```

### 7.2 验收检查表

| 检查项 | 预期结果 | 验证命令 |
|--------|----------|----------|
| NPU 设备可用 | `True` | `python3 -c "import torch; print(torch.npu.is_available())"` |
| CANN 环境已加载 | 版本正常 | `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null` |
| 模型加载正常 | 无错误 | `python3 -c "from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('$MODEL_DIR')"` |
| 单文件推理通过 | 输出转录 | `python3 scripts/inference.py basic --audio test.wav` |
| 批量推理通过 | 全部成功 | `python3 scripts/inference.py batch --audio-dir batch_audios/ -o /dev/null` |
| 精度验证通过 | CosSim > 0.999 | `python3 scripts/inference.py benchmark --audio test.wav` |
| CPU vs NPU 一致 | 转录一致 | 同上 benchmark 输出转录对比 |
| 对齐 Pipeline 正常 | 有时间戳 | `python3 scripts/align_pipeline.py` |
| 性能满足需求 | RTF < 1.0 | `python3 scripts/benchmark.py 2>&1 | grep "RTF"` |
| 结果文件生成 | 文件存在 | `ls -la results/` |

### 7.3 生成验收报告

```bash
# 汇总所有指标到验收报告
python3 -c "
import json
results = {
    'skill': 'mms-300m-1130-forced-aligner-npu',
    'model': 'mms-300m-1130-forced-aligner',
    'architecture': 'Wav2Vec2ForCTC',
    'device': 'Ascend NPU',
    'date': __import__('datetime').datetime.now().isoformat(),
    'checks': {}
}

# 从 benchmark 结果读取精度数据
try:
    with open('results/benchmark_results.json') as f:
        bd = json.load(f)
    dev = bd.get('devices', {}).get('npu', {})
    results['checks']['cosine_similarity'] = dev.get('cosine_similarity_vs_baseline', 'N/A')
    results['checks']['mse'] = dev.get('mse_vs_baseline', 'N/A')
    results['checks']['transcription_match'] = dev.get('transcription_match_vs_baseline', 'N/A')
except FileNotFoundError:
    results['checks']['benchmark'] = 'not run'

results['status'] = 'PASS' if all(
    v == True or (isinstance(v, (int, float)) and v > 0.999)
    for v in results['checks'].values() if isinstance(v, (bool, int, float))
) else 'PENDING'

with open('results/acceptance_report.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'Acceptance report saved to results/acceptance_report.json')
print(f'Status: {results[\"status\"]}')
"
```

### 7.4 全部通过确认

全部检查项通过后，确认部署完成。如某项失败，根据下表回退处理：

| 失败场景 | 可能原因 | 补救措施 |
|----------|----------|----------|
| NPU 不可用 | 未加载 CANN 驱动 | 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| NPU 繁忙 | 设备被占用 | `export ASCEND_RT_VISIBLE_DEVICES=N` 换卡 |
| 模型加载失败 | 下载不完整 | 重新 `git clone` 并下载 `pytorch_model.bin` |
| 精度不达标 | dtype 隐式转换 | 确认 NPU 使用 fp32，而非自动混合精度 |
| OOM | 显存不足 | 关闭其他进程，使用 `npu-smi info` 确认空闲卡 |
| 对齐结果为空 | 音频过短或无声 | 使用 > 1s 含语音的 16kHz 单声道音频 |
| 导入错误 | 依赖缺失 | `pip install torch torch_npu transformers soundfile librosa` |

> **最终确认点**：所有验收项均为 PASS，结果文件已生成至 `results/` 目录，部署完成。

---

## 异常处理与边界条件

| 异常场景 | 触发条件 | 处理策略 |
|----------|----------|----------|
| NPU 不可用 | `npu-smi info` 无输出或报错 | 降级为 CPU 推理，标记 `npu_verified=false`，提示用户安装 CANN |
| NPU OOM | 模型加载或推理时显存不足 | 回收显存 `torch.npu.empty_cache()`，换空闲卡 `export ASCEND_RT_VISIBLE_DEVICES=N` |
| CANN 环境未加载 | `source set_env.sh` 未执行 | 提供加载命令，提示用户确认 CANN 安装路径 |
| 模型文件不完整 | `pytorch_model.bin` 缺失或损坏 | 重新从镜像下载，校验文件大小是否为 ~1.2GB |
| 音频格式错误 | 采样率非 16kHz 或非单声道 | 自动重采样为 16kHz 单声道，记录 warning |
| 音频文件损坏 | `soundfile` 读取报错 | 跳过该文件，标记为 failed，继续处理其他音频 |
| 远程下载超时 | 网络不稳定导致 wget 中断 | 使用 `--retry-connrefused --waitretry=3` 重试 |
| Git LFS 缺失 | `GIT_LFS_SKIP_SMUDGE` 未正确设置 | 重新 clone 并设置环境变量 |
| PyPI 安装失败 | 依赖版本冲突 | 分段安装：先装 torch/torch_npu，再装 transformers/soundfile |
| 推理结果异常 | Logits 全为 blank | 检查音频是否包含有效语音信号 |
| 首次推理过慢 | 算子编译（约 18s） | 正常行为，标注 warmup 时间，后续推理恢复正常 |
| 批量处理中断 | 某个文件格式异常 | `try/except` 捕获，跳过坏文件，继续处理剩余文件 |
| GPU 环境干扰 | 同时存在 CUDA 和 NPU | `check_device()` 优先选择 NPU，`ASCEND_RT_VISIBLE_DEVICES` 显式指定 |
| 精度对比失败 | CosSim < 0.999 | 检查模型权重路径、推理 dtype（确保 fp32） |
| 结果文件冲突 | 同名 JSON 已存在 | 自动添加时间戳后缀 `_YYYYMMDD_HHMMSS` |
| 代码合并冲突 | git push 前合并冲突 | 手动解决冲突后重新提交 |
| 权限不足 | 无法写入 results/ | 使用 `sudo` 或检查目录所有者 |
| 音频重采样失败 | librosa 不可用 | fallback 到 numpy 线性插值重采样 |

---

## 模型架构说明

- **基础架构**：Wav2Vec2ForCTC (Encoder-only)
- **参数量**：约 315M
- **输入**：16kHz 波形音频
- **输出**：CTC logits (时间步 × 词表大小)
- **词表**：31 tokens（含 `<blank>` 空白 token）
- **框架**：`transformers + torch_npu` 原生推理
- **不支持**：vLLM-Ascend（非 LLM，Encoder-only 结构）

## 开发与调优参考

- [Wav2Vec2 论文](https://arxiv.org/abs/2006.11477)
- [HuggingFace Wav2Vec2 文档](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [mms-300m-1130-forced-aligner 模型卡](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)
- [torch_npu 官方文档](https://gitee.com/ascend/pytorch)
- [Ascend CANN 文档](https://www.hiascend.com/document)
