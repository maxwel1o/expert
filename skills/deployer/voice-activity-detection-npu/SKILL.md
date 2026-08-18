---
name: voice-activity-detection-npu
description: "pyannote/segmentation (PyanNet) 语音活动检测模型在华为昇腾 NPU 上的完整部署、推理、CPU/NPU 精度对比与 VAD 管线处理。适用于：NPU 部署、昇腾推理、精度验证、语音端点检测。触发词：VAD 语音活动检测 NPU、pyannote segmentation 昇腾、voice activity detection npu 部署、pyannote 精度对比、VAD NPU 推理。"
---

# pyannote/segmentation 语音活动检测模型昇腾 NPU 部署 Skill

> 在昇腾 NPU 上自动部署 `pyannote/segmentation`（PyanNet, 1.47M 参数）语音活动检测（VAD）模型，完成环境初始化、依赖安装、NPU 基础验证、模型推理、VAD 管线处理、CPU/NPU 精度对比验证、性能基准测试和验收确认的全流程。支持串行执行各步骤，每步完成后再进入下一步。

## 概述

本 Skill 用于自动完成 **pyannote/segmentation 语音活动检测模型** 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证和 VAD 管线处理。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910 系列) |
| 框架版本 | PyTorch 2.0+, torch_npu >= 2.9.0, pyannote.audio |
| 精度目标 | CPU 与 NPU 推理 VAD 决策一致率 > 95% |
| 执行方式 | 串行逐步骤执行，完成后再进入下一步 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 模型参数量 | 1,473,265 (PyanNet: SincNet + LSTM + DNN) |

## 支持的模型

| 模型名称 | 参数量 | 架构 | 输入 | 输出 |
|:---|:---:|:---|:---|:---|
| `pyannote/segmentation` | 1,473,265 | PyanNet (SincNet + LSTM + DNN) | 16kHz 单声道音频 | 7-class Powerset LogSoftmax |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.9+ 环境，昇腾 NPU 驱动，CANN >= 8.0。

**动作**:
1. 检查 Python 版本，确认 >= 3.9：

```bash
python3 --version
```

2. 加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3. 运行 NPU 检测：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

4. 选择空闲 NPU 卡并设置环境变量：

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

5. 设置华为 pip 镜像（国内加速）：

```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，CANN 环境已加载，空闲 NPU 卡已选择。

### Step 2: 依赖安装

**输入**: PIP 镜像源已配置。

**动作**:
6. 安装 torch_npu：

```bash
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
```

7. 安装 pyannote.audio 及其他依赖：

```bash
pip install pyannote.audio soundfile torchaudio -i https://repo.huaweicloud.com/repository/pypi/simple/
```

8. 确认 torch 与 torch_npu 版本一致（如均为 2.2.0）：

```python
import torch
import torch_npu
print(f"torch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__}")
```

**输出**: 依赖已安装完成，版本兼容性已确认。

### Step 3: NPU 基础验证

**输入**: torch_npu 已安装。

**动作**:
9. 运行 NPU 环境验证：

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
"
```

10. 确认输出包含 `device='npu:0'` 的 Tensor 且无报错。

11. 若报错 `No module named 'torch_npu'`，检查 CANN 环境是否已加载，或重装 torch_npu。

**输出**: NPU 设备可用确认，Tensor 在 NPU 上正常运算。

### Step 4: 基础推理验证

**输入**: NPU 环境可用，模型权重可访问。

**动作**:
12. 加载 pyannote/segmentation 模型：

```python
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/segmentation", map_location="cpu")
print(f"Model loaded: {type(model).__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

13. 创建基础推理脚本（使用 `scripts/vad_npu_infer.py`）：

```bash
cp scripts/vad_npu_infer.py ./
```

14. 运行 NPU 推理：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python3 vad_npu_infer.py
```

15. 确认推理输出：
    - 模型前向推理成功，无 NPU 报错
    - VAD score 在 [0, 1] 范围内
    - Output shape 正确

**输出**: 基础推理结果，包含 VAD score 范围和输出 shape。

### Step 5: VAD 管线处理

**输入**: 模型推理正常，测试音频就绪。

**动作**:
16. 使用 pyannote.audio Inference 构建 VAD 管线（使用 `scripts/vad_npu_pipeline.py`）：

```bash
cp scripts/vad_npu_pipeline.py ./
```

17. 运行 VAD 管线：

```bash
python3 vad_npu_pipeline.py
```

18. 确认管线输出：
    - VAD 得分计算正常，得分范围在 [0, 1]
    - 输出说话段起止时间
    - 静音段正确过滤
    - 滞后阈值判决正常工作（onset=0.5, offset=0.5）

19. 处理自定义音频文件：

```bash
python3 vad_npu_pipeline.py --audio /path/to/file.wav
```

**输出**: VAD 分割结果，包含检测到的说话段列表（起止时间）。

### Step 6: CPU/NPU 精度对比验证

**输入**: 模型在 CPU 和 NPU 上的推理结果。

**动作**:
20. 使用 200 个随机种子信号进行精度对比（含 5 种类型：正弦波、多频混合、纯噪声、低信噪比、振幅调制），在 CPU 和 NPU 上分别推理。
21. **Level 1** VAD score 逐帧对比：比较 CPU 和 NPU 推理输出的原始 softmax score：

```python
import torch, torch_npu
import numpy as np
from pyannote.audio import Model

model_cpu = Model.from_pretrained("pyannote/segmentation", map_location="cpu")
model_npu = Model.from_pretrained("pyannote/segmentation", map_location="npu:0")
model_cpu.eval()
model_npu.eval()

scores_cpu = []
scores_npu = []
sr = 16000

for seed in range(200):
    np.random.seed(seed)
    t = np.linspace(0, 2.0, int(2.0 * sr))
    # 随机信号类型
    sig_type = seed % 5
    if sig_type == 0:
        s = np.sin(2 * np.pi * 400 * t)                     # 正弦波
    elif sig_type == 1:
        s = sum(np.sin(2 * np.pi * f * t) for f in [200,400,600])  # 多频混合
    elif sig_type == 2:
        s = np.random.randn(len(t))                          # 纯噪声
    elif sig_type == 3:
        s = np.sin(2 * np.pi * 400 * t) + 0.5 * np.random.randn(len(t))  # 低信噪比
    else:
        s = np.sin(2 * np.pi * 400 * t) * (1 + 0.8 * np.sin(2 * np.pi * 2 * t))  # 振幅调制
    s = s / np.abs(s).max()

    w = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out_cpu = model_cpu(w).exp().max(-1).values
        out_npu = model_npu(w.npu()).cpu().exp().max(-1).values
    scores_cpu.append(out_cpu)
    scores_npu.append(out_npu)

all_cpu = torch.cat(scores_cpu)
all_npu = torch.cat(scores_npu)
abs_err = (all_cpu - all_npu).abs()
mean_err = abs_err.mean().item()
max_err = abs_err.max().item()
consistent = (all_cpu > 0.5) == (all_npu > 0.5)
consistency_rate = consistent.float().mean().item()

print(f"VAD score mean error: {mean_err*100:.2f}%")
print(f"VAD score max error: {max_err*100:.2f}%")
print(f"VAD decision consistency: {consistency_rate*100:.2f}%")
```

22. **Level 2** 完整 VAD 管线决策一致率：使用 `scripts/vad_npu_pipeline.py` 中的 `scores_to_segments` 函数，对比 CPU 和 NPU 的二值说话段输出。

23. 检查精度指标：

| 指标 | 期望值 |
|------|--------|
| VAD score 平均误差 | ~1.8% |
| VAD score 最大误差 | ~12%（边缘信号，不影响 VAD 决策） |
| 平均 VAD 决策一致率 | > 99% |
| 最低 VAD 决策一致率 | > 93% |
| NaN/Inf | 无 |

24. 若 `平均 VAD 决策一致率 > 95%` 标记 `PRECISION_PASS=true`；否则标记 `PRECISION_FAIL`。

**输出**: 精度对比结果，包含逐帧 VAD score 平均/最大误差和 VAD 决策一致率。

### Step 7: 性能基准测试

**输入**: 模型已加载，推理正常。

**动作**:
25. 对不同时长的音频进行性能基准测试：

```python
import torch_npu
import torch, time, numpy as np
from pyannote.audio import Model

device = torch.device("npu:0")
model = Model.from_pretrained("pyannote/segmentation", map_location=str(device))
model.eval()
sr = 16000

for duration in [2.0, 5.0, 10.0, 30.0]:
    t = torch.linspace(0, duration, int(duration * sr))
    signal = 0.5 * torch.sin(2 * np.pi * 400 * t)
    waveform = signal.unsqueeze(0).unsqueeze(0).float().to(device)
    
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(waveform)
            times.append(time.time() - start)
    
    avg_latency = np.mean(times)
    rtf = avg_latency / duration
    print(f"{duration:.1f}s: {avg_latency*1000:.0f}ms, RTF={rtf:.4f}, {1/rtf:.0f}x realtime")
```

26. 检查性能指标（参考值）：

| 音频时长 | 平均延迟 | RTF | 倍实时 |
|----------|----------|-----|--------|
| 2.0s | ~47ms | 0.0236 | 42x |
| 5.0s | ~46ms | 0.0092 | 108x |
| 10.0s | ~46ms | 0.0046 | 217x |
| 30.0s | ~57ms | 0.0019 | 523x |

27. 记录性能数据到 `benchmark_results.json`。

**输出**: `benchmark_results.json`，包含各时长下的延迟、RTF 和倍实时指标。

### Step 8: 验收确认

**输入**: 所有步骤完成。

**动作**:
28. 执行最终验收检查清单：

```bash
# NPU 设备检查
npu-smi info

# torch_npu 可用性
python3 -c "import torch_npu; print(f'NPU available: {torch.npu.is_available()}')"

# 推理验证
python3 vad_npu_infer.py

# VAD 管线验证
python3 vad_npu_pipeline.py

# 精度验证
python3 -c "
import torch, torch_npu
import numpy as np
from pyannote.audio import Model
model_cpu = Model.from_pretrained('pyannote/segmentation', map_location='cpu')
model_npu = Model.from_pretrained('pyannote/segmentation', map_location='npu:0')
model_cpu.eval(); model_npu.eval()
sr = 16000; consistencies = []
for seed in range(200):
    np.random.seed(seed)
    t = np.linspace(0, 2.0, int(2.0 * sr))
    s = np.sin(2*np.pi*400*t) if seed%5 in (0,4) else (np.random.randn(len(t)) if seed%5==2 else 0.5*np.sin(2*np.pi*400*t)+0.5*np.random.randn(len(t)))
    s = s / np.abs(s).max()
    w = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        cpu_s = model_cpu(w).exp().max(-1).values
        npu_s = model_npu(w.npu()).cpu().exp().max(-1).values
    consistencies.append(((cpu_s>0.5)==(npu_s>0.5)).float().mean().item())
print(f'Avg VAD decision consistency: {np.mean(consistencies)*100:.2f}%')
"
```

29. 确认以下检查项全部通过：
    - [ ] `npu-smi info` 显示设备正常
    - [ ] `import torch_npu` 无报错
    - [ ] 模型推理正常，输出 shape 正确
    - [ ] VAD 管线输出说话段起止时间
    - [ ] 精度验证平均决策一致率 > 95%
    - [ ] 无 NaN/Inf 异常值
    - [ ] 性能基准测试数据已记录

30. 汇总部署报告到 `deployment_report.json`。

**输出**: 完整的部署报告，包含各步骤状态、精度结论和性能数据。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，CANN 版本是否正确 | 暂停，提示安装 torch_npu 或检查 NPU 驱动和 CANN 环境 |
| 2 | CP-2: 模型下载检查点 | 模型下载完成后 | 模型名称、参数量是否正确，权重文件是否存在 | 检查网络连接或切换镜像源后重试 |
| 3 | CP-3: NPU 推理验证检查点 | 基础推理完成后 | NPU 推理是否成功、VAD score 范围是否合理 | 检查 NPU 显存和驱动状态后重试 |
| 4 | CP-4: VAD 管线检查点 | VAD 管线处理后 | 说话段起止时间是否正确，静音段是否过滤 | 调整阈值参数（onset/offset）后重试 |
| 5 | CP-5: 精度验证检查点 | 精度对比完成后 | 平均决策一致率是否 > 95%，NaN/Inf 是否存在 | 检查推理脚本和随机种子一致性后重试 |
| 6 | CP-6: 性能测试检查点 | 性能基准测试后 | 延迟和 RTF 是否在合理范围内 | 检查 NPU 负载，释放其他进程后重试 |
| 7 | CP-7: 验收确认检查点 | 全流程完成时 | 所有检查项是否通过，报告是否完整 | 返回未通过步骤重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查 NPU 驱动和 CANN 环境 |
| NPU 显存 OOM | 推理时报内存不足 | 释放缓存后重试，减少 batch 或分段处理 | CP-3 | 释放其他进程占用的 NPU 显存后重试 |
| 模型加载异常 | `Model.from_pretrained` 抛出异常 | 打印错误堆栈，提示网络或模型名是否正确 | CP-2 | 检查 HF 镜像源或从 GitCode 镜像下载 |
| 下载网络超时 | pip/curl 长时间无响应 | 提示使用华为镜像源，重试最多 3 次 | CP-1 | 切换镜像源或设置 HF_ENDPOINT 环境变量 |
| 模型权重不存在 | HuggingFace 缓存未命中 | 自动回退到镜像源下载 | CP-2 | 检查 HF_ENDPOINT 或使用 GitCode 镜像 |
| VAD 管线异常 | Inference 预处理报错 | 输出错误堆栈，检查音频格式和采样率 | CP-4 | 确认音频为 16kHz 单声道格式后重试 |
| 精度超标异常 | 平均决策一致率 <= 95% | 记录偏差明细，分析 SincNet 算子差异 | CP-5 | 检查推理脚本和数据一致性，调整验证策略 |
| 精度含 NaN/Inf | 推理结果出现异常值 | 标记 PRECISION_FAIL，输出详细环境信息 | CP-5 | 检查算子兼容性问题，更新 torch_npu 版本 |
| 多卡抢占冲突 | 默认都用 0 号卡导致冲突 | 提示用 `npu-smi info` 选空闲卡 | CP-1 | 设置 `ASCEND_RT_VISIBLE_DEVICES` 指向空闲卡 |
| 磁盘空间不足 | 模型权重下载失败 | 提示清理磁盘空间后重试 | CP-1 | 释放磁盘空间后重试 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/vad_npu_infer.py` | 基础 NPU 推理脚本：加载模型、生成测试信号、执行前向推理 |
| `scripts/vad_npu_pipeline.py` | 完整 VAD 管线脚本：滑动窗口推理、Powerset 转换、滞后阈值判决 |
| `requirements.txt` | Python 依赖清单（torch_npu, pyannote.audio, soundfile, torchaudio） |
| `skill.json` | 技能元数据：模型触发词、描述、参数定义 |
| `test-prompts.json` | 结构评测用测试提示词 |
| `results/vad_scores_cpu.json` | CPU VAD score 推理结果（运行后生成） |
| `results/vad_scores_npu.json` | NPU VAD score 推理结果（运行后生成） |
| `results/compare_results.json` | CPU/NPU 精度对比结果（运行后生成）：VAD score 误差 + 决策一致率 |
| `results/benchmark_results.json` | 性能基准测试结果（运行后生成）：延迟 + RTF + 倍实时 |
| `results/deployment_report.json` | 完整部署报告（运行后生成）：各步骤状态 + 精度结论 + 性能数据 |

## 精度要求

- NPU 与 CPU 推理 VAD 决策一致率必须 > 95%
- VAD score 平均误差 < 5%（SincNet 算子差异可接受最大 ~12%）
- 对比指标：VAD score 逐帧平均误差、最大误差、决策一致率
- 结论标记：`平均决策一致率 > 95%` 时记为 `PRECISION_PASS`

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 否 | `pyannote/segmentation` | 要加载的模型名称 |
| `test_duration` | float | 否 | 2.0 | 测试信号时长（秒） |
| `sample_rate` | int | 否 | 16000 | 音频采样率 |
| `validate_samples` | int | 否 | 200 | 精度验证使用的随机种子数 |
| `benchmark_durations` | list | 否 | [2, 5, 10, 30] | 性能测试的音频时长列表（秒） |
| `vad_onset` | float | 否 | 0.5 | VAD 起始阈值 |
| `vad_offset` | float | 否 | 0.5 | VAD 结束阈值 |
| `min_duration_on` | float | 否 | 0.055 | 最短语音段（秒） |
| `min_duration_off` | float | 否 | 0.098 | 最短静音段（秒） |

## 使用约束

1. 使用华为镜像源（repo.huaweicloud.com）或 GitCode 镜像下载模型权重（HF 官方可能无法访问）。
2. 精度验证通过前不标记 `PRECISION_PASS`（必须有 "平均决策一致率 > 95%" 确认）。
3. SincNet 在 Ascend NPU 上因底层算子实现差异，原始 softmax score 最大差异可达 ~12%，但 VAD 二值决策一致率超过 99%，对实际使用无影响。
4. 每次执行前确认 `ASCEND_RT_VISIBLE_DEVICES` 指向空闲 NPU 卡，避免多卡抢占冲突。
5. 长时间音频文件建议分段处理，避免 NPU 显存溢出（OOM）。
6. 测试前确认 Ascend910 驱动和 CANN 环境已正确安装。
7. 音频输入必须为 16kHz 单声道 WAV 格式（或自动重采样）。
8. VAD 阈值参数（onset/offset）可根据实际场景调整。
