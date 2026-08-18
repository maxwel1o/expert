---
name: speaker-diarization-npu-deploy
description: >
  Pyannote Speaker Diarization (speaker-diarization / speaker-diarization-community-1)
  说话人日志模型在华为昇腾 NPU 上的完整部署与推理验证 Skill。
  涵盖环境准备、依赖安装、NPU 适配补丁（torchcodec 替换、FFT 路由、设备检测）、
  推理验证、精度对比、性能评测的全流程。可在 Ascend910 系列服务器上一键复现。
  当用户提到 pyannote 说话人日志 NPU 部署、说话人分离 NPU、speaker diarization NPU 时触发。
metadata:
  short-description: Pyannote Speaker Diarization 昇腾 NPU 部署与推理验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, speaker-diarization, pyannote, audio, pytorch, inference]
---

# Pyannote Speaker Diarization 昇腾 NPU 部署 Skill

本 Skill 提供 pyannote.audio `speaker-diarization` 和 `speaker-diarization-community-1`
说话人日志模型在华为昇腾 Ascend NPU 上的完整部署、推理验证、精度对比和性能评测的标准化可复现流程。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910 系列，64GB HBM) |
| 框架版本 | PyTorch 2.0+, torch_npu, pyannote.audio 4.0.4 |
| 精度目标 | NPU vs CPU 输出 DER < 1% |
| 执行方式 | 逐模型串行推理，确保 NPU 显存稳定 |
| 音频格式 | 16kHz 单声道 WAV |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 模型 | speaker-diarization-community-1 / basic |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0 |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重 |
| 磁盘 | 模型权重约 35MB，推理结果约 100KB |

## 执行工作流

### 1. 环境初始化与 NPU 检测

**输入**: Python 3.9-3.13 环境，昇腾 NPU 驱动 (CANN >= 8.0)。

**动作**:
1. 加载 CANN 环境：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2. 选择空闲 NPU 卡：
```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```
3. 设置国内 pip 镜像（加速）：
```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```
4. 确认 NPU 驱动版本和状态，验证 `npu-smi` 命令返回正常。

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，环境变量已配置。

### 2. 安装依赖

**输入**: pip 镜像源已配置。

**动作**:
5. 安装 torch_npu：
```bash
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
```
6. 确认 torch_npu 版本：
```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```
7. 安装 pyannote.audio 与 soundfile：
```bash
pip install pyannote.audio==4.0.4 soundfile pyannote.metrics -i https://repo.huaweicloud.com/repository/pypi/simple/
```
8. 验证依赖包安装成功：
```bash
python3 -c "
import torch, torch_npu, soundfile
import pyannote.audio
print('All dependencies installed successfully')
"
```

**输出**: 所有依赖已安装完成，版本兼容。

### 3. NPU 基础验证

**输入**: torch_npu 已安装，CANN 环境已加载。

**动作**:
9. 运行 NPU availability 检测：
```bash
python3 -c "
import torch
import torch_npu
print('NPU available:', torch.npu.is_available())
print('NPU device:', torch.npu.get_device_name(0))
a = torch.randn(3, 4).npu()
print(a + a)
"
```
10. 确认输出包含 `device='npu:0'` 的 Tensor 且无报错。
11. 若 `torch.npu.is_available()` 返回 False，则标记 `NPU_FALLBACK=true` 并回退 CPU。

**输出**: NPU 可用确认，基础 Tensor 操作验证通过。

### 4. 下载模型权重

**输入**: `models/` 目录已创建，网络可访问 GitCode。

**动作**:
12. 从 GitCode 下载模型权重：
```bash
mkdir -p models
git lfs clone https://gitcode.com/hf_mirrors/pyannote/speaker-diarization-community-1.git models/speaker-diarization-community-1
```
13. 确认模型目录结构完整：
```bash
ls -la models/speaker-diarization-community-1/
```
14. 验证关键文件存在（config.yaml, embedding/pytorch_model.bin, segmentation/pytorch_model.bin, plda/plda.npz）。
15. 若下载失败（网络超时或磁盘空间不足），重试最多 3 次后提示用户确认。

**输出**: 模型权重已就绪，目录结构验证通过。

### 5. 推理验证

**输入**: 模型权重已下载，NPU 环境就绪。

**动作**:
16. 应用 NPU 适配补丁（torchcodec 替换、FFT 路由到 CPU、设备检测替换、安全反序列化）。
17. 运行推理验证：
```bash
python3 scripts/inference.py audio.wav --model community-1 --output result.json
```
18. 支持可选参数：
    - `--device`: `auto` / `npu` / `cpu` / `cuda`，默认 `auto`
    - `--num-speakers`: 强制指定说话人数
    - `--min-speakers` / `--max-speakers`: 说话人数范围
19. 输出应为说话人片段时间段和 Embedding 形状 `[1, 256]`。
20. 若 NPU 推理报错（OOM 或算子不支持），自动回退 CPU 并记录 fallback 日志。

**输出**: `result.json` 包含说话人日志分段（speaker, start, end）和 Embedding 信息。

### 6. 精度对比验证

**输入**: 已完成 NPU 推理，需要对比 CPU 基线。

**动作**:
21. 运行精度评测脚本：
```bash
python3 scripts/evaluate.py
```
22. 评估指标：
    - DiarizationErrorRate (DER)：NPU vs CPU 输出一致性，通过标准 DER < 1%
    - NPU 推理时间 vs CPU 推理时间对比
    - 说话人数一致性检查
23. 若 DER >= 1% 则标记 `PRECISION_FAIL` 并记录偏差明细。
24. 若 DER < 1% 则标记 `PRECISION_PASS=true`。

**输出**: 精度对比结果，含 DER 百分比和推理耗时。

### 7. 性能评测

**输入**: 精度验证通过的模型 pipeline。

**动作**:
25. 运行性能基准测试（10s / 30s / 60s 音频）：
```bash
python3 scripts/evaluate.py
```
26. 记录 NPU 推理 RTF（Real-Time Factor）和实时倍数。
27. 记录 CPU 推理时间用于加速比计算。
28. 验证 RTF 在 0.003~0.006 范围内，实时倍数 > 100x。
29. 若 RTF 异常（> 0.01），检查 NPU 驱动版本和显存使用情况后重试。

**输出**: 性能数据汇总表，含 RTF、吞吐量和加速比。

### 8. 验收确认

**输入**: 推理、精度和性能数据已就绪。

**动作**:
30. 逐项确认验收清单：
    - `torch.npu.is_available()` 返回 True
    - 模型加载后子模型在 `npu:0` 上
    - Embedding 前向传播返回 `(1, 256)` 无报错
    - 推理完成，正确输出说话人片段时间
    - NPU vs CPU 精度对比 DER < 1%
    - RTF 在 0.003~0.006 范围内
31. 所有检查项通过后标记 `DEPLOY_PASS=true`。
32. 若任一检查项失败，记录具体失败原因并供用户确认后重试。

**输出**: 验收报告，标记全部通过或列出失败项。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查 NPU 驱动 |
| 2 | CP-2: 模型权重检查点 | 模型权重下载后 | 模型目录结构和文件完整性 | 返回重新下载或检查网络连接 |
| 3 | CP-3: NPU 推理检查点 | NPU 推理完成后 | 说话人分段结果和 Embedding 形状 | 检查补丁应用和 NPU 显存后重试 |
| 4 | CP-4: 精度验证检查点 | 精度对比完成后 | DER 是否 < 1%，推理日志是否完整 | 检查精度不达标原因，调整后重试 |
| 5 | CP-5: 性能验证检查点 | 性能评测完成后 | RTF 和实时倍数是否在正常范围 | 检查 NPU 驱动版本和显存后重试 |
| 6 | CP-6: 验收审批检查点 | 验收确认前 | 所有检查项是否通过 | 记录失败原因，提示用户排查后重试 |
| 7 | CP-7: 全量完成检查点 | 全部验证流程完毕 | 最终报告是否完整，结果是否可复现 | 返回未通过步骤重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查 NPU 驱动 |
| torchcodec 加载失败 | NPU 环境无 CUDA NVRTC | soundfile 替代解码器自动生效 | CP-3 | 已自动处理，无需用户干预 |
| FFT 复数运算异常 | Ascend NPU 的 aclnnAbs 不支持 complex64 | FFT 自动路由到 CPU 执行 | CP-3 | 已自动处理，会产生 CPU↔NPU 传输开销 |
| 模型下载失败 | 网络超时或 LFS 授权失败 | 重试最多 3 次，记录失败原因 | CP-2 | 检查网络或使用镜像源后重试 |
| 显存溢出 (OOM) | 推理时报显存溢出错误 | 释放缓存后重试，若仍失败回退 CPU | CP-3 | `torch.npu.empty_cache()` 或减少 batch size |
| PyTorch 反序列化错误 | weights_only=True 下自定义类未注册 | 自动注册 pyannote 自定义类 | CP-1 | 检查 torch 版本和 pyannote.audio 版本 |
| 精度超标异常 | DER >= 1% | 记录偏差明细，标记 PRECISION_FAIL | CP-4 | 检查推理脚本和数据一致性后重试 |
| 音频格式不匹配 | 采样率或声道数不符合要求 | 提示转换音频格式后重试 | CP-3 | 使用 16kHz 单声道 WAV |
| 磁盘空间不足 | 模型权重下载或结果保存失败 | 提示清理磁盘空间后重试 | CP-2 | 释放磁盘空间后重试 |
| 推理命令执行超时 | inference.py 长时间无响应 | kill 进程后重试，最多 2 次 | CP-3 | 检查 NPU 驱动和模型加载状态后重试 |
| pip 安装失败 | 依赖版本冲突或下载超时 | 切换镜像源并重试安装 | CP-1 | 使用国内镜像源或离线安装 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | 说话人日志推理脚本：支持 community-1 和 basic 模型，含 NPU 适配补丁 |
| `scripts/evaluate.py` | 精度与性能评测脚本：DER 计算、RTF 基准、NPU vs CPU 对比 |
| `test-prompts.json` | 结构评测用测试提示词（含 NPU 回退场景） |
| `models/speaker-diarization-community-1/` | 模型权重目录（需下载）：含 config.yaml、embedding、segmentation、plda |
| `results/` | 推理结果输出目录（运行后生成）：含说话人分段 JSON |
| `result.json` | 推理结果文件（运行后生成）：说话人日志分段、Embedding 信息 |
| `evals.json` | 评测汇总数据（运行后生成）：各模型精度、性能指标 |
| `references/` | 参考文档：Pyannote Speaker Diarization 论文和相关技术文档 |

## NPU 适配核心补丁

推理脚本自动应用以下 NPU 适配补丁：

1. **torchcodec 替换**: NPU 环境无 CUDA NVRTC 库，torchcodec 的 `AudioDecoder` 不可用。使用 `soundfile` 实现替代解码器。
2. **FFT 路由到 CPU**: WeSpeaker ResNet 的 `compute_fbank` 使用 `torch.fft.rfft` 产生 `complex64` 张量，Ascend NPU 的 `aclnnAbs` 不支持复数输入。参考 MPS 方案将 FFT 路由到 CPU。
3. **设备检测替换**: `pyannote.audio.pipelines.utils.getter.get_devices()` 硬编码 `torch.cuda.device_count()`，替换为 `torch.npu.device_count()`。
4. **安全反序列化**: PyTorch 2.6+ 的 `weights_only=True` 需要注册 pyannote 的自定义类。

## 预期性能数据

### NPU 推理性能（Ascend 910B4）

| 模型 | 10s RTF | 30s RTF | 60s RTF | 峰值吞吐 |
|------|---------|---------|---------|---------|
| community-1 | 0.0052 (192x) | 0.0029 (341x) | 0.0026 (386x) | 386x realtime |
| basic | 0.0052 (192x) | 0.0025 (405x) | - | 405x realtime |

### CPU 对比

| 模型 | 10s CPU | 30s CPU | NPU 加速比 |
|------|---------|---------|-----------|
| community-1 | 0.733s | 7.214s | **14x~100x** |
| basic | 0.737s | 7.234s | **14x~100x** |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `audio_file` | string | 是 | — | 音频文件路径，推荐 16kHz 单声道 WAV |
| `--model` | string | 否 | community-1 | 模型类型: community-1 / basic |
| `--device` | string | 否 | auto | 运行设备: auto / npu / cpu / cuda |
| `--output` | string | 否 | — | 结果 JSON 输出路径 |
| `--num-speakers` | int | 否 | — | 强制指定说话人数 |
| `--min-speakers` | int | 否 | — | 最少说话人数 |
| `--max-speakers` | int | 否 | — | 最多说话人数 |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `result.json` | JSON | 说话人日志结果：含说话人分段、Embedding 信息 |
| `evals.json` | JSON | 评测汇总：DER 精度、RTF 性能、NPU vs CPU 对比数据 |
| `scripts/` | Python | 推理脚本和评测脚本源码 |
| 性能报告 | Markdown | RTF 基准测试结果和加速比汇总 |

## 使用约束

1. 推荐使用 16kHz 单声道 WAV 格式，与模型训练数据一致。
2. NPU 环境下 torchcodec 加载失败属于正常现象，本 Skill 已通过 soundfile 替换解决。
3. WeSpeaker Fbank 计算的 FFT 路由到 CPU 会导致 Embedding 推理时存在 CPU↔NPU 数据传输，但对整体性能影响较小。
4. community-1 模型约 35MB（embedding 26.6MB + segmentation 5.9MB + PLDA 134KB），下载快速。
5. 当前验证基于单 NPU 卡，多卡场景可通过 `ASCEND_RT_VISIBLE_DEVICES` 环境变量控制。
6. NPU 推理精度验证（DER < 1%）通过前不标记 DEPLOY_PASS。
7. 全程 FP32 精度，无需混合精度。
