---
name: wav2vec2-urdu-npu-deploy
description: >
  wav2vec2-large-xls-r-300m-Urdu 乌尔都语语音识别模型在昇腾 NPU 上的
  完整部署与推理验证 Skill。涵盖环境准备、模型下载、transfer_to_npu 自动迁移、
  NPU 推理验证、精度对比验证（NPU vs CPU）的全流程。
  可在任意 Ascend910 系列服务器上一键复现。
  当用户提到 wav2vec2 部署昇腾、Urdu ASR NPU、wav2vec2 NPU 推理、
  乌尔都语语音识别 NPU 时触发。
metadata:
  short-description: wav2vec2 Urdu ASR 昇腾 NPU 部署
  category: NPU-Model-Deploy
  tags: [ascend, npu, wav2vec2, urdu, asr, speech-recognition, pytorch, inference]
---

# wav2vec2-large-xls-r-300m-Urdu 昇腾 NPU 部署 Skill

> 在昇腾 NPU 上部署 `kingabzpro/wav2vec2-large-xls-r-300m-Urdu` 乌尔都语 ASR 模型，完成环境初始化、模型下载、NPU 推理、CPU/NPU 精度对比验证、性能基准测试和验收确认的全流程。适用于乌尔都语语音识别、Wav2Vec2 昇腾迁移、NPU 精度对比场景。

## 概述

本 Skill 用于在华为昇腾 NPU 上自动部署 **wav2vec2-large-xls-r-300m-Urdu** 乌尔都语自动语音识别模型，实现 NPU 无代码修改推理（通过 `torch_npu.contrib.transfer_to_npu` 自动映射 CUDA API），并与 CPU 基线进行精度对比验证。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910 系列, 29.5GB+ HBM) |
| 框架版本 | PyTorch 2.0+, torch_npu>=2.9.0, transformers>=4.16.0 |
| 精度目标 | Token 预测一致率 >= 99%，最大相对误差 < 1% |
| 执行方式 | 按步骤串行执行，每步确认后再进入下一步 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 模型参数 | 315.5M 参数 Wav2Vec2ForCTC 模型 |

## 前置条件

| 项目 | 要求 |
|:---|:---|
| 硬件 | Ascend910 系列（至少 1 卡，29.5GB+ HBM） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.10 – 3.12 |
| 网络 | 首次运行需联网下载模型权重（~2GB） |

## 执行工作流

### 1. 环境初始化与 NPU 检测

**输入**: Python 3.10-3.12 环境，昇腾 NPU 驱动 (CANN >= 8.0)。

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
3. 设置华为 pip 镜像（国内加速）：
```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```
4. 选择空闲 NPU 卡：
```bash
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```
5. 验证 NPU 基础算子通过：
```bash
python3 -c "
import torch
import torch_npu
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
print(f'NPU device: {torch.npu.get_device_name(0)}')
a = torch.randn(3, 4).npu()
print(f'Basic tensor test: {a.device}')
"
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，依赖环境已配置。

### 2. 安装依赖与模型下载

**输入**: 网络连接正常，Python 和 pip 可用。

**动作**:
6. 安装 torch_npu（如未安装）：
```bash
pip install torch_npu
```
7. 安装其他依赖包：
```bash
pip install transformers soundfile pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip  # 可选，用于 LM 解码
```
8. 设置 HuggingFace 镜像并下载模型权重：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'kingabzpro/wav2vec2-large-xls-r-300m-Urdu',
    local_dir='./model_files/wav2vec2-urdu',
    local_dir_use_symlinks=False,
)
"
```
9. 验证模型文件完整性（~2GB 含 model.safetensors 1.2GB + 5gram.arpa 765MB）：
```bash
ls -lh ./model_files/wav2vec2-urdu/
```

**输出**: 模型文件下载至 `./model_files/wav2vec2-urdu/`。

### 3. CPU 基线推理

**输入**: 模型权重文件、推理脚本 `scripts/inference.py`。

**动作**:
10. 在 CPU 上运行推理，指定使用合成音频测试：
```bash
cd model_files/wav2vec2-urdu
python3 scripts/inference.py --test
```
11. 验证 CPU 推理输出包含转录结果，记录推理延迟。
12. 若使用真实音频文件，验证采样率为 16kHz，不一致时触发重采样警告。

**输出**: CPU 推理转录文本和延迟数据作为精度对比基线。

### 4. NPU 推理

**输入**: 模型权重文件、`transfer_to_npu` 已启用。

**动作**:
13. 检查 `NPU_AVAILABLE` 状态，若 false 则跳过并标记 `NPU_FALLBACK=true`。
14. 在脚本顶部添加 `import torch_npu; from torch_npu.contrib import transfer_to_npu` 确保 CUDA API 自动映射至 NPU。
15. 执行 NPU 推理（代码使用 `cuda:0` 设备，由 transfer_to_npu 自动映射到 `npu:0`）：
```bash
cd model_files/wav2vec2-urdu
python3 inference.py --test
```
16. 处理失败：NPU 不可用时自动回退 CPU，OOM 时释放缓存后重试：
```python
import gc
gc.collect()
torch.npu.empty_cache()
```
17. 验证模型参数在 NPU 设备上加载 (`[INFO] Device: npu:0`)。

**输出**: NPU 推理转录文本和延迟数据。

### 5. CPU/NPU 精度对比验证

**输入**: CPU 基线结果和 NPU 推理结果（由 `accuracy_run.py` 自动管理）。

**动作**:
18. 执行一键精度验证脚本（自动运行 CPU 基线 + NPU 推理 + logits 对比）：
```bash
python3 scripts/accuracy_run.py
```
19. 计算以下精度指标：
    - Token 预测一致率（argmax 一致性），通过标准：>= 99%
    - 最大绝对误差，通过标准：非零元素最大相对误差 < 1%
    - 平均绝对误差
    - Top-1 in Top-5 命中率
    - 平均推理时间（NPU）
20. 若 `token_match_pct >= 99%` 且 `max_rel_diff < 1%` 标记 `PRECISION_PASS=true`；否则标记 `PRECISION_FAIL` 中止流程。

**输出**: 精度验证报告，保存至 `scripts/accuracy_check/accuracy_results.json`。

### 6. 性能基准测试

**输入**: 精度验证通过的模型，NPU 推理环境。

**动作**:
21. 运行性能基准测试脚本：
```bash
python3 scripts/accuracy_run_perf.py
```
22. 测试不同音频长度（3s、5s、10s、30s）在 NPU 上的推理延迟。
23. 计算 RTF（实时率 = 推理时间 / 音频时长），记录到性能报告。
24. 验证结果与预期范围一致：
    - 3s 音频 ~32ms 延迟（RTF ~0.01）
    - 30s 音频 ~56ms 延迟（RTF ~0.002）

**输出**: 性能数据保存至 `scripts/accuracy_check/perf_results.json`。

### 7. 验收确认

**输入**: 精度验证结果和性能测试结果。

**动作**:
25. 运行验收确认脚本：
```bash
python3 scripts/check_accuracy_run_perf.py
```
26. 检查精度验证状态：
    - Token 预测一致率 >= 99%
    - 最大绝对误差在可接受范围
27. 检查性能数据完整性：
    - 所有音频长度（3s、5s、10s、30s）均有数据
    - RTF 值在合理范围
28. 若所有检查通过，标记验收状态为「通过」。

**输出**: 最终验收报告，确认适配状态为「完成」。

### 8. 交付物整理

**输入**: 所有验证通过的脚本和结果文件。

**动作**:
29. 确认以下交付文件完备：
    - `SKILL.md` — 完整部署与验证流程文档
    - `scripts/inference.py` — NPU 推理脚本（支持单音频/目录/合成音频）
    - `scripts/accuracy_run.py` — 一键精度验证脚本
    - `scripts/accuracy_run_perf.py` — 性能基准测试脚本
    - `scripts/check_accuracy_run_perf.py` — 验证结果检查脚本
30. 确认精度和性能结果文件已生成：
```bash
ls -la scripts/accuracy_check/
```
31. 整理工作目录文件结构，确保无临时文件遗留。

**输出**: 完整的技能目录，包含流程文档、脚本和验证结果。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，Python 版本是否符合 3.10-3.12 | 暂停，提示检查 CANN 驱动或安装 torch_npu |
| 2 | CP-2: 模型下载完成检查点 | 模型权重下载完成后 | 文件大小是否符合预期（~2GB），目录结构是否正确 | 重新运行 `snapshot_download` 或切换镜像源 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理转录结果是否合理，采样率是否为 16kHz | 检查测试音频并重采样为 16kHz |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功，设备是否为 npu:0，耗时是否合理 | 检查 NPU 显存和驱动状态后重试 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | Token 预测一致率是否 >= 99%，最大相对误差是否 < 1% | 检查 transfer_to_npu 映射和数据预处理一致性 |
| 6 | CP-6: 性能验证确认检查点 | 性能基准测试完成后 | 各长度音频的 RTF 是否在合理范围（0.001-0.01） | 检查 NPU 负载和后台进程后重试 |
| 7 | CP-7: 验收审批检查点 | 全部确认完成后 | 所有验证项是否通过，报告是否完整 | 返回未通过项重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 `NPU_FALLBACK=true` | CP-1 | 安装 torch_npu 或检查 NPU 驱动后重启 |
| NPU 显存 OOM | 推理时报 `out of memory` 错误 | 释放 NPU 缓存（gc.collect + torch.npu.empty_cache），减少 batch | CP-4 | 关闭其他 NPU 进程后重试 |
| 模型下载失败 | `snapshot_download` 抛出网络异常 | 重试最多 3 次，每次间隔 5 秒 | CP-2 | 切换 `HF_ENDPOINT` 镜像源或离线导入 |
| 模型加载异常 | `from_pretrained` 抛出异常 | 打印错误堆栈，提示模型名是否正确或路径是否完整 | CP-2 | 检查模型目录结构和文件完整性 |
| 精度超标异常 | Token 预测一致率 < 99% 或最大相对误差 >= 1% | 记录偏差明细到日志，中止精度验证，不通过验收 | CP-5 | 检查推理脚本和数据预处理一致性后重试 |
| 音频采样率不匹配 | 输入音频采样率 != 16000 | 打印警告信息提示重采样，自动使用 sf.read 返回的原始采样率 | CP-3 | 使用 `librosa.resample` 或 `sox` 重采样至 16kHz |
| LM 解码依赖缺失 | `pyctcdecode` 或 `kenlm` 未安装 | 处理器自动降级为无 LM 解码，使用纯 argmax 解码 | CP-3 | `pip install pyctcdecode kenlm` 后重新运行 |
| 磁盘空间不足 | 下载过程中磁盘已满 | 提示清理磁盘空间（至少 7GB 空闲），打印当前磁盘使用情况 | CP-2 | 清理磁盘后重试下载 |
| fp64 不支持 | Ascend910 不支持 double 精度 | torch_npu 自动降级为 fp32，不影响推理精度，打印提示信息 | CP-1 | 无需操作，自动兼容 |
| 脚本依赖错误 | `import torch_npu` 或 `transfer_to_npu` 失败 | 打印错误堆栈，提示安装正确版本 torch_npu | CP-1 | `pip install torch_npu>=2.9.0` 后重试 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | NPU/CPU 推理执行入口，支持单音频文件、音频目录和合成音频测试 |
| `scripts/accuracy_run.py` | 一键精度验证脚本：自动执行 CPU 基线推理、NPU 推理、logits 对比和报告生成 |
| `scripts/accuracy_run_perf.py` | 性能基准测试脚本：在不同音频长度（3s/5s/10s/30s）上测试 NPU 推理延迟 |
| `scripts/check_accuracy_run_perf.py` | 验收确认脚本：检查精度验证和性能测试结果并输出最终报告 |
| `scripts/accuracy_check/accuracy_results.json` | 精度验证结果（运行后生成）：包含模型名、NPU 设备、各精度指标和通过状态 |
| `scripts/accuracy_check/perf_results.json` | 性能测试结果（运行后生成）：各音频长度的平均延迟、标准差和 RTF |
| `model_files/wav2vec2-urdu/` | 模型权重文件（首次运行下载）：model.safetensors、config.json、5gram.arpa 等 |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--audio` | string | 否 | — | 音频文件路径或包含音频的目录路径 |
| `--test` | boolean | 否 | false | 使用合成测试音频运行推理 |
| `--dtype` | string | 否 | fp32 | 推理精度（当前仅支持 fp32） |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `{transcription}` | text | 推理转录文本（乌尔都语 Unicode） |
| `accuracy_results.json` | JSON | 精度验证结果：Token 预测一致率、最大/平均绝对误差 |
| `perf_results.json` | JSON | 性能测试结果：各长度音频的平均延迟和 RTF |
| 验收报告 | text | 最终适配验证检查结果，标记通过/失败 |

## 使用约束

1. **音频采样率**: 模型训练于 16kHz 音频，输入音频需重采样至 16kHz。使用 `soundfile` 读取时自动获取采样率，非匹配时打印警告。
2. **transfer_to_npu**: 必须在脚本最顶部 `import torch_npu; from torch_npu.contrib import transfer_to_npu`，确保 CUDA API 自动映射。
3. **LM 解码依赖**: `Wav2Vec2ProcessorWithLM` 需要 `kenlm` + `pyctcdecode`，否则自动降级为无 LM argmax 解码。
4. **fp64 不支持**: Ascend910 不支持 double 精度，torch_npu 自动降级为 fp32，不影响推理精度。
5. **单卡推理**: 当前仅支持单卡推理，不支持多卡并行。
6. **transfer_to_npu 限制**: 模型代码中使用 `cuda:0` 设备字符串，由 `transfer_to_npu` 运行时映射为 `npu:0`，无需修改模型代码。
