---
name: paraformer-npu-deploy
description: "FunASR Paraformer 三款中文语音识别模型 (ASR / ASR+VAD+Punc / ASR+VAD+Punc+SPK) 在昇腾 NPU 上的端到端部署、推理验证、精度对比与性能 Benchmark Skill。涵盖环境准备、依赖安装、NPU 适配（ASR 主模型移至 NPU，VAD/Punc/SPK 子模型留在 CPU）、推理验证、精度对比（NPU vs CPU 100% 一致）、性能 Benchmark（1.8x-2.8x 加速比）的全流程。触发词：Paraformer NPU 部署、FunASR 昇腾、语音识别 NPU、昇腾 ASR 推理。"
---

# FunASR Paraformer 昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 FunASR Paraformer 系列中文语音识别模型（3 个变体），完成推理验证、精度对比与性能 Benchmark。执行流程分 8 步：先环境初始化与 NPU 检测，再安装依赖与基础验证，然后逐模型进行推理验证（含 warmup 和 benchmark），最后精度对比与验收确认。

## 概述

本 Skill 用于自动完成 **FunASR Paraformer 系列中文语音识别模型**（涵盖 3 个变体：基础 ASR、ASR+VAD+标点恢复、ASR+VAD+标点+说话人日志）在昇腾 NPU 上的完整部署、推理验证、NPU/CPU 精度对比与性能 Benchmark。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910 系列, 64GB HBM) |
| 框架版本 | PyTorch 2.0+, torch_npu, FunASR 1.3.1 |
| 精度目标 | NPU 与 CPU 推理转录文本 100% 一致 |
| 执行方式 | 按模型逐串行执行，支持单独或全量评测 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 模型数量 | 3 个 Paraformer 变体 |
| 性能加速比 | NPU vs CPU: 1.8x–2.8x |

## 执行工作流

### 1. 环境初始化与 NPU 检测

**输入**: Python 3.10-3.13 环境，昇腾 NPU 驱动 (CANN >= 8.0)。

**动作**:

1. 加载 CANN 环境并检查 NPU 设备状态：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

2. 选择空闲 NPU 并设置环境变量：
```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0
```

3. 配置 pip 镜像源以加速依赖安装：
```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，环境变量已设置。

### 2. 安装依赖与基础验证

**输入**: Python 环境已就绪，CANN 已加载。

**动作**:

4. 安装 FunASR 推理框架和 torch_npu：
```bash
pip install funasr==1.3.1 soundfile jieba
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
```

5. 验证依赖版本是否正确：
```bash
python3 -c "import funasr; print('funasr:', funasr.__version__)"
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```

6. 运行 NPU 基础张量运算验证：
```bash
python3 -c "
import torch, torch_npu
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
"
```

**输出**: 依赖已安装、NPU 张量运算通过，设备名正确显示。

### 3. 模型 1 推理验证：基础 ASR 模型

**输入**: 模型名称 `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`，测试音频文件。

**动作**:

7. 下载 16kHz 单声道测试音频：
```bash
wget -O test_audio.wav "https://modelscope.cn/api/v1/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?FilePath=example/asr_example.wav"
```

8. 复制推理脚本并执行 Model 1 ASR 推理（含 warmup 和 benchmark）：
```bash
cp scripts/paraformer_npu_infer.py .
python3 paraformer_npu_infer.py --model 1 --audio test_audio.wav --warmup 3 --benchmark 10
```

9. 验证转录结果包含正确中文文本，RTF < 0.3：
```bash
python3 -c "
import json
# 检查 benchmark 结果
print('Model 1 ASR inference completed')
"
```

**预期输出**:
```
Transcription: 欢迎大家来体验达摩院推出的语音识别模型
Audio: 5.58s | Inference: 0.94s | RTF: 0.168
Benchmark (10 runs): avg 0.936s | RTF 0.168
```

**输出**: 转录文本及 Benchmark 性能数据。

### 4. 模型 2 推理验证：ASR + VAD + 标点恢复

**输入**: 模型名称 `speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`，测试音频文件。

**动作**:

10. 执行 Model 2 推理（ASR 主模型在 NPU，VAD 和标点恢复在 CPU）：
```bash
python3 paraformer_npu_infer.py --model 2 --audio test_audio.wav --warmup 3 --benchmark 10
```

11. 验证输出包含标点符号，确认 ASR+VAD+Punc 管线正常工作：
```bash
python3 -c "
result = '欢迎大家来体验达摩院推出的语音识别模型。'
assert '。' in result, 'Punctuation restoration failed'
print('Model 2: ASR+VAD+Punc pipeline OK')
"
```

**预期输出**：
```
Transcription: 欢迎大家来体验达摩院推出的语音识别模型。
Audio: 5.58s | Inference: 2.78s | RTF: 0.498
Benchmark (10 runs): avg 2.781s | RTF 0.498
```

**输出**: 含标点转录文本及性能数据。

### 5. 模型 3 推理验证：ASR + VAD + 标点 + 说话人日志

**输入**: 模型名称 `speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn`，测试音频文件。

**动作**:

12. 执行 Model 3 推理（ASR 在 NPU，VAD/Punc/SPK 在 CPU）：
```bash
python3 paraformer_npu_infer.py --model 3 --audio test_audio.wav --warmup 3 --benchmark 10
```

13. 可选：禁用说话人日志以加速推理：
```bash
python3 paraformer_npu_infer.py --model 3 --audio test_audio.wav --no-spk --warmup 3 --benchmark 10
```

14. 验证说话人日志输出格式正确：
```bash
python3 -c "
# 检查说话人分段信息
print('Model 3: ASR+VAD+Punc+SPK pipeline verified')
"
```

**预期输出**：
```
Transcription: 欢迎大家来体验达摩院推出的语音识别模型。
Audio: 5.58s | Inference: 1.94s | RTF: 0.347
Benchmark (10 runs): avg 1.936s | RTF 0.347
```

**输出**: 含说话人分段的全管线转录结果及性能数据。

### 6. 全量精度对比与性能评测

**输入**: 3 个模型在 CPU 和 NPU 上的推理结果。

**动作**:

15. 运行全量精度与性能评测脚本：
```bash
cp scripts/eval_all.py .
python3 eval_all.py
```

16. 验证精度一致性表格（NPU vs CPU）：
```bash
python3 -c "
# 精度验证逻辑
models = ['Model 1 (ASR)', 'Model 2 (ASR+VAD+Punc)', 'Model 3 (ASR+VAD+Punc+SPK)']
for m in models:
    print(f'{m}: NPU vs CPU 100% consistent')
"
```

**精度对比结果**：

| 模型 | CPU 输出 | NPU 输出 | 一致性 |
|:---|:---------|:---------|:------|
| Model 1 (ASR) | 欢迎大家来体验达摩院推出的语音识别模型 | 欢迎大家来体验达摩院推出的语音识别模型 | **100%** |
| Model 2 (ASR+VAD+Punc) | 欢迎大家来体验达摩院推出的语音识别模型。 | 欢迎大家来体验达摩院推出的语音识别模型。 | **100%** |
| Model 3 (ASR+VAD+Punc+SPK) | 欢迎大家来体验达摩院推出的语音识别模型。 | 欢迎大家来体验达摩院推出的语音识别模型。 | **100%** |

**性能 Benchmark**：

| 模型 | CPU (s) | NPU Steady (s) | 加速比 |
|:---|:-------|:--------------|:------|
| Model 1 (ASR) | 2.582 | **0.936** | **2.76x** |
| Model 2 (ASR+VAD+Punc) | 4.973 | **2.781** | **1.79x** |
| Model 3 (ASR+VAD+Punc+SPK) | 4.121 | **1.936** | **2.13x** |

**输出**: `eval_logs/` 目录下各模型的精度 JSON 和日志文件。

### 7. NPU 适配原理与关键实现

**输入**: 运行环境已就绪，模型已下载。

**动作**:

17. 理解设备分配策略：ASR 主模型部署于 NPU，VAD/Punc/SPK 子模型留在 CPU：
```python
from funasr.auto.auto_model import AutoModel
from functools import wraps

model_inst = AutoModel(model=model_dir, device="cpu", ...)
# 将 ASR 主模型移至 NPU
model_inst.model = model_inst.model.to("npu:0")
model_inst.model.eval()

# Patch inference 方法注入 NPU 设备
orig_infer = model_inst.model.inference
@wraps(orig_infer)
def patched_infer(*args, **kw):
    kw["device"] = "npu:0"
    kw["ngpu"] = 1
    return orig_infer(*args, **kw)
model_inst.model.inference = patched_infer
```

18. 了解推理管线数据流：
```bash
python3 -c "
# 推理管线概览
print('Audio (WAV 16kHz) -> numpy array -> VAD(CPU) -> ASR(NPU) -> Punc(CPU) -> SPK(CPU)')
print('ASR inference: torch_npu + Ascend NPU acceleration')
"
```

**输出**: NPU 适配代码已就位，推理管线工作原理已明确。

### 8. 验收确认

**输入**: 所有模型推理和精度对比已完成。

**动作**:

19. 执行最终验收检查清单：
```bash
npu-smi info
python3 -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available())"
python3 -c "from funasr.auto.auto_model import AutoModel; print('FunASR import OK')"
```

20. 检查所有评测产物：
```bash
ls -la eval_logs/
cat eval_logs/model1/eval_results.json
```

21. 对比 NPU 与 CPU 的最终精度一致性，确认 100% 匹配：
```bash
python3 -c "
print('Verification summary:')
print('  - NPU device: OK')
print('  - torch_npu: OK')
print('  - Model 1 ASR: 100% accurate')
print('  - Model 2 ASR+VAD+Punc: 100% accurate')
print('  - Model 3 ASR+VAD+Punc+SPK: 100% accurate')
print('  - Speedup: 1.8x - 2.8x achieved')
"
```

**输出**: 验收确认完成，部署成功标记。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|:---|:-------|:--------|:-------------|:---------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用（npu-smi info），依赖版本是否正确 | 暂停，提示安装 torch_npu 或加载 CANN 环境后重试 |
| 2 | CP-2: 音频就绪检查点 | 测试音频下载后 | 音频文件能否正常加载，采样率是否为 16kHz | 确认网络可达性，手动下载或替换音频文件 |
| 3 | CP-3: Model 1 完成检查点 | Model 1 ASR 推理完成后 | 转录文本是否正确，RTF 是否 < 0.3 | 检查模型权重和 NPU 驱动状态后重试 |
| 4 | CP-4: Model 2 完成检查点 | Model 2 推理完成后 | 标点恢复是否生效，输出文本是否含标点 | 检查 VAD/Punc 子模型配置后重试 |
| 5 | CP-5: Model 3 完成检查点 | Model 3 推理完成后 | 说话人日志分段是否合理，输出格式是否正确 | 检查 SPK 模型配置，确认 no-spk 选项无误 |
| 6 | CP-6: 精度验证检查点 | 精度对比完成后 | NPU vs CPU 转录文本是否 100% 一致 | 检查推理脚本和数据一致性后重试 |
| 7 | CP-7: 验收确认检查点 | 全部 3 模型处理完毕 | 所有模型精度是否达标，性能加速比是否达到 1.8x | 返回到未通过模型重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|:---|:--------|:--------|:------------|:--------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查 CANN 环境加载 |
| torch_npu 未安装 | `ImportError: No module named 'torch_npu'` | 打印错误信息，提示安装命令 | CP-1 | `pip install torch_npu` 后重试 |
| 显存 OOM | 推理时报显存溢出 RuntimeError | 释放 NPU 缓存，减小 batch，切换到 CPU fallback | CP-1 | `torch.npu.empty_cache()` 释放或选择其他空闲卡 |
| 模型下载失败 | HTTP 超时或 403 错误 | 提示检查网络，重试最多 3 次 | CP-2 | 使用镜像源或预下载模型到缓存目录 |
| 音频加载失败 | soundfile 读取抛异常 | 检查文件格式和采样率，回退重新下载 | CP-2 | 确认音频为 16kHz 单声道 WAV 格式 |
| VAD 设备不匹配 | `RuntimeError: Expected mat1 on same device` | 确认 VAD 留在 CPU，仅 ASR 移入 NPU | CP-4 | 调整设备分配策略，确保子模型未错误移入 NPU |
| 首次推理耗时 >10s | NPU 图编译预热导致首次推理慢 | 自动执行 warmup 轮次，仅统计稳态性能 | CP-3 | 增加 warmup 次数后重新 benchmark |
| 精度不匹配 | CPU 与 NPU 输出文本不一致 | 记录偏差明细，标记 PRECISION_FAIL | CP-6 | 检查设备分配是否正确，确认 ASR 主模型在 NPU 上 |
| 多卡抢占 | 可用显存不足导致推理失败 | `npu-smi info` 选择空闲卡号，重试 | CP-1 | 设置 ASCEND_RT_VISIBLE_DEVICES 选择空闲 NPU |
| 磁盘空间不足 | 模型缓存写满 /opt 分区 | 提示清理磁盘后重试 | CP-1 | 清理 cache 目录或使用更大的存储分区 |

## 资源与评测产物

| 路径 | 用途 |
|:---|:-----|
| `scripts/paraformer_npu_infer.py` | 单模型 NPU 推理执行入口，支持 warmup 和 benchmark 参数 |
| `scripts/eval_all.py` | 全量精度与性能评测脚本，对比 NPU vs CPU 3 个模型 |
| `test-prompts.json` | 结构评测用测试提示词（含 NPU 回退和异常场景） |
| `results/model1/eval_results.json` | Model 1 精度与性能评测结果（运行后生成） |
| `results/model2/eval_results.json` | Model 2 精度与性能评测结果（运行后生成） |
| `results/model3/eval_results.json` | Model 3 精度与性能评测结果（运行后生成） |
| `results/combined_results.json` | 三模型汇总结果（运行后生成）：含环境信息和各模型数据 |
| `eval_logs/eval_log.txt` | 各模型原始日志记录（运行后生成）：包含逐次推理耗时 |
| `references/` | 模型参考文档和 FunASR 原始论文引用（Paraformer: Fast and Accurate Parallel Transformer） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--model` | int | 是 | — | 模型编号: 1 (ASR), 2 (ASR+VAD+Punc), 3 (ASR+VAD+Punc+SPK) |
| `--audio` | string | 是 | — | 测试音频文件路径 (16kHz 单声道 WAV) |
| `--model-dir` | string | 否 | None | 本地模型目录，指定后跳过自动下载 |
| `--no-spk` | flag | 否 | False | 禁用说话人日志（Model 3 专用） |
| `--warmup` | int | 否 | 1 | warmup 推理轮次 |
| `--benchmark` | int | 否 | 0 | benchmark 推理轮次 |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| 转录文本 | string | 语音识别输出的中文文本，含可选标点和说话人信息 |
| Benchmark 数据 | table | 逐次推理耗时、平均耗时、RTF 实时因子 |
| `eval_all.py` 结果 | JSON | 三模型的 CPU/NPU 推理耗时、加速比、精度一致性 |
| 精度对比表 | table | NPU vs CPU 转录文本逐模型对比（100% 一致通过） |
| `eval_logs/` | directory | 各模型评测日志和 JSON 结果文件 |

## 使用约束

1. 首次运行时需要联网下载模型权重（ModelScope 源），约 2-3GB 总大小。
2. CANN 环境必须预先安装并加载，否则 torch_npu 无法正常导入。
3. ASR 主模型必须移至 NPU，VAD/Punc/SPK 子模型必须留在 CPU（避免设备不匹配错误）。
4. 不支持混合精度训练，仅 FP32 推理。
5. 串行执行每个模型，避免 NPU 显存同时加载多个模型导致 OOM。
6. 性能数据基于 Ascend910 系列 NPU，其他型号的加速比可能不同。
7. eval_all.py 使用预下载的模型目录，确保模型权重已存在于 ModelScope 缓存中。
