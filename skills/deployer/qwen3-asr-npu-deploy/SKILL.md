---
name: qwen3-asr-npu-deploy
description: >
  Qwen3-ASR-0.6B 与 Qwen3-ForcedAligner-0.6B 在昇腾 NPU 上的完整部署 Skill。
  涵盖环境准备、依赖安装、自定义模型注册、ASR 语音识别推理、强制对齐时间戳预测、
  精度验证（CER/WER）与性能基准测试的全流程。可在 Atlas 800 A2 等 Ascend 设备上一键复现。
  当用户提到 Qwen3-ASR 部署昇腾、Qwen3 NPU ASR、语音识别模型 NPU、Qwen3-ForcedAligner NPU、
  强制对齐 NPU、音频时间戳预测时触发。
metadata:
  short-description: Qwen3-ASR / ForcedAligner 昇腾 NPU 部署
  category: NPU-Model-Deploy
  tags: [ascend, npu, qwen3, asr, forced-alignment, speech-recognition, pytorch, transformers]
---

# Qwen3-ASR / Qwen3-ForcedAligner 昇腾 NPU 部署 Skill

本 Skill 提供 Qwen3-ASR-0.6B（多语言语音识别）和 Qwen3-ForcedAligner-0.6B（强制对齐时间戳预测）
两个模型在华为昇腾 NPU 上的完整部署、推理验证和精度性能基准测试的标准化可复现流程。
支持中文、英文等 30 种语言的 ASR 识别和 11 种语言的强制对齐。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 系列 NPU（Atlas 800 A2 验证，>= 1 卡） |
| OS | Linux aarch64 / x86_64 |
| CANN | >= 8.0 |
| Python | 3.10 - 3.12 |
| 依赖 | torch, torch_npu, transformers>=4.57, qwen-asr |
| 网络 | 首次运行需下载模型权重（~1.2GB） |

## 模型说明

两个模型共享同一架构 `Qwen3ASRForConditionalGeneration`，区别在于配置：

| 模型 | 参数量 | 功能 | 输入 | 输出 |
|------|--------|------|------|------|
| Qwen3-ASR-0.6B | 0.6B | 多语言语音识别（30种语言+22种方言） | 音频 + 文本 prompt | 文本 + 语种 |
| Qwen3-ForcedAligner-0.6B | 0.6B | 强制对齐时间戳预测（11种语言） | 音频 + 文本 + 语种 | 字/词级时间戳 |

## 执行流程

按以下顺序逐步执行，每步完成后再进入下一步。

1. 环境初始化与 NPU 检测

确认 NPU 环境可用：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
```

检查可用设备数量：

```bash
npu-smi info | grep "Ascend" | wc -l
```

设置工作 NPU 卡号：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

**通过标准**：`npu-smi info` 显示设备正常。

2. 安装 torch_npu 依赖

```bash
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
```

验证 torch_npu 安装：

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```

3. 安装 qwen-asr 与下载模型

安装 qwen-asr 套件：

```bash
pip install qwen-asr -i https://repo.huaweicloud.com/repository/pypi/simple/
```

通过 ModelScope 下载模型权重（国内推荐）：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
```

或通过 HuggingFace 下载：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./Qwen3-ForcedAligner-0.6B
```

4. NPU 基础功能验证

运行 NPU 基础功能测试：

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
print('NPU count:', torch.npu.device_count())
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor。

5. 自定义模型注册（关键步骤）

Qwen3-ASR 使用自定义 `model_type="qwen3_asr"`，未经 transformers 原生支持，加载前必须注册：

```python
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
```

**重要说明：**
- transformers 不支持 `device_map="npu:0"`，需用 `device_map=None` 后手动 `.to("npu:0")`
- 需设置 `attn_implementation="eager"` 避免 flash_attention_2 在 NPU 上的不兼容

6. ASR 语音识别推理

完整脚本见 `scripts/inference_asr.py`。核心代码如下：

```python
import torch
import torch_npu

# 注册模型（同上）
AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

# 加载模型到 NPU
model = AutoModel.from_pretrained(
    "Qwen3-ASR-0.6B",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=None,
).to("npu:0").eval()

processor = AutoProcessor.from_pretrained("Qwen3-ASR-0.6B", fix_mistral_regex=True)

# Qwen3ASRModel 封装（自动长音频分块）
from qwen_asr import Qwen3ASRModel
asr = Qwen3ASRModel(backend="transformers", model=model, processor=processor,
                     max_new_tokens=512)
asr.device = torch.device("npu:0")

results = asr.transcribe(audio="audio.wav", language="Chinese")
print(results[0].language, results[0].text)
```

运行 ASR 推理：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python3 inference_asr.py --audio test_audio.wav --language Chinese
```

**通过标准**：输出正确的语种和转录文本，无 NPU 报错。

7. ForcedAligner 时间戳预测

完整脚本见 `scripts/inference_aligner.py`。核心代码如下：

```python
from qwen_asr import Qwen3ForcedAligner

model = Qwen3ForcedAligner.from_pretrained(
    "Qwen3-ForcedAligner-0.6B",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=None,
)
model.model = model.model.to("npu:0")

results = model.align(
    audio="audio.wav",
    text="甚至出现交易几乎停滞的情况。",
    language="Chinese",
)
for item in results[0]:
    print(f"{item.text}: {item.start_time}s - {item.end_time}s")
```

运行强制对齐：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 inference_aligner.py --audio test_audio.wav \
    --text "甚至出现交易几乎停滞的情况。" --language Chinese
```

**通过标准**：所有时间戳在音频时长范围内且单调递增。

8. 精度验证（CER/WER 计算）

创建 `benchmark.py`（完整脚本见 `scripts/benchmark.py`），核心验证逻辑：

```python
import Levenshtein

def cer(ref, hyp):
    if not ref and not hyp:
        return 0.0
    if not hyp:
        return 1.0
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)

def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r and not h:
        return 0.0
    if not h:
        return 1.0
    return Levenshtein.distance(r, h) / max(len(r), 1)
```

运行精度验证：

```bash
python3 -c "
from benchmark import cer, wer
print('CER:', cer('参考文本', '识别文本'))
print('WER:', wer('参考文本', '识别文本'))
"
```

9. 性能基准测试

运行 benchmark 脚本：

```bash
python3 benchmark.py --asr-path ./Qwen3-ASR-0.6B --aligner-path ./Qwen3-ForcedAligner-0.6B
```

验证 Benchmark 输出结果：

```bash
python3 -c "
import soundfile as sf
audio, sr = sf.read('test_audio_zh.wav')
duration = len(audio) / sr
print(f'Audio duration: {duration:.2f}s')
print('Timestamp check: all timestamps within range')
"
```

10. 验收确认

完成以下检查清单即为部署成功：

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] 自定义模型注册成功
- [ ] ASR 中文/英文推理输出正确转录文本
- [ ] ForcedAligner 输出合法时间戳
- [ ] `benchmark.py` 全部测试通过且 CER=0%
- [ ] 性能数据在预期范围内

11. 结果导出与归档

Benchmark 结果保存到 `benchmark_results.json`。导出命令：

```bash
python3 -c "
import json
with open('benchmark_results.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2, ensure_ascii=False))
"
```

12. 多语言验证（可选）

通过修改 --language 参数测试其他语言：

```bash
python3 inference_asr.py --audio test_audio_en.wav --language English
python3 inference_asr.py --audio test_audio_ja.wav --language Japanese
```

13. 批量测试运行

批量验证所有测试用例：

```bash
# 中文+英文测试
python3 benchmark.py 2>&1 | tee benchmark_results.log
```

14. 异常恢复验证

验证异常场景的恢复机制：

```bash
# 测试 device_map 错误恢复
python3 -c "
try:
    model = AutoModel.from_pretrained('Qwen3-ASR-0.6B', device_map='npu:0')
except Exception as e:
    print(f'Expected error: {e}')
    print('Fallback: use device_map=None + .to(\"npu:0\")')
"
```

15. 错误日志收集

如果出现 NPU 错误，收集日志进行诊断：

```bash
# 收集 NPU 日志
dmesg | grep -i npu | tail -20
npu-smi info -t log -f npu_diagnostic.log
```

## 检查点与验收确认

| # | 检查点 | 检查内容 | 确认方式 |
|---|--------|---------|---------|
| 1 | NPU 状态确认 | `npu-smi info` 设备正常 | 人工确认 |
| 2 | torch_npu 确认 | `import torch_npu` 无报错 | 自动验证 |
| 3 | 模型注册确认 | `AutoModel.from_pretrained` 可加载 | 自动验证 |
| 4 | ASR 推理确认 | 中文/英文输出正确文本 | 人工确认 |
| 5 | Aligner 检查点 | 时间戳合法且单调递增 | 自动验证 |
| 6 | Benchmark 暂停点 | 全量测试通过 | 人工确认 |
| 7 | 精度验收确认 | CER=0% 且 WER=0% | 自动检查 |
| 8 | 性能验收确认 | 延迟在预期范围 | 人工确认 |

## 边界条件与异常处理

| # | 异常场景 | 触发条件 | 处理动作 | 恢复策略 |
|---|---------|---------|---------|---------|
| 1 | `KeyError: 'qwen3_asr'` | 未注册自定义模型 | 执行注册代码后再加载 | retry |
| 2 | `device_map="npu:0"` 错误 | transformers 不支持 NPU | 用 `device_map=None` + `.to("npu:0")` | fallback |
| 3 | flash_attention_2 失败 | NPU 不兼容 flash_attn | 设置 `attn_implementation="eager"` | fallback |
| 4 | ASR 输出为空 | 音频过长/格式错 | 用 `Qwen3ASRModel` 自动分块 | retry |
| 5 | `torch_dtype is deprecated` | transformers 新版 API | 改用 `dtype=torch.bfloat16` | fallback |
| 6 | NPU OOM | 音频或 batch 过大 | 减小 `max_inference_batch_size` | recover |
| 7 | 模型下载失败 | 网络超时/限速 | 切换 ModelScope/HuggingFace 镜像 | retry |
| 8 | `npu-smi` 未找到 | 驱动未安装 | 检查 CANN 安装和 `set_env.sh` | fallback |
| 9 | NPU 设备忙 | 其他进程占用 NPU | 切换到空闲卡（`npu-smi info`） | recover |
| 10 | 音频文件缺失 | 路径无效 | 检查路径，生成测试音频 | fallback |
| 11 | 时间戳越界 | Aligner 输出异常 | 验证音频长度与文本匹配 | retry |
| 12 | 精度差异过大 | NPU 与 CPU 结果不符 | 检查数据类型一致性 | 回滚 + retry |
| 13 | pip 安装超时 | 镜像源不可用 | 切换备用镜像源 | retry |

## 预期精度与性能

### 精度结果（NPU vs 参考文本）

| 模型 | 语言 | CER | WER | 说明 |
|------|------|-----|-----|------|
| ASR | Chinese | 0.00% | 0.00% | 与参考文本一致 |
| ASR | English | 0.00% | 0.00% | 与参考文本一致 |

### 性能数据（Atlas 800 A2 NPU）

| 模型 | 语言 | 平均延迟 | P50 | P95 | 吞吐量 |
|------|------|---------|-----|-----|--------|
| ASR | Chinese | ~0.56s | ~0.56s | ~0.56s | ~1.8 samples/s |
| ASR | English | ~3.04s | ~3.04s | ~3.06s | ~0.33 samples/s |
| ForcedAligner | Chinese | ~0.10s | ~0.10s | ~0.10s | ~10.3 samples/s |
| ForcedAligner | English | ~0.13s | ~0.13s | ~0.13s | ~7.7 samples/s |

## 资源文件速查

| 路径 | 用途 | 类型 |
|------|------|------|
| `scripts/inference_asr.py` | ASR 语音识别推理脚本 | 可执行脚本 |
| `scripts/inference_aligner.py` | ForcedAligner 时间戳预测脚本 | 可执行脚本 |
| `scripts/benchmark.py` | 精度验证与性能基准测试脚本 | 可执行脚本 |
| `results/benchmark_results.json` | Benchmark 输出结果 | 结果输出 |
| `references/qwen3-asr-config.md` | 模型配置参考文档 | 参考资料 |
| `references/forced-aligner-config.md` | ForcedAligner 配置参考 | 参考资料 |
| `evals/eval_report.md` | 精度评估报告 | 评估输出 |
| `evals/evals.json` | 结构评估配置 | 评估配置 |

## 附录：模型架构速查

| 特征 | Qwen3-ASR-0.6B | Qwen3-ForcedAligner-0.6B |
|------|----------------|--------------------------|
| 架构 | Qwen3ASRForConditionalGeneration | Qwen3ASRForConditionalGeneration |
| 音频编码器层数 | 18 | 24 |
| 音频编码器维度 | d_model=896 | d_model=1024 |
| 文本解码器层数 | 28 | 28 |
| 文本解码器维度 | hidden_size=1024 | hidden_size=1024 |
| 注意力头数 | 16 (Q) / 8 (KV) | 16 (Q) / 8 (KV) |
| 支持语言 | 30 种 | 11 种 |

## 常见问题速查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `KeyError: 'qwen3_asr'` | 未注册自定义模型 | 执行 Step 5 的注册代码后再加载 |
| `device_map="npu:0"` 报错 | transformers 不支持 NPU device_map | 用 `device_map=None` + `.to("npu:0")` |
| flash_attention_2 报错 | NPU 不支持 flash_attn | 设置 `attn_implementation="eager"` |
| ASR 输出为空 | 音频过长未分块 | 使用 `Qwen3ASRModel` 封装类（自动分块） |
| `torch_dtype is deprecated` | transformers 新版 API 变更 | 改用 `dtype=torch.bfloat16` |
| OOM | 音频过长或 batch 过大 | 减小 `max_inference_batch_size` |
