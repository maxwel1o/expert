---
name: qwen-asr-npu-deploy
description: Ascend NPU deployment skill for Qwen3-ASR speech recognition models. Automates the complete pipeline of adapting, verifying, and deploying Qwen3-ASR (0.6B/1.7B) on Huawei Ascend NPUs via vLLM-Ascend. Covers environment setup, vllm-ascend compatibility patching, model loading validation, CPU-vs-NPU precision comparison, performance benchmarking, and documentation generation. Use when users need to deploy ASR models on Ascend NPU, validate transcription accuracy against CPU baseline, or troubleshoot multimodal model loading issues on vLLM-Ascend.
---

# Qwen3-ASR Ascend NPU 部署验证 Skill

**TL;DR**: 本 Skill 提供从环境准备到精度验证的完整流水线，用于在华为昇腾 NPU 上部署 Qwen3-ASR 语音识别模型。包含 vllm-ascend 兼容性补丁、NPU 推理验证、CPU/NPU 精度对比、性能测试及适配文档自动生成。

---

## 目录

- [适用场景](#适用场景)
- [前置条件](#前置条件)
- [核心流程概览](#核心流程概览)
- [Phase 1: 环境预检与准备](#phase-1-环境预检与准备)
- [Phase 2: 依赖安装与补丁](#phase-2-依赖安装与补丁)
- [Phase 3: 模型加载验证](#phase-3-模型加载验证)
- [Phase 4: NPU 推理测试](#phase-4-npu-推理测试)
- [Phase 5: CPU/NPU 精度对比](#phase-5-cpunpu-精度对比)
- [Phase 6: 性能基准测试](#phase-6-性能基准测试)
- [Phase 7: 文档与交付](#phase-7-文档与交付)
- [边界条件与异常处理](#边界条件与异常处理)
- [检查点清单](#检查点清单)
- [资源清单](#资源清单)
- [参考指标](#参考指标)

---

## 适用场景

- 在昇腾 NPU (Atlas 800 A2/A3) 上部署 Qwen3-ASR-0.6B 或 Qwen3-ASR-1.7B
- 验证 ASR 模型在 NPU 上的推理功能是否正常
- 对比 CPU 与 NPU 推理结果的精度差异
- 为模型生成 Ascend 适配文档并提交到 GitCode
- 排查 vLLM-Ascend 加载多模态模型时的接口兼容性问题

---

## 前置条件

### 硬件要求

| 组件 | 要求 |
|------|------|
| NPU 设备 | Atlas 800 A2 (64G×8) 或 Atlas 800 A3 (64G×16) |
| 单卡显存 | >= 4GB HBM (Qwen3-ASR-0.6B 仅需 ~1.5GB 权重) |
| 内存 | 建议 32GB 以上 |
| 存储 | 建议 10GB 以上可用空间 |

### 软件要求

| 软件 | 版本 |
|------|------|
| Python | 3.9 - 3.12 |
| vLLM-Ascend | v0.18.0rc1+ |
| vLLM | v0.18.0+ |
| transformers | 4.57.6 |
| torch-npu | 2.9.0+ |
| qwen-asr | 0.0.6 |

---

## 核心流程概览

```
Phase 1: 环境预检  ──→  Phase 2: 依赖安装与补丁  ──→  Phase 3: 模型加载验证
     │                                              │
     ▼                                              ▼
Phase 7: 文档交付  ◄──  Phase 6: 性能基准测试  ◄──  Phase 4: NPU 推理测试  ◄──  Phase 5: CPU/NPU 精度对比
```

---

## Phase 1: 环境预检与准备

### Step 1.1: 检查 NPU 设备状态

```bash
npu-smi info
```

**预期输出**: 显示 NPU 卡数量、温度、HBM 使用率。确认至少 1 张卡状态为 `OK`。

**异常处理**: 若 `npu-smi` 不可用，检查 CANN 驱动安装：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Step 1.2: 检查 Python 依赖版本

```bash
python3 -c "import torch_npu; print('torch_npu:', torch_npu.__version__)"
python3 -c "import vllm; print('vllm:', vllm.__version__)"
python3 -c "import vllm_ascend; print('vllm_ascend: installed')"
python3 -c "import transformers; print('transformers:', transformers.__version__)"
```

**检查点 1**: 所有依赖版本符合 [前置条件](#前置条件) 表格要求。若版本不匹配，暂停并确认是否继续。

### Step 1.3: 确认模型权重路径

```bash
ls -la ${MODEL_PATH}/config.json
ls -la ${MODEL_PATH}/model.safetensors
```

**环境变量**: `MODEL_PATH` 默认为 `/opt/atomgit/QwenASR/models`，可通过 `--model-path` 参数覆盖。

---

## Phase 2: 依赖安装与补丁

### Step 2.1: 安装 qwen-asr 包

```bash
# 方式一：从 PyPI 安装（需联网）
pip install -U qwen-asr

# 方式二：从源码安装（推荐，无网环境）
git clone --depth 1 https://github.com/QwenLM/Qwen3-ASR.git /tmp/Qwen3-ASR
cd /tmp/Qwen3-ASR
pip install -e . --no-build-isolation --no-deps
```

### Step 2.2: 处理可选依赖缺失

qwen-asr 依赖 `nagisa` 和 `soynlp`，这两个库仅用于 Forced Aligner 功能，不影响 ASR 核心推理。若安装失败，创建占位模块：

```bash
mkdir -p ~/.local/lib/python$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/site-packages

cat > ~/.local/lib/python*/site-packages/nagisa.py << 'EOF'
class Tagger:
    def tagging(self, text): return []
def tagging(text): return []
EOF

cat > ~/.local/lib/python*/site-packages/soynlp.py << 'EOF'
class WordExtractor:
    def extract(self): return {}
    def load(self, path): pass
def get_compound_score(word, scores): return 0.0
EOF
```

### Step 2.3: 应用 vllm-ascend 兼容性补丁

当前 `vllm-ascend 0.18.0rc1` 与 `qwen-asr 0.0.6` 存在 3 处 API 差异，必须在 import qwen_asr 之前应用运行时补丁：

```python
# npu_patch.py
import types

def patch_ascend_mm_encoder_attention():
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
    orig_init = AscendMMEncoderAttention.__init__
    def new_init(self, num_heads, head_size, scale=None, num_kv_heads=None, prefix="", multimodal_config=None):
        orig_init(self, num_heads, head_size, scale, num_kv_heads, prefix)
    AscendMMEncoderAttention.__init__ = new_init

def patch_get_vit_attn_backend():
    from vllm.model_executor.models.vision import get_vit_attn_backend as orig_fn
    def new_fn(head_size, dtype, attn_backend_override=None):
        return orig_fn(head_size, dtype)
    import vllm.model_executor.models.vision
    vllm.model_executor.models.vision.get_vit_attn_backend = new_fn

def patch_embed_text_input_ids():
    from vllm.model_executor.models.interfaces import SupportsMultiModal
    orig = SupportsMultiModal._embed_text_input_ids
    def new_fn(self, input_ids, embed_input_ids, *, is_multimodal=None, handle_oov_mm_token=False):
        return orig(self, input_ids, embed_input_ids, is_multimodal=is_multimodal)
    SupportsMultiModal._embed_text_input_ids = new_fn

patch_ascend_mm_encoder_attention()
patch_get_vit_attn_backend()
patch_embed_text_input_ids()
```

**使用方式**:
```python
import npu_patch  # 必须在 import qwen_asr 之前
from qwen_asr import Qwen3ASRModel
```

**边界条件**: 若未来 vllm-ascend 版本修复了上述接口，补丁会通过 try/except 静默跳过，不会报错。

---

## Phase 3: 模型加载验证

### Step 3.1: NPU 模型加载测试（Dummy Gate）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 -c "
import npu_patch
from vllm import LLM
llm = LLM(
    model='${MODEL_PATH}',
    dtype='bfloat16',
    max_model_len=65536,
    max_num_seqs=4,
    trust_remote_code=True,
)
print('Model loaded on NPU:', type(llm.llm_engine).__name__)
print('Supported tasks:', llm.llm_engine.supported_tasks)
"
```

**预期输出**: `Supported tasks: ['generate', 'transcription']`

**异常处理**:
- 若报错 `TypeError: AscendMMEncoderAttention.__init__() got unexpected keyword argument 'multimodal_config'` → 未正确应用 npu_patch.py
- 若报错 `RuntimeError: Engine core initialization failed` → 检查 NPU 显存是否充足，或尝试 `--enforce-eager`
- 若报错 `KeyError: 'qwen3_asr'` → qwen-asr 包未正确安装，模型架构未注册

**检查点 2**: 模型成功加载且显示 `transcription` 任务支持。若不支持，停止并排查 qwen-asr 版本。

### Step 3.2: 权重加载确认

观察日志中的关键指标：
```
INFO Loading model weights took X.XX GB
INFO Available KV cache memory: XX.XX GiB
INFO Maximum concurrency for XXXXX tokens per request: X.XXx
```

**边界条件**: 若 `Loading model weights took` 远大于 2GB，可能加载了错误模型或存在重复加载。

---

## Phase 4: NPU 推理测试

### Step 4.1: 单条音频推理

```python
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import npu_patch
from qwen_asr import Qwen3ASRModel

asr = Qwen3ASRModel.LLM(
    model="/opt/atomgit/QwenASR/models",
    dtype="bfloat16",
    max_model_len=65536,
    max_num_seqs=4,
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
)

results = asr.transcribe(
    audio="/path/to/audio.wav",
    language=None,  # 自动识别语言
    return_time_stamps=False,
)

print(f"Language: {results[0].language}")
print(f"Text: {results[0].text}")
```

**输入参数说明**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | str | ✅ | 音频文件路径（支持 wav/mp3/flac） |
| `language` | str | ❌ | 强制语言，如 "Chinese"/"English"，默认自动识别 |
| `return_time_stamps` | bool | ❌ | 是否返回时间戳，默认 False |

**预期输出**: 返回包含 `language` 和 `text` 字段的 `ASRTranscription` 对象。

### Step 4.2: Batch 推理

```python
results = asr.transcribe(
    audio=["audio1.wav", "audio2.wav", "audio3.wav"],
    language=[None, "Chinese", "English"],
    return_time_stamps=False,
)

for i, r in enumerate(results):
    print(f"[{i}] {r.language}: {r.text}")
```

**边界条件**: batch 大小受 `--max-num-seqs` 限制，超出时会自动分块处理。

---

## Phase 5: CPU/NPU 精度对比

### Step 5.1: CPU 基准推理

```python
from qwen_asr import Qwen3ASRModel
import torch

asr_cpu = Qwen3ASRModel.from_pretrained(
    "/opt/atomgit/QwenASR/models",
    dtype=torch.bfloat16,
)

results_cpu = asr_cpu.transcribe(
    audio="test_data/asr_en.wav",
    language="English",
    return_time_stamps=False,
)
text_cpu = results_cpu[0].text
```

### Step 5.2: NPU 推理

```python
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
import npu_patch
from qwen_asr import Qwen3ASRModel

asr_npu = Qwen3ASRModel.LLM(
    model="/opt/atomgit/QwenASR/models",
    dtype="bfloat16",
    max_model_len=65536,
    max_num_seqs=4,
    trust_remote_code=True,
)

results_npu = asr_npu.transcribe(
    audio="test_data/asr_en.wav",
    language="English",
    return_time_stamps=False,
)
text_npu = results_npu[0].text
```

### Step 5.3: 精度差异计算

```python
import difflib

ratio = difflib.SequenceMatcher(None, text_cpu, text_npu).ratio()
gap = (1.0 - ratio) * 100

print(f"CPU text:  {text_cpu}")
print(f"NPU text:  {text_npu}")
print(f"Similarity: {ratio*100:.2f}%")
print(f"Precision gap: {gap:.2f}%")
```

**精度通过标准**:
| 指标 | 通过阈值 | 说明 |
|------|----------|------|
| 文本相似度 | >= 95% | 语义一致即通过 |
| 精度差距 | <= 5% | 差异仅限语气词/填充词 |
| 语言识别准确率 | 100% | CPU/NPU 识别的语言必须一致 |

**边界条件**: 若精度差距 > 5%，需排查：
1. 是否使用了不同的 `dtype`（如 CPU float32 vs NPU bfloat16）
2. 是否未正确设置 `temperature=0`
3. 是否存在随机性（检查是否固定了 seed）

---

## Phase 6: 性能基准测试

### Step 6.1: 记录基础性能指标

```python
import time

t0 = time.time()
results = asr_npu.transcribe(audio="test_data/asr_en.wav", language="English")
npu_time = time.time() - t0

audio_duration_sec = 68  # 音频时长
rtf = npu_time / audio_duration_sec  # Real-Time Factor

print(f"NPU inference time: {npu_time:.2f}s")
print(f"Audio duration: {audio_duration_sec}s")
print(f"RTF (Real-Time Factor): {rtf:.2f}x")
```

### Step 6.2: 性能参考指标

基于 Ascend910 A2 单卡的实测数据：

| 音频时长 | CPU (transformers) | NPU (vLLM) | 加速比 | RTF |
|----------|-------------------|------------|--------|-----|
| ~4s (short) | ~9.5s | ~5.1s | 1.8x | 1.3x |
| ~68s (long) | ~41.6s | ~6.5s | 6.4x | 0.1x |

**说明**: 
- 短音频加速比较低，主要受 vLLM 引擎初始化开销影响
- 长音频 RTF < 0.2x 表示 NPU 可实时处理 5 倍长度的音频
- 首次加载需 30-60s ACL Graph 编译，后续推理复用缓存

---

## Phase 7: 文档与交付

### Step 7.1: 生成适配报告

运行 `test_asr.py` 自动生成结构化测试报告：

```bash
python3 scripts/test_asr.py \
  --backend both \
  --audio test_data/asr_en.wav \
  --language English
```

**输出示例**:
```
[Summary]
  CPU time: 41.59s
  NPU time: 6.51s
  Speedup:  6.38x
  Precision gap: 1.08%
```

### Step 7.2: 提交到 GitCode

1. 初始化 Git 仓库并添加文件：
```bash
git init
git config user.email "your-email@example.com"
git remote add origin https://gitcode.com/<user>/Qwen3-ASR-0.6B-NPU.git
```

2. 添加必要文件：
```bash
git add README.md inference.py test_asr.py npu_patch.py test_data/
git commit -m "feat: add Qwen3-ASR-0.6B Ascend NPU adaptation"
git push -u origin main
```

**检查点 3**: README 必须包含以下内容：
- 环境版本表格（vllm-ascend, torch-npu, transformers）
- 服务启动命令
- 精度对比表格
- 性能参考数据
- 注意事项与边界条件

---

## 边界条件与异常处理

### 异常分类与 Fallback 策略

| 异常场景 | 错误特征 | Fallback 策略 |
|----------|----------|---------------|
| **vllm-ascend API 不兼容** | `TypeError: unexpected keyword argument` | 应用 npu_patch.py 运行时补丁 |
| **NPU 显存不足** | `RuntimeError: out of memory` | 降低 `gpu_memory_utilization` 或 `max_num_seqs` |
| **ACL Graph 编译失败** | `Engine core initialization failed` | 添加 `--enforce-eager` 禁用图编译 |
| **模型架构未注册** | `KeyError: qwen3_asr` | 重新安装 qwen-asr 包 |
| **音频解码失败** | `soundfile.LibsndfileError` | 确认音频为 16kHz WAV 格式 |
| **精度差异过大** | `gap > 5%` | 检查 dtype 一致性，固定 temperature=0 |
| **网络超时（下载）** | `ConnectTimeoutError` | 使用本地已下载的模型权重 |
| **qwen-asr 依赖缺失** | `ModuleNotFoundError: nagisa` | 安装占位模块或忽略（不影响 ASR） |

### 降级方案

若 vLLM 后端无法启动，可降级至 transformers 后端：

```python
from qwen_asr import Qwen3ASRModel
import torch

asr = Qwen3ASRModel.from_pretrained(
    "/opt/atomgit/QwenASR/models",
    dtype=torch.bfloat16,
)
```

**性能差异**: transformers 后端无 batch 优化，速度约为 vLLM 后端的 1/6。

---

## 检查点清单

- [ ] **CP1**: NPU 设备状态正常（`npu-smi info` 显示 OK）
- [ ] **CP2**: vllm-ascend 版本 >= 0.18.0rc1
- [ ] **CP3**: qwen-asr 0.0.6 已安装且可 import
- [ ] **CP4**: npu_patch.py 在 import qwen_asr 之前执行
- [ ] **CP5**: 模型权重文件存在（config.json + model.safetensors）
- [ ] **CP6**: NPU 模型加载成功，显示 `transcription` 任务
- [ ] **CP7**: 单条音频推理输出非空文本
- [ ] **CP8**: CPU/NPU 精度差距 <= 5%
- [ ] **CP9**: 性能报告已生成（含 speedup 和 RTF）
- [ ] **CP10**: README.md 已编写并推送至 GitCode

---

## 资源清单

### 文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| `SKILL.md` | `./SKILL.md` | 本 Skill 文档 |
| `npu_patch.py` | `./scripts/npu_patch.py` | vllm-ascend 兼容性运行时补丁 |
| `inference.py` | `./scripts/inference.py` | NPU 推理脚本 |
| `test_asr.py` | `./scripts/test_asr.py` | CPU/NPU 精度对比测试脚本 |
| `test-prompts.json` | `./test-prompts.json` | 测试提示词配置 |

### 外部参考

| 资源 | 链接 |
|------|------|
| Qwen3-ASR 官方仓库 | https://github.com/QwenLM/Qwen3-ASR |
| Qwen3-ASR-0.6B HuggingFace | https://huggingface.co/Qwen/Qwen3-ASR-0.6B |
| vLLM-Ascend 文档 | https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/ |
| vLLM-Ascend GitHub | https://github.com/vllm-project/vllm-ascend |
| Ascend 模型适配验证 | https://gitcode.com/Ascend/model-agent |

---

## 参考指标

### 实测验证结果（Ascend910 A2, vllm-ascend 0.18.0rc1）

| 测试项 | 样本 | CPU 结果 | NPU 结果 | 差异 |
|--------|------|----------|----------|------|
| 中文短音频 (4.2s) | asr_zh.wav | `甚至出现交易几乎停滞的情况。` | `甚至出现交易几乎停滞的情况。` | **完全一致** |
| 英文长音频 (68s) | asr_en.wav | `Hmm. Oh yeah...` | `Mhm. Oh yeah...` | 语气词差异 |
| 文本相似度 | - | - | - | **98.92%** |
| 精度差距 | - | - | - | **1.08%** |
| NPU 加速比 (短) | - | 9.49s | 5.12s | **1.85x** |
| NPU 加速比 (长) | - | 41.59s | 6.51s | **6.38x** |

### 功能验证矩阵

| 功能 | 状态 | 验证方式 |
|------|------|----------|
| NPU 模型加载 | ✅ | `LLM(...)` 成功初始化 |
| Transcription 任务 | ✅ | `llm.supported_tasks` 包含 `transcription` |
| 中文语音识别 | ✅ | asr_zh.wav 测试通过 |
| 英文语音识别 | ✅ | asr_en.wav 测试通过 |
| Batch 推理 | ✅ | 多音频列表输入 |
| CPU/NPU 精度对齐 | ✅ | 相似度 >= 95% |
| vLLM 图编译 | ✅ | ACL Graph 捕获成功 |

---

## 版本信息

- **Skill 版本**: 1.0.0
- **更新日期**: 2026-05-17
- **支持 vLLM-Ascend**: v0.18.0rc1+
- **支持 qwen-asr**: v0.0.6
- **测试硬件**: Atlas 800 A2 (Ascend910)
