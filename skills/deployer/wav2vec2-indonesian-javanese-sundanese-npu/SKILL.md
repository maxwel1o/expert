---
name: wav2vec2-indonesian-javanese-sundanese-npu
description: >
  wav2vec2-indonesian-javanese-sundanese 多语种语音识别模型在昇腾 NPU 上的部署与推理 Skill。
  涵盖环境准备、模型下载、transfer_to_npu 自动迁移、NPU 推理验证、
  CPU/NPU 精度对比、性能基准测试的全流程。可在任意 Ascend910 系列服务器上一键复现。
  当用户提到 wav2vec2 昇腾部署、印尼语语音识别 NPU、语音模型 Ascend 时触发。
metadata:
  short-description: wav2vec2 印尼语多语种语音识别昇腾 NPU 部署与推理
  category: NPU-Model-Deploy
  tags: [ascend, npu, wav2vec2, speech-recognition, indonesian, pytorch, inference]
---

# wav2vec2-indonesian-javanese-sundanese 昇腾 NPU 部署与推理 Skill

本 Skill 提供 `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` 多语种语音识别模型
（印尼语/爪哇语/巽他语）在华为昇腾 NPU 上的完整部署、推理验证和 CPU/NPU 精度对比的标准化流程。

## 支持的模型信息

| 属性 | 说明 |
|------|------|
| 模型名称 | `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` |
| 基础架构 | Wav2Vec2ForCTC (基于 `facebook/wav2vec2-large-xlsr-53`) |
| 参数量 | 315,469,470 (~1.2GB) |
| 语言 | 印尼语 (id)、爪哇语 (jv)、巽他语 (su) |
| 解码方式 | CTC + 4gram KenLM Beam Search |
| 适配策略 | 零算子替换，`model.npu()` 即可迁移 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡，32GB HBM） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.9 – 3.13 |
| NPU 内存 | 模型 ~1.2GB HBM，推理余量充裕 |
| 网络 | 首次运行需联网下载模型权重（~1.2GB + 2.2GB LM） |

## 工作流程总览

1. 环境初始化：加载 CANN 环境，选择 NPU 卡，设置镜像源。
2. 安装依赖：安装 torch_npu、transformers、kenlm 等必要组件。
3. NPU 基础验证：确认 NPU 设备和 torch_npu 可用。
4. 模型下载：从 hf-mirror 下载 wav2vec2 权重和 4gram 语言模型。
5. 基础推理验证：使用 inference.py 执行 NPU 推理并验证结果。
6. CPU/NPU 精度对比验证：使用 benchmark.py 运行全自动精度和性能评测。
7. 性能基准测试：测量延迟、RTF 和批处理扩展性。
8. 验收确认：运行完整检查清单确认部署成功。

按以下各节顺序执行，每步完成后再进入下一步。

---

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|---|---|---|---|
| CP-1 环境检查点 | `npu-smi info` 或 `torch_npu` 导入检查完成后 | 用户确认 Ascend910 环境正常、NPU 可用 | 暂停确认，记录为 dry-run，不写入真实 NPU 结论 |
| CP-2 依赖检查点 | pip install torch_npu 完整执行后 | 用户确认 torch_npu、transformers、soundfile、kenlm 均安装正确 | retry 一次安装，失败则输出依赖安装日志并退出 |
| CP-3 模型下载检查点 | 首次下载模型权重前 | 用户确认模型名、权重来源（hf-mirror）和缓存目录 | 切换镜像源或复用本地缓存，失败则跳过推理步骤 |
| CP-4 NPU 推理检查点 | inference.py 执行完成后 | 用户确认推理无报错、RTF < 0.05、解码文本有意义 | 标记验证失败，保留推理日志，跳过精度对比 |
| CP-5 精度检查点 | benchmark.py CPU/NPU 对比完成后 | 用户确认余弦相似度 > 0.9999 且解码文本一致 | 标记验证失败，保留日志，不输出通过结论 |
| CP-6 性能检查点 | 性能基准测试完成后 | 用户确认 RTF 达标且批处理性能符合预期 | 记录异常，降低 batch_size 或分块策略 retry |
| CP-7 验收确认 | 全部流程完成后 | 用户确认 checklist 中所有项目通过 | 回滚至上一状态，追加修复方案 |

---

## 步骤 0：环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=4   # 替换为实际空闲卡号

# 设置环境变量优化 NPU 内存分配
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# HuggingFace 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 步骤 1：安装依赖

#### 1.1 基础依赖

```bash
pip install torch_npu transformers soundfile librosa datasets
```

#### 1.2 安装 kenlm（4gram 语言模型解码器）

kenlm 需从源码编译安装：

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

安装完成后验证：

```bash
python3 -c "import kenlm; print('kenlm OK')"
```

---

## 步骤 2：NPU 基础验证

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU device:', torch.npu.get_device_name(0))
a = torch.randn(3, 4).npu()
print(a + a)
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错。

---

## 步骤 3：模型下载

使用 hf-mirror 镜像下载完整模型权重（含 4gram LM）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    'indonesian-nlp/wav2vec2-indonesian-javanese-sundanese',
    local_dir='./wav2vec2-indonesian-javanese-sundanese'
)
print(f'Downloaded to: {path}')
"
```

或直接使用 transformers 下载并缓存：

```bash
python3 -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
model = Wav2Vec2ForCTC.from_pretrained('indonesian-nlp/wav2vec2-indonesian-javanese-sundanese')
processor = Wav2Vec2ProcessorWithLM.from_pretrained('indonesian-nlp/wav2vec2-indonesian-javanese-sundanese')
print('Download OK')
"
```

---

## 步骤 4：基础推理验证

在模型目录下创建 `inference.py`：

核心里面逻辑：

```python
import torch_npu
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

MODEL_PATH = "./wav2vec2-indonesian-javanese-sundanese"

processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).npu().eval()

# 加载音频 (16kHz single-channel)
audio, sr = sf.read("audio.wav")
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# NPU 推理
inputs = processor(audio.astype(np.float32), sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values.npu()).logits.cpu().numpy()

# CTC 解码（含 4gram 语言模型）
text = processor.decode(logits[0]).text
print(f"识别结果: {text}")
```

运行推理脚本：

```bash
python3 scripts/inference.py --audio test_audio.wav --backend npu --verbose
```

**通过标准**：
- 模型加载成功（315,469,470 参数）
- NPU 推理无报错
- RTF < 0.05（即推理速度 > 20x 实时）

---

## 步骤 5：CPU/NPU 精度对比验证

运行评测脚本，自动对比 CPU 与 NPU 推理精度：

```bash
cd eval && python3 scripts/benchmark.py
```

评测脚本将运行以下测试：
1. 环境检查
2. 模型加载测试
3. 多时长精度测试（1s / 3s / 5s / 10s 合成音频）
4. 性能基准测试（20 iterations）
5. 批处理性能测试（batch_size=1/2/4/8）
6. 解码器集成测试
7. CPU vs NPU 完整对比

#### 5.1 精度结果

在 Ascend 910B4 (CANN 8.5.1) 上实测：

| 音频时长 | CPU耗时 | NPU耗时 | 加速比 | Cosine Similarity | 结论 |
| --- | --- | --- | --- | --- | --- |
| 1.0s | 2,581 ms | 29.1 ms | 88.7x | 0.99999509 | PASS |
| 3.0s | 6,426 ms | 31.0 ms | 207.5x | 0.99999791 | PASS |
| 5.0s | 10,448 ms | 31.7 ms | 329.7x | 0.99999896 | PASS |
| 10.0s | 21,944 ms | 33.2 ms | 660.6x | 0.99999870 | PASS |

**CPU vs NPU 完整对比（5s 音频）：**

| 指标 | 数值 |
| --- | --- |
| CPU 总耗时 | 10,553 ms |
| NPU 总耗时 | 35.8 ms |
| NPU 加速比 | 294.9x |
| Cosine 相似度 | 0.99999860 |
| 解码文本一致性 | True |

**通过标准**：余弦相似度 > 0.9999，NPU/CPU 解码文本完全一致。

---

## 步骤 6：性能基准测试

测试条件：合成音频 / 20 iterations / batch_size=1

| 音频时长 | 均值延迟 | P95延迟 | P99延迟 | RTF |
| --- | --- | --- | --- | --- |
| 1.0s | 39.1 ms | 40.6 ms | 43.3 ms | 0.0391 |
| 5.0s | 40.0 ms | 42.0 ms | 49.1 ms | 0.0080 |
| 10.0s | 40.5 ms | 41.5 ms | 42.2 ms | 0.0041 |
| 30.0s | 60.8 ms | 62.9 ms | 123.9 ms | 0.0020 |

**批处理性能：**

| Batch Size | 均值延迟 | 单样本延迟 |
| --- | --- | --- |
| 1 | 38.6 ms | 38.6 ms |
| 2 | 37.7 ms | 18.8 ms |
| 4 | 40.0 ms | 10.0 ms |
| 8 | 45.8 ms | 5.7 ms |

**通过标准**：RTF < 0.05，批处理单样本延迟随 batch_size 递减。

---

## 步骤 7：验收确认

完成以下检查清单即为部署成功：

```bash
# 环境确认
npu-smi info                          # 确认 NPU 设备正常
python3 -c "import torch_npu; print(torch_npu.__version__)"  # 确认 torch_npu 可用
python3 -c "import kenlm; print('kenlm OK')"                 # 确认 kenlm 可用

# 下载确认
ls -lh ./wav2vec2-indonesian-javanese-sundanese/             # 确认权重下载完整

# 推理验证
python3 scripts/inference.py --audio test_audio.wav --backend npu --verbose

# 完整评测
cd eval && python3 scripts/benchmark.py
```

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] `import kenlm` 无报错
- [ ] 模型权重下载完成（含 4gram.bin LM）
- [ ] `inference.py` 可对真实音频文件进行 NPU 推理
- [ ] `benchmark.py` 全部测试 PASS
- [ ] 精度对比 Cosine Similarity > 0.9999
- [ ] RTF < 0.05
- [ ] 批处理单样本延迟随 batch_size 递减

---

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|---|---|---|---|
| CANN 环境未加载 | `source set_env.sh` 未执行，torch_npu 导入失败 | 提示加载 CANN 环境脚本，retry 一次环境检查 | 环境检查日志记录 CANN 路径状态 |
| NPU 设备不可用 | `npu-smi info` 失败或无 Ascend910 | fallback 到 CPU dry-run，禁止写入真实 NPU 结论 | README 标记 `dry_run`，`evals.json` 记录 `npu_unavailable` |
| 模型权重下载失败 | hf-mirror 网络超时或模型不存在 | retry 2 次，随后切换 `HF_ENDPOINT=https://huggingface.co` 或本地缓存 | `results.tsv` 记录权重来源和缓存路径 |
| kenlm 编译安装失败 | pip install kenlm 报错或系统缺少编译器 | 回退不使用 LM：纯贪心解码（argmax），记录 fallback | `evals.json` 记录 `kenlm_fallback` |
| 输入音频格式错误 | 采样率不是 16kHz 或多通道立体声 | 自动重采样至 16kHz 单声道，retry 推理 | 推理日志含 "Audio resampled" 信息 |
| 音频文件不存在 | `--audio` 指定的路径不存在 | 退出并打印路径错误，要求用户确认正确路径 | 错误信息回显至终端 |
| 长音频 OOM | 音频过长导致 NPU 显存溢出 | 启用 `--chunk-length-s 10.0` 分块推理，设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | `evals.json` 记录 chunk 数量和 retry 次数 |
| 多卡抢占冲突 | 默认卡被其他进程占用 | 自动扫描 `npu-smi info` 输出选择空闲卡，设置 `ASCEND_RT_VISIBLE_DEVICES` | 推理日志记录最终使用的卡号 |
| 精度不达标 | Cosine Similarity < 0.9999 | 保留 CPU/NPU logits 供人工分析，标记 failed，不生成通过结论 | 精度表显示失败原因和差异详情 |
| 解码文本为空 | 音频内容静音或非目标语言 | 提示更换测试音频，retry 最多 2 次 | `evals.json` 记录 `empty_decode` |

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `No module named 'torch_npu'` | 未安装或 CANN 环境未加载 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后重装 torch_npu |
| `NameError: name 'kenlm' is not defined` | kenlm 未安装 | `pip install https://github.com/kpu/kenlm/archive/master.zip` |
| HuggingFace 下载超时 | 网络不通 | 设置 `export HF_ENDPOINT=https://hf-mirror.com` |
| `Wav2Vec2ProcessorWithLM` 加载失败 | 缺少语言模型文件 | 确认下载目录包含 `language_model/4gram.bin` |
| 解码文本为空 | 输入音频采样率不正确 | 模型要求 16kHz 单声道，需预处理 `librosa.resample` |
| OOM | 音频过长 | 使用 `--chunk-length-s 10.0` 分块推理 |
| 多卡抢占冲突 | 默认都用同一卡 | `npu-smi info` 选空闲卡，设置 `ASCEND_RT_VISIBLE_DEVICES` |

## 文件结构与资源

| 路径 | 用途 |
|---|---|
| `SKILL.md` | 本 Skill 文档，包含完整部署流程、检查点、异常处理和测试提示 |
| `test-prompts.json` | 提供重复评估本 Skill 的测试提示 |
| `scripts/inference.py` | NPU/CPU 推理脚本，支持分块推理和 CTC+LM 解码 |
| `scripts/benchmark.py` | 全自动评测脚本：环境检查、模型加载、精度对比、性能基准、批处理测试、解码器集成 |
| `eval/results.json` | 结构化保存环境、精度、性能和解码结果 |
| `eval/benchmark.log` | 详细的评估运行日志（时间戳 + 步骤） |
| `references/` | 模型来源和目标仓库参考信息 |

## 附录：wav2vec2 NPU 适配要点速查

| 特征 | wav2vec2 值 | 对适配的影响 |
|------|-------------|-------------|
| 模型架构 | Wav2Vec2ForCTC | transformers 直接支持，无需自定义代码 |
| CNN 特征提取 | 7 层 Conv1d | torch_npu 原生支持 |
| Transformer | 24 层 / 16 heads / 1024 hidden | 标准 attention + FFN，无需额外替换 |
| 激活函数 | GELU (内置) | torch.nn.GELU 已在 NPU 上优化 |
| LayerNorm | do_stable_layer_norm=True | torch.nn.LayerNorm 原生支持 |
| 语言模型 | 4gram KenLM | CPU 端解码，不占 NPU 资源 |
| 输入格式 | 16kHz 单声道 raw waveform | 需预处理重采样 |
| 输出格式 | CTC logits → Beam Search | 解码与推理解耦，不影响 NPU 精度 |
