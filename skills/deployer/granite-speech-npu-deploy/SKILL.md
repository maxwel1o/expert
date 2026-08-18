---
name: granite-speech-npu-deploy
description: >
  Granite Speech 4.1 (2B/2B-Plus) 语音理解模型在昇腾 NPU 上的完整部署 Skill。
  涵盖环境准备、模型加载、Smoke 测试、性能基准和精度对比的全流程。
  支持 base 版本（granite_speech，transformers 原生支持）和 plus 版本
  （granite_speech_plus，含 cat_hidden_layers 中间层拼接增强）。
  当用户提到 Granite Speech 昇腾 NPU 部署、语音模型 NPU 推理时触发。
metadata:
  short-description: Granite Speech 4.1 昇腾 NPU 部署与推理
  category: NPU-Model-Deploy
  tags: [ascend, npu, granite, speech, asr, conformer, qformer, ibm, pytorch, inference]
  core-scripts: inference.py,benchmark.py,test_npu.py,test_accuracy.py
  skill-type: model-deploy
  input-params:
    model-path: 模型权重路径（默认 ./granite-speech-4.1-2b）
    audio-path: 输入音频路径（16kHz 单声道 WAV）
    prompt-text: 推理提示文本（含 <|audio|> 标记）
    max-new-tokens: 生成最大 token 数（默认 128）
    benchmark-runs: 基准测试重复次数（默认 50）
  output-artifacts:
    - output: NPU 推理结果文本
    - benchmark_results.json: 性能基准数据
    - accuracy_report: 精度对比日志
  allowed-tools: Bash(*) Python3(*)
---

# Granite Speech 4.1 昇腾 NPU 部署 Skill

本 Skill 提供 Granite Speech 4.1（2B / 2B-Plus）语音理解模型在华为昇腾 NPU 上的完整部署流程，包括环境配置、模型加载、推理验证、性能基准测试和精度对比。

## 模型简介

Granite Speech 是 IBM 开源的语音理解模型系列，基于 Conformer 音频编码器 + QFormer 投影层 + Granite-4.0-1B-base 语言模型架构。支持语音识别（ASR）和语音理解任务。

| 变体 | 参数量 | 特点 |
|------|--------|------|
| granite-speech-4.1-2b (base) | 2,313M | transformers 4.57.6 原生支持 |
| granite-speech-4.1-2b-plus (plus) | 2,112M | cat_hidden_layers=[3] 拼接增强，需注册模型类型 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 910B4 或 910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.5.1 |
| Python | 3.10 – 3.11 |
| 依赖 | transformers == 4.57.6, torch >= 2.9.0, torch_npu >= 2.9.0 |
| 磁盘 | 模型权重约 4.5GB（每个变体） |

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model-path | string | 是 | 模型权重路径（如 ./granite-speech-4.1-2b） |
| audio-path | string | 否 | 输入音频路径（16kHz 单声道 WAV） |
| prompt-text | string | 否 | 推理提示文本，需含 `<|audio|>` 标记 |
| max-new-tokens | int | 否 | 生成最大 token 数（默认 128） |
| benchmark-runs | int | 否 | 基准测试重复次数（默认 50） |

## 执行总览

1. 确认 NPU 硬件就绪，加载 CANN 环境，选择空闲 NPU 卡，设置 NPU 内存分配参数。
2. 从 ModelScope 下载 granite-speech-4.1-2b 和 granite-speech-4.1-2b-plus 模型权重到本地。
3. 安装 PyTorch、torch_npu、transformers 4.57.6 及基础依赖，通过简单张量运算验证 NPU 可用性。
4. 使用 torch.npu 的 API 对 NPU 设备执行信息查询，确认卡数、名称和显存容量。
5. 对 base 变体使用 transformers 原生 GraniteSpeechForConditionalGeneration 加载模型并执行 forward pass smoke test。
6. 对 plus 变体执行 register() 注册后加载 GraniteSpeechPlusForConditionalGeneration，验证 cat_hidden_layers 拼接逻辑。
7. 使用 inference.py 加载实际音频文件，执行 generate 端到端推理并记录转录结果。
8. 使用 benchmark.py 测量不同序列长度下的 forward pass 耗时与吞吐量，保存 benchmark_results.json。
9. 使用 test_accuracy.py 对比 NPU bfloat16 与 CPU float32 推理结果，确认余弦相似度 >0.999。
10. 完成验收检查清单，确认所有步骤通过后部署流程结束。

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.9.0 |
| torch_npu | >= 2.9.0 |
| transformers | 4.57.6 |
| CANN | >= 8.5.1 |
| NPU | Ascend910B4 / 910 (32GB) |
| soundfile | >= 0.12 |
| librosa | >= 0.10 |

## 工作流程

### 1. 环境准备

1. 加载 CANN 环境，配置 NPU 相关的系统环境变量。
2. 使用 npu-smi info 查看 NPU 设备状态，确认设备健康。
3. 选择空闲 NPU 卡，设置 ASCEND_RT_VISIBLE_DEVICES。
4. 配置 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True 和 TASK_QUEUE_ENABLE=1。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
```

### 2. 下载模型权重

从 ModelScope（推荐，国内加速）或 HuggingFace 下载。

```bash
pip install modelscope
python3 -c "
from modelscope import snapshot_download
snapshot_download('ibm-granite/granite-speech-4.1-2b', local_dir='./granite-speech-4.1-2b')
snapshot_download('ibm-granite/granite-speech-4.1-2b-plus', local_dir='./granite-speech-4.1-2b-plus')
"
```

如 ModelScope 失败，可切换到 HuggingFace 或 GitCode 镜像。

```bash
# HuggingFace 方式
# huggingface-cli download ibm-granite/granite-speech-4.1-2b --local-dir ./granite-speech-4.1-2b
```

### 3. 环境配置与依赖安装

1. 安装 PyTorch 与 torch_npu（使用华为云 PyPI 镜像加速）。
2. 安装 transformers 4.57.6 以及 soundfile、librosa 等音频处理库。
3. 运行验证脚本确认 NPU 张量计算正常。

```bash
pip install torch torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
pip install transformers==4.57.6 soundfile librosa -i https://repo.huaweicloud.com/repository/pypi/simple/
python3 -c "
import torch
import torch_npu
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
"
```

**通过标准**: 输出包含 `device='npu:0'` 的 Tensor 且无报错。

### 4. NPU 基础验证

确认 NPU 环境可用的快速验证。

```bash
python3 -c "
import torch
import torch_npu
assert torch.npu.is_available()
print(f'NPU count: {torch.npu.device_count()}')
print(f'NPU name: {torch.npu.get_device_name(0)}')
print(f'NPU memory: {torch.npu.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### 5. 模型加载验证（Smoke Test）

#### 5.1 Base 版本

Base 版本使用 transformers 原生 `GraniteSpeechForConditionalGeneration`。

```python
import torch
import torch_npu
from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
from transformers import AutoProcessor

model_path = "./granite-speech-4.1-2b"
model = GraniteSpeechForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
).npu()
model.eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
```

#### 5.2 Plus 版本

Plus 版本需要先注册 `granite_speech_plus` 模型类型。

```python
import torch
import torch_npu
from transformers import AutoProcessor
from granite_speech_plus.register import register
register()
from granite_speech_plus import GraniteSpeechPlusForConditionalGeneration

model_path = "./granite-speech-4.1-2b-plus"
model = GraniteSpeechPlusForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
).npu()
model.eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
```

#### 5.3 Plus 适配包说明

`granite_speech_plus` 运行时适配包提供以下组件。

| 组件 | 说明 |
|------|------|
| `GraniteSpeechPlusEncoder` | 支持 `cat_hidden_layers=[3]`，拼接第 3 层中间层隐藏态 |
| `GraniteSpeechPlusConfig` | 使用 `GraniteSpeechPlusEncoderConfig` 的配置类 |
| `GraniteSpeechPlusForConditionalGeneration` | 完整模型类，继承基类推理逻辑 |
| `register.py` | 运行时注册到 transformers auto 系统 |

适配包位于 `scripts/granite_speech_plus/` 目录，使用前需将其复制到 Python 可搜索路径或添加到 `sys.path`。

#### 5.4 完整 Smoke Test 脚本

运行 Smoke Test。

```bash
# 测试 base 版本
python3 scripts/test_npu.py --model-path ./granite-speech-4.1-2b --mode smoke

# 测试 plus 版本
python3 scripts/test_npu.py --model-path ./granite-speech-4.1-2b-plus --mode smoke
```

### 6. 推理命令

使用 `inference.py` 进行端到端推理。

```bash
# Base 版本
python3 scripts/inference.py \
    --model-path ./granite-speech-4.1-2b \
    --audio-path sample.wav \
    --text "<|audio|> Transcribe the audio." \
    --max-new-tokens 128

# Plus 版本
python3 scripts/inference.py \
    --model-path ./granite-speech-4.1-2b-plus \
    --audio-path sample.wav \
    --text "<|audio|> Transcribe the audio." \
    --max-new-tokens 128
```

#### 输入要求
- 音频：16kHz 单声道 WAV 格式
- 文本：需包含 `<|audio|>` 标记指定音频嵌入插入位置
- 推荐使用 `bfloat16` 精度以平衡性能与精度

### 7. 性能基准测试

使用 `benchmark.py` 进行 forward pass 和生成性能评估。

```bash
# 同时测试两个模型
python3 scripts/benchmark.py
```

#### 测试条件
- Ascend 910B4 x 1 逻辑卡
- bfloat16 精度
- 50 轮平均

#### 预期性能参考

**Base 版本 (granite-speech-4.1-2b)**

| 序列长度 | 平均耗时 (ms) | Throughput (tok/s) |
|----------|---------------|-------------------|
| 32 | 134.01 | 238.78 |
| 64 | 139.03 | 460.32 |
| 128 | 136.33 | 938.89 |

显存占用：~4.4 GB (batch_size=1)

**Plus 版本 (granite-speech-4.1-2b-plus)**

| 序列长度 | 平均耗时 (ms) | Throughput (tok/s) |
|----------|---------------|-------------------|
| 32 | 149.24 | 214.42 |
| 64 | 134.99 | 474.11 |
| 128 | 139.29 | 918.97 |

显存占用：~4.0 GB (batch_size=1)

### 8. 精度对比验证

使用 `test_accuracy.py` 对比 NPU bfloat16 与 CPU float32 推理精度。

```bash
# 测试 base 版本
python3 scripts/test_accuracy.py --model-path ./granite-speech-4.1-2b

# 测试 plus 版本
python3 scripts/test_accuracy.py --model-path ./granite-speech-4.1-2b-plus
```

#### 预期精度

| 指标 | Base | Plus |
|------|------|------|
| 余弦相似度 | > 0.9999 | > 0.9999 |
| 平均绝对误差 | 0.030 | 0.025 |
| 最大绝对误差 | 0.401 | 0.246 |

**通过标准**: 余弦相似度 > 0.999，差异仅来自 bfloat16 精度差异。

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|--------|---------|-------------|--------------|
| CP-1 环境检查点 | 执行 npu-smi info 完成后 | 用户确认 NPU 设备健康、CANN 版本 >= 8.5.1 | 暂停，提示加载 CANN 或更换设备，不满足则停止 |
| CP-2 下载检查点 | 调用 snapshot_download 前 | 用户确认模型名、权重来源和缓存目录 | 切换镜像源至 HuggingFace 或 GitCode，retry 最多 2 次，失败则跳过 |
| CP-3 安装检查点 | pip install 完成后 | 用户确认 PyTorch/torch_npu 安装正确，验证脚本通过 | 回滚到上一步，检查 Python 版本或 PyPI 镜像配置后 retry |
| CP-4 环境验证检查点 | NPU 基础验证完成后 | 用户确认 torch.npu.is_available() 为 True，显存充足 | 标记环境不满足，记录到 evals.json，退出当前流程 |
| CP-5 烟雾测试检查点 | Smoke test 完成后 | 用户确认两个变体加载成功且 forward pass 正常 | 检查模型路径和 config.json，确认 dtype 兼容性后 retry |
| CP-6 推理验证检查点 | inference.py 执行完成后 | 用户确认 generate 输出合理，音频转录/理解结果正确 | 检查音频格式，确认 sampling_rate=16000，调整 prompt 后 retry |
| CP-7 基准测试检查点 | benchmark.py 完成后 | 用户确认性能数据与参考值在合理范围 | 记录异常数据，检查 NPU 负载或其他进程占用，reclaim 后 retry |
| CP-8 精度验证检查点 | test_accuracy.py 完成后 | 用户确认余弦相似度 >0.999 且误差在预期范围 | 标记精度验证失败，保留日志，禁止写入通过结论 |
| CP-9 最终验收 | 所有步骤完成后 | 用户验收所有结果并确认部署完成 | 记录验收状态到 evals.json，补充缺失步骤 |

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|------|---------|-------------------------------|---------|
| CANN 环境未加载 | ASCEND_HOME 未设置或 npu-smi 失败 | 提示用户 source set_env.sh，retry 1 次环境检查 | 日志记录 env_check_failed |
| NPU 设备不可用 | torch.npu.is_available() 返回 False | fallback 到 CPU dry-run 模式，禁止写入 NPU 通过结论 | evals.json 记录 npu_unavailable，输出标记 dry_run |
| 依赖安装失败 | pip install torch/torch_npu 网络超时或版本冲突 | retry 3 次，切换 PyPI 镜像为华为云或清华源 | 安装日志和 requirements 版本快照 |
| 权重下载超时 | ModelScope 下载中途断开或鉴权失败 | retry 2 次，随后切换到 HuggingFace 或 GitCode 镜像源 | 权重来源记录在 download.log |
| LFS 指针文件错误 | 权重为 LFS 指针而非真实文件（SafetensorError header too large） | 重新使用 ModelScope 下载完整权重，不使用 git lfs pull | 校验文件大小，确认 >100MB 为真实权重 |
| Plus 注册失败 | KeyError: 'granite_speech_plus' 或 register() 导入错误 | 确认 sys.path 包含 scripts/ 目录，检查 register.py 语法 | 错误日志记录 reg_failure |
| 模型 OOM | 加载模型时显存不足（CUDA out of memory / NPU OOM） | 设置 expandable_segments:True，释放其他进程占用，reclaim 后 retry | 显存使用记录，OOM 前后对比 |
| 音频格式错误 | soundfile 读取失败或采样率不是 16kHz | 自动尝试 librosa 重新采样，retry 1 次 | 日志记录 audio_format_fallback |
| 推理结果为空 | generate 返回空文本或特殊 token | 检查 prompt 格式，确认 <|audio|> 位置正确，retry 调整指令 | 推理日志含 input_ids 和 attention_mask |
| 精度不达标 | 余弦相似度 < 0.999 或 max_rel_error >= 1% | 保留 CPU/NPU logits，标记 failed，不输出通过结论 | 精度表显示失败原因，evals.json 记录 accuracy_fail |
| 基准测试异常 | 性能数据偏离参考值 50% 以上 | 检查 NPU 负载和卡温度，确认无其他进程抢占资源，reclaim 后 retry | benchmark_results.json 记录异常标记 |
| 显存泄漏 | 连续运行后 alloc 持续增长不下降 | 执行 torch.npu.empty_cache() 和 gc.collect()，重启 Python 进程 | 显存增长曲线记录 |
| 脚本依赖缺失 | ModuleNotFoundError 如 soundfile、librosa | 自动 pip install 缺失依赖，retry 最多 2 次 | install.log 记录补充安装 |
| 模型路径错误 | OSError: Unable to load weights from path | 检查路径是否存在和目录结构，验证 config.json 文件完整性 | 路径检查日志，建议用户重新下载 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | 端到端推理入口，支持 base 和 plus 变体加载、音频预处理与 generate 生成 |
| `scripts/benchmark.py` | 性能基准测试，测量 forward pass 耗时、吞吐量、生成速度和显存占用 |
| `scripts/test_npu.py` | NPU 烟雾测试和精度验证，支持 --mode smoke 和 --mode accuracy |
| `scripts/test_accuracy.py` | CPU float32 vs NPU bfloat16 精度对比，输出误差和余弦相似度 |
| `scripts/granite_speech_plus/` | Plus 变体适配包，含自定义 Encoder、Config、Model 类和 register 注册脚本 |
| `scripts/granite_speech_plus/register.py` | 运行时将 GraniteSpeechPlus 注册到 transformers auto 系统 |
| `scripts/granite_speech_plus/modeling_granite_speech_plus.py` | 自定义建模逻辑，支持 cat_hidden_layers=[3] 拼接增强 |
| `scripts/granite_speech_plus/configuration_granite_speech_plus.py` | 自定义配置类，继承基础配置并添加 encoder 参数 |
| `references/benchmark_results.json` | 实录性能基准数据，含 npu-smi 信息和两个变体的 forward/generate/memory 指标 |
| `references/readme-base.md` | Base 变体的参考 README 模板和部署说明 |
| `references/readme-plus.md` | Plus 变体的参考 README 模板和 cat_hidden_layers 说明 |
| `test-prompts.json` | 提供重复评估本 Skill 的测试提示 |
| `results.tsv` | 各变体的推理耗时、精度指标和状态汇总（生成目标） |
| `evals.json` | 结构化保存环境检查、重试记录、验证结果和验收状态（生成目标） |

## 文件结构

```
granite-speech-npu/
├── SKILL.md                       # 本 Skill 文档
├── test-prompts.json              # 结构化测试提示
├── scripts/
│   ├── inference.py               # 端到端推理脚本
│   ├── benchmark.py               # 性能基准测试
│   ├── test_npu.py                # NPU 烟雾测试与精度验证
│   ├── test_accuracy.py           # CPU vs NPU 精度对比
│   └── granite_speech_plus/       # Plus 变体适配包
│       ├── __init__.py
│       ├── register.py
│       ├── modeling_granite_speech_plus.py
│       └── configuration_granite_speech_plus.py
└── references/
    ├── benchmark_results.json     # 实录基准数据
    ├── readme-base.md             # Base 变体 README 模板
    └── readme-plus.md             # Plus 变体 README 模板
```

## 验收确认

完成以下检查清单即为部署成功。

- [ ] `npu-smi info` 显示 NPU 设备正常，Health 状态 OK
- [ ] `import torch_npu` 无报错，`torch.npu.is_available()` 返回 True
- [ ] 模型加载至 NPU 成功，参数数量输出正常
- [ ] Forward 前向传播正常，无算子兼容性报错
- [ ] Generate 文本生成正常，输出合理转录/理解结果
- [ ] 性能基准测试完成，结果与参考值在合理范围
- [ ] 精度对比余弦相似度 > 0.999
- [ ] Plus 版本 cat_hidden_layers 验证通过

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `AutoModel` 无法识别模型类型 | 用了 `AutoModel` 而非具体 Model 类 | 使用 `GraniteSpeechForConditionalGeneration.from_pretrained` 替代 |
| `ImportError: cannot import name 'MODEL_NAMES_MAPPING'` | 导入路径错误 | 使用 `from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES` |
| `safetensors_rust.SafetensorError: header too large` | LFS 指针文件代替了真实权重 | 使用 ModelScope 下载完整权重 |
| 模型加载到 NPU 后 OOM | 显存不足或未设置 expandable_segments | 设置环境变量 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| `KeyError: 'granite_speech_plus'` | plus 版本未注册模型类型 | 先调用 `register()` 再导入 Model 类 |
| 设备 ID 错误 | torch_npu 中只存在 device 0 | 使用 `model.npu()` 而非 `model.to('npu:7')` |
| 精度对比误差过大 | 随机种子不一致或输入生成方式不同 | 在 CPU 上生成输入后再 `.to(device)` |

## 附录

### 模型架构

```
Input Audio (16kHz mono WAV)
    ↓
Conformer Encoder (cat_hidden_layers=[3] for plus)
    ↓
QFormer Projection Layer
    ↓
Granite-4.0-1B-base Language Model
    ↓
Text Output (transcription/understanding)
```

### 权重获取地址

- ModelScope: https://modelscope.cn/models/ibm-granite/granite-speech-4.1-2b
- HuggingFace: https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- GitCode 镜像: https://gitcode.com/hf_mirrors/ibm-granite/granite-speech-4.1-2b
