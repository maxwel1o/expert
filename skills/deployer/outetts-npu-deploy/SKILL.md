---
name: outetts-npu-deploy
description: >
  OuteTTS 系列文本转语音模型在昇腾 NPU 上的部署与推理 Skill。
  支持 OuteTTS-0.1-350M (LLaMA)、OuteTTS-0.2-500M (Qwen2)、
  OuteTTS-0.3-500M (Qwen2) 三个模型版本，
  涵盖环境准备、模型下载、NPU/PyTorch 直接推理、
  精度评测验证（NPU vs CPU 波形一致性）的全流程。
  适用于 Ascend 910B NPU 单卡推理场景。
  当用户提到 OuteTTS 部署昇腾、OuteTTS NPU 推理、TTS 模型昇腾适配时触发。
metadata:
  short-description: OuteTTS 语音合成模型昇腾 NPU 部署与推理
  category: NPU-Model-Deploy
  tags: [ascend, npu, outetts, tts, text-to-speech, torch-npu, llama, qwen2, inference]
---

# OuteTTS 系列昇腾 NPU 部署与推理 Skill

本 Skill 提供 OuteTTS 系列模型（OuteTTS-0.1-350M、OuteTTS-0.2-500M、OuteTTS-0.3-500M）
在华为昇腾 NPU 上的完整适配、推理验证和精度评测的标准化可复现流程。

所有模型通过 `outetts` 库 + `torch_npu` 直接推理（无需 vLLM），利用 PyTorch 的 device-agnostic
特性将模型和 WavTokenizer 音频编解码器迁移至 NPU。

## 支持的模型

| 模型 | 架构 | 参数量 | HF 原始模型 | GitCode NPU 仓库 |
|------|------|--------|------------|------------------|
| OuteTTS-0.1-350M-npu | LlamaForCausalLM | 350M | [OuteAI/OuteTTS-0.1-350M](https://huggingface.co/OuteAI/OuteTTS-0.1-350M) | [gcw_C8PI9e90/OuteTTS-0.1-350M-npu](https://gitcode.com/gcw_C8PI9e90/OuteTTS-0.1-350M-npu) |
| OuteTTS-0.2-500M-npu | Qwen2ForCausalLM | 500M | [OuteAI/OuteTTS-0.2-500M](https://huggingface.co/OuteAI/OuteTTS-0.2-500M) | [gcw_C8PI9e90/OuteTTS-0.2-500M-npu](https://gitcode.com/gcw_C8PI9e90/OuteTTS-0.2-500M-npu) |
| OuteTTS-0.3-500M-npu | Qwen2ForCausalLM | 500M | [OuteAI/OuteTTS-0.3-500M](https://huggingface.co/OuteAI/OuteTTS-0.3-500M) | [gcw_C8PI9e90/OuteTTS-0.3-500M-npu](https://gitcode.com/gcw_C8PI9e90/OuteTTS-0.3-500M-npu) |

**版本差异：**

| 特性 | 0.1-350M | 0.2-500M | 0.3-500M |
|------|----------|----------|----------|
| 基底架构 | LLaMA | Qwen2 | Qwen2 |
| 接口版本 | V1 | V2 | V2 |
| 隐藏层 | 20 | 24 | 24 |
| 注意力头 | 16 | 14 | 14 |
| 词表大小 | 67,584 | 157,696 | 157,696 |
| 推荐优先级 | 中 | 低 | **高** |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 910B (>= 4GB HBM，单卡) |
| OS | openEuler / Ubuntu (aarch64) |
| CANN | 8.5.1 |
| Python | 3.11.x |
| torch | >= 2.5.0 |
| torch_npu | >= 2.5.0 |
| transformers | >= 4.46.0 |
| outetts | == 0.4.4 |
| 网络 | 首次需下载 WavTokenizer 模型权重 (~500MB) |

## 流程总览

```mermaid
graph LR
    A[步骤 1: 安装依赖] -->|check: import success| B[步骤 2: 下载模型权重]
    B -->|check: config.json + weights| C[步骤 3: NPU 推理]
    C -->|check: output.wav generated| D[步骤 4: 精度评测]
    D -->|check: MSE < 1e-5, SNR > 40 dB| E[步骤 5: 清理]
    E --> F[部署完成]
```

按以下顺序执行，每步完成后确认检查点，通过后进入下一步。任一阶段失败则停止并执行对应异常处理流程。

## 工作流阶段汇总

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 步骤 1: 安装依赖 | 安装 outetts/torch/torch_npu 等依赖 | Python 3.11+、CANN 8.5.1+ | `pip install outetts==0.4.4 torch transformers torch_npu soundfile numpy` | 依赖安装完成 | `python -c "import outetts; import torch; import torch_npu; print(torch.npu.is_available())"` | `torch.npu.is_available()=True`，导入无报错 |
| 步骤 2: 下载模型权重 | 从 HF 下载 OuteTTS + WavTokenizer 权重 | 网络连接 | `huggingface-cli download OuteAI/OuteTTS-0.3-500M --local-dir ...` | 本地模型目录 + WavTokenizer cache | `ls .../config.json` | 模型文件完整可加载 |
| 步骤 3: NPU 推理 | 加载模型并生成语音 WAV 文件 | 本地模型路径 + 输入文本 | `python inference.py --text "..." --output output.wav --model_path ...` | `output.wav` 音频文件 | `ls output.wav` | WAV 文件存在且时长 > 0 |
| 步骤 4: 精度评测 | NPU vs CPU 波形一致性对比 | 本地模型路径 + 测试文本 | `python eval_accuracy.py --model_path ... --output_json eval_results.json` | `eval_results.json` | `cat eval_results.json` | MSE < 1e-5，余弦相似度 > 0.999，SNR > 40 dB |
| 步骤 5: 清理 | 释放 NPU 显存 | 已运行的 NPU 进程 | 退出 Python 进程；`npu-smi info` | 显存释放确认 | `npu-smi info` | 无残留 NPU 进程占用 |

## 步骤 1：安装依赖

**执行步骤**

1. 确认 NPU 环境就绪：`npu-smi info` 和 `python -c "import torch_npu; print(torch.npu.is_available())"`
2. 安装核心依赖：`pip install outetts==0.4.4 torch transformers torch_npu soundfile numpy`（使用国内镜像加速）
3. 验证所有依赖可导入：`python -c "import outetts, torch, torch_npu, soundfile; print('OK')"`
4. 确认 `torch.npu.is_available()` 返回 True
5. 若 pip 安装失败，检查网络连接或更换镜像源

| 项目 | 内容 |
|------|------|
| **输入** | Python 3.11+、CANN 8.5.1+、NPU 驱动就绪 |
| **输出** | Python 依赖安装完成，`torch.npu.is_available()=True` |
| **失败终止条件** | pip 安装失败或 outetts 版本与 torch_npu 不兼容 |
| **前置依赖** | CANN 已安装、NPU 硬件在线 |

```bash
# 核心依赖
pip install outetts==0.4.4 torch transformers torch_npu soundfile numpy \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 可选：音频播放
pip install sounddevice pygame \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 步骤 2：下载模型权重

**执行步骤**

1. 设置 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
2. 下载主模型权重：`huggingface-cli download OuteAI/OuteTTS-0.3-500M --local-dir /path/to/OuteTTS-0.3-500M`（推荐 0.3 版本）
3. 下载 WavTokenizer 编解码器（自动下载或手动执行 `huggingface-cli download OuteAI/wavtokenizer-large-75token-interface`）
4. 确认模型目录下存在 `config.json` 和 safetensors 权重文件
5. 确认磁盘空间充足（模型约需 1–2GB，WavTokenizer 约 500MB）

| 项目 | 内容 |
|------|------|
| **输入** | 模型 ID（如 `OuteAI/OuteTTS-0.3-500M`） |
| **输出** | 本地模型目录（含 config.json + safetensors 权重）+ WavTokenizer cache |
| **失败终止条件** | 网络超时、磁盘空间不足或模型 404 |
| **前置依赖** | 步骤 1 通过 |

```bash
# 方式一：从 HuggingFace 下载（推荐使用镜像）
export HF_ENDPOINT=https://hf-mirror.com

# 下载主模型（任选一个或全部）
huggingface-cli download OuteAI/OuteTTS-0.1-350M --local-dir /path/to/OuteTTS-0.1-350M
huggingface-cli download OuteAI/OuteTTS-0.2-500M --local-dir /path/to/OuteTTS-0.2-500M
huggingface-cli download OuteAI/OuteTTS-0.3-500M --local-dir /path/to/OuteTTS-0.3-500M

# WavTokenizer 编解码器（首次运行 inference.py 会自动下载）
# 或手动下载：
huggingface-cli download OuteAI/wavtokenizer-large-75token-interface \
  --local-dir ~/.cache/outeai/tts/wavtokenizer_75_token_interface
```

## 步骤 3：NPU 推理

**执行步骤**

1. 进入模型仓库目录：`cd OuteTTS-0.3-500M-npu`（推荐 0.3 版本）
2. 执行推理命令：`python inference.py --text "Hello, this is a test." --output output.wav --model_path /path/to/OuteTTS-0.3-500M --temperature 0.1 --repetition_penalty 1.1 --max_length 4096`
3. 等待模型加载和推理完成（首次加载 ~150s，后续 ~24s）
4. 确认 `output.wav` 文件已生成且音频时长 > 0
5. 若报 `LazySetDevice 507033` 错误，检查 NPU 占用进程并 kill

| 项目 | 内容 |
|------|------|
| **输入** | 本地模型路径、输入文本、生成参数（temperature/repetition_penalty） |
| **输出** | `output.wav` 音频文件 |
| **失败终止条件** | LazySetDevice 507033 错误或 NPU OOM |
| **前置依赖** | 步骤 2 通过（模型权重已下载） |

每个模型仓库包含独立的 `inference.py`，接口一致：

```bash
# 通用推理命令（修改 model_path 切换模型）
cd OuteTTS-0.3-500M-npu  # 或其他版本目录

python inference.py \
  --text "Hello, this is a test of the OuteTTS model on Ascend NPU." \
  --output output.wav \
  --model_path /path/to/OuteTTS-0.3-500M \
  --temperature 0.1 \
  --repetition_penalty 1.1 \
  --max_length 4096
```

参数说明：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--text` | 输入英文文本 | 任意文本 |
| `--output` | 输出 WAV 文件路径 | `output.wav` |
| `--model_path` | 模型权重目录 | 本地路径 |
| `--temperature` | 采样温度 | `0.1` (稳定性) / `0.7` (多样性) |
| `--repetition_penalty` | 重复惩罚 | `1.1` |
| `--max_length` | 最大 token 长度 | `4096` |
| `--top_k` | Top-K 采样 | `40` |
| `--top_p` | Top-P 采样 | `0.9` |

## 推理输出示例

```
[NPU] Device: Ascend910B4
[NPU] torch_npu version: 2.9.0.post1+gitee7ba04
[Model] Loading OuteTTS-0.3-500M (Qwen2 base) ...
[Model] Loaded successfully in 150.07s
[Inference] Input: Hello, this is a test...
[Result] Audio saved to: output.wav
[Result] Generation time: 23.75s (avg)
[Result] Audio duration: 3.41s (avg)
```

## 关键实现细节

模型通过 `outetts` 的 `ModelConfig` 指定 NPU 设备：

```python
from outetts.models.config import ModelConfig
from outetts.models.info import Backend, InterfaceVersion

config = ModelConfig(
    model_path="/path/to/model",
    interface_version=InterfaceVersion.V2,  # 0.1 -> V1, 0.2/0.3 -> V2
    backend=Backend.HF,                     # HuggingFace 后端
    device="npu:0",                         # NPU 设备
    dtype=torch.float16,                    # float16 推理
    max_seq_length=4096,
)
interface = Interface(config)
output = interface.generate(gen_config)
```

## 步骤 4：精度评测

**执行步骤**

1. 执行精度评测脚本：`python eval_accuracy.py --model_path /path/to/OuteTTS-0.3-500M --output_json eval_results.json --num_tests 5`
2. 等待脚本运行完成，检查生成的 `eval_results.json` 文件
3. 确认平均 MSE < 1e-5，相对误差 < 1%
4. 确认余弦相似度 > 0.999，SNR > 40 dB
5. 若精度不达标，检查模型加载完整性和生成参数设置

| 项目 | 内容 |
|------|------|
| **输入** | 本地模型路径、测试文本（默认 5 组） |
| **输出** | `eval_results.json`（MSE、相对误差、余弦相似度、SNR） |
| **合格标准** | MSE < 1e-5，相对误差 < 1%，余弦相似度 > 0.999，SNR > 40 dB |
| **前置依赖** | 步骤 3 通过，推理可正常执行 |

评测脚本对比 NPU (float16) 与 CPU (float32) 推理输出的波形一致性：

```bash
python eval_accuracy.py \
  --model_path /path/to/OuteTTS-0.3-500M \
  --output_json eval_results.json \
  --num_tests 5
```

## 精度结果汇总

| 模型 | 平均 MSE | 相对误差 | 余弦相似度 | SNR (dB) | 评估状态 |
|------|---------|---------|-----------|---------|---------|
| OuteTTS-0.1-350M | 9.83e-07 | 0.012% | 0.99997 | 49.2 | **PASS** |
| OuteTTS-0.2-500M | 1.15e-06 | 0.019% | 0.99994 | 47.1 | **PASS** |
| OuteTTS-0.3-500M | 1.12e-06 | 0.018% | 0.99995 | 47.6 | **PASS** |

> 所有模型的 NPU float16 vs CPU float32 波形一致性相对误差均远低于 1% 的接受阈值，精度验证通过。

## CPU Baseline 参考数据

| 模型 | 加载时间 (CPU) | 推理时间 (CPU) | 音频时长 | RTF |
|------|---------------|---------------|---------|-----|
| OuteTTS-0.1-350M | 149.17s | 31.42s | 2.09s | 15.0x |
| OuteTTS-0.2-500M | 150.33s | 140.53s | 26.37s* | 5.3x |
| OuteTTS-0.3-500M | 150.88s | 34.52s | 2.17s | 15.9x |

> *0.2 版本在 temperature=0.1 下可能生成异常长音频。建议使用 0.3 版本或调高 temperature 至 0.4-0.7。
> 测试条件：CPU float32, text="Hello, this is a test of the OuteTTS model.", max_length=2048

## NPU 性能数据

| 模型 | 加载时间 (NPU) | 平均推理时间 | 平均音频时长 | RTF |
|------|---------------|-------------|------------|-----|
| OuteTTS-0.1-350M | 149.25s | 22.87s | 3.54s | 6.5x |
| OuteTTS-0.2-500M | 151.07s | 110.42s | 21.3s | 5.2x |
| OuteTTS-0.3-500M | 150.07s | 23.75s | 3.41s | 7.0x |

> 测试条件：NPU float16, temperature=0.1 (0.2 use 0.5), repetition_penalty=1.1, max_length=2048

## 步骤 5：清理

| 项目 | 内容 |
|------|------|
| **输入** | 已运行的 NPU 推理进程 |
| **输出** | NPU 显存释放确认 |
| **失败终止条件** | 无（清理步骤可安全重复执行） |
| **前置依赖** | 步骤 4 完成 |

```bash
# 释放 Python 进程占用的 NPU 显存
# 推理脚本在单个 Python 进程中运行，退出自动释放

# 验证显存状态
npu-smi info
```

## 常见问题

**Q: 运行时报 `LazySetDevice: error code 507033`？**
A: NPU 设备被占用。检查是否有 vLLM 或其他进程占用：
```bash
npu-smi info  # 查看 NPU 使用情况
kill -9 <PID>  # 停止占用进程
```

**Q: WavTokenizer 下载失败？**
A: 使用 HF 镜像或手动下载到指定目录：
```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p ~/.cache/outeai/tts/wavtokenizer_75_token_interface
huggingface-cli download OuteAI/wavtokenizer-large-75token-interface \
  --local-dir ~/.cache/outeai/tts/wavtokenizer_75_token_interface
```

**Q: torch_npu 导入报 warning？**
A: CANN 版本不匹配导致的 UserWarning 可忽略，不影响推理功能。

**Q: 0.2 版本生成音频太长/太短？**
A: temperature 和 repetition_penalty 对 0.2 版本影响较大。推荐：
- 短文本使用 temperature=0.4-0.7
- 长文本使用 temperature=0.1 并适当降低 max_length

**Q: 三个版本怎么选？**
A: 推荐优先级：**0.3 > 0.1 > 0.2**。0.3 基于 Qwen2 架构，生成质量最稳定。

**Q: GGUF 版本能用吗？**
A: GGUF 版本（.gguf 文件）需要 llama.cpp 后端，当前 NPU 适配仅支持 HuggingFace 格式。

## 交付件清单

| 文件 | 说明 |
|------|------|
| inference.py | NPU 推理脚本：加载模型 → 文本转语音 → 保存 WAV |
| eval_accuracy.py | 精度评测脚本：NPU vs CPU 波形对比（MSE, CosSim, SNR） |
| eval_results.json | 精度评测结果数据 |
| readme.md | 部署文档（中文），含环境要求、步骤、评测数据、问题排查 |

## 相关仓库

- GitCode NPU 适配仓库：
  - [OuteTTS-0.1-350M-npu](https://gitcode.com/gcw_C8PI9e90/OuteTTS-0.1-350M-npu)
  - [OuteTTS-0.2-500M-npu](https://gitcode.com/gcw_C8PI9e90/OuteTTS-0.2-500M-npu)
  - [OuteTTS-0.3-500M-npu](https://gitcode.com/gcw_C8PI9e90/OuteTTS-0.3-500M-npu)
- 原始模型：[OuteAI on HuggingFace](https://huggingface.co/OuteAI)
- 推理库：[OuteTTS GitHub](https://github.com/edwko/OuteTTS)
- outetts PyPI：[outetts 0.4.4](https://pypi.org/project/outetts/)
- Ascend CANN：[华为昇腾文档](https://www.hiascend.com/document/)

---

## 执行检查点与用户确认

在每个步骤执行完毕后，向用户展示确认信息并等待确认后再进入下一步。

| 步骤 | 检查点 | 确认内容 | 通过条件 |
|------|--------|---------|---------|
| 步骤 1: 安装依赖 | Python 包导入验证 | `import outetts, torch, torch_npu, soundfile` 无报错 | 所有包版本与 CANN 兼容 |
| 步骤 2: 下载模型权重 | 模型文件完整性 | 目录下存在 `config.json` 和权重文件（safetensors/bin） | 模型文件完整且可加载 |
| 步骤 3: NPU 推理 | 推理输出验证 | `inference.py` 运行完成生成 output.wav | WAV 文件可播放，时长 > 0 |
| 步骤 4: 精度评测 | 精度指标验证 | MSE < 1e-5、余弦相似度 > 0.999、SNR > 40 dB | `eval_results.json` 中指标达标 |
| 步骤 5: 清理 | 显存释放确认 | `npu-smi info` 显示显存已释放 | 无残留 NPU 进程占用 |

## 异常处理与回滚策略

| 场景 | 检测方式 | 处理动作 | 回滚策略 |
|------|---------|---------|---------|
| NPU 设备不可见 | `npu-smi info` 无输出或报错 | 停止部署，提示检查 CANN 驱动安装 | 无（需管理员修复底层环境） |
| LazySetDevice 507033 错误 | 运行时报 `LazySetDevice: error code 507033` | 检查 NPU 占用进程并 `kill -9 <PID>` | `torch.npu.empty_cache()` 释放显存 |
| torch_npu 版本不匹配 | 导入报 warning 或 error | 按 CANN 版本重装对应 torch_npu | `pip uninstall torch_npu` 后重装匹配版本 |
| WavTokenizer 下载失败 | `huggingface-cli download` 失败 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 后手动下载到 cache 目录 | 删除不完整 cache 后重新下载 |
| 模型下载超时/失败 | huggingface-cli 返回非零退出码 | 切换 HF 镜像或检查网络 | 删除已下载的不完整目录，重新下载 |
| 生成音频异常（过长/过短） | 音频时长异常或内容异常 | 调整 temperature（0.4-0.7）和 repetition_penalty | 修改生成参数后重新推理 |
| 磁盘空间不足 | `df -h` 显示 < 2GB | 清理 pip 缓存或模型缓存 | 删除不需要的模型版本 |
| 0.2 版本生成异常 | 在 temperature=0.1 下生成超长音频 | 推荐使用 0.3 版本或调高 temperature 至 0.4-0.7 | 终止当前生成，调整参数重试 |

## 资源与评测产物

| 步骤 | 产物文件 | 格式 | 用途说明 |
|------|---------|------|---------|
| 步骤 2: 下载模型权重 | `{model_name}/` | 目录（config.json + 权重文件） | 本地缓存模型权重与配置文件 |
| 步骤 3: NPU 推理 | `output.wav` | WAV 音频文件 | 生成的语音输出 |
| 步骤 4: 精度评测 | `eval_results.json` | JSON | NPU vs CPU 波形对比结果（MSE, CosSim, SNR） |
| 步骤 5: 清理 | `npu-smi info` 输出 | 文本 | 显存使用状态验证 |
| 脚本资源 | `inference.py` | Python | NPU 推理主脚本 |
| 脚本资源 | `eval_accuracy.py` | Python | 精度评测脚本 |
| 脚本资源 | `readme.md` | Markdown | 部署文档 |
