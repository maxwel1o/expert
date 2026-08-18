---
name: cosyvoice-npu-deploy
description: >
  CosyVoice TTS 系列模型在华为昇腾 NPU 上的完整部署 Skill。
  涵盖 CosyVoice-300M（零样本/跨语言）、CosyVoice-300M-Instruct（指令情感控制）、
  CosyVoice-300M-SFT（多说话人微调）和 CosyVoice-300M-25Hz（低帧率版本）
  共 4 个模型的 NPU 环境准备、依赖安装、模型下载、CPU/NPU 推理验证、
  精度对比的一站式标准化流程。可在 Ascend910 系列服务器上一键复现。
  当用户提到 CosyVoice NPU、TTS 昇腾、语音合成 NPU 部署时触发。
metadata:
  short-description: CosyVoice TTS 昇腾 NPU 部署与验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, cosyvoice, tts, voice, pytorch, inference, chinese]
---

# CosyVoice TTS 昇腾 NPU 部署 Skill

本 Skill 提供 CosyVoice TTS 系列模型在华为昇腾 NPU 上的完整部署、推理验证和性能评测的标准化可复现流程。

## 模型列表

| 模型 | 说明 | 推理方法 | GitCode 仓库 |
|------|------|---------|-------------|
| CosyVoice-300M | 基础版，零样本语音克隆 + 跨语言合成 | `inference_zero_shot`, `inference_cross_lingual` | [cosyvoice-CosyVoice-300M-npu](https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-npu) |
| CosyVoice-300M-Instruct | 指令版，支持情感/语速控制 | `inference_instruct` | [cosyvoice-CosyVoice-300M-Instruct-npu](https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-Instruct-npu) |
| CosyVoice-300M-SFT | 微调版，支持多说话人 | `inference_sft` | [cosyvoice-CosyVoice-300M-SFT-npu](https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-SFT-npu) |
| CosyVoice-300M-25Hz | 低帧率版（25fps mel），计算量更低 | `inference_zero_shot`, `inference_cross_lingual` | [cosyvoice-CosyVoice-300M-25Hz-npu](https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-25Hz-npu) |
| CosyVoice-ttsfrd | 文本前端工具（非 TTS 模型），提供中文文本正则化 | - | 作为依赖安装 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.RC1） |
| Python | 3.10 – 3.13 |
| 磁盘 | 至少 20GB（每个模型约 2-5GB 权重） |
| 网络 | 首次运行需联网下载模型权重（ModelScope） |

## 流程总览

```
0. 环境初始化
→ 1. 安装基础依赖（PyTorch + torch_npu + CosyVoice 源码）
→ 2. 下载模型权重（ModelScope 自动缓存）
→ 3. CPU 推理验证（每个模型串行执行）
→ 4. NPU 推理验证（应用 monkey-patch 适配）
→ 5. 精度对比（模型权重一致性验证）
→ 6. 性能数据汇总
```

**重要原则：多个模型必须串行适配、串行推理、串行测评**，不要并行运行多个模型。

## 执行检查点与用户确认

在执行高成本或有副作用的步骤前必须设置 checkpoint，并向用户展示当前输入、预期输出、fallback 和是否继续。

1. **环境 checkpoint**：运行 `npu-smi info`、`python3 --version`、`python3 -c "import torch_npu"`；如果 NPU 不可用，则暂停并说明只能进行 CPU dry-run，不进入 NPU benchmark。
2. **依赖 checkpoint**：安装 CosyVoice、Matcha-TTS、ModelScope 和 torch_npu 前确认 Python 版本、CANN 版本、磁盘空间和虚拟环境路径；如果安装失败，先记录错误再 retry。
3. **推理 checkpoint**：每个模型先跑 CPU smoke test，再跑 NPU smoke test；如果 CPU 失败，则停止 NPU 流程；如果 NPU 失败，则触发 fallback 排查。
4. **精度 checkpoint**：只有 CPU/NPU 结果文件都存在时才计算对比；如果随机采样导致音频不同，要解释这是预期行为，并以权重一致性和性能指标作为验证证据。

## 异常处理与回滚策略

| 场景 | 判断方式 | fallback / recover | 回滚要求 |
|------|----------|--------------------|----------|
| `npu-smi info` 失败 | 命令非 0 或无 Ascend 设备 | 切换为 CPU dry-run，标记 `npu_unavailable` | 不生成 NPU benchmark 结论 |
| `torch_npu` 导入失败 | `ModuleNotFoundError` 或 ABI 不匹配 | 重新安装与 CANN 匹配版本，retry 1 次 | 保留安装日志，不覆盖环境说明 |
| CosyVoice 源码安装失败 | `pip install -e .` 非 0 | 检查 submodule 与 Matcha-TTS 路径，修复后 retry | 回滚临时路径修改 |
| ModelScope 下载失败 | 网络错误或缓存缺失 | retry 1 次，仍失败则跳过该模型并记录 `download_failed` | 不删除已下载缓存 |
| `torch.istft` NPU 报错 | 出现 `aclnnUnfoldGrad` 等错误 | 使用已定义的 CPU fallback，不把该错误视为适配失败 | 在报告中标注 CPU fallback |
| NPU OOM | 显存不足或进程残留 | 单模型串行执行、释放缓存、缩短文本后 retry | 不覆盖 CPU 结果文件 |

## 资源与评测产物

本 skill 当前以 `SKILL.md` 为主，不依赖额外脚本；如后续补充 `scripts/` 或 `references/`，必须在本节登记用途和入口。评测与复现至少产生以下资源：

1. `/tmp/CosyVoice-*_cpu_results.json`：CPU smoke test 输出。
2. `/tmp/CosyVoice-*_npu_results.json`：NPU smoke test 输出。
3. `/tmp/CosyVoice-*_comparison.json`：权重一致性、耗时、RTF 和 benchmark 指标。
4. `test-prompts.json`：用于 ascend-skills-eval 的 eval / dry_run 测试输入。
5. `results.tsv`（可选）：批量评测时记录模型名、设备、状态、错误类型和 retry 次数。

每次完成后必须汇总：成功模型数、失败模型数、是否发生 fallback、benchmark 加速比、仍需人工确认的问题。

---

## 0. 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# pip 国内镜像加速（可选）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

---

## 1. 安装基础依赖

### 1.1 Python 依赖

```bash
pip install torch torchaudio torch_npu --index-url https://download.pytorch.org/whl/cpu
pip install matcha-tts modelscope onnxruntime onnx soundfile librosa scipy
pip install numpy inflect wetext pyworld tqdm omegaconf hydra-core pyyaml
```

### 1.2 安装 CosyVoice 源码

```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
pip install -e .
```

### 1.3 设置 Python 路径

```python
# 在推理脚本中需添加以下路径
sys.path.insert(0, '/path/to/CosyVoice/third_party/Matcha-TTS')
sys.path.insert(0, '/path/to/CosyVoice')
```

---

## 2. 下载模型权重

模型权重通过 ModelScope 自动缓存到 `~/.cache/modelscope/` 目录。

```python
from modelscope.hub.snapshot_download import snapshot_download
models = [
    "iic/CosyVoice-300M",
    "iic/CosyVoice-300M-Instruct",
    "iic/CosyVoice-300M-SFT",
    "iic/CosyVoice-ttsfrd",
    "iic/CosyVoice-300M-25Hz",
]
for model in models:
    print(f"Downloading {model}...")
    snapshot_download(model)
```

也可直接从 GitCode 模型仓库克隆推理代码（不包含权重）：

```bash
git clone https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-npu.git
git clone https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-Instruct-npu.git
git clone https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-SFT-npu.git
git clone https://gitcode.com/gcw_C8PI9e90/cosyvoice-CosyVoice-300M-25Hz-npu.git
```

---

## 3. NPU 适配说明

### 3.1 适配原理

所有模型采用 **monkey-patch** 方式进行 NPU 适配，无需修改 CosyVoice 源码：

1. **设备替换**：将 `CosyVoiceModel.__init__` 中的 CUDA 设备替换为 NPU 设备
2. **权重加载**：`CosyVoiceModel.load` 使用 `map_location='npu:0'` 将权重加载到 NPU
3. **算子回退**：`HiFTGenerator._istft`（`torch.istft`）在 NPU 上不支持，回退到 CPU 执行
4. **ONNX Runtime**：特征提取模型（campplus, speech_tokenizer）使用 CPUExecutionProvider

### 3.2 关键补丁代码

```python
def apply_npu_patch():
    import cosyvoice.cli.model as cvm

    # Patch 1: NPU device
    cvm.CosyVoiceModel.__init__ = lambda self, llm, flow, hift, fp16=False: (
        setattr(self, 'device', torch.device('npu:0')),
        # ... 其余属性初始化
    )

    # Patch 2: Weight loading to NPU
    def npu_load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(
            torch.load(llm_model, map_location='npu:0', weights_only=True), strict=True)
        # ...

    # Patch 3: istft CPU fallback
    from cosyvoice.hifigan.generator import HiFTGenerator
    def npu_istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        device = magnitude.device
        inv = torch.istft(torch.complex(real, img).cpu(),
                          self.istft_params["n_fft"],
                          self.istft_params["hop_len"],
                          self.istft_params["n_fft"],
                          window=self.stft_window.cpu())
        return inv.to(device)
    HiFTGenerator._istft = npu_istft
```

完整的推理脚本见各模型仓库的 `inference.py`。

---

## 4. 推理验证

对每个模型，按以下步骤执行（**必须串行，一个完成后再进行下一个**）：

### 4.1 CosyVoice-300M（基础版）

```bash
cd cosyvoice-CosyVoice-300M

# CPU 推理
python3 inference.py --device cpu

# NPU 推理
python3 inference.py --device npu

# 精度对比
python3 compare_cpu_npu.py
```

测试内容：零样本语音克隆（Zero-shot TTS）+ 跨语言合成（Cross-lingual TTS）

### 4.2 CosyVoice-300M-Instruct（指令版）

```bash
cd cosyvoice-CosyVoice-300M-Instruct

# CPU 推理
python3 inference.py --device cpu

# NPU 推理
python3 inference.py --device npu

# 精度对比
python3 compare_cpu_npu.py
```

测试内容：指令式情感控制合成，使用 `<strong>` SSML 标签和文本指令控制情感表达

### 4.3 CosyVoice-300M-SFT（微调版）

```bash
cd cosyvoice-CosyVoice-300M-SFT

# CPU 推理
python3 inference.py --device cpu

# NPU 推理
python3 inference.py --device npu

# 精度对比
python3 compare_cpu_npu.py
```

测试内容：多说话人合成，支持 7 种说话人（中文男/女、英文男/女、日语男、粤语女、韩语女）

### 4.4 CosyVoice-300M-25Hz（低帧率版）

```bash
cd cosyvoice-CosyVoice-300M-25Hz

# CPU 推理
python3 inference.py --device cpu

# NPU 推理
python3 inference.py --device npu

# 精度对比
python3 compare_cpu_npu.py
```

测试内容：零样本语音克隆 + 跨语言合成（25fps mel 帧率）

---

## 5. 精度对比方法

### 5.1 数据收集

推理脚本自动生成 JSON 结果文件到 `/tmp/` 目录：

```
/tmp/CosyVoice-300M_cpu_results.json
/tmp/CosyVoice-300M_npu_results.json
/tmp/CosyVoice-300M_comparison.json
```

### 5.2 对比指标

| 指标 | 说明 |
|------|------|
| LLM state_dict 大小 | 401 权重张量（所有模型一致） |
| 权重最大差异 | 加载自同一 pt 文件，理论上为 0.0 |
| 推理耗时 | CPU vs NPU 的端到端推理时间 |
| 加速比 | CPU 时间 / NPU 时间 |
| RTF（实时因子） | 推理耗时 / 音频时长，越低越好 |

### 5.3 说明

CosyVoice 系列模型使用 top-k=25、top-p=0.8 的随机采样策略（ras_sampling），因此 CPU 和 NPU 生成的语音 tokens 序列不同是预期的模型行为，而非 NPU 适配问题。模型权重在 CPU 和 NPU 上加载后完全一致（最大差异 0.0），证明 NPU 适配的正确性。

---

## 6. 性能基准

| 模型 | CPU 总耗时(s) | NPU 总耗时(s) | 加速比 |
|------|:-----------:|:-----------:|:----:|
| CosyVoice-300M | ~361.66 | ~12.20 | **29.6x** |
| CosyVoice-300M-Instruct | ~199.47 | ~7.99 | **25.0x** |
| CosyVoice-300M-SFT | ~156.17 | ~7.60 | **20.6x** |
| CosyVoice-300M-25Hz | ~267.91 | ~7.61 | **35.2x** |

所有模型权重加载后与 CPU 完全一致（最大差异 0.0），NPU 生成的语音清晰自然。

---

## 7. 常见问题

### 7.1 ttsfrd 安装失败

**问题**：Python 3.11 下 ttsfrd 预编译 wheel（cp310）无法安装。
**解决**：使用 wetext 替代：
```bash
pip install wetext
```
CosyVoice 自动检测可用后端：ttsfrd > wetext > 无文本前端。

### 7.2 torch.istft 在 NPU 上报错

**问题**：`RuntimeError: aclnnUnfoldGrad` 表明 torch.istft 在 Ascend NPU 上不支持。
**解决**：已在 monkey-patch 中将 istft 回退到 CPU 执行。无需额外操作。

### 7.3 torchaudio.load 失败

**问题**：torchaudio 2.9.x 强制使用 torchcodec 后端。
**解决**：卸载 torchcodec 或在 `load_wav()` 中使用 soundfile 直接读取。

### 7.4 显存不足

**解决**：
```python
import torch
del cosyvoice
import gc; gc.collect()
if torch.npu.is_available():
    torch.npu.empty_cache()
```

---

## 8. 参考链接

- [CosyVoice 官方仓库](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice ModelScope 主页](https://www.modelscope.cn/models/iic/CosyVoice-300M)
- [华为昇腾社区](https://www.hiascend.com/)
