---
name: wav2vec2-large-xlsr-53-greek-npu
description: "wav2vec2-large-xlsr-53-greek 希腊语语音识别模型在昇腾 NPU 上的自动迁移部署、CPU/NPU 推理精度验证与适配。适用于：昇腾部署、NPU 推理、语音识别 ASR、精度验证。触发词：wav2vec2 希腊语 NPU 部署、wav2vec2-large-xlsr-53-greek 昇腾推理、精度对比、transfer_to_npu。"
---

# wav2vec2-large-xlsr-53-greek 昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 wav2vec2-large-xlsr-53-greek 希腊语语音识别模型，完成推理、精度验证，并发布适配结果。

## 概述

本 Skill 提供 wav2vec2-large-xlsr-53-greek 希腊语 ASR 模型在昇腾 NPU 上的自动化部署、CPU 基线推理、NPU 推理（基于 transfer_to_npu 自动迁移）、CPU/NPU 精度对比验证的全流程能力。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend 910B/910) |
| 模型 | wav2vec2-large-xlsr-53-greek（1.2B 参数，CTC 架构） |
| 框架版本 | PyTorch 2.0+, torch_npu, transformers, librosa |
| 精度目标 | CPU 与 NPU 推理结果 Argmax 一致率 100%，相对误差 < 1% |
| CANN 版本 | 8.0+（推荐 8.2+） |
| 迁移方式 | transfer_to_npu 自动迁移（无需手动修改 CUDA 代码） |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.8-3.11 环境，昇腾 NPU 驱动，CANN 8.0+。

**动作**:
1. 加载 CANN 环境并检测 NPU 设备状态：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

2. 设置 NPU 设备：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

**输出**: NPU 可用状态确认，环境变量已设置。

### Step 2: 安装依赖

**输入**: 上一步确认的 NPU 可用环境。

**动作**:
3. 安装核心依赖：

```bash
pip install torch_npu transformers datasets librosa soundfile -i https://repo.huaweicloud.com/repository/pypi/simple/
```

4. 验证版本一致性：

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```

**输出**: torch 与 torch_npu 版本一致，无 ImportError。

### Step 3: NPU 基础验证

**输入**: 依赖安装完成的环境。

**动作**:
5. 执行 NPU Tensor 运算验证：

```bash
python3 -c "
import torch
import torch_npu
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU available:', torch.npu.is_available())
print('NPU device count:', torch.npu.device_count())
"
```

**通过标准**: 输出包含 `device='npu:0'` 的 Tensor 且无报错。

**输出**: NPU 运算正常，设备计数正确。

### Step 4: 下载模型权重

**输入**: 模型名称 wav2vec2-large-xlsr-53-greek。

**动作**:
6. 从 ModelScope 下载模型权重（约 1.2GB）：

```bash
python3 -c "
from modelscope import snapshot_download
snapshot_download(
    'jonatasgrosman/wav2vec2-large-xlsr-53-greek',
    cache_dir='./modelscope_cache'
)
"
```

7. 检查文件完整性：

```bash
ls -lh ./modelscope_cache/jonatasgrosman/wav2vec2-large-xlsr-53-greek/pytorch_model.bin
```

**输出**: 模型权重文件已就绪，缓存完整性确认。

### Step 5: CPU 基线推理

**输入**: 模型权重目录、设备类型 `cpu`。

**动作**:
8. 执行 CPU 基线推理并保存参考 logits：

```bash
python3 scripts/baseline_cpu.py
```

9. 验证推理输出包含 logits 文件和参考结果：

```bash
ls -la /opt/atomgit/wav2vec2-npu-adapt/ref_logits_cpu.npy
```

**输出**: CPU 推理 logits (`ref_logits_cpu.npy` 和 `ref_output_cpu.json`)，形状为 `[1, 149, 41]`。

### Step 6: NPU 推理（transfer_to_npu 自动迁移）

**输入**: 模型权重、设备类型 `npu:0`。

**动作**:
10. 执行 NPU 推理（transfer_to_npu 自动完成 CUDA->NPU 映射）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python3 scripts/inference_npu.py
```

11. 验证 NPU 推理输出：

```bash
ls -la /opt/atomgit/wav2vec2-npu-adapt/logits_npu.npy
```

核心适配逻辑说明：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu

# transfer_to_npu 会自动完成以下映射：
# - torch.cuda.is_available() -> 返回 True
# - .cuda() -> .npu()
# - torch.device('cuda') -> torch.device('npu')
# - torch.cuda.* -> torch.npu.*
```

**输出**: NPU 推理 logits (`logits_npu.npy` 和 `output_npu.json`)，设备标注为 `npu:0`。

### Step 7: CPU/NPU 精度对比验证

**输入**: CPU、NPU 推理 logits 文件。

**动作**:
12. 执行精度对比脚本：

```bash
python3 scripts/accuracy_compare.py /opt/atomgit/wav2vec2-npu-adapt/ref_logits_cpu.npy /opt/atomgit/wav2vec2-npu-adapt/logits_npu.npy
```

13. 验证对比指标：

| 指标 | 阈值 | 说明 |
|------|:----:|------|
| Argmax 一致率 | = 100% | 解码后预测结果完全一致 |
| 全范围相对误差 | < 1% | 相对输出动态范围的误差率 |
| 余弦相似度 | >= 0.9999 | 输出分布方向一致 |
| 最大绝对误差 | 记录 | 逐元素最大差值 |
| 均方根误差 | 记录 | 总体误差水平 |

**输出**: 精度对比报告（OVERALL: PASS/FAIL），包含误差指标表格。

### Step 8: 验收确认与结果发布

**输入**: 精度验证通过的完整结果。

**动作**:
14. 验证检查清单：

```bash
# 确认精度对比通过
grep "OVERALL: PASS" accuracy_output.txt

# 确认 Argmax 一致率 100%
grep "Argmax Match Rate" accuracy_output.txt

# 确认 NPU 推理日志
cat /opt/atomgit/wav2vec2-npu-adapt/output_npu.json | head -5
```

15. 记录 NFT 测试结果和适配要点（附录速查表），生成适配报告。

**输出**: 适配报告、精度验证结论、验收确认清单。

## 执行检查点与用户确认

每个关键步骤设有检查点 checkpoint，用户必须暂停确认后才能继续。以下 7 个检查点覆盖从环境准备到验收发布的完整流程：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测后 | NPU 设备是否可用，CANN 版本是否正确 | 暂停并提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 依赖安装检查点 | 依赖安装后 | torch 与 torch_npu 版本是否一致 | 暂停并检查 pip 镜像源后重试 |
| 3 | CP-3: 权重下载检查点 | 权重下载后 | 权重文件大小是否符合预期（~1.2GB） | 暂停并重新下载或切换镜像 |
| 4 | CP-4: CPU 基线检查点 | CPU 推理后 | CPU 推理 logits 形状是否为 `[1, 149, 41]` | 暂停并检查模型加载 |
| 5 | CP-5: NPU 推理确认 checkpoint | NPU 推理后 | NPU 推理是否成功，logits 设备是否为 npu:0 | 暂停并检查 NPU 显存和算子编译状态 |
| 6 | CP-6: 精度确认检查点 | 精度对比后 | Argmax 一致率是否 100%，相对误差是否 < 1% | 暂停并检查推理脚本一致性后重试 |
| 7 | CP-7: 验收审批检查点 | 验收确认前 | 所有验证指标是否通过，适配报告是否完整 | 暂停并修正后重新申请用户确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 提示加载 CANN 环境或安装 torch_npu，标记 fallback | `source set_env.sh` 后重新检测 |
| 算子编译失败 | 首次 NPU 推理报 SetPrecisionMode 错误 | 等待算子编译完成后重试 | 二次运行使用缓存，无需特殊处理 |
| NPU 显存 OOM | 推理时报 OOM 错误 | 释放缓存后重试，检查其他进程占用 | 关闭其他 NPU 进程，重启推理 |
| 模型加载异常 | transformers.from_pretrained 抛出异常 | 打印错误堆栈，检查权重缓存 | 确认 modelscope 缓存完整性，重新下载 |
| 网络超时 | pip/curl 下载失败（HuggingFace 超时） | 重试最多 3 次，每次间隔 5 秒 | 切换 ModelScope 国内镜像重试 |
| 精度超标 | Argmax 一致率 < 100% 或相对误差 >= 1% | 记录偏差明细，中止发布流程 | 检查输入数据一致性和随机种子 |
| 权重文件损坏 | pytorch_model.bin 加载异常 | 删除缓存目录后重新下载 | 确认磁盘空间充足后重试 |
| 用户确认超时 | 检查点等待确认超过 300 秒 | 暂停流程，保留当前进度，通知用户 | 用户恢复后从当前 checkpoint 继续 |
| CPU/NPU 形状不匹配 | ref_logits.shape != npu_logits.shape | 打印形状差异，检查推理输入一致性 | 确保使用相同输入音频和预处理流程 |
| Python 版本不兼容 | Python 版本 < 3.8 或 > 3.11 | 提示安装 Python 3.8 - 3.11 | 使用 conda 创建正确版本环境 |
| CANN 环境未加载 | torch_npu 导入失败或报环境错误 | 打印缺失的环境变量 | 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 首次推理耗时过长 | NPU 算子编译耗时 2-3 分钟 | 提示用户等待，显示编译进度 | 二次运行无需等待 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/baseline_cpu.py` | CPU 基线推理脚本，生成参考 logits |
| `scripts/inference_npu.py` | NPU 推理脚本（transfer_to_npu 自动迁移） |
| `scripts/accuracy_compare.py` | CPU/NPU 精度对比脚本，输出多维度指标 |
| `SKILL.md` | 本技能文档（当前文件） |
| `test-prompts.json` | 本技能测试提示词 |
| `ref_logits_cpu.npy` | CPU 推理参考 logits（运行后生成） |
| `logits_npu.npy` | NPU 推理 logits（运行后生成） |
| `ref_output_cpu.json` | CPU 推理输出 JSON（运行后生成） |
| `output_npu.json` | NPU 推理输出 JSON（运行后生成） |
| `modelscope_cache/` | 模型权重缓存目录（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 否 | wav2vec2-large-xlsr-53-greek | 模型名称（当前仅支持此模型） |
| `device` | string | 否 | npu:0 | 推理设备（npu:0 或 cpu） |
| `skip_download` | boolean | 否 | false | 跳过模型权重下载步骤 |
| `skip_cpu_baseline` | boolean | 否 | false | 跳过 CPU 基线推理步骤 |

## 使用约束

1. 使用 ModelScope 下载模型权重，优先使用国内镜像加速（华为云镜像源）。
2. 首次 NPU 推理需编译算子，耗时约 2-3 分钟，属正常现象。
3. transfer_to_npu 自动迁移适用于标准 PyTorch 模型（无自定义 CUDA kernel）。
4. 精度验证通过前不应提交适配结果。
5. CPU/NPU 对比需使用相同随机种子和预处理流程（seed=42）。
6. 权重文件较大（~1.2GB），确保磁盘空间充足（建议 > 5GB）。
