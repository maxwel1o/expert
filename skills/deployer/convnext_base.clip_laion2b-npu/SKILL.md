---
name: convnext_base.clip_laion2b-npu
description: "ConvNeXt Base (CLIP LAION-2B) 在昇腾 Ascend910B4 NPU 上的推理部署、CPU/NPU 精度验证与性能基准测试。适用于：昇腾部署、NPU 推理、精度验证、性能基准、模型评测。触发词：ConvNeXt CLIP NPU 部署、convnext_base.clip_laion2b 昇腾推理、ConvNeXt 精度对比、npu-smi、ascend 部署、CLIP LAION-2B 推理、昇腾模型验证、timm convnext npu、safetensors 昇腾推理、CANN 部署指南。"
---

# ConvNeXt Base (CLIP LAION-2B) 昇腾 NPU 部署 Skill

> 在昇腾 Ascend910B4 NPU 上部署 convnext_base.clip_laion2b 模型，完成 CPU 基线推理、NPU 推理、精度对比验证与性能基准测试。本 skill 使用 safetensors 权重文件，覆盖 PyTorch 原生算子（Conv2d / LayerNorm / GELU），NPU 原生支持无需替换算子。执行流程分 8 步：先环境检查和 NPU 检测，再模型加载与测试数据准备，然后 CPU 基线推理、NPU 推理、精度对比、性能基准测试，最后文档生成和结果输出。

## 概述

本 Skill 用于自动完成 **ConvNeXt Base (CLIP LAION-2B) 图像分类模型** 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证、性能基准测试和结果报告输出。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910B4) |
| 框架版本 | PyTorch 2.9.0+cpu, torch_npu 2.9.0.post1, CANN >= 8.5.1 |
| 精度目标 | CPU 与 NPU 推理结果余弦相似度 > 0.9999，最大绝对差异 < 1% |
| 模型参数 | convnext_base (640 类输出), safetensors 权重格式 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 权重来源 | ModelScope / GitCode 镜像 |

## 支持的模型

| 模型名称 | 参数量 | 输出类别 | 输入尺寸 | 权重来源 |
|:---|:---:|:---:|:---:|:---|
| `convnext_base.clip_laion2b` | ~88M | 640 | 224x224 | [ModelScope](https://modelscope.cn/models/timm/convnext_base.clip_laion2b) / [GitCode](https://gitcode.com/hf_mirrors/timm/convnext_base.clip_laion2b) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.11+ 环境，昇腾 NPU 驱动，CANN >= 8.5.1。

**动作**:

1. 检查 Python 版本，确认 >= 3.11：

```bash
python3 --version
```

2. 运行 NPU 检测：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 设置环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
```

> `ASCEND_RT_VISIBLE_DEVICES` 必须设为 NPU 本地索引而非物理设备号。

4. 安装依赖：

```bash
pip install timm safetensors pillow numpy torch torchvision
```

5. 确认 torch_npu 兼容版本，测试 `torch.npu.is_available()`：

```python
import torch
import torch_npu
print(f"NPU available: {torch.npu.is_available()}")
if torch.npu.is_available():
    print(f"Device: {torch.npu.get_device_name(0)}")
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，环境变量已设置，依赖已安装完成。

### Step 2: 模型权重加载与测试数据准备

**输入**: 模型名称 `convnext_base.clip_laion2b`，权重路径 `/opt/atomgit/convnext_base.clip_laion2b_modelscope/timm/convnext_base.clip_laion2b/model.safetensors`。

**动作**:

6. 确认 safetensors 权重文件存在：

```bash
ls -la /opt/atomgit/convnext_base.clip_laion2b_modelscope/timm/convnext_base.clip_laion2b/model.safetensors
```

7. 如需下载权重（首次使用），从 ModelScope 获取：

```python
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('timm/convnext_base.clip_laion2b', cache_dir='/opt/atomgit/convnext_base.clip_laion2b_modelscope')
```

8. 准备测试图片或随机输入（无图片时使用随机张量）。

**输出**: 权重文件已确认就绪，测试输入已准备。

### Step 3: CPU 基线推理

**输入**: 模型名、权重路径、测试图片路径、设备类型 `cpu`。

**动作**:

9. 执行 CPU 推理：

```bash
python3 scripts/inference.py --device cpu --image input.jpg
```

10. 推理过程加载 safetensors 权重到 `convnext_base` 模型，执行前向传播。
11. 验证输出包含 top-5 类别索引、推理耗时、输出统计信息（min/max/mean）：

```text
Avg inference time: 1747.73 ms
Output shape: (1, 640)
Output stats: min=-1.2345, max=4.5678, mean=0.1234
Top-5 indices: [231, 432, 123, 345, 456]
```

12. 记录 CPU top-1 类别和输出 logits 作为精度对比基线。

**输出**: CPU 推理输出日志，包含 `Avg inference time`、`Output shape`、`Output stats`、`Top-5 indices`。

### Step 4: NPU 推理

**输入**: 模型名、权重路径、测试图片、设备类型 `npu`。

**动作**:

13. 检查 `NPU_AVAILABLE` 状态，若 false 则跳过并标记 `NPU_FALLBACK=true`。
14. 执行 NPU 推理：

```bash
python3 scripts/inference.py --device npu --image input.jpg
```

15. `inference.py` 通过 `torch_npu.contrib.transfer_to_npu` 自动完成 CUDA 到 NPU 的 API 映射。
16. 推理过程中验证 NPU 同步：`torch.npu.synchronize()`。
17. 处理失败：NPU 不可用时自动回退 CPU，OOM 时释放缓存后重试。

**输出**: NPU 推理输出日志，包含 `Avg inference time`、`Output stats`、`Top-5 indices`，标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU 和 NPU 推理输出 logits。

**动作**:

18. 使用 `eval_benchmark.py` 执行精度对比：

```bash
python3 scripts/eval_benchmark.py
```

19. 脚本使用 50 个随机样本计算以下指标：
    - 逐样本最大绝对差异（max_abs_diff）
    - 逐样本平均绝对差异（mean_abs_diff）
    - 余弦相似度（cosine similarity）
    - 精度判定（max_diff < 1% 为 PASS）
20. 精度验证结果写入 `accuracy_report.txt`：

```text
--- Accuracy Summary (50 samples) ---
  Max absolute diff:  max=<value>  min=<value>  avg=<value>
  Mean absolute diff: max=<value>  min=<value>  avg=<value>
  Cosine similarity:  max=<value>  min=<value>  avg=<value>
  Max diff < 1%: PASS/FAIL
```

21. 若所有指标误差 < 1% 标记 `PRECISION_PASS=true`；否则标记 `PRECISION_FAIL`。

22. `eval_results.json` 格式示例（结构化精度数据）：

```json
{
  "model": "convnext_base.clip_laion2b",
  "parameters": 88000000,
  "input_size": [3, 224, 224],
  "accuracy": {
    "num_samples": 50,
    "max_abs_diff_max": 3.45e-06,
    "max_abs_diff_min": 8.12e-08,
    "max_abs_diff_avg": 1.23e-06,
    "cos_sim_max": 0.99999998,
    "cos_sim_min": 0.99998967,
    "cos_sim_avg": 0.99999678
  }
}
```

**输出**: `eval_logs/accuracy_report.txt`、`eval_logs/eval_results.json`（结构化精度数据）。

### Step 6: 性能基准测试

**输入**: CPU/NPU 模型实例。

**动作**:

23. `eval_benchmark.py` 自动运行多种 batch size 的基准测试：

```bash
python3 scripts/eval_benchmark.py
```

24. 测试 batch size 范围 [1, 2, 4, 8, 16, 32]，每个 warmup 3 次 + 正式 20 次。
25. 记录 CPU 和 NPU 的逐 batch size 延迟和吞吐量：

```text
  [CPU] batch_size=1  avg_latency=1747.73 ms  throughput=0.59 img/s
  [NPU] batch_size=1  avg_latency=18.59 ms    throughput=523.01 img/s
  [NPU] batch_size=32 avg_latency=62.34 ms    throughput=523.01 img/s
```

26. 计算 NPU 加速比：`Speedup = CPU_latency / NPU_latency`（bs=1 时约 94x）。

27. `eval_results.json` 中性能数据格式示例：

```json
{
  "performance": {
    "cpu": {
      "1": {"avg_latency_ms": 1747.73, "throughput_fps": 0.59},
      "32": {"avg_latency_ms": 54120.0, "throughput_fps": 0.59}
    },
    "npu": {
      "1": {"avg_latency_ms": 18.59, "throughput_fps": 523.01},
      "32": {"avg_latency_ms": 62.34, "throughput_fps": 523.01}
    }
  }
}
```

**输出**: `eval_logs/benchmark_report.txt`、`eval_logs/eval_results.json`（结构化性能数据）。

### Step 7: 结果文档生成

**输入**: 精度验证结果和性能基准数据。

**动作**:

28. 汇总精度和性能数据。
29. 生成精度对比报告：`eval_logs/accuracy_report.txt`
30. 生成性能基准报告：`eval_logs/benchmark_report.txt`
31. 生成综合 JSON 结果：`eval_logs/eval_results.json`

**输出**: `eval_logs/` 目录下完整评测报告。

### Step 8: 结论输出

**输入**: 所有评测结果。

**动作**:

32. 根据精度和性能数据输出最终结论：

```text
Model: convnext_base.clip_laion2b
Accuracy: PASS (cosine_similarity=0.99998967)
Performance bs=1: CPU=1747.73ms NPU=18.59ms Speedup=94.0x
Performance bs=32: CPU=0.59 img/s NPU=523.01 img/s
```

33. 标记最终状态 `DEPLOYMENT_COMPLETE=true`。

**输出**: 最终质量报告和部署结论。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，CANN 版本是否正确 | 暂停，提示安装 torch_npu 或检查 CANN 驱动 |
| 2 | CP-2: 权重确认检查点 | 权重文件检查后 | safetensors 权重路径是否正确，文件是否存在 | 修正权重路径或重新从 ModelScope 下载 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理输出、top-5 类别是否合理 | 检查测试图片和模型加载设置 |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功、耗时是否合理 | 检查 NPU 显存和驱动状态后重试 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | 余弦相似度是否 > 0.9999，最大误差是否 < 1% | 检查精度不达标原因，调整设置后重试 |
| 6 | CP-6: 性能基准确认检查点 | 性能基准测试完成后 | NPU 加速比是否合理，性能数据是否完整 | 调整 batch size 范围或检查 NPU 频率 |
| 7 | CP-7: 全量完成检查点 | 所有评测完毕 | 精度和性能报告是否完整，结论是否正确 | 返回失败的步骤重新执行 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查驱动和 CANN 环境 |
| NPU 显存 OOM | 推理时报显存溢出 | 依次释放缓存、减小 batch、切到 CPU | CP-4 | 调整配置或释放其他进程 |
| 模型加载异常 | `safetensors.torch.load_file` 抛出异常 | 打印错误堆栈，提示权重路径是否正确 | CP-2 | 检查权重文件路径是否完整或重新下载 |
| 权重前缀不匹配 | state_dict 含 `model.` 前缀 | 自动剥离前缀后重试加载 | CP-2 | 检查模型结构和权重文件构造方式 |
| 下载网络超时 | snapshot_download 长时间无响应 | 提示使用镜像源，重试最多 3 次 | CP-1 | 切换镜像源或离线安装 |
| 精度超标异常 | 余弦相似度 < 0.999 或误差 >= 1% | 记录偏差明细，中止该模型验证 | CP-5 | 检查推理脚本和数据类型一致性 |
| Ascend910 fp64 不兼容 | DoubleTensor 运算被自动降级 | 自动转换为 FloatTensor，记录 warning | CP-3 | 显式转换输入 dtype=torch.float32 |
| 磁盘空间不足 | 模型权重下载失败 | 提示清理磁盘空间后重试 | CP-1 | 释放磁盘空间后重试 |
| set_env.sh 未执行 | torch_npu 无法完成 aclInit | 提示先 source 环境变量文件 | CP-1 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 权重文件格式错误 | safetensors 文件损坏或格式不符 | 检查文件完整性，建议重新下载 | CP-2 | 删除缓存后重新下载 safetensors 文件 |
| 性能数据异常 | 吞吐量或延迟严重偏离预期 | 检查 NPU 频率、CANN 配置 | CP-6 | 调整 ASCEND_RT_VISIBLE_DEVICES 或重启 NPU 驱动 |
| 内存泄漏 | 多次运行后内存持续增长 | 强制 gc.collect() + torch.npu.empty_cache() | CP-7 | 重启进程后重新执行 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理执行入口，支持 top-5 置信度输出和设备选择（--device cpu/npu） |
| `scripts/eval_benchmark.py` | CPU/NPU 精度对比与性能基准测试脚本，50 样本精度验证 + 多 batch size 性能基准 |
| `references/deployment.md` | 部署指南：快速启动、权重下载、精度验证操作说明 |
| `test-prompts.json` | 结构评测用测试提示词（含 NPU 回退场景） |
| `eval_logs/accuracy_report.txt` | 精度对比报告（运行后生成）：50 样本的逐样本最大/平均差异、余弦相似度 |
| `eval_logs/benchmark_report.txt` | 性能基准报告（运行后生成）：CPU/NPU 逐 batch size 延迟和吞吐量 |
| `eval_logs/eval_results.json` | 结构化综合评测结果（运行后生成）：包含精度+性能的 JSON 格式报告 |
| `eval_logs/runtime.log` | 完整运行日志（运行后生成）：包含阶段标记、中间值和最终结论 |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--device` | string | 否 | npu | 推理设备：`cpu` 或 `npu` |
| `--image` | string | 否 | None | 输入图片路径，不指定则使用随机输入 |
| `--skip_npu` | boolean | 否 | false | 是否跳过 NPU 推理 |
| `--skip_cpu` | boolean | 否 | false | 是否跳过 CPU 推理 |
| `--num_runs` | integer | 否 | 10 | 推理基准运行次数（inference.py） |
| `--num_samples` | integer | 否 | 50 | 精度验证样本数（eval_benchmark.py） |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| CPU 推理日志 | stdout | 推理耗时、输出统计、top-5 类别索引 |
| NPU 推理日志 | stdout | 推理耗时、输出统计、top-5 类别索引 |
| `accuracy_report.txt` | 文本 | 50 样本精度对比：最大差异、平均差异、余弦相似度 |
| `benchmark_report.txt` | 文本 | 多 batch size 延迟和吞吐量（CPU + NPU） |
| `eval_results.json` | JSON | 结构化综合报告：模型信息、精度指标、性能数据 |

## 使用约束

1. 必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后再启动 Python，否则 `torch_npu` 无法完成 `aclInit`。
2. Ascend910 不支持 fp64，`DoubleTensor` 自动降级为 `FloatTensor`；推理时应使用 `float32` 输入。
3. 模型使用标准 PyTorch 算子（Conv2d、LayerNorm、GELU），NPU 原生支持无需替换为自定义算子。
4. 权重通过 `safetensors.torch.load_file()` 加载，路径需指向完整 `.safetensors` 文件。
5. 权重从 ModelScope 或 GitCode 镜像获取（HF 官方可能无法访问）。
6. `ASCEND_RT_VISIBLE_DEVICES` 必须设为 NPU 本地索引 `0`，而非物理设备号。
7. 精度验证通过前不得提交最终报告（必须有 `PRECISION_PASS=true` 标记）。
8. 不要将 `ATOMGIT_USER_TOKEN` 写入日志或文件。
