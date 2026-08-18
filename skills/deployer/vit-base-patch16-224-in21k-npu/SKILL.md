---
name: vit-base-patch16-224-in21k-npu
description: Google ViT-Base-Patch16-224-IN21K Vision Transformer 在昇腾 NPU 上的推理适配与部署 Skill。涵盖环境准备、ModelScope 权重下载、transfer_to_npu 自动迁移、CPU/NPU 精度对比验证、性能基准测试的全流程。当用户提到 ViT 昇腾部署、Vit NPU 推理、视觉模型 NPU、google/vit-base-patch16-224-in21k 适配时触发。
---

# ViT-Base-Patch16-224-IN21K 昇腾 NPU 部署 Skill

本 Skill 提供 `google/vit-base-patch16-224-in21k` Vision Transformer 模型在华为昇腾 NPU 上的完整推理适配、部署验证和精度对比的标准化可复现流程。

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型名称 | `google/vit-base-patch16-224-in21k` |
| 架构 | ViT-Base (Patch Size 16, Hidden Size 768, Layers 12, Heads 12) |
| 参数量 | ~86M |
| 输入尺寸 | 224×224 (RGB) |
| 权重来源 | ModelScope: `google/vit-base-patch16-224-in21k` / HuggingFace 镜像 |
| 模型仓库 | [vit-base-patch16-224-in21k-npu](https://gitcode.com/m0_74196153/vit-base-patch16-224-in21k-npu) |

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 否 | 模型名称，默认 `google/vit-base-patch16-224-in21k` |
| `action` | string | 否 | 执行操作：inference / compare / benchmark / all（默认 all） |

## Skill 输出结果

| 输出 | 格式 | 说明 |
|------|------|------|
| inference_result | JSON | CPU/NPU 推理结果，含 logits 输出和 Top-5 预测 |
| accuracy_report | JSON | CPU/NPU 精度对比结果，含 max_abs_diff / mean_rel_diff / top1_match_rate |
| benchmark_report | JSON | 性能基准测试结果，含 CPU/NPU 各 batch 延迟 |
| evals.json | JSON | 结构化评估记录，含环境检查、重试、验证结果 |

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| NPU | Ascend910 系列（Atlas 800 A2，至少 1 卡） |
| CANN | >= 8.0（推荐 8.5.1+） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| Python | 3.9 – 3.13 |
| PyTorch | >= 2.0.0 |
| torch_npu | >= 2.0.0 |
| transformers | == 4.57.6 |
| modelscope | == 1.35.3 |
| 网络 | 首次运行需联网下载模型权重（~330MB） |

## 工作流程

### 执行总览

1. **环境初始化与 NPU 验证**：加载 CANN 环境，检查 NPU 设备状态，验证 torch_npu 导入和 NPU 可用性。
2. **安装 Python 依赖**：安装 torch_npu、transformers、modelscope 等依赖包。
3. **模型权重下载**：通过 ModelScope 或 HuggingFace 下载 `google/vit-base-patch16-224-in21k` 预训练权重。
4. **CPU Baseline 推理**：在 CPU 设备上执行推理，保存 logits 和 Top-K 分类结果到文件。
5. **NPU 推理**：通过 `transfer_to_npu` 自动迁移 CUDA API 至 NPU，在 NPU 设备上执行推理，保存 NPU logits。
6. **CPU/NPU 精度对比**：对比 CPU 与 NPU 推理结果，计算最大绝对误差、相对误差、Top-1 匹配率等指标。
7. **性能基准测试**：测量 CPU 与 NPU 在不同 batch size 下的推理延迟和吞吐。
8. **结果汇总与验收确认**：汇总精度和性能数据，生成 evals.json 和 benchmark_report，通过验收检查清单。

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|--------|---------|-------------|---------------|
| CP-1 环境检查点 | npu-smi info 执行后 | 确认当前 Ascend/CANN/NPU 环境正常 | 标记为 dry-run 模式，不写入真实 NPU 结论 |
| CP-2 依赖安装检查点 | 依赖安装完成后 | 确认 torch_npu 和 transformers 版本兼容 | 重试安装或指定兼容版本号 |
| CP-3 权重下载检查点 | ModelScope 下载前 | 确认模型名称和权重来源 | 切换至 HuggingFace 镜像或复用本地缓存 |
| CP-4 CPU 推理检查点 | CPU 推理完成后 | 确认 logits 输出合理且 shape 正确 | 检查模型加载和输入预处理参数 |
| CP-5 NPU 推理检查点 | NPU 推理完成后 | 确认 NPU 推理无异常，logits shape 为 [1,2] | 释放 NPU 显存后重试推理 |
| CP-6 精度验证检查点 | CPU/NPU 对比完成后 | 确认 mean_rel_diff < 1% 且 top1_match_rate = 100% | 标记精度验证失败，保留 logits 日志 |
| CP-7 最终验收检查点 | 所有步骤完成后 | 确认 benchmark 数据和精度结论完整 | 补充缺失数据或重新运行相关步骤 |

## 执行步骤详解

### 1. NPU 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0

# 验证 NPU 可用性
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错。

### 2. 安装依赖

```bash
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
pip install transformers==4.57.6 modelscope==1.35.3 Pillow numpy==1.26.4 -i https://repo.huaweicloud.com/repository/pypi/simple/
```

### 3. 下载模型权重

从 ModelScope 下载 `google/vit-base-patch16-224-in21k` 预训练权重：

```python
from modelscope import snapshot_download

model_id = 'google/vit-base-patch16-224-in21k'
local_dir = snapshot_download(model_id, cache_dir='./model_weights')
print(f"Model downloaded to: {local_dir}")
```

也可从 HuggingFace 镜像下载：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download google/vit-base-patch16-224-in21k --local-dir ./model_weights
```

### 4. CPU Baseline 推理

```bash
# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts

# 运行 CPU 推理
python3 scripts/infer_cpu.py
```

核心代码（`scripts/infer_cpu.py`）：

```python
import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

torch.manual_seed(42)
np.random.seed(42)

processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

device = torch.device('cpu')
model = model.to(device)

image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"[CPU] Output logits shape: {logits.shape}")
print(f"[CPU] Output logits sample: {logits[0, :5].tolist()}")
```

**预期输出**：logits shape 为 `[1, 2]`（IN21K 预训练模型无分类头，输出维度为 2）。

### 5. NPU 推理

`transfer_to_npu` 自动迁移必须在所有其他 import 之前注入：

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# 运行 NPU 推理
python3 scripts/infer_npu.py
```

核心代码（`scripts/infer_npu.py`）：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu  # 必须在所有 import 之前

import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

torch.manual_seed(42)
np.random.seed(42)

processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

device = torch.device('npu:0')
model = model.to(device)

image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"[NPU] Output logits shape: {logits.shape}")
print(f"[NPU] Output logits sample: {logits[0, :5].cpu().tolist()}")
```

**通过标准**：
- 程序正常退出（exit code 0）
- logits shape 为 `[1, 2]`
- 无 NPU 算子报错或 fallback 警告
- 算子编译缓存已生成，二次运行延迟稳定在 ~12 ms

### 6. CPU/NPU 精度对比

```bash
python3 scripts/compare_accuracy.py
```

精度对比脚本（`scripts/compare_accuracy.py`）读取 CPU 和 NPU 的推理结果，计算以下指标：

| 指标 | 计算方法 | 通过标准 |
|------|---------|---------|
| max_abs_diff | max(|cpu_logits - npu_logits|) | - |
| mean_abs_diff | mean(|cpu_logits - npu_logits|) | - |
| max_rel_diff | max(|cpu_logits - npu_logits| / |cpu_logits|) | - |
| mean_rel_diff | mean(|cpu_logits - npu_logits| / |cpu_logits|) | < 1% |
| top1_match_rate | sum(argmax(cpu) == argmax(npu)) / N * 100 | 100% |

**实测结果（Ascend910B4 + CANN 8.5.1）**：

| 指标 | 数值 |
| --- | --- |
| `max_abs_diff` | `7.54e-04` |
| `mean_abs_diff` | `4.52e-04` |
| `max_rel_diff` | `1.26%` |
| `mean_rel_diff` | `0.76%` |
| `top1_match_rate` | `100%` |

**通过标准**：Mean Relative Diff < 1%，Top-1 Match Rate = 100%。

### 7. 性能基准测试

```bash
python3 scripts/benchmark.py
```

基准测试脚本（`scripts/benchmark.py`）测量 CPU 与 NPU 在不同 batch size 下的推理延迟：

| 指标 | CPU (batch=1) | NPU (batch=1) | NPU (batch=4) | NPU (batch=8) |
| --- | --- | --- | --- | --- |
| `mean_latency_ms` | `1761.40` | `12.42` | `12.03` | `11.91` |
| `median_latency_ms` | `1760.89` | `12.21` | `11.72` | `11.81` |
| `min_latency_ms` | `1751.59` | `11.83` | `11.53` | `11.66` |
| `max_latency_ms` | `1775.22` | `15.38` | `16.81` | `12.62` |
| `std_ms` | — | `0.60` | `1.11` | `0.27` |

加速比（相对 CPU）：batch=1 约 **142x**，batch=4 约 **568x**（单样本等效）。

**通过标准**：NPU 单图延迟 < 20 ms（二次运行）。

### 8. 结果汇总与验收

```bash
# 查看 benchmark 报告
cat benchmark_results/benchmark_report.json

# 查看精度对比日志
python3 scripts/compare_accuracy.py
```

**验收检查清单**：

- [ ] npu-smi info 显示设备正常
- [ ] import torch_npu 无报错
- [ ] infer_npu.py 输出 logits 且无 NPU 报错
- [ ] 精度对比 mean_rel_diff < 1%，top1_match_rate = 100%
- [ ] 性能测试 NPU 单图延迟 < 20 ms（二次运行）
- [ ] evals.json 记录完整的环境检查和验证结论

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|------|---------|-------------------------------|---------|
| NPU 设备不可用 | npu-smi info 失败或无 Ascend910 | fallback 到 CPU dry-run 模式，禁止写入真实 NPU 结论 | evals.json 记录 npu_unavailable |
| CANN 环境缺失 | ASCEND_HOME 或 LD_LIBRARY_PATH 未设置 | 提示加载 CANN 环境脚本，retry 一次环境检查 | 环境检查日志输出 set_env.sh |
| torch_npu 导入失败 | import torch_npu 抛出 ImportError | 提示省略版本安装 `pip install torch_npu`，retry 一次 | evals.json 记录环境错误 |
| torch_npu 版本不兼容 | import 后版本不匹配 | 指定兼容版本 `pip install torch_npu==2.x.x` | 版本日志输出 |
| 模型下载失败 | ModelScope 网络超时或模型不存在 | retry 2 次，切换 HuggingFace 镜像或使用本地缓存 | results.tsv 记录权重来源 |
| transfer_to_npu 未注入 | infer_npu.py 中 import 顺序错误 | 将 `import torch_npu` / `transfer_to_npu` 移至文件顶部，retry | 推理日志记录注入状态 |
| NPU 算子编译失败 | 首次运行编译器报错 | 正常现象，等待编译完成，缓存自动落盘 | 日志记录编译耗时 |
| NPU OOM | 显存不足异常 | 执行 `gc.collect()` 和 `torch.npu.empty_cache()` 后 retry | evals.json 记录 retry 次数 |
| CPU/NPU 推理结果为空 | logits 输出全部为零或 NaN | 检查随机种子和模型加载，重新推理 | 推理日志记录错误原因 |
| 精度不达标 | mean_rel_diff >= 1% 或 top1_match_rate < 100% | 保留 CPU/NPU logits，标记 failed，禁止发布通过结论 | 精度对比表显示失败指标 |
| benchmark 结果异常 | benchmark 延迟 > 100 ms | 检查 NPU 设备负载，确保无其他进程占用 | benchmark 日志记录异常 |

## 关键适配点

| 特征 | ViT 值 | 适配说明 |
|------|--------|----------|
| 框架 | PyTorch + Transformers | 标准架构，transfer_to_npu 自动迁移即可 |
| 注意力类型 | 双向自注意力 | 无需修改 mask 逻辑 |
| 位置编码 | 1D Learnable Position Embedding | 无 Rotary/Cos-Sin 依赖 |
| 激活函数 | GELU | torch.nn.GELU 原生支持 |
| CUDA 专有 API | 无 | 不涉及 `get_device_properties`、`flash_attn`、`.cu` 等 |
| 分类头 | 随机初始化（预训练权重不含） | 直接推理的分类结果无实际语义 |
| 输入归一化 | `(0.5, 0.5, 0.5)` mean/std | ViTImageProcessor 自动处理 |

**唯一注意点**：`transfer_to_npu` 必须在所有其他 `import` 之前注入，否则 `.cuda()` 等 API 不会被自动替换。

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| scripts/infer_cpu.py | CPU Baseline 推理脚本，保存 logits 到 cpu_output.pkl |
| scripts/infer_npu.py | NPU 推理脚本，通过 transfer_to_npu 自动迁移，保存 logits 到 npu_output.pkl |
| scripts/compare_accuracy.py | CPU/NPU 精度对比脚本，计算误差和匹配率指标 |
| scripts/benchmark.py | 综合性能基准测试脚本，生成 benchmark_report.json |
| scripts/download_model.py | ModelScope 模型权重下载脚本 |
| benchmark_results/benchmark_report.json | 性能基准测试报告（JSON） |
| evals.json | 结构化保存环境检查、重试、验证和精度结论 |
| test-prompts.json | 提供重复评估本 Skill 的测试提示 |

## 使用约束

1. **transfer_to_npu 注入顺序**：NPU 推理脚本中必须将 `import torch_npu` 和 `from torch_npu.contrib import transfer_to_npu` 放在所有其他 import 之前，否则 `.cuda()` 等 API 不会被自动替换为 `.npu()`。
2. **随机种子一致**：CPU 和 NPU 推理脚本必须使用相同的随机种子（`torch.manual_seed(42)`、`np.random.seed(42)`），否则精度对比结果不可信。
3. **权重来源**：优先使用 ModelScope 下载模型权重。若下载失败，切换至 HuggingFace 镜像（`hf-mirror.com`）或本地缓存。
4. **NPU 资源释放**：多个推理任务之间建议执行 `torch.npu.empty_cache()` 释放显存。
5. **精度标准**：NPU 与 CPU 推理结果 mean_rel_diff < 1% 即为通过。若不达标，保留推理日志并禁止发布通过结论。
6. **数据真实性**：精度结论和 benchmark 数据必须来自真实测试，严禁编写无数据支撑的总结。
7. **首次算子编译**：初次在 NPU 上运行推理时，AICore kernel 编译需要额外时间（约 1-3 分钟），属正常现象，编译缓存会自动落盘，二次运行无需编译。
