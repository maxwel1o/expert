---
name: webssl-mae-npu-deploy
description: >
  WebSSL MAE (300M / 700M / 1B) 视觉表征模型在昇腾 NPU 上的推理部署 Skill。
  涵盖环境准备、权重下载、transfer_to_npu 自动迁移、NPU 推理验证、
  CPU 基线精度对比的全流程。支持 facebook/webssl-mae300m-full2b-224、
  webssl-mae700m-full2b-224、webssl-mae1b-full2b-224 三个尺度模型。
  当用户提到 WebSSL MAE 昇腾部署、MAE NPU 推理、视觉 encoder NPU 时触发。
metadata:
  short-description: WebSSL MAE 昇腾 NPU 推理部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, webssl, mae, vit, vision, pytorch, inference, feature-extraction]
---

# WebSSL MAE 昇腾 NPU 推理部署与精度验证 Skill

本 Skill 提供 WebSSL MAE 系列模型（300M / 700M / 1B）在华为昇腾 NPU 上的
标准化推理部署与精度验证流程。三个模型架构相同，仅参数量不同，
共用同一套适配脚本。

## 前置条件

### 硬件与软件需求

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（300M~1.1GB） |

### 磁盘空间

1. 确认磁盘空间 >= 10GB（模型缓存约 1.1GB * 3 = 3.3GB + 临时空间）
2. 推理日志与精度报告约需 1GB

### 前置条件确认

- [确认] 使用 `npu-smi info` 确认 NPU 驱动已加载，至少一张卡状态为 Normal
- [确认] 使用 `python3 --version` 确认 Python 版本 >= 3.9
- [确认] 使用 `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` 确认 CANN 版本 >= 8.0

## 支持模型

| 模型 | HuggingFace ID | 参数量 | 隐藏维度 | 输出 Token 数 |
|------|---------------|--------|---------|--------------|
| MAE 300M | facebook/webssl-mae300m-full2b-224 | 303M | 1024 | 197 |
| MAE 700M | facebook/webssl-mae700m-full2b-224 | 631M | 1280 | 257 |
| MAE 1B | facebook/webssl-mae1b-full2b-224 | 1135M | 1536 | 257 |

## 工作流程

### 流程总览

```
0. 环境初始化
→ 用户确认：NPU 状态正常
→ 1. 安装依赖（torch_npu + transformers + pillow + numpy）
→ 用户确认：依赖安装成功
→ 2. NPU 验证
→ 用户确认：NPU 可用
→ 3. 权重下载
→ 4. NPU 推理验证（inference.py）
→ 用户确认：推理结果合理
→ 5. CPU 基线精度对比（verify_accuracy.py）
→ 6. 验收确认
```

按以下各节顺序执行，每步完成后再进入下一步。**每步执行后检查对应通过标准**，不通过则排查修复或回退至上一步。

---

## 0. 环境初始化

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU（先用 npu-smi info 查看各卡占用，选空闲卡）
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 设置 HuggingFace 镜像（中国大陆推荐）
export HF_ENDPOINT=https://hf-mirror.com
```

### 错误处理

- 若 `source set_env.sh` 失败：确认 Ascend Toolkit 已安装，路径为 `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- 若 `npu-smi info` 报错：确认驱动已安装，执行 `npu-smi -v` 查看驱动版本
- 若所有卡均被占用：等待其他任务释放，或使用 `npu-smi reset -t 0`（需 root 权限）重置设备
- 若 `HF_ENDPOINT` 设置后仍无法下载：检查网络连通性 `curl -I https://hf-mirror.com`

### Checkpoint：环境初始化通过标准

- [确认] `npu-smi info` 显示至少一张卡状态为 Normal
- [确认] `echo $ASCEND_RT_VISIBLE_DEVICES` 输出正确卡号
- [确认] 网络可访问 hf-mirror.com 或 huggingface.co

---

## 1. 安装依赖

```bash
pip install torch torch_npu transformers pillow numpy
```

安装完成后版本应匹配：
- `torch` >= 2.0
- `torch_npu` 与 `torch` 主版本一致
- `transformers` >= 4.40

### 错误处理

- 若 `pip install torch_npu` 失败：确认 CANN 版本与 torch_npu 版本匹配
- 若 pip 超时：使用华为云镜像 `pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ torch torch_npu ...`
- 若版本冲突：创建独立虚拟环境 `python3 -m venv webssl-env && source webssl-env/bin/activate` 后重试
- 若 `ImportError: libcudart.so` 等 CUDA 库缺失：transfer_to_npu 模式不需要 CUDA，确认已加载 CANN 环境

### Checkpoint：安装验证

- [确认] `python3 -c "import torch; import torch_npu; print(torch.__version__, torch_npu.__version__)"` 无报错
- [确认] `python3 -c "import transformers; import PIL; import numpy"` 无报错

---

## 2. NPU 基础验证

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU device:', torch.npu.get_device_name(0))
"
```

**通过标准**：`NPU available: True` 且输出设备名称无报错。

### 错误处理

- 若 `torch.npu.is_available()` 返回 False：检查 CANN 环境是否加载、`ASCEND_RT_VISIBLE_DEVICES` 是否设置、驱动是否安装
- 若 `ImportError: No module named 'torch_npu'`：重新安装 `pip install torch_npu`，确认 torch 和 torch_npu 版本匹配
- 若出现 `Segmentation fault`：尝试升级 CANN 至 8.5.1 及以上

### Checkpoint：NPU 验证确认

- [确认] `NPU available: True`
- [确认] 设备名称为 Ascend910 系列

---

## 3. 权重下载

脚本首次运行时会自动从 HuggingFace Hub 下载权重到本地缓存目录。

如需手动预下载：

```bash
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoImageProcessor, AutoModel
for model_name in [
    'facebook/webssl-mae300m-full2b-224',
    'facebook/webssl-mae700m-full2b-224',
    'facebook/webssl-mae1b-full2b-224',
]:
    AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print(f'Downloaded: {model_name}')
"
```

### 错误处理

- 若下载超时/失败：检查 `HF_ENDPOINT` 是否正确，尝试直接设置 `export HF_ENDPOINT=https://huggingface.co` 使用官方源
- 若磁盘空间不足：清理缓存 `rm -rf ~/.cache/huggingface/hub` 后重试
- 若 `trust_remote_code=True` 安全警告：确认模型来源为 facebook 官方仓库，可安全放行

### Checkpoint：权重下载确认

- [确认] 三个模型的权重均已成功下载
- [确认] 缓存目录中可找到对应模型文件（`~/.cache/huggingface/hub/`）

---

## 4. NPU 推理验证

### 4.1 模型加载

使用 `transfer_to_npu` 自动将模型迁移到 NPU：

1. 导入 `from torch_npu.contrib import transfer_to_npu`
2. 使用 `AutoImageProcessor` 加载图像处理器
3. 使用 `AutoModel` 加载模型权重
4. 调用 `model.cuda()` 自动映射到 NPU（transfer_to_npu 拦截 .cuda() 调用）
5. 将输入张量同样 `.cuda()` 映射到 NPU

### 4.2 推理脚本

<!-- EMBED:scripts/inference.py -->

脚本文件位于 `scripts/inference.py`，支持通过环境变量 `MODEL_NAME` 切换模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_NAME=facebook/webssl-mae300m-full2b-224

python3 inference.py --device npu --runs 10
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `npu` | 推理设备：`npu` / `cpu` / `cuda` |
| `--warmup` | `3` | warmup 轮数 |
| `--runs` | `10` | 正式计时轮数 |
| `--cache-dir` | `model_cache` | 权重缓存目录 |

### 4.3 核心代码逻辑

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained(
    "facebook/webssl-mae300m-full2b-224",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    "facebook/webssl-mae300m-full2b-224",
    trust_remote_code=True,
)
model.eval()
model = model.cuda()  # transfer_to_npu 自动映射到 NPU

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
```

### 4.4 预期性能（Ascend910B4, CANN 8.5.1）

| 模型 | 平均延迟 | NPU 内存 |
|------|---------|---------|
| MAE 300M | ~22 ms | ~1164 MB |
| MAE 700M | ~31 ms | ~2479 MB |
| MAE 1B | ~38 ms | ~4340 MB |

### 错误处理

- 若 `transfer_to_npu` 报 `AssertionError`：确认已执行 `import torch_npu`，transfer_to_npu 依赖 torch_npu 初始化
- 若推理时 OOM：减少 batch size（默认 1），或切换更小模型
- 若 `model.cuda()` 不生效：确认为 `torch_npu.contrib.transfer_to_npu` 而非标准 `torch`，且 import 顺序正确
- 若输出 NaN：确认输入图像正常，dtype 匹配

### Checkpoint：推理确认

- [确认] 推理脚本成功输出 `last_hidden_state` 且 shape 正确
- [确认] 延迟在预期范围内
- [确认] NPU 内存占用合理且无泄漏

---

## 5. CPU 基线精度对比

### 5.1 精度指标说明

脚本对比 NPU 与 CPU 的 `last_hidden_state` 输出，计算以下指标：

- **L2 relative error**：向量级差异标准，作为主要通过指标
- **Norm relative error**：输出范数相对差异
- **Cosine similarity**：方向一致性
- **Max / Mean absolute error**：元素级绝对误差

### 5.2 精度验证脚本

<!-- EMBED:scripts/verify_accuracy.py -->

脚本文件位于 `scripts/verify_accuracy.py`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_NAME=facebook/webssl-mae300m-full2b-224

python3 verify_accuracy.py --threshold 1.0
```

### 5.3 验证结果参考

| 模型 | L2 relative error | Norm relative error | Cosine similarity | 结果 |
|------|------------------|---------------------|-------------------|------|
| MAE 300M | 0.5283% | 0.0021% | 1.000001 | PASS |
| MAE 700M | 0.6436% | 0.1047% | 1.000018 | PASS |
| MAE 1B | 0.7152% | 0.0385% | 1.000005 | PASS |

**通过标准**：L2 relative error < 1.0% 且 Cosine similarity > 0.999。

### 错误处理

- 若 CPU 推理 OOM：1B 模型在 CPU 上可能需要 32GB+ 内存，尝试 `--skip-cpu` 跳过 CPU 对比
- 若 L2 error 超过 1%：检查 bf16/float32 精度是否一致，确认 `model.eval()` 已调用
- 若 Cosine similarity 异常低：检查 dropout/training 模式是否关闭，seed 是否一致
- 若 NPU 推理结果与 CPU 偏差大：检查 `transfer_to_npu` 是否正确映射算子

### Checkpoint：精度验证确认

- [确认] 每个模型的 L2 relative error < 1%
- [确认] Cosine similarity > 0.999
- [确认] 全部报告 PASSED（或使用了 --skip-cpu 确认 NPU 推理功能正常）

---

## 6. 验收确认

### 检查清单

完成以下检查清单即为部署成功：

1. [ ] `npu-smi info` 显示设备正常
2. [ ] `import torch_npu` 无报错
3. [ ] `inference.py` 在 NPU 上正常输出特征向量
4. [ ] `verify_accuracy.py` 精度验证通过（L2 relative error < 1%）

### 回退策略

若任一检查项未通过：
1. 记录失败项和错误信息到日志文件
2. 根据失败项定位到对应的步骤
3. 执行该步骤的故障排除流程
4. 修复后重新运行验收确认

### 最终输出

部署成功后的交付物：
1. 精度验证报告（`accuracy_report.json`）
2. 推理性能日志（`benchmark_logs/` 目录）
3. 模型配置快照

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 权重下载超时 | HuggingFace 官方网络不通 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `transfer_to_npu` 警告 | 正常行为 | 不影响推理，可忽略 |
| NPU 内存不足 | 模型过大或前序任务未释放 | `torch.npu.empty_cache()` 后重试 |
| 多卡抢占冲突 | 默认使用 0 号卡 | `npu-smi info` 选空闲卡，设置 `ASCEND_RT_VISIBLE_DEVICES` |
| import torch_npu 后 segfault | CANN/torch 版本不匹配 | 升级 CANN 至 8.5.1，安装匹配的 torch_npu |
| `model.cuda()` 未映射到 NPU | 未 import transfer_to_npu | 添加 `from torch_npu.contrib import transfer_to_npu` |
| L2 error > 1% | 精度对齐问题 | 关闭 dropout，确认 eval 模式，seed 一致 |

---

## 附录：资源文件

| 文件 | 用途 | 路径 |
|------|------|------|
| 推理脚本 | 模型加载与推理基准测试 | `scripts/inference.py` |
| 精度验证脚本 | NPU vs CPU 精度对比 | `scripts/verify_accuracy.py` |
| 测试用例 | 验证 prompt 与预期结果 | `test-prompts.json` |

---

## 附录：模型配置速查

| 特征 | MAE 300M | MAE 700M | MAE 1B |
|------|---------|---------|--------|
| hidden_size | 1024 | 1280 | 1536 |
| num_hidden_layers | 24 | 32 | 32 |
| num_attention_heads | 16 | 16 | 24 |
| intermediate_size | 4096 | 5120 | 6144 |
| patch_size | 16 | 14 | 14 |
| image_size | 224 | 224 | 224 |
| model_type | vit | vit | vit |
| mask_ratio | 0.0 | 0.0 | 0.0 |
