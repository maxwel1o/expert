---
name: timm-mixnet-npu
description: "timm MixNet 系列 3 个模型 (tf_mixnet_s/m/l.in1k) 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证、终端截图生成与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：MixNet NPU 部署、timm mixnet 昇腾推理、精度对比、npu-smi。"
---

# timm MixNet NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 timm MixNet 系列图像分类模型，完成推理、精度验证，并发布模型仓库。

## 概述

本 Skill 提供 timm MixNet 系列 3 个模型在昇腾 NPU 上的自动化部署、推理测试、CPU/NPU 精度对比、终端截图生成和模型仓库发布能力。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend 910B/910) |
| 模型数量 | 3 个（tf_mixnet_s.in1k, tf_mixnet_m.in1k, tf_mixnet_l.in1k） |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 0.9+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| CANN 版本 | 8.5.1+ |

### 支持的模型

| 模型名称 | 架构 | 参数量 | 输入尺寸 | GitCode 仓库 |
|----------|------|--------:|---------|-------------|
| `tf_mixnet_s.in1k` | MixNet Small | ~2.0M | 3×224×224 | [tf_mixnet_s.in1k-npu](https://gitcode.com/m0_74196153/tf_mixnet_s.in1k-npu) |
| `tf_mixnet_m.in1k` | MixNet Medium | ~3.4M | 3×224×224 | [tf_mixnet_m.in1k-npu](https://gitcode.com/m0_74196153/tf_mixnet_m.in1k-npu) |
| `tf_mixnet_l.in1k` | MixNet Large | ~5.0M | 3×224×224 | [tf_mixnet_l.in1k-npu](https://gitcode.com/m0_74196153/tf_mixnet_l.in1k-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.10+ 环境，昇腾 NPU 驱动，CANN 8.5.1+。

**动作**:
1. 检查 Python 版本：

```bash
python3 --version
```

2. 检测 NPU 状态：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch>=2.0.0 torchvision>=0.15.0 timm>=0.9.0 safetensors>=0.4.0 torch_npu
```

**输出**: NPU 可用状态，依赖已安装。

### Step 2: 模型选择与数据准备

**输入**: 模型名称（tf_mixnet_s.in1k / tf_mixnet_m.in1k / tf_mixnet_l.in1k），权重文件目录。

**动作**:
4. 校验模型名是否在支持列表内。
5. 检查权重文件，确认存在 `model.safetensors` 或 `pytorch_model.bin`。

```bash
ls -la ${checkpoint_dir}/model.safetensors 2>/dev/null || ls -la ${checkpoint_dir}/pytorch_model.bin 2>/dev/null
```

6. 生成随机 RGB 测试张量（3×224×224），模拟输入数据。

**输出**: 模型名和测试数据已就绪。

### Step 3: CPU 基线推理

**输入**: 模型名、设备类型 `cpu`。

**动作**:
7. 执行 CPU 推理：

```bash
python3 scripts/inference.py \
  --model tf_mixnet_s.in1k \
  --checkpoint-dir /path/to/weights \
  --device cpu
```

8. 验证推理输出包含 top-5 预测结果和 logits 张量。

**输出**: CPU 推理结果 `output_cpu.json`（top-5 类别和概率），`logits_cpu.pt`（全量 logits）。

### Step 4: NPU 推理

**输入**: 模型名、设备类型 `npu`。

**动作**:
9. 检查 NPU 可用性，不可用时自动回退 CPU。

```python
import torch
if not torch.npu.is_available():
    print("WARNING: NPU not available, falling back to CPU")
    device = "cpu"
```

10. 执行 NPU 推理：

```bash
python3 scripts/inference.py \
  --model tf_mixnet_s.in1k \
  --checkpoint-dir /path/to/weights \
  --device npu
```

11. 处理 OOM：捕获异常后释放缓存再重试。

```python
try:
    output = model(input_tensor)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        torch.npu.empty_cache()
        gc.collect()
        output = model(input_tensor)
```

**输出**: NPU 推理结果 `output_npu.json`，`logits_npu.pt`，标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU、NPU 推理结果的 logits 张量。

**动作**:
12. 执行精度对比脚本：

```bash
python3 scripts/compare_cpu_npu.py
```

13. 验证以下指标均通过：

| 指标 | 通过标准 |
|------|---------|
| MAE/Mean(\|CPU\|) | < 1% |
| Cosine Similarity | > 0.999 |
| Top-1 Match | CPU 与 NPU 一致 |
| Top-5 Overlap | CPU 与 NPU 一致 |

14. 若误差超标则记录偏差明细并中止发布。

**输出**: 精度对比报告 `comparison_results.json`（含 MAE、MSE、Cosine Similarity、Top-1/Top-5 匹配）。

### Step 6: 串行批量执行

**输入**: 剩余模型列表。

**动作**:
15. 使用 `run_all.sh` 对每个模型串行执行 Step 3-5，防止 NPU 显存溢出：

```bash
bash scripts/run_all.sh /path/to/models/dir
```

16. 每完成一个模型释放资源：

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

17. 结果归档到各模型的 results/ 目录。

**输出**: 每个模型的 CPU/NPU 推理结果和精度报告。

### Step 7: 生成终端截图

**输入**: inference.py 运行时录制的终端输出文本。

**动作**:
18. 使用截图工具生成终端输出截图：

```bash
python3 /path/to/terminal_screenshot.py \
  --input terminal_output.txt \
  --output terminal_screenshot.png
```

**输出**: PNG 格式的终端截图 `terminal_screenshot.png`。

### Step 8: 调用 GitCode API 提交模型仓库

**输入**: ATOMGIT_USER_TOKEN 环境变量，模型仓库数据。

**动作**:
19. 创建 GitCode 模型仓库：

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "name": "tf_mixnet_s.in1k-npu",
    "description": "tf_mixnet_s.in1k adapted for Ascend NPU",
    "visibility": "public",
    "repository_type": "model"
  }'
```

20. 推送代码到模型仓库：

```bash
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/<repo-name>.git
git branch -M main
git push -u origin main
```

**输出**: GitCode 模型仓库已创建并推送成功。

## 执行检查点与用户确认

每个关键步骤设有检查点 checkpoint，用户必须暂停确认后才能继续。以下 7 个检查点覆盖从环境准备到仓库发布的完整流程：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测后 | NPU 设备是否可用，CANN 版本是否正确 | 提示安装 torch_npu 或检查驱动 |
| 2 | CP-2: 模型确认检查点 | 模型选择后 | 选定的模型名称是否正确 | 返回重新选择 |
| 3 | CP-3: CPU 基线检查点 | CPU 推理后 | CPU 推理 top-5 是否合理 | 检查模型加载 |
| 4 | CP-4: NPU 推理检查点 | NPU 推理后 | NPU 推理是否成功 | 检查 NPU 显存并重试 |
| 5 | CP-5: 精度确认检查点 | 精度对比后 | 误差是否 < 1%，余弦相似度是否 > 0.999 | 检查精度原因后重试 |
| 6 | CP-6: 截图确认检查点 | 截图生成后 | 终端截图内容是否完整 | 重新生成截图 |
| 7 | CP-7: 发布前审批检查点 | 推送仓库前 | 仓库名和 README 是否正确 | 修改后重新申请用户确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 回退 CPU 推理，标记 fallback | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | NPU 推理时报 `out of memory` | 释放缓存后重试，最多重试 3 次 | 调整配置或释放进程 |
| 模型加载异常 | `create_model()` 或 `load_state_dict()` 异常 | 打印错误堆栈，跳过该模型 | 修正模型名或检查权重文件 |
| 权重文件缺失 | `model.safetensors` / `pytorch_model.bin` 不存在 | 打印详细错误路径和目录列表 | 下载权重或修正 checkpoint_dir |
| 网络超时 | pip/curl 下载失败 | 重试最多 3 次，切换镜像源 | 切换到清华镜像源 |
| 精度超标 | MAE/Mean(\|CPU\|) >= 1% 或 Cosine Similarity <= 0.999 | 记录偏差明细，中止发布流程 | 检查推理脚本一致性和权重版本 |
| API 调用失败 | GitCode API 返回非 200 状态码 | 打印错误状态码和响应体 | 检查 ATOMGIT_USER_TOKEN 权限 |
| 用户确认超时 | checkpoint 等待超过 300 秒 | 暂停流程并通知用户，保留当前进度 | 用户恢复后从当前检查点继续 |
| 权重加载多 key | checkpoint 含 model 无权重的额外 key | 过滤 unexpected keys 后继续加载 | 自动处理，仅加载匹配权重 |
| 截图生成失败 | terminal_screenshot.py 执行异常 | 跳过截图步骤，继续执行 | 手动执行截图脚本 |
| Git 推送冲突 | 远程仓库已存在非空内容 | 打印冲突详情，建议强制推送 | 确认后使用 git push -f |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理脚本，支持 tf_mixnet_s/m/l.in1k，输出 top-5 结果和 logits 持久化 |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本，计算 MAE、MSE、Cosine Similarity、Top-1/Top-5 匹配，输出 comparison_results.json |
| `scripts/run_all.sh` | 串行批量执行脚本，依次处理所有模型的 CPU/NPU 推理和精度对比，释放资源 |
| `examples/example_inference.sh` | 单模型 NPU 推理示例，含 ModelScope 权重下载 |
| `skill.json` | 技能元数据（模型列表、输入参数、输出产物、依赖版本、硬件要求） |
| `test-prompts.json` | 本技能测试提示词（含 NPU 回退场景和批量执行场景） |
| `comparison_results.json` | CPU/NPU 精度对比结果（运行后生成，含 MAE、Cosine Similarity、Top-1/Top-5 指标） |
| `output_cpu.json` | CPU 推理结果 evals.json（运行后生成） |
| `output_npu.json` | NPU 推理结果 evals.json（运行后生成） |
| `logits_cpu.pt` / `logits_npu.pt` | CPU 和 NPU 的完整 logits 张量（运行后生成） |
| `references/` | 精度验证参考指标和 benchmark（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 是 | tf_mixnet_s.in1k | 模型名称（支持列表中的 3 个模型） |
| `checkpoint_dir` | string | 是 | - | 模型权重文件所在目录（包含 model.safetensors） |
| `device` | string | 否 | npu | 推理设备: cpu 或 npu |
| `seed` | int | 否 | 42 | 随机种子 |

## 验证结果摘要

| 模型 | NPU 推理 | CPU/NPU 误差 | Top-1 匹配 | NPU 加速比 |
| --- | :---: | :---: | :---: | :---: |
| tf_mixnet_s.in1k | OK 通过 | 0.158% | OK 一致 | 4.12x |
| tf_mixnet_m.in1k | OK 通过 | 0.054% | OK 一致 | 4.48x |
| tf_mixnet_l.in1k | OK 通过 | 0.053% | OK 一致 | 5.51x |

## 使用约束

1. 使用 ModelScope 或 hf-mirror.com 下载模型权重。
2. 精度验证通过前不提交模型仓库。
3. 串行执行避免 NPU 显存溢出（NPU OOM）。
4. 首次使用需配置 `ATOMGIT_USER_TOKEN` 环境变量用于 GitCode API 调用。
