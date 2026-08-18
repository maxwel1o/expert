---
name: volo-npu-deploy
description: "VOLO (Vision Outlooker) 系列 11 个模型 (volo_d1~volo_d5) 在昇腾 NPU 上的完整部署、推理验证、CPU/NPU 精度对比与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：VOLO NPU 部署、volo 昇腾推理、精度对比、npu-smi。"
---

# VOLO (Vision Outlooker) NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 VOLO 系列图像分类模型，完成推理、精度验证，并发布模型仓库。

## 概述

本 Skill 提供 VOLO (Vision Outlooker) 系列 11 个模型在昇腾 NPU 上的自动化部署、推理测试、CPU/NPU 精度对比、README 生成、终端截图生成和模型仓库发布能力。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend 910B/910) |
| 模型数量 | 11 个（volo_d1 ~ volo_d5，含 224/384/448/512 分辨率变体） |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 1.0+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% (max prob error) |
| CANN 版本 | 8.0+ |

### 支持的模型

| 模型名称 | 输入尺寸 | 参数量 | 任务类型 | GitCode 仓库 |
|---------|---------|--------|---------|-------------|
| volo_d1_224.sail_in1k | 224x224 | 27M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d1_224-npu) |
| volo_d1_384.sail_in1k | 384x384 | 27M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d1_384-npu) |
| volo_d2_224.sail_in1k | 224x224 | 59M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d2_224-npu) |
| volo_d2_384.sail_in1k | 384x384 | 59M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d2_384-npu) |
| volo_d3_224.sail_in1k | 224x224 | 87M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d3_224-npu) |
| volo_d3_448.sail_in1k | 448x448 | 87M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d3_448-npu) |
| volo_d4_224.sail_in1k | 224x224 | 116M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d4_224-npu) |
| volo_d4_448.sail_in1k | 448x448 | 116M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d4_448-npu) |
| volo_d5_224.sail_in1k | 224x224 | 297M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d5_224-npu) |
| volo_d5_448.sail_in1k | 448x448 | 297M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d5_448-npu) |
| volo_d5_512.sail_in1k | 512x512 | 297M | 图像分类 | [repo](https://gitcode.com/m0_74196153/volo_d5_512-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.9+ 环境，昇腾 NPU 驱动，CANN 8.0+。

**动作**:
1. 检查 Python 版本并确认满足 3.9-3.11 要求：

```bash
python3 --version
```

2. 检测 NPU 状态，确认 npu-smi 可正常输出：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 配置环境变量，选择空闲 NPU 卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torch_npu timm Pillow numpy safetensors scipy modelscope
```

**输出**: NPU 可用状态确认，依赖已安装。

### Step 2: 模型选择与数据准备

**输入**: 模型名称（从支持列表中选择）。

**动作**:
4. 校验模型名是否在 11 个支持模型列表中，不在列表中则报错并提示可用模型。
5. 检查测试图片是否存在，缺失时自动生成 224x224 RGB 测试图：

```bash
python3 -c "from PIL import Image; Image.new('RGB', (224,224), color=(100,100,200)).save('/tmp/test_volo.jpg')"
```

**输出**: 模型名已校验，测试图片已就绪。

### Step 3: CPU 基线推理

**输入**: 模型名、设备类型 `cpu`。

**动作**:
6. 配置 ModelScope 环境变量，从本地缓存加载模型权重：

```bash
export MODELSCOPE_CACHE=/opt/atomgit/.cache/modelscope
```

7. 执行 CPU 推理，输出 top-5 分类结果和推理耗时：

```bash
python3 scripts/inference.py --model volo_d1_224.sail_in1k --device cpu --num_runs 10
```

8. 验证推理输出包含 top-5 类别和概率。
9. 结果保存至 `results_cpu.json`。

**输出**: CPU 推理结果（top-5 类别和概率、平均耗时）。

### Step 4: NPU 推理

**输入**: 模型名、设备类型 `npu`。

**动作**:
10. 检查 NPU 可用性，若 `torch.npu.is_available()` 为 False 则自动回退 CPU 并标记 fallback。
11. 执行 NPU 推理：

```bash
python3 scripts/inference.py --model volo_d1_224.sail_in1k --device npu --num_runs 10
```

12. 若遇 OOM 异常：调用 `torch.npu.empty_cache()` 释放缓存后重试，最多 3 次 retry。
13. 结果保存至 `results_npu.json`。

**输出**: NPU 推理结果（top-5 类别和概率、平均耗时），标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU 和 NPU 推理结果。

**动作**:
14. 执行精度对比脚本：

```bash
python3 scripts/compare_cpu_npu.py --model volo_d1_224.sail_in1k --num_runs 10
```

15. 自动计算指标：Logits 最大/平均绝对误差、余弦相似度、Softmax 概率最大/平均差异、Top-1 一致性、Top-5 重合率。
16. 若 `max_prob_error < 1%` 标记通过（PASSED），否则标记失败（FAILED）并记录偏差明细到 `compare_results.json`。
17. 结果保存至 `compare_results.json`。

**输出**: 精度对比报告（误差指标表格、PASSED/FAILED 结论）。

### Step 6: 串行批量执行与资源释放

**输入**: 剩余模型列表（11 个模型串行执行防止 NPU OOM）。

**动作**:
18. 对每个模型依次执行 Step 3-5 的推理和精度对比动作。
19. 每完成一个模型后立即释放资源：

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

20. 将全部模型的推理和精度结果汇总到 `references/evals.json`。

**输出**: 全部 11 个模型的推理结果、精度报告和汇总 evals.json。

### Step 7: README 生成与仓库发布

**输入**: 精度通过（PASSED）的模型结果。

**动作**:
21. 为每个模型生成完整的 README.md 文档（含模型介绍、NPU 适配说明、推理命令、精度测试结果表格和性能对比）。
22. 使用 GitCode API 创建 model 类型仓库：

```bash
curl -X POST --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "volo_d1_224-npu", "private": false, "repository_type": "model"}' \
  "https://api.gitcode.com/api/v5/user/repos"
```

23. 推送 inference.py、compare_cpu_npu.py、requirements.txt、readme.md 和 compare_results.json 到仓库 main 分支。

**输出**: 每个模型的 GitCode model 仓库已发布。

## 执行检查点与用户确认

每个关键步骤设有检查点 checkpoint，用户必须暂停确认后才能继续。以下 7 个检查点覆盖从环境准备到仓库发布的完整流程：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测后 | NPU 设备是否可用，CANN 版本是否正确 | 提示安装 torch_npu 或检查驱动后重新确认 |
| 2 | CP-2: 模型确认检查点 | 模型选择后 | 选定的模型名称是否正确 | 返回 Step 2 重新选择模型 |
| 3 | CP-3: CPU 基线检查点 | CPU 推理后 | CPU 推理 top-5 结果是否合理 | 检查模型权重加载路径和 modelscope 缓存 |
| 4 | CP-4: NPU 推理检查点 | NPU 推理后 | NPU 推理是否成功，是否触发 fallback | 检查 NPU 显存占用并释放后重试 |
| 5 | CP-5: 精度确认检查点 | 精度对比后 | 误差是否 < 1% | 检查 CPU 和 NPU 推理脚本是否一致后重试 |
| 6 | CP-6: 批量进度检查点 | 每 3 个模型完成后 | 当前批量进度是否正常 | 查看 evals.json 确认失败原因后继续 |
| 7 | CP-7: 发布前审批检查点 | 推送仓库前 | 仓库名和 README 内容是否正确 | 修改后重新申请用户确认 |

## 异常处理与回滚策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|------|---------|---------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 回退 CPU 推理，输出标记 fallback=true | 安装 torch_npu 或检查驱动，重新执行 |
| NPU 显存 OOM | 推理时报 OutOfMemoryError | 释放 torch.npu.empty_cache() 后重试，最多 3 次 retry | 调整 batch_size=1 或串行执行模型 |
| 模型权重加载失败 | safetensors/bin 文件不存在或格式错误 | 打印完整错误堆栈，跳过该模型 | 从 ModelScope 重新 snapshot_download 下载 |
| 权重 module. 前缀不匹配 | load_state_dict 报错 missing keys | 自动去除 `module.` 前缀，使用 strict=False 恢复 | 使用 `{k.replace("module.",""):v}` 处理 |
| 网络超时 | pip install 或 curl 下载超时 | 自动重试最多 3 次，切换镜像源 | 使用清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple` |
| HuggingFace 不可达 | modelscope 下载失败或连接超时 | 回退本地缓存加载或使用 `pretrained=False` | 切换 hf-mirror.com 或直接使用 ModelScope 国内源 |
| 精度超标 | max_prob_error >= 1% | 记录偏差明细到 compare_results.json，中止发布 | 检查 CPU/NPU 输入预处理一致性 |
| GitCode API 失败 | curl 返回非 200 状态码 | 打印错误状态码和响应体 | 检查 ATOMIGIT_USER_TOKEN 权限和有效期 |
| CPU 推理过慢 | 大模型 448/512 推理耗时 > 30s | 提示使用 NPU 推理可获 100-700x 加速 | 减少 num_runs 或切换到 NPU 执行 |
| 用户确认超时 | 检查点等待超过 60 秒 | 暂停流程，保存当前进度到 `references/checkpoint.json` | 用户恢复后从当前检查点继续 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理脚本，支持 ModelScope 本地缓存权重加载和 top-5 输出 |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本，输出 logits 误差、概率一致性和加速比 |
| `examples/volo_inference_example.py` | VOLO 单模型推理示例脚本（含 module. 前缀处理和 NPU 资源释放） |
| `skill.json` | 技能元数据（模型列表、依赖版本、硬件要求、CANN 版本） |
| `test-prompts.json` | 本技能测试提示词（含 NPU 回退、串行批量、精度验证场景） |
| `references/evals.json` | 全部模型的推理和精度汇总结果（串行执行后生成） |
| `references/checkpoint.json` | 用户确认检查点状态持久化文件（暂停恢复用） |
| `results_cpu.json` | CPU 推理结果（运行后生成） |
| `results_npu.json` | NPU 推理结果（运行后生成） |
| `compare_results.json` | CPU/NPU 精度对比结果（运行后生成） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--model` | string | 是 | volo_d1_224.sail_in1k | timm 模型名称（支持列表中的 11 个模型） |
| `--device` | string | 否 | npu | 推理设备: cpu 或 npu，NPU 不可用时自动回退 CPU |
| `--image` | string | 否 | /tmp/test_volo.jpg | 输入图像路径，缺失时自动生成 |
| `--num_runs` | int | 否 | 10 | 推理重复次数，用于计算平均耗时 |

## 使用约束

1. 使用 ModelScope (`modelscope.hub.snapshot_download`) 或本地缓存加载模型权重，避免直接访问 HuggingFace。
2. 精度验证 (`max_prob_error < 1%`) 通过前不允许提交模型仓库。
3. 全部 11 个模型必须串行执行，禁止并行以防范 NPU 显存溢出 OOM。
4. VOLO 模型权重可能包含 `module.` 前缀，加载时必须使用 `strict=False` 或主动去除前缀。
5. 每个模型测试完成后必须调用 `gc.collect()` 和 `torch.npu.empty_cache()` 释放资源。
6. 发布模型仓库时需确保 `${ATOMGIT_USER_TOKEN}` 环境变量已设置且有效。
