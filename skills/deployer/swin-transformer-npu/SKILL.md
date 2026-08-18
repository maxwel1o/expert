---
name: swin-transformer-npu
description: "Swin Transformer 系列图像分类模型 (tiny/small/s3/base/large, 224/384 输入) 在昇腾 NPU 上的完整部署、推理验证、CPU/NPU 精度对比、README 生成与模型仓库发布 Skill。适用于：昇腾部署、NPU 推理、精度验证、模型发布。触发词：Swin Transformer NPU 部署、swin 昇腾推理、Swin Transformer 精度对比、npu-smi、ascend 部署。"
---

# Swin Transformer 系列模型昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 Swin Transformer 系列图像分类模型（19 个变体），完成推理、精度验证，并发布模型仓库。执行流程分 8 步：先环境检查和 NPU 检测，再模型选择和测试数据准备，然后单模型 CPU 基线推理、NPU 推理、CPU/NPU 精度对比、剩余模型串行批量处理，最后文档生成和模型仓库发布。

## 概述

本 Skill 用于自动完成 **Swin Transformer 系列图像分类模型**（涵盖 19 个 Swin 变体，包括 tiny/small/s3/base/large，支持 224 和 384 输入尺寸）在昇腾 NPU 上的完整部署、推理验证、CPU/NPU 精度对比、README 文档生成、终端截图生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910 系列, 64GB HBM) |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 1.0+ |
| 精度目标 | CPU 与 NPU 推理结果 L2 误差 < 1% |
| 执行方式 | 串行逐模型执行，避免显存爆炸 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 模型数量 | 19 个 Swin Transformer 变体 |

## 执行工作流

### 1. 环境初始化与 NPU 检测

**输入**: Python 3.9-3.13 环境，昇腾 NPU 驱动 (CANN >= 8.0)。

**动作**:
1. 检查 Python 版本，确认符合要求：
```bash
python3 --version
```
2. 加载 CANN 环境并运行 NPU 检测：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```
3. 设置镜像源并安装依赖：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision timm Pillow requests numpy
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/
```
4. 验证依赖包安装成功：
```bash
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}, torch_npu: {torch_npu.__version__}'); print(f'NPU available: {torch.npu.is_available()}')"
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，依赖已安装完成。

### 2. 模型选择与测试数据准备

**输入**: 目标模型名称（必填），如 `swin_tiny_patch4_window7_224.ms_in1k`。

**动作**:
5. 校验模型名是否在 19 个 Swin Transformer 变体支持列表内（tiny/small/s3/base/large 系列），不在则打印可用列表。
6. 检查测试图片是否存在，缺失则自动生成合成测试图像：
```bash
python3 -c "
from PIL import Image
img = Image.new('RGB', (224, 224), color='red')
img.save('test_image.jpg')
print('test_image.jpg generated')
"
```
7. 验证图片文件已就绪：
```bash
file test_image.jpg
```

**输出**: 测试图片和模型名已准备就绪。

### 3. CPU 基线推理

**输入**: 模型名、测试图片路径、设备类型 `cpu`。

**动作**:
8. 执行 CPU 推理，生成基线结果：
```bash
python3 scripts/swin_npu_infer.py --model swin_tiny_patch4_window7_224.ms_in1k --device cpu --image test_image.jpg
```
9. 验证输出文件 `{model_name}_cpu_output.pt` 包含 logits 数据。
10. 记录 CPU 推理耗时和 top-5 类别作为精度对比基线。

**输出**: `{model_name}_cpu_output.pt`，形状 `[1, 1000]`，包含 1000 个 ImageNet 类别的 logits 值。

### 4. NPU 推理

**输入**: 模型名、测试图片、设备类型 `npu`。

**动作**:
11. 检查 `NPU_AVAILABLE` 状态，若 false 则跳过并标记 `NPU_FALLBACK=true`。
12. 执行 NPU 推理：
```bash
python3 scripts/swin_npu_infer.py --model swin_tiny_patch4_window7_224.ms_in1k --device npu --image test_image.jpg
```
13. 处理失败：NPU 不可用时自动回退 CPU，OOM 时释放缓存后重试：
```python
import gc
gc.collect()
torch.npu.empty_cache()
```
14. 使用 `torch.npu.synchronize()` 确保准确测量 NPU 推理耗时。

**输出**: `{model_name}_npu_output.pt`，形状 `[1, 1000]`，同 CPU 输出。

### 5. CPU/NPU 精度对比

**输入**: `{model_name}_cpu_output.pt`、`{model_name}_npu_output.pt`。

**动作**:
15. 执行精度对比脚本：
```bash
python3 scripts/swin_npu_compare.py --model swin_tiny_patch4_window7_224.ms_in1k
```
16. 计算以下精度指标：
    - L2 相对误差 = ||cpu-npu||₂ / ||cpu||₂ * 100%，通过标准：< 1%
    - Cosine Similarity，通过标准：> 0.9999
    - Top-1 一致性，通过标准：最大预测类别相同
    - Top-5 重叠率
    - 最大概率差异
    - RMSE 均方根误差
17. 若 `max_error < 1%` 标记 `PRECISION_PASS=true`；否则标记 `PRECISION_FAIL` 并中止该模型发布。

**输出**: 精度对比结果，记录 L2 误差、余弦相似度、Top-1 一致性等指标。

### 6. 串行多模型批量执行

**输入**: 剩余模型列表（19 个 Swin Transformer 变体）。

**动作**:
18. 从列表中移除已处理模型。
19. 对每个剩余模型串行执行 Step 3-5。
20. 每完成一个模型释放 NPU 显存：
```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```
21. 等待 2 秒确保资源完全释放后处理下一个模型。
```bash
python3 scripts/swin_npu_batch.py --all
```
22. 批量脚本自动记录每个模型的结果到 JSON 文件并输出汇总表格。

**输出**: 每个模型的推理结果和精度对比数据，汇总至 `result.json`。

### 7. 文档与截图生成

**输入**: 各模型精度对比结果 JSON。

**动作**:
23. 汇总所有通过模型的精度数据。
24. 生成每个模型的 README.md（含精度结论表格）：
```bash
python3 scripts/swin_npu_generate_readme.py --model swin_tiny_patch4_window7_224.ms_in1k --input-size 224 --result result.json
```
25. 生成终端截图：
```bash
python3 scripts/terminal_screenshot.py --text "推理结果文本" --output terminal_screenshot.png
```

**输出**: 各模型目录下的 `README.md` 和 `terminal_screenshot.png`。

### 8. 模型仓库发布

**输入**: 精度验证通过的模型名、GitCode API token (`${ATOMGIT_USER_TOKEN}`)。

**动作**:
26. 确认 `PRECISION_PASS=true`，否则拒绝发布。
27. 调用 GitCode API 创建模型仓库：
```bash
curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "{model_name}-npu", "description": "Swin Transformer NPU adaptation - {model_name}", "visibility": "public", "repository_type": "model"}'
```
28. 初始化本地仓库并推送：
```bash
mkdir -p {model_name}-npu && cd {model_name}-npu
git init && git checkout -b main
cp ../README.md . && git add README.md
git commit -m "docs: add {model_name} NPU deployment model"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/${USERNAME}/{model_name}-npu.git"
git push -u origin main 2>&1 | tail -3
```

**输出**: 模型仓库已推送至 `https://gitcode.com/{username}/{model_name}-npu`。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查 NPU 驱动 |
| 2 | CP-2: 模型确认检查点 | 选择模型后 | 选定的模型名称和参数是否正确 | 返回重新选择模型 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理输出、top-5 类别是否合理 | 检查测试图片和模型加载设置 |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功、耗时是否合理 | 检查 NPU 显存和驱动状态后重试 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | L2 误差是否 < 1%，日志是否完整 | 检查精度不达标原因，调整设置后重试 |
| 6 | CP-6: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、README 内容是否正确 | 修改仓库配置后再次申请用户确认 |
| 7 | CP-7: 全量完成检查点 | 全部 19 个模型处理完毕 | 所有模型精度是否达标，报告是否完整 | 返回未通过模型重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查 NPU 驱动 |
| NPU 显存 OOM | 推理时报显存溢出错误 | 依次释放缓存、减小 batch、切换到 CPU | CP-4 | 调整配置或释放其他 NPU 进程 |
| 模型加载异常 | timm.create_model 抛出异常 | 打印错误堆栈，提示模型名是否正确 | CP-2 | 修正模型名或检查网络连接 |
| 下载网络超时 | pip/curl 长时间无响应 | 提示使用镜像源，重试最多 3 次 | CP-1 | 切换镜像源或离线安装 |
| 精度超标异常 | L2 误差 >= 1% | 记录偏差明细，中止该模型发布流程 | CP-5 | 检查推理脚本和数据一致性后重试 |
| 磁盘空间不足 | 模型权重下载失败 | 提示清理磁盘空间后重试 | CP-1 | 释放磁盘空间后重试 |
| API 调用失败 | GitCode API 返回非 200 状态码 | 打印响应状态码和错误体信息 | CP-6 | 检查 token 权限和 API 地址 |
| 端口被占用 | 推理服务端口冲突 | 提示更换端口或结束占用进程 | CP-1 | `lsof -i :port` 查看后处理 |
| 截图生成失败 | Python 截图模块异常 | 跳过截图步骤，记录 warning 日志 | CP-7 | 手动安装截图依赖后重试 |
| 输入尺寸不匹配 | `device-side assert` 触发 | 检查输入尺寸与模型匹配的尺寸（224 或 384） | CP-2 | 使用正确的输入尺寸重新执行 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/swin_npu_infer.py` | 单模型 NPU/CPU 推理执行入口，支持 logits 输出和 JSON 持久化 |
| `scripts/swin_npu_compare.py` | CPU 与 NPU 推理结果对比脚本，输出 L2 误差、余弦相似度等精度指标 |
| `scripts/swin_npu_batch.py` | 串行执行多个模型推理和精度验证的完整 batch 脚本 |
| `scripts/swin_npu_generate_readme.py` | 生成每个模型的 README.md（含真实性能数据和精度结论表格） |
| `skill.json` | 技能元数据：支持的模型列表、参数定义、输出说明 |
| `test-prompts.json` | 结构评测用测试提示词（含 NPU 回退场景） |
| `{model_name}_cpu_output.pt` | CPU 推理结果（运行后生成）：形状 [1, 1000] 的 logits 张量 |
| `{model_name}_npu_output.pt` | NPU 推理结果（运行后生成）：形状 [1, 1000] 的 logits 张量 |
| `result.json` | 批量推理汇总结果（运行后生成）：各模型性能数据和精度指标 |
| `terminal_screenshot.png` | 终端输出截图（运行后生成）：展示推理结果和精度统计 |
| `references/` | 模型参考文档和原始论文引用（Swin Transformer: Hierarchical Vision Transformer using Shifted Windows） |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--model` | string | 是 | — | 模型名称: 19 个 Swin 变体之一 |
| `--device` | string | 否 | npu | 运行设备: cpu 或 npu |
| `--image` | string | 否 | test_image.jpg | 测试图像路径 |
| `--input-size` | int | 否 | 224 | 输入尺寸: 224 或 384 |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `{model_name}_cpu_output.pt` | PyTorch Tensor | CPU 推理 logits，形状 `[1, 1000]` |
| `{model_name}_npu_output.pt` | PyTorch Tensor | NPU 推理 logits，形状 `[1, 1000]` |
| `result.json` | JSON | 批量处理汇总结果，含各模型性能数据和精度指标 |
| `terminal_screenshot.png` | PNG | 终端输出截图 |
| `README.md` | Markdown | 模型中文文档（含精度结论表格） |

## 使用约束

1. 使用 hf-mirror.com 镜像下载模型权重（HF 官方可能无法访问）。
2. 精度验证通过前不提交模型仓库（必须有 `PRECISION_PASS=true` 标记）。
3. 每个模型独立提交仓库，不混合多个模型到同一仓库。
4. 模型仓库使用 `main` 分支。
5. 串行执行必须使用 `swin_npu_batch.py` 避免 NPU 显存溢出。
6. 全程 FP32 精度，无需混合精度。
