---
name: batch21-spnasnet100-npu-deploy
description: "SPNASNet_100 (spnasnet_100.rmsp_in1k) 图像分类模型在昇腾 NPU 上的自动部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证、模型发布。触发词：SPNASNet NPU 部署、spnasnet 昇腾推理、SPNASNet 精度对比、npu-smi、ascend 部署。"
---

# SPNASNet_100 昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 SPNASNet_100 图像分类模型，完成推理、精度验证，并发布模型仓库。执行流程分 8 步：先环境检查和 NPU 检测，再模型下载和数据准备，然后 CPU 基线推理、NPU 推理、精度对比，最后文档生成和仓库发布。

## 概述

本 Skill 用于自动完成 **SPNASNet_100 图像分类模型** 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证、README 文档生成、终端截图生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend) |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 0.9+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| 执行方式 | 单模型执行，支持标准精度验证流程 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |

## 支持的模型

| 模型名称 | 参数量 | 输入尺寸 | 模型仓库 |
|:---|:---:|:---:|:---|
| `spnasnet_100.rmsp_in1k` | 4.4M | 224x224 | [spnasnet_100.rmsp_in1k-npu](https://gitcode.com/m0_74196153/spnasnet_100.rmsp_in1k-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.9+ 环境，昇腾 NPU 驱动。

**动作**:
1. 检查 Python 版本，确认 >= 3.9：

```bash
python3 --version
```

2. 运行 NPU 检测：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm torch_npu pillow numpy
```

4. 确认依赖包版本兼容。

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，依赖已安装完成。

### Step 2: 模型下载与测试数据准备

**输入**: 目标模型名称 `spnasnet_100.rmsp_in1k`。

**动作**:
5. 使用 ModelScope 下载模型权重：

```bash
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('timm/spnasnet_100.rmsp_in1k', cache_dir='./model')
"
```

6. 检查测试图片是否存在，缺失则自动生成：

```bash
if [ ! -f test_image.jpg ]; then
    python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (224, 224), (128, 128, 128))
draw = ImageDraw.Draw(img)
draw.ellipse([50, 50, 174, 174], fill=(200, 100, 50))
draw.rectangle([20, 20, 100, 100], fill=(50, 150, 200))
img.save('test_image.jpg')
print('Test image created')
"
fi
```

**输出**: 模型权重文件已下载到本地缓存，测试图片已就绪。

### Step 3: CPU 基线推理

**输入**: 模型名 `spnasnet_100.rmsp_in1k`、测试图片路径、设备类型 `cpu`。

**动作**:
7. 复制推理脚本：

```bash
cp /path/to/scripts/inference.py ./
cp /path/to/scripts/requirements.txt ./
```

8. 执行 CPU 推理：

```bash
python3 inference.py --model spnasnet_100.rmsp_in1k --image test_image.jpg
```

9. 验证输出 JSON 包含 top5、confidence、latency_ms 字段。

**输出**: `results/inference_results.json`（含 CPU 推理结果），格式如下：

```json
{
  "cpu": {
    "device": "cpu",
    "model": "spnasnet_100.rmsp_in1k",
    "avg_time_ms": 123.4,
    "top5": [{"class": 0, "label": "...", "probability": 95.2}]
  }
}
```

### Step 4: NPU 推理

**输入**: 模型名 `spnasnet_100.rmsp_in1k`、测试图片路径、设备类型 `npu`。

**动作**:
10. 检查 `NPU_AVAILABLE` 状态，若 false 则跳过并标记 `NPU_FALLBACK=true`。
11. 执行 NPU 推理：

```bash
python3 inference.py --model spnasnet_100.rmsp_in1k --image test_image.jpg
```

12. 处理失败：NPU 不可用时自动回退 CPU，OOM 时释放缓存后重试。

**输出**: `results/inference_results.json`（含 CPU + NPU 双端结果），标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: `results/logits.pt`（CPU 和 NPU 的 logits 张量）。

**动作**:
13. 复制对比脚本并执行：

```bash
cp /path/to/scripts/compare_cpu_npu.py ./
python3 compare_cpu_npu.py
```

14. 读取两组 logits，计算以下指标：
    - Logits 最大/平均绝对差异
    - Logits MSE
    - 概率最大/平均差异
    - 余弦相似度
    - Top-1 / Top-5 分类一致性

15. 若 `max_probability_diff < 1%` 标记 `PRECISION_PASS=true`；否则标记 `PRECISION_FAIL`。

**输出**: `results/comparison_results.json`：

```json
{
  "max_probability_diff_pct": 0.418,
  "cosine_similarity": 0.999998,
  "top1_match": true,
  "top5_overlap": 5,
  "verdict": "PASS"
}
```

### Step 6: 资源清理与释放

**输入**: 模型推理和精度验证完成后的运行环境。

**动作**:
16. 清理模型权重缓存以节省磁盘空间：

```bash
rm -rf ./model
```

17. 释放 NPU 显存资源：

```python
import gc
import torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

18. 清理临时中间文件。

**输出**: 磁盘和 NPU 显存资源已释放完毕。

### Step 7: 文档与截图生成

**输入**: 精度对比结果 JSON。

**动作**:
19. 汇总精度数据，确认验证状态。
20. 生成 README.md（含精度结论表格）。
21. 生成终端截图：

```bash
python3 /opt/atomgit/terminal_screenshot.py \
    --text "SPNASNet_100 NPU Inference Results\n" \
    --output terminal_screenshot.png
```

**输出**: `README.md` 和 `terminal_screenshot.png`。

### Step 8: 模型仓库发布

**输入**: 精度验证通过的模型名、GitCode API token。

**动作**:
22. 确认 `PRECISION_PASS=true`，否则拒绝发布。
23. 调用 GitCode API 创建仓库：

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "spnasnet_100.rmsp_in1k-npu", "license": "cc-by-4.0"}'
```

24. 初始化本地仓库并推送：

```bash
mkdir -p spnasnet_100.rmsp_in1k-npu && cd spnasnet_100.rmsp_in1k-npu
git init && git checkout -b main
cp ../README.md . && git add README.md
cp ../inference.py ../compare_cpu_npu.py ../requirements.txt . && git add .
git commit -m "docs: add spnasnet_100.rmsp_in1k NPU deployment model"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/{username}/spnasnet_100.rmsp_in1k-npu.git"
git push -u origin main 2>&1 | tail -3
```

**输出**: 模型仓库已推送至 `https://gitcode.com/{username}/spnasnet_100.rmsp_in1k-npu`。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查 NPU 驱动 |
| 2 | CP-2: 模型下载确认检查点 | 模型权重下载完成后 | 模型名称和缓存路径是否正确 | 返回重新下载或检查网络连接 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理输出、top-5 类别是否合理 | 检查测试图片和模型加载设置 |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功、耗时是否合理 | 检查 NPU 显存和驱动状态后重试 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | 精度误差是否 < 1%，日志是否完整 | 检查精度不达标原因，调整设置后重试 |
| 6 | CP-6: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、README 内容是否正确 | 修改仓库配置后再次申请用户确认 |
| 7 | CP-7: 全量完成检查点 | 全部流程执行完毕 | 所有步骤是否成功，报告是否完整 | 返回失败步骤重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 CUDA out of memory | 依次释放缓存、减小 batch、切换到 CPU 重试 | CP-4 | 调整配置或释放其他进程 |
| 模型加载异常 | timm.create_model 抛出异常 | 打印错误堆栈，提示模型名是否正确 | CP-2 | 修正模型名或检查网络 |
| 下载网络超时 | pip/curl 长时间无响应 | 提示使用镜像源，重试最多 3 次后失败 | CP-1 | 切换镜像源或离线安装 |
| 精度超标异常 | 误差 >= 1% | 记录偏差明细，中止发布流程 | CP-5 | 检查推理脚本和数据一致性 |
| 磁盘空间不足 | 模型权重下载失败 | 提示清理磁盘空间后重试 | CP-1 | 释放磁盘空间后重试 |
| API 调用失败 | GitCode API 返回非 200 | 打印响应状态码和错误体 | CP-6 | 检查 token 权限和 API 地址 |
| 端口被占用 | 推理服务端口冲突 | 提示更换端口或结束占用进程 | CP-1 | `lsof -i :port` 查看后处理 |
| 截图生成失败 | terminal_screenshot 模块异常 | 跳过截图步骤，记录 warning | CP-7 | 手动安装截图依赖后重试 |
| safetensors 加载失败 | 模型权重格式不兼容 | 回退到 PyTorch 格式加载，记录 warning | CP-2 | 检查模型权重文件完整性 |
| 内存不足 | 模型加载时 OOM | 减小 batch 或升级硬件配置 | CP-3 | 增加 swap 或释放内存 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理执行入口，支持 top-5 置信度输出和 JSON 持久化 |
| `scripts/compare_cpu_npu.py` | CPU 与 NPU 推理结果对比脚本，输出误差百分比和验证结论 |
| `scripts/requirements.txt` | Python 依赖清单 |
| `examples/run_model.sh` | 端到端执行示例脚本：下载模型、推理、对比、截图 |
| `evals.json` | 结构评测参考配置（按需生成） |
| `references/` | 模型参考文档目录（可选） |
| `results/inference_results.json` | CPU/NPU 推理结果（运行后生成）：top-5 类别索引、置信度、推理耗时 |
| `results/logits.pt` | CPU 和 NPU 的 logits 张量（运行后生成） |
| `results/comparison_results.json` | CPU/NPU 精度对比结果（运行后生成）：逐类误差 + 汇总统计 |
| `terminal_screenshot.png` | 终端输出截图（运行后生成）：展示推理结果和精度统计 |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `--model` | string | 否 | spnasnet_100.rmsp_in1k | 模型名称 |
| `--device` | string | 否 | npu | 运行设备: cpu 或 npu |
| `--image` | string | 否 | None | 测试图像路径，为空则自动生成随机图 |
| `--runs` | int | 否 | 10 | 推理次数（用于计算平均耗时） |
| `--cache_dir` | string | 否 | .modelscope_cache/... | 模型权重缓存路径 |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `results/inference_results.json` | JSON | CPU/NPU 推理结果：top-5 类别索引、置信度、推理耗时 |
| `results/logits.pt` | PyTorch | CPU 和 NPU 的 logits 张量 |
| `results/comparison_results.json` | JSON | CPU/NPU 逐类精度对比 + 最大/平均误差百分比 |
| `terminal_screenshot.png` | PNG | 终端输出截图 |
| `README.md` | Markdown | 模型中文文档（含精度结论表格） |

## 使用约束

1. 使用 ModelScope 或 hf-mirror.com 下载模型权重（HF 官方可能无法访问）。
2. 精度验证通过前不提交模型仓库（必须有 `PRECISION_PASS=true` 标记）。
3. 模型仓库使用 `main` 分支，单模型独立提交。
4. 推理前确保已安装 torch_npu 和对应的 CANN 工具包。
5. 测试图片分辨率需与模型输入尺寸（224x224）匹配。
