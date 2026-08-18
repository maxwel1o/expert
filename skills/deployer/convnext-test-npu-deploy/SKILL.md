---
name: convnext-test-npu-deploy
description: "ConvNeXt 系列测试模型 (test_convnext.r160_in1k / test_convnext2.r160_in1k / test_convnext3.r160_in1k) 在昇腾 NPU 上的自动部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证、模型发布。触发词：ConvNeXt NPU 部署、convnext 昇腾推理、ConvNeXt 精度对比、npu-smi、ascend 部署。"
---

# ConvNeXt 测试模型昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 ConvNeXt 系列测试图像分类模型，完成推理、精度验证，并发布模型仓库。执行流程分 8 步：先环境检查和 NPU 检测，再模型下载和数据准备，然后 CPU 基线推理、NPU 推理、精度对比、剩余模型串行处理，最后文档生成和仓库发布。

## 概述

本 Skill 用于自动完成 **ConvNeXt 系列测试图像分类模型** 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证、README 文档生成、终端截图生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend) |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 1.0+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| 执行方式 | 串行逐模型执行，避免显存爆炸 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |

## 支持的模型

| 模型名称 | 参数量 | 输入尺寸 | 模型仓库 |
|:---|:---:|:---:|:---|
| `test_convnext.r160_in1k` | 0.3M | 160x160 | [test_convnext.r160_in1k-npu](https://gitcode.com/m0_74196153/test_convnext.r160_in1k-npu) |
| `test_convnext2.r160_in1k` | 0.48M | 160x160 | [test_convnext2.r160_in1k-npu](https://gitcode.com/m0_74196153/test_convnext2.r160_in1k-npu) |
| `test_convnext3.r160_in1k` | 0.47M | 160x160 | [test_convnext3.r160_in1k-npu](https://gitcode.com/m0_74196153/test_convnext3.r160_in1k-npu) |

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
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision timm modelscope pillow numpy scipy
```

4. 确认 torch_npu 兼容版本，测试 `torch.npu.is_available()`。

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，依赖已安装完成。

### Step 2: 模型下载与测试数据准备

**输入**: 目标模型名称（test_convnext.r160_in1k / test_convnext2.r160_in1k / test_convnext3.r160_in1k）。

**动作**:
5. 校验模型名是否在支持列表内，不在则打印可用列表。
6. 使用 ModelScope 下载模型权重：

```bash
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('timm/test_convnext.r160_in1k', cache_dir='./model')
"
```

7. 检查测试图片是否存在，缺失则自动下载测试图片。

**输出**: 模型权重缓存目录 `./model/timm/test_convnext___r160_in1k/`，测试图片已就绪。

### Step 3: CPU 基线推理

**输入**: 模型名、测试图片路径、设备类型 `cpu`。

**动作**:
8. 执行 CPU 推理脚本：

```bash
python3 scripts/inference.py
```

9. 修改 `MODEL_NAME` 和 `MODEL_PATH` 变量控制目标模型。
10. 验证输出 JSON 包含 top-5、confidence、latency_ms 字段。
11. 记录 CPU top-1 类别和置信度作为精度对比基线。

**输出**: `cpu_results.json`，格式如下：

```json
{"model": "test_convnext.r160_in1k", "device": "cpu", "top5": [{"class": "...", "confidence": 0.95}], "latency_ms": 123.4}
```

### Step 4: NPU 推理

**输入**: 模型名、测试图片、设备类型 `npu`。

**动作**:
12. 检查 `NPU_AVAILABLE` 状态，若 false 则跳过并标记 `NPU_FALLBACK=true`。
13. 执行 NPU 推理（调用同一份 `inference.py`，自动检测 NPU）：

```bash
python3 scripts/inference.py
```

14. 处理失败：NPU 不可用时自动回退 CPU，OOM 时释放缓存后重试。

**输出**: `npu_results.json`，格式同 CPU 结果，标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU 和 NPU 推理结果。

**动作**:
15. 执行精度对比脚本：

```bash
python3 scripts/compare_cpu_npu.py
```

16. 计算 logits 最大/平均绝对差异、概率最大/平均差异、Top-5 一致率、余弦相似度。
17. 若所有指标误差 < 1% 标记 `PRECISION_PASS=true`；否则标记 `PRECISION_FAIL`。

**输出**: `compare_results.json`：

```json
{"model": "test_convnext.r160_in1k", "max_error_pct": 0.039, "avg_error_pct": 0.01, "cosine_similarity": 0.99999984, "top5_agreement": "5/5 (100%)", "status": "pass"}
```

### Step 6: 串行多模型执行

**输入**: 剩余模型列表。

**动作**:
18. 从列表中移除已处理模型。
19. 对每个剩余模型串行执行 Step 3-5。
20. 每完成一个模型释放资源：

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

21. 等待 2 秒确保资源完全释放后处理下一个模型。

```bash
bash examples/run_all_models.sh
```

**输出**: 每个模型的 `cpu_results.json`、`npu_results.json`、`compare_results.json`。

### Step 7: 文档与截图生成

**输入**: 各模型精度对比结果。

**动作**:
22. 汇总所有通过模型的精度数据。
23. 生成每个模型的 README.md（含精度结论表格：概率最大差异、余弦相似度、Top-5 一致率、NPU 加速比）。
24. 生成终端截图：

```bash
python3 /opt/atomgit/terminal_screenshot.py --input screenshot_terminal.txt --output terminal_screenshot.png
```

**输出**: 各模型目录下的 `README.md` 和 `terminal_screenshot.png`。

### Step 8: 模型仓库发布

**输入**: 精度验证通过的模型名、GitCode API token（`ATOMGIT_USER_TOKEN` 环境变量）。

**动作**:
25. 确认 `PRECISION_PASS=true`，否则拒绝发布。
26. 调用 GitCode API 创建仓库：

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "{model_name}-npu", "license": "cc-by-4.0"}'
```

27. 初始化本地仓库并推送：

```bash
mkdir -p {model_name}-npu && cd {model_name}-npu
git init && git checkout -b main
cp ../README.md . && git add README.md
git commit -m "docs: add {model_name} NPU deployment model"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/{username}/{repo}.git"
git push -u origin main 2>&1 | tail -3
```

**输出**: 模型仓库已推送至 `https://gitcode.com/{username}/{model_name}-npu`。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查 NPU 驱动 |
| 2 | CP-2: 模型确认检查点 | 模型下载完成后 | 选定的模型名称和缓存路径是否正确 | 修正模型名或重新下载 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理输出、top-5 类别是否合理 | 检查测试图片和模型加载设置 |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功、耗时是否合理 | 检查 NPU 显存和驱动状态后重试 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | 精度误差是否 < 1%，日志是否完整 | 检查精度不达标原因，调整设置后重试 |
| 6 | CP-6: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、README 内容是否正确 | 修改仓库配置后再次申请用户确认 |
| 7 | CP-7: 全量完成检查点 | 全部模型处理完毕 | 所有模型精度是否达标，报告是否完整 | 返回未通过模型重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报 CUDA out of memory | 依次释放缓存、减小 batch、切到 CPU | CP-4 | 调整配置或释放其他进程 |
| 模型加载异常 | timm.create_model 抛出异常 | 打印错误堆栈，提示模型名是否正确 | CP-2 | 修正模型名或检查网络 |
| 下载网络超时 | pip/curl 长时间无响应 | 提示使用镜像源，重试最多 3 次 | CP-1 | 切换镜像源或离线安装 |
| 精度超标异常 | 误差 >= 1% | 记录偏差明细，中止该模型发布 | CP-5 | 检查推理脚本和数据一致性 |
| 磁盘空间不足 | 模型权重下载失败 | 提示清理磁盘空间后重试 | CP-1 | 释放磁盘空间后重试 |
| API 调用失败 | GitCode API 返回非 200 | 打印响应状态码和错误体 | CP-6 | 检查 token 权限和 API 地址 |
| 端口被占用 | 推理服务端口冲突 | 提示更换端口或结束占用进程 | CP-1 | `lsof -i :port` 查看后处理 |
| 截图生成失败 | Python 截图模块异常 | 跳过截图步骤，记录 warning | CP-7 | 手动安装截图依赖后重试 |
| 模型脚本路径不匹配 | sed 替换模型名后路径不正确 | 提示检查 MODEL_PATH 拼接逻辑 | CP-2 | 手动修正 MODEL_PATH 或恢复默认值 |
| 内存泄漏 | 连续运行多个模型后内存持续增长 | 强制 gc.collect() + torch.npu.empty_cache()，增加间隔时间 | CP-7 | 重启进程后重新执行 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | CPU/NPU 推理执行入口，支持 top-5 置信度输出和 JSON 持久化 |
| `scripts/compare_cpu_npu.py` | CPU 与 NPU 推理结果对比脚本，输出误差百分比、余弦相似度等指标 |
| `scripts/requirements.txt` | Python 依赖列表：torch, torchvision, timm, modelscope, pillow, numpy, scipy |
| `examples/run_all_models.sh` | 串行执行三个 ConvNeXt 模型推理和精度验证的完整 batch 脚本 |
| `test-prompts.json` | 结构评测用测试提示词（含 NPU 回退场景） |
| `cpu_results.json` | CPU 推理结果（运行后生成）：top-5 类别索引、置信度、推理耗时 |
| `npu_results.json` | NPU 推理结果（运行后生成）：top-5 类别索引、置信度、推理耗时 |
| `compare_results.json` | CPU/NPU 精度对比结果（运行后生成）：logits 差异、概率差异、余弦相似度 |
| `terminal_screenshot.png` | 终端输出截图（运行后生成）：展示推理结果和精度统计 |
| `evals.json` | 精度评测记录（运行后生成）：各模型精度验证汇总 |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 否 | test_convnext.r160_in1k | 模型名称 |
| `--skip_npu` | boolean | 否 | false | 是否跳过 NPU 推理 |
| `--skip_cpu` | boolean | 否 | false | 是否跳过 CPU 推理 |
| `output_dir` | string | 否 | ./output | 输出目录 |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `cpu_results.json` | JSON | top-5 类别索引、置信度分数、推理耗时 |
| `npu_results.json` | JSON | top-5 类别索引、置信度分数、推理耗时 |
| `compare_results.json` | JSON | CPU/NPU 逐类精度对比 + 最大/平均误差百分比 |
| `terminal_screenshot.png` | PNG | 终端输出截图 |
| `README.md` | Markdown | 模型中文文档（含精度结论表格） |

## 使用约束

1. 使用 ModelScope 或 hf-mirror.com 下载模型权重（HF 官方可能无法访问）。
2. 精度验证通过前不提交模型仓库（必须有 `PRECISION_PASS=true` 标记）。
3. 每个模型独立提交仓库，不混合多个模型到同一仓库。
4. 模型仓库使用 `main` 分支。
5. 串行执行必须使用 `run_all_models.sh` 避免 NPU 显存溢出。
6. 多个模型串行执行时，每个模型执行完毕后需调用 `gc.collect()` 和 `torch.npu.empty_cache()` 手动释放资源。
7. 不要将 `ATOMGIT_USER_TOKEN` 写入日志或文件。
