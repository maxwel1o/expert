---
name: xception-npu
description: "Xception 模型在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证与模型仓库发布。适用于：昇腾部署、NPU 推理、精度验证。触发词：Xception NPU 部署、昇腾推理、精度对比、npu-smi。"
---

# Xception 系列模型昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 Xception 系列 6 个图像分类模型（xception71.tf_in1k / xception65p.ra3_in1k / xception65.tf_in1k / xception65.ra3_in1k / xception41p.ra3_in1k / xception41.tf_in1k），串行完成推理、CPU/NPU 精度对比、文档生成和模型仓库发布。执行流程分 8 步：先环境检查和 NPU 检测，再串行执行每个模型的 CPU 基线推理、NPU 推理、精度对比和资源释放，最后文档生成和仓库发布。

## 概述

本 Skill 用于自动完成 **Xception 系列图像分类模型** 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证、README 文档生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910) |
| 模型数量 | 6 个 Xception 变体 |
| 框架版本 | PyTorch 2.0+, torch_npu, timm 0.9+ |
| 精度目标 | CPU 与 NPU 推理结果误差 < 1% |
| CANN 版本 | CANN >= 8.0 |

## 支持的模型

| 模型名称 | 参数量 | 输入尺寸 | 模型仓库 |
|:---|:---:|:---:|:---|
| `xception71.tf_in1k` | 42.3M | 299x299 | [xception71.tf_in1k-npu](https://gitcode.com/m0_74196153/xception71.tf_in1k-npu) |
| `xception65p.ra3_in1k` | 39.8M | 299x299 | [xception65p.ra3_in1k-npu](https://gitcode.com/m0_74196153/xception65p.ra3_in1k-npu) |
| `xception65.tf_in1k` | 39.8M | 299x299 | [xception65.tf_in1k-npu](https://gitcode.com/m0_74196153/xception65.tf_in1k-npu) |
| `xception65.ra3_in1k` | 39.8M | 299x299 | [xception65.ra3_in1k-npu](https://gitcode.com/m0_74196153/xception65.ra3_in1k-npu) |
| `xception41p.ra3_in1k` | 25.6M | 299x299 | [xception41p.ra3_in1k-npu](https://gitcode.com/m0_74196153/xception41p.ra3_in1k-npu) |
| `xception41.tf_in1k` | 25.6M | 299x299 | [xception41.tf_in1k-npu](https://gitcode.com/m0_74196153/xception41.tf_in1k-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.9+ 环境，昇腾 NPU 驱动，CANN >= 8.0。

**动作**:

1. 检查 Python 版本，确认 >= 3.9：

```bash
python3 --version
```

2. 安装依赖：

```bash
pip install torch torchvision timm Pillow modelscope safetensors numpy
```

3. 运行 NPU 检测：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

4. 确认依赖包版本兼容，检查 torch_npu 可用性：

```python
import torch
print(f"NPU available: {torch.npu.is_available()}")
if torch.npu.is_available():
    print(f"NPU device count: {torch.npu.device_count()}")
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，依赖已安装完成。

### Step 2: 模型选择与测试数据准备

**输入**: 目标模型名称（从 6 个支持模型中选择）。

**动作**:

5. 校验模型名是否在支持列表内，不在则打印可用列表。
6. 检查测试图片是否存在，缺失则自动创建：

```bash
TEST_IMAGE="/tmp/test_xception.jpg"
if [ ! -f "$TEST_IMAGE" ]; then
    python3 -c "
from PIL import Image
Image.new('RGB', (299, 299), color=(100,100,200)).save('$TEST_IMAGE')
print('Dummy test image created')
"
fi
```

7. 创建模型专属工作目录：

```bash
mkdir -p "$model"
cp scripts/inference.py "$model/"
cp scripts/compare_cpu_npu.py "$model/"
```

**输出**: 测试图片就绪，模型工作目录已创建。

### Step 3: CPU 基线推理

**输入**: 模型名、测试图片路径、推理次数（默认 10）。

**动作**:

8. 执行 CPU 推理：

```bash
python3 inference.py --model xception71.tf_in1k --device cpu --num_runs 10
```

9. 验证输出 `results_cpu.json` 包含 top-5 预测、推理耗时字段。
10. 记录 CPU top-1 类别和置信度作为精度对比基线。

**输出**: `results_cpu.json`，包含模型名、设备、top-5 索引与概率、平均推理时间。

### Step 4: NPU 推理

**输入**: 模型名、测试图片路径、设备类型 `npu`。

**动作**:

11. 检查 `NPU_AVAILABLE` 状态，若 false 则跳过并标记 `NPU_FALLBACK=true`。
12. 执行 NPU 推理：

```bash
python3 inference.py --model xception71.tf_in1k --device npu --num_runs 10
```

13. 处理失败：NPU 不可用时自动回退 CPU，OOM 时释放缓存后重试。

**输出**: `results_npu.json`，格式同 CPU 结果，标注 fallback 状态。

### Step 5: CPU/NPU 精度对比

**输入**: CPU 和 NPU 推理结果。

**动作**:

14. 执行精度对比脚本：

```bash
python3 compare_cpu_npu.py --model xception71.tf_in1k --num_runs 10
```

15. 对比指标包括：
    - Logits 余弦相似度、最大/平均绝对误差
    - 概率分布最大/平均绝对误差
    - Top-1 类别一致性
    - Top-5 结果重叠数

16. 若 `max_prob_error < 0.01`（即 < 1%）标记 `PRECISION_PASS=true`；否则标记 `PRECISION_FAIL`。

**输出**: `compare_results.json`，包含完整精度对比指标和结论。

### Step 6: 串行多模型执行

**输入**: 剩余未处理模型列表。

**动作**:

17. 对每个剩余模型串行执行 Step 3-5。
18. 每完成一个模型释放资源：

```python
import gc, torch
gc.collect()
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.empty_cache()
```

19. 等待 2 秒确保资源完全释放后处理下一个模型。

```bash
for model in xception71.tf_in1k xception65p.ra3_in1k xception65.tf_in1k xception65.ra3_in1k xception41p.ra3_in1k xception41.tf_in1k; do
    echo "=== Processing $model ==="
    python3 compare_cpu_npu.py --model $model --num_runs 10
    python3 -c "import gc, torch; gc.collect(); torch.npu.empty_cache()"
    sleep 2
done
```

**输出**: 每个模型的 `results_cpu.json`、`results_npu.json`、`compare_results.json`。

### Step 7: 文档与 README 生成

**输入**: 各模型精度对比结果。

**动作**:

20. 汇总所有通过模型的精度数据。
21. 生成每个模型的 README.md（含精度结论表格）：

```bash
python3 scripts/generate_readme.py --model_name xception71.tf_in1k --output readme.md
```

22. 生成终端截图（可选）：

```bash
python3 scripts/generate_screenshot.py --log inference.log --output screenshot.html
```

**输出**: 各模型目录下的 `README.md` 和 `screenshot.html`。

### Step 8: 模型仓库发布

**输入**: 精度验证通过的模型名、GitCode API token。

**动作**:

23. 确认 `PRECISION_PASS=true`，否则拒绝发布。
24. 调用 GitCode API 创建仓库：

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "{model_name}-npu", "license": "cc-by-4.0"}'
```

25. 初始化本地仓库并推送：

```bash
mkdir -p {model_name}-npu && cd {model_name}-npu
git init && git checkout -b main
cp ../README.md . && git add README.md
git commit -m "docs: add {model_name} NPU deployment model"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/${USERNAME}/${model_name}-npu.git"
git push -u origin main 2>&1 | tail -3
```

**输出**: 模型仓库已推送至 `https://gitcode.com/{username}/{model_name}-npu`。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: NPU 就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，依赖版本是否正确 | 暂停，提示安装 torch_npu 或检查 NPU 驱动 |
| 2 | CP-2: 模型确认检查点 | 选择模型后 | 选定的模型名称和参数是否正确 | 返回重新选择模型 |
| 3 | CP-3: CPU 基线完成检查点 | CPU 推理完成后 | CPU 推理输出、top-5 类别是否合理 | 检查测试图片和模型加载设置 |
| 4 | CP-4: NPU 推理完成检查点 | NPU 推理完成后 | NPU 推理是否成功、耗时是否合理 | 检查 NPU 显存和驱动状态后重试 |
| 5 | CP-5: 精度验证确认检查点 | 精度对比完成后 | 精度误差是否 < 1%，日志是否完整 | 检查精度不达标原因，调整设置后重试 |
| 6 | CP-6: 串行执行进度检查点 | 每个模型完成后 | 资源释放是否成功，准备处理下一个模型 | 手动执行资源释放后继续 |
| 7 | CP-7: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、README 内容是否正确 | 修改仓库配置后再次申请用户确认 |
| 8 | CP-8: 全量完成检查点 | 全部模型处理完毕 | 所有模型精度是否达标，报告是否完整 | 返回未通过模型重新验证 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查驱动 |
| NPU 显存 OOM | 推理时报内存不足错误 | 依次释放缓存、减小 batch、切到 CPU | CP-4 | 调整配置或释放其他进程 |
| 模型加载异常 | timm.create_model 抛出异常 | 打印错误堆栈，提示模型名是否正确 | CP-2 | 修正模型名或检查网络 |
| 下载网络超时 | pip/curl 长时间无响应 | 提示使用镜像源，重试最多 3 次 | CP-1 | 切换镜像源或离线安装 |
| 本地权重不存在 | modelscope 缓存未命中 | 自动回退到 hub 下载 | CP-2 | 检查 modelscope 缓存路径 |
| 精度超标异常 | 误差 >= 1% | 记录偏差明细，中止该模型发布 | CP-5 | 检查推理脚本和数据一致性 |
| 磁盘空间不足 | 模型权重下载失败 | 提示清理磁盘空间后重试 | CP-1 | 释放磁盘空间后重试 |
| API 调用失败 | GitCode API 返回非 200 | 打印响应状态码和错误体 | CP-7 | 检查 token 权限和 API 地址 |
| 截图生成失败 | Python 截图模块异常 | 跳过截图步骤，记录 warning | CP-8 | 手动安装截图依赖后重试 |
| 运行超时 | 单模型推理超过 600 秒 | 终止该进程并记录超时失败 | CP-3 | 检查 NPU 负载或减小 num_runs |
| Python 版本不兼容 | Python < 3.9 | 停止执行并提示升级 | CP-1 | 安装 Python 3.9+ |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/run_all_serial.py` | 串行执行全部 6 个模型推理、精度验证和资源释放的入口脚本 |
| `scripts/generate_screenshot.py` | 从推理日志生成终端截图 HTML 页面 |
| `scripts/inference.py` | CPU/NPU 推理执行入口，支持 top-5 置信度输出和 JSON 持久化 |
| `scripts/compare_cpu_npu.py` | CPU 与 NPU 推理结果对比脚本，输出误差百分比和精度结论 |
| `scripts/generate_readme.py` | 为每个模型从 compare_results.json 自动生成 README 文档 |
| `examples/README.md` | Xception NPU 部署使用示例，包含 CPU/NPU 推理、精度对比和发布流程代码 |
| `skill.json` | 技能元数据：模型触发词、描述、参数定义 |
| `test-prompts.json` | 结构评测用测试提示词 |
| `results_cpu.json` | CPU 推理结果（运行后生成）：top-5 类别索引、置信度、推理耗时 |
| `results_npu.json` | NPU 推理结果（运行后生成）：top-5 类别索引、置信度、推理耗时 |
| `compare_results.json` | CPU/NPU 精度对比结果（运行后生成）：logits 误差 + 概率误差 + 分类一致性 |
| `README.md` | 每个模型的中文文档（运行后生成）：含精度结论表格和推理命令 |
| `references/` | 参考文档目录，存放 NPU 部署相关说明 |

## 精度要求

- NPU 与 CPU 推理结果误差必须 < 1%
- 对比指标：Logits 余弦相似度、最大/平均绝对误差、Top-1/Top-5 一致性
- 结论标记：`max_prob_error < 0.01` 时记为 `PRECISION_PASS`

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 否 | 全部 6 个 | 要处理的模型名称 |
| `device` | string | 否 | npu:0 | 推理设备（cpu / npu:0） |
| `batch_size` | int | 否 | 1 | 批处理大小 |
| `skip_accuracy` | bool | 否 | false | 是否跳过精度测试 |
| `skip_push` | bool | 否 | false | 是否跳过 GitCode 推送 |
| `test_image` | string | 否 | /tmp/test_xception.jpg | 测试图像路径 |
| `num_runs` | int | 否 | 10 | 推理重复次数 |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `inference.py` | Python | 模型推理脚本 |
| `compare_cpu_npu.py` | Python | CPU/NPU 精度对比脚本 |
| `requirements.txt` | Text | 依赖清单 |
| `results_cpu.json` | JSON | CPU 推理结果：top-5 类别索引、置信度分数、推理耗时 |
| `results_npu.json` | JSON | NPU 推理结果：top-5 类别索引、置信度分数、推理耗时 |
| `compare_results.json` | JSON | CPU/NPU 逐类精度对比 + 最大/平均误差百分比 + 结论 |
| `screenshot.html` | HTML | 模拟终端输出截图 |
| `inference.log` | Text | 完整推理日志 |
| `README.md` | Markdown | 模型中文文档（含精度结论表格、环境要求、部署示例） |

## 使用约束

1. 使用 ModelScope 或 hf-mirror.com 下载模型权重（HF 官方可能无法访问）。
2. 精度验证通过前不提交模型仓库（必须有 `PRECISION_PASS=true` 标记）。
3. 每个模型独立提交仓库，不混合多个模型到同一仓库。
4. 模型仓库使用 `main` 分支。
5. 串行执行必须逐模型释放资源，避免 NPU 显存溢出。
6. 测试前确认 Ascend910 驱动和 CANN 环境已正确安装。
7. 推理时若报 OOM 错误需减小 batch_size 至 1。
