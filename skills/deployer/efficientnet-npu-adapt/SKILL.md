---
name: efficientnet-npu-adapt
description: 将 PyTorch Vision 的 EfficientNet 系列图像分类模型（B0-B6）适配到华为昇腾 NPU 的一站式流程。支持模型推理、精度验证（CPU vs NPU）和性能基准测试。已验证 7 个模型，所有模型精度误差 < 0.005。
keywords:
    - EfficientNet
    - 图像分类
    - Ascend
    - NPU
    - PyTorch Vision
    - 精度验证
    - 性能基准
    - torch_npu
---

# efficientnet-npu-adapt

## 功能

自动完成 PyTorch Vision 的 EfficientNet 模型在华为昇腾 NPU 上的适配、推理验证、精度对齐和性能评估。

## 已验证模型（7 个）

| 模型 | 参数量 | 输入尺寸 | ImageNet Top-1 | Ascend910 延迟 (bs=1) | 最大吞吐 |
|---|---|---|---|---|---|
| EfficientNet-B0 | 5.3M | 224×224 | 77.69% | 6.04ms | 2419 img/s |
| EfficientNet-B1 | 7.8M | 240×240 | 79.24% | 8.99ms | 1683 img/s |
| EfficientNet-B2 | 9.2M | 260×260 | 80.06% | 8.56ms | 1278 img/s |
| EfficientNet-B3 | 12M | 300×300 | 81.48% | 9.88ms | 767 img/s |
| EfficientNet-B4 | 19M | 380×380 | 82.93% | 12.49ms | 429 img/s |
| EfficientNet-B5 | 30M | 456×456 | 83.44% | 16.79ms | 204 img/s |
| EfficientNet-B6 | 43M | 528×528 | 83.95% | 22.38ms | 118 img/s |

所有模型精度验证通过：余弦相似度 > 0.99999，Top-1 一致率 100%。

## 环境要求

- **硬件**: 华为昇腾 NPU (Atlas 800 A2/A3 系列，建议 Ascend910)
- **Python**: 3.9+
- **PyTorch**: 2.9.0 (昇腾适配版)
- **torch_npu**: 2.9.0.post1
- **torchvision**: 0.24.0

## 安装命令

```bash
pip install torch torchvision torch_npu pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 完整执行流程（分步指南）

本 Skill 按「环境准备 → 模型获取 → 推理验证 → 精度对比 → 性能测试 → 文档生成」六阶段组织执行。

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 检测 NPU 并安装依赖 | CANN 环境, Python | python import, pip install | 可用 NPU 环境 | python -c "import torch_npu; print(torch.npu.is_available())" | NPU 可用且 import 无报错 |
| 模型获取 | 获取模型代码 | 模型名称 | git clone 或使用预训练权重 | 推理脚本 | ls efficientnet_b{n}-npu/ | 目录包含 inference.py |
| 推理验证 | 执行 NPU/CPU 推理 | 模型代码, 测试图片 | inference.py --device npu/cpu | Top-5 预测输出 | python inference.py | 推理正常完成 |
| 精度验证 | CPU vs NPU 对比 | 推理脚本 | accuracy_run.py | accuracy_report.json | cat accuracy_report.json | 余弦相似度 > 0.99999 |
| 性能测试 | 多 batch size 测试 | 推理脚本 | accuracy_run_perf.py | perf_report.json | cat perf_report.json | OOM 的 bs 被跳过记录 |
| 文档生成 | 生成 README 和截图 | 精度和性能报告 | generate_screenshot.py | readme.md, PNG | ls readme.md | 包含精度和性能数据 |

## 第一阶段：环境准备

**步骤 1.1 — 检测 NPU 环境**
```bash
python3 -c "import torch; print(f'NPU available: {torch.npu.is_available()}'); print(f'PyTorch: {torch.__version__}'); import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"
```
→ **检查点**：输出显示 `NPU available: True`

**步骤 1.2 — 安装依赖**
```bash
pip install torch torchvision torch_npu pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**执行步骤**：
1. 执行 `python -c "import torch; import torch_npu; print(torch.npu.is_available())"` 确认 NPU 可用
2. 检查 PyTorch 和 torch_npu 版本是否与 CANN 版本匹配
3. 执行 `pip install torch torchvision torch_npu pillow` 安装依赖包
4. 再次运行环境检测确认所有组件正常工作

## 第二阶段：获取模型代码

从预构建的模型仓库克隆或直接使用 PyTorch Vision 预训练权重：

| 模型 | 仓库地址 |
|---|---|
| EfficientNet-B0 | `https://gitcode.com/gcw_C8PI9e90/efficientnet_b0-npu` |
| EfficientNet-B1 | `https://gitcode.com/gcw_C8PI9e90/efficientnet_b1-npu` |
| EfficientNet-B2 | `https://gitcode.com/gcw_C8PI9e90/efficientnet_b2-npu` |
| EfficientNet-B3 | `https://gitcode.com/gcw_C8PI9e90/efficientnet_b3-npu` |
| EfficientNet-B4 | `https://gitcode.com/gcw_C8PI9e90/efficientnet_b4-npu` |
| EfficientNet-B5 | `https://gitcode.com/gcw_C8PI9e90/efficientnet_b5-npu` |
| EfficientNet-B6 | `https://gitcode.com/gcw_C8PI9e90/efficientnet_b6-npu` |

```bash
# 克隆单个模型仓库
git clone https://gitcode.com/gcw_C8PI9e90/efficientnet_b0-npu
cd efficientnet_b0-npu
```

**执行步骤**：
1. 选择要适配的 EfficientNet 模型版本（B0-B6），从预构建仓库克隆代码
2. 或直接使用 PyTorch Vision 预训练权重，无需额外下载
3. 进入模型目录：`cd efficientnet_b{n}-npu`
4. 确认目录包含 inference.py 等核心脚本文件

## 第三阶段：推理验证

```bash
# 使用随机输入进行 NPU 推理
python3 inference.py

# 使用自定义图片进行 CPU 推理
python3 inference.py --image cat.jpg --device cpu
```

参数：
- `--image`: 输入图片路径（可选，默认使用随机张量）
- `--device`: 设备（`npu:0` 或 `cpu`，默认自动检测 NPU）
- `--batch_size`: 批量大小（默认 1）

→ **检查点**：推理正常完成，输出 Top-5 预测结果和延迟数据

**执行步骤**：
1. 执行 `python3 inference.py` 使用随机输入进行 NPU 推理
2. 执行 `python3 inference.py --image cat.jpg --device cpu` 使用自定义图片进行 CPU 推理
3. 检查 Top-5 预测结果是否合理，确认推理正常完成
4. 记录延迟数据用于后续性能对比

## 第四阶段：精度验证

```bash
python3 accuracy_run.py
```

执行流程：
1. 生成 5 组不同随机种子的输入张量
2. 在 CPU (FP32) 上推理得到 baseline logits
3. 在 NPU (FP32) 上推理得到对比 logits
4. 计算并输出：max_abs_error、mean_abs_error、cosine_similarity、top1_match
5. 结果保存到 `accuracy_report.json`

→ **检查点**：所有模型余弦相似度 > 0.99999，Top-1 一致率 100%

**执行步骤**：
1. 执行 `python3 accuracy_run.py` 运行精度验证脚本
2. 脚本自动生成 5 组不同随机种子的输入张量，分别在 CPU 和 NPU 上推理
3. 检查输出的 max_abs_error、cosine_similarity、top1_match 三个核心指标
4. 确认所有模型的余弦相似度 > 0.99999，Top-1 一致率 100%
5. 精度报告自动保存到 `accuracy_report.json`

## 第五阶段：性能基准测试

```bash
python3 accuracy_run_perf.py
```

测试覆盖的 batch size 配置：1, 2, 4, 8, 16, 32
输出包含每种配置的：
- 平均延迟（ms）
- 最大吞吐（img/s）
- 结果保存到 `perf_report.json`

→ **检查点**：所有 batch size 配置完成测试，OOM 的配置被跳过并记录

**执行步骤**：
1. 执行 `python3 accuracy_run_perf.py` 运行多 batch size 性能基准测试
2. 脚本自动测试 batch size 1/2/4/8/16/32 的延迟和吞吐数据
3. 检查输出结果中每种配置的平均延迟（ms）和最大吞吐（img/s）
4. OOM 的 batch size 配置会自动跳过并记录到 perf_report.json

## 第六阶段：生成交付件

执行完成后自动生成：
- `readme.md` — 模型适配验证报告
- `terminal_screenshot.png` — 终端运行截图（通过 `generate_screenshot.py` 生成）

**执行步骤**：
1. 执行 `python3 generate_screenshot.py` 生成终端运行截图 `terminal_screenshot.png`
2. 检查生成的 readme.md 是否包含模型名称、精度结果和性能数据等必要字段
3. 检查截图是否清晰包含推理命令与结果
4. 确认所有交付件（inference.py、accuracy_run.py、accuracy_report.json、perf_report.json、readme.md、terminal_screenshot.png）完整

## 目录结构

```
efficientnet_b{n}-npu/
├── inference.py            # NPU 推理脚本
├── accuracy_run.py         # 精度验证脚本（CPU vs NPU）
├── accuracy_run_perf.py    # 性能基准测试脚本
├── accuracy_report.json    # 精度验证报告
├── perf_report.json        # 性能测试报告
├── terminal_screenshot.png # 终端运行截图
└── readme.md               # 部署文档
```

## 精度结果汇总

| 模型 | 最大绝对误差 | 平均绝对误差 | 最小余弦相似度 | Top-1 一致率 |
|---|---|---|---|---|
| EfficientNet-B0 | 2.56e-03 | 4.57e-04 | 0.99999869 | 100% |
| EfficientNet-B1 | 1.72e-03 | 3.56e-04 | 0.99999976 | 100% |
| EfficientNet-B2 | 2.53e-03 | 4.03e-04 | 0.99999887 | 100% |
| EfficientNet-B3 | 1.57e-03 | 2.05e-04 | 0.99999994 | 100% |
| EfficientNet-B4 | 4.60e-04 | 9.54e-05 | 1.00000000 | 100% |
| EfficientNet-B5 | 3.58e-03 | 4.60e-04 | 0.99999970 | 100% |
| EfficientNet-B6 | 2.39e-03 | 2.83e-04 | 0.99999988 | 100% |

所有模型精度误差 < 1%，满足部署要求。

## 性能数据 (Ascend910, FP32, bs=1)

| 模型 | 延迟 (ms) | 吞吐 (img/s) |
|---|---|---|
| EfficientNet-B0 | 6.04 | 165.68 |
| EfficientNet-B1 | 8.99 | 111.19 |
| EfficientNet-B2 | 8.56 | 116.88 |
| EfficientNet-B3 | 9.88 | 101.25 |
| EfficientNet-B4 | 12.49 | 80.05 |
| EfficientNet-B5 | 16.79 | 59.55 |
| EfficientNet-B6 | 22.38 | 44.68 |

## 注意事项

- 所有脚本使用 PyTorch Vision 预训练权重，首次运行自动下载（约 5-50MB 取决于模型）
- NPU 需要至少 4GB 显存（B6 最大输入 528×528，batch_size=1 约需 2GB）
- 精度验证时 CPU/NPU 各推理一次，并保存精度报告到 accuracy_report.json
- 如遇到 OOM，尝试减小 batch_size，B5/B6 建议 bs ≤ 8
- 终端截图使用 `generate_screenshot.py` 工具生成

## Token约束

- 推理日志：关注设备信息、Top-5 预测结果和延迟数据
- 精度报告：关注 max_abs_error、cosine_similarity、top1_match 三个核心指标
- 错误提示：优先检查 torch.npu.is_available() 和 torch_npu 版本兼容性

## 执行检查点与用户确认

| # | 检查点 | 阶段 | 确认内容 | 预期结果 | 失败处理 |
|---|--------|:----:|---------|:--------:|:--------:|
| 1 | NPU 环境检测 | 初始化 | 确认 NPU 驱动、CANN 版本、PyTorch NPU 支持 | `torch.npu.is_available()` 返回 `true`，`npu-smi info` 显示设备正常 | 输出环境诊断报告，提示用户安装 CANN 与 torch_npu |
| 2 | 预训练权重下载 | 数据准备 | 确认 PyTorch Vision 预训练权重自动下载完成 | 首次运行自动下载约 5-50MB 权重文件到缓存目录 | 检查网络连接，手动下载后放置到缓存目录 |
| 3 | CPU 基线推理 | 精度验证 | 确认 CPU (FP32) 推理结果已生成 | 5 组不同随机种子的 CPU logits 保存完成 | 标记为 CPU baseline 生成失败，检查模型加载是否正确 |
| 4 | NPU 推理执行 | 精度验证 | 确认 NPU (FP32) 推理正常完成，无 OOM | 与 CPU 相同输入的 NPU logits 保存完成，与 CPU 输出形状一致 | OOM 时自动减小 batch_size 重试，记录失败的 batch_size 值 |
| 5 | 精度对比 | 精度验证 | 确认 CPU 与 NPU 精度一致性 | 余弦相似度 > 0.99999，Top-1 一致率 100% | 输出详细对比报告，标注异常模型，暂停流程等待用户确认 |
| 6 | 性能基准测试 | 性能评估 | 确认多 batch size 性能数据已采集 | 所有 batch size (1,2,4,8,16,32) 的延迟和吞吐数据写入 perf_report.json | 跳过 OOM 的 batch size 配置，记录可用最大 batch size |
| 7 | README 与截图生成 | 文档生成 | 确认生成的 README 和截图包含完整信息 | README 包含模型名、精度结果、性能数据；截图文字清晰 | 补全缺失字段，检查中文字体支持 |

用户确认时机：
- **检查点 1 之前**：确认 NPU 环境已正确安装配置
- **检查点 5 之后**：查看精度报告，确认是否接受精度结果
- **检查点 6 之后**：确认性能数据是否符合上线要求

## 异常处理与回滚策略

| 异常场景 | 可能原因 | 检测方式 | 处理动作 | 回滚策略 |
|:---------|:---------|:---------|:---------|:---------|
| NPU 设备不可用 | 驱动未加载、CANN 未安装 | `torch.npu.is_available()` 返回 false | 输出诊断信息，提示用户检查安装 | 回退到纯 CPU 模式，仅执行 CPU 推理和文档生成 |
| 预训练权重下载失败 | 网络超时、磁盘空间不足 | `torch.hub.load` 抛出异常或文件不完整 | 自动重试 3 次（间隔 3 秒），若仍失败则提示手动下载 | 跳过该模型权重下载，记录失败日志 |
| NPU OOM | batch_size 过大、显存不足 | 推理时抛出显存不足异常 | 自动将 batch_size 减半重试（最多 3 次），最小支持 bs=1 | 释放显存 (`torch.npu.empty_cache()` + `gc.collect()`)，继续处理 |
| 精度对比不达标 | 算子精度差异、随机性影响 | 余弦相似度 < 0.9999 或 Top-1 一致率 < 100% | 输出逐样本对比报告，详细列出偏差较大的样本 | 保存精度报告到 `accuracy_issue.log`，由用户决定是否继续 |
| 算子兼容性错误 | NPU 不支持特定 PyTorch 算子 | 推理时抛出 RuntimeError（算子未实现） | 捕获错误，记录不兼容算子名称和行号 | 生成算子兼容性报告，建议替换为等价实现 |
| 性能数据异常 | 设备负载过高、降频 | 延迟波动超过均值 3σ 或吞吐量明显低于预期 | 重新测试 3 次取中位数 | 记录环境负载信息，给出建议的空闲时间窗口 |

所有异常日志写入 `{model_name}/execution.log`，格式为 `[TIMESTAMP] [LEVEL] [MODEL] message`。

## 资源与评测产物

| 资源类型 | 文件/目录 | 说明 | 用途 |
|:---------|:----------|:-----|:-----|
| 脚本 | `inference.py` | NPU/CPU 推理脚本，支持 `--device` 参数切换 | 模型推理入口，验证推理正确性 |
| 脚本 | `accuracy_run.py` | 5 组随机种子精度对比脚本（CPU FP32 vs NPU FP32） | 精度指标计算与对比报告生成 |
| 脚本 | `accuracy_run_perf.py` | 多 batch size 性能基准测试脚本 | 性能数据采集（延迟与吞吐） |
| 结果 | `accuracy_report.json` | 精度验证指标汇总 | 精度结果存档和 README 数据源 |
| 结果 | `perf_report.json` | 多 batch size 性能测试数据 | 性能基准数据存档 |
| 结果 | `accuracy_issue.log` | 精度不达标时的详细差异日志 | 精度问题排查依据 |
| 报告 | `readme.md` | 模型适配验证报告 | GitCode 模型仓库文档 |
| 截图 | `terminal_screenshot.png` | 终端运行截图 | README 可视化展示 |
| 日志 | `execution.log` | 全流程执行日志 | 执行过程回溯与问题排查 |

评测产物保留策略：
- 默认保留所有产物到 `{model_name}-npu/` 目录
- 性能测试中间文件（如原始 profiling 数据）超过 100MB 时自动清理
- 精度报告和性能报告为必留文件
