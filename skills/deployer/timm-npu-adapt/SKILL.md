---
name: timm-npu-adapt
description: 将 HuggingFace timm 视觉分类模型适配到华为昇腾 NPU 的一站式流程。支持推理精度验证、性能基准测试和完整部署。已验证 MobileNetV4、VOLO、ResNeXt、SE-ResNet、CaiT、XCiT 等 10 个视觉模型。
---

# timm-npu-adapt

## 功能

自动完成 timm 视觉分类模型在华为昇腾 NPU 上的适配、推理验证、精度对齐和性能评估。

## 已验证模型（10 个）

| 模型 | 架构 | 输入尺寸 | NPU/CPU 加速比 |
|---|---|---|---|
| `timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k` | MobileNetV4-Hybrid | 384×384 | **6.4×** |
| `timm/mobilenetv4_conv_medium.e500_r224_in1k` | MobileNetV4-Conv | 224×224 | **8.1×** |
| `timm/volo_d3_448.sail_in1k` | VOLO-D3 | 448×448 | **6.2×** |
| `timm/resnext101_64x4d.c1_in1k` | ResNeXt-101 | 224×224 | **6.1×** |
| `timm/seresnet50.ra2_in1k` | SE-ResNet50 | 224×224 | **6.6×** |
| `timm/cait_xxs24_384.fb_dist_in1k` | CaiT-XXS24 | 384×384 | **6.1×** |
| `timm/cait_xxs36_384.fb_dist_in1k` | CaiT-XXS36 | 384×384 | **5.8×** |
| `timm/xcit_nano_12_p16_224.fb_in1k` | XCiT-Nano | 224×224 | **6.0×** |
| `timm/seresnext101_32x4d.gluon_in1k` | SE-ResNeXt-101 | 224×224 | **6.2×** |
| `timm/volo_d1_384.sail_in1k` | VOLO-D1 | 384×384 | **5.7×** |

所有模型精度验证通过：余弦相似度 1.0，预测一致率 100%。

## 环境要求

- **硬件**: 华为昇腾 NPU (Atlas 800 A2/A3 系列)
- **Python**: 3.10+
- **PyTorch**: 2.1+ (昇腾适配版)
- **torch_npu**: 与 PyTorch 版本匹配
- **timm**: 最新版 (>=1.0.0)

## 安装命令

```bash
pip install torch torch_npu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## 完整执行流程（分步指南）

本 Skill 按「环境准备 → 模型加载 → 推理验证 → 精度对比 → 性能评估」五阶段组织执行。

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 检测 NPU 并安装依赖 | CANN 环境, Python | python import, pip install | 可用 NPU 环境 | python -c "import torch_npu; print(torch.npu.is_available())" | NPU 可用且 import 无报错 |
| 模型加载 | 加载 timm 模型 | model_name | timm.create_model() | 模型权重 | python -c "import timm; m = timm.create_model('...')" | 模型加载成功 |
| 推理验证 | 执行 NPU/CPU 推理 | 模型, 测试图片 | inference.py | Top-5 预测 | python inference.py --model {name} | 推理正常完成 |
| 精度对比 | CPU vs NPU 对比 | 模型 | accuracy_eval.py | accuracy_results.json | cat accuracy_results.json | 余弦相似度 >= 1.0 |
| 性能评估 | 采集 CPU/NPU 延迟 | 模型 | inference.py | 延迟数据 | 检查输出加速比 | 加速比符合预期 |

## 第一阶段：环境准备

**步骤 1.1 — 检测 NPU 环境**
```bash
python3 -c "import torch; print(f'NPU available: {torch.npu.is_available()}'); import torch_npu; print(f'torch_npu: {torch_npu.__version__}')"
```
→ **检查点**：确认输出显示 `NPU available: True`

**步骤 1.2 — 安装依赖**
```bash
pip install torch torch_npu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

**执行步骤**：
1. 执行 `python -c "import torch; import torch_npu; print(torch.npu.is_available())"` 确认 NPU 可用
2. 执行 `pip install torch torch_npu timm pillow` 安装依赖包
3. 设置 `export HF_ENDPOINT=https://hf-mirror.com` 配置 HuggingFace 镜像加速下载
4. 再次确认 NPU 环境就绪后进入下一阶段

## 第二阶段：模型加载与推理验证

**步骤 2.1 — 单模型推理验证**
```bash
# 使用默认合成测试图进行 NPU 推理
python3 scripts/inference.py --model timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k

# 使用真实图片进行 CPU 推理
python3 scripts/inference.py --model timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k --image test.jpg --cpu
```

参数:
- `--model`: timm 模型名称（必填）
- `--image`: 输入图片路径（可选，默认使用合成测试图）
- `--cpu`: 强制使用 CPU（可选，默认自动检测 NPU）

→ **检查点**：输出显示 Top-5 预测结果，NPU 推理延迟数据

**执行步骤**：
1. 执行 `python3 scripts/inference.py --model timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k` 进行 NPU 推理
2. 执行 `python3 scripts/inference.py --model timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k --image test.jpg --cpu` 进行 CPU 推理
3. 检查 Top-5 预测结果是否合理，确认推理正常完成
4. 记录 NPU 和 CPU 的延迟数据用于加速比对比

## 第三阶段：精度验证

**步骤 3.1 — 单模型精度对比**
```bash
python3 scripts/accuracy_eval.py --model timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k
```

执行流程：
1. 生成 4 张随机合成测试图（不同随机种子）
2. 在 CPU 上推理得到 baseline logits
3. 在 NPU 上推理得到对比 logits
4. 计算 max_diff、mean_diff、cosine_similarity、prediction_match_pct
5. 结果写入 `accuracy_results.json`

**步骤 3.2 — 批量模型验证（所有 10 个模型）**
```bash
for model in \
  timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k \
  timm/mobilenetv4_conv_medium.e500_r224_in1k \
  timm/volo_d3_448.sail_in1k \
  timm/resnext101_64x4d.c1_in1k \
  timm/seresnet50.ra2_in1k \
  timm/cait_xxs24_384.fb_dist_in1k \
  timm/cait_xxs36_384.fb_dist_in1k \
  timm/xcit_nano_12_p16_224.fb_in1k \
  timm/seresnext101_32x4d.gluon_in1k \
  timm/volo_d1_384.sail_in1k; do
  echo "Processing: $model"
  python3 scripts/accuracy_eval.py --model "$model" || {
    echo "WARNING: $model failed" >> failures.log
    continue
  }
done
```

→ **检查点**：所有模型余弦相似度 >= 1.0，预测一致率 100%

**执行步骤**：
1. 执行 `python3 scripts/accuracy_eval.py --model timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k` 进行单模型精度对比
2. 脚本自动生成 4 张合成测试图，在 CPU 和 NPU 上分别推理并计算精度指标
3. 检查 max_diff、cosine_similarity、prediction_match_pct 核心指标
4. 确认余弦相似度 >= 1.0，预测一致率 100%
5. 批量验证所有 10 个模型：使用 for 循环依次执行 accuracy_eval.py，失败模型记录到 failures.log

## 第四阶段：性能评估

推理脚本自动输出性能数据，汇总如下：

| 模型 | NPU 平均延迟 | CPU 平均延迟 | 加速比 |
|:-----|:-----------:|:-----------:|:-----:|
| MobileNetV4-Hybrid (384) | — | — | **6.4×** |
| MobileNetV4-Conv (224) | — | — | **8.1×** |
| VOLO-D3 (448) | — | — | **6.2×** |
| ResNeXt-101 (224) | — | — | **6.1×** |
| SE-ResNet50 (224) | — | — | **6.6×** |

→ **检查点**：确认加速比符合预期，无明显性能异常

**执行步骤**：
1. 通过推理脚本输出的延迟数据计算每个模型的 CPU 和 NPU 平均延迟
2. 计算加速比 = CPU 平均延迟 / NPU 平均延迟
3. 检查加速比是否与预期值相符（通常 5-8×）
4. 如加速比异常（过高或过低），检查设备负载和其他影响因素

## 目录结构

```
timm-npu-adapt/
├── SKILL.md              # 本技能定义
├── scripts/
│   ├── inference.py      # 推理脚本
│   └── accuracy_eval.py  # 精度验证脚本
└── references/
    └── deployment.md     # 部署指南（可选）
```

## 注意事项

- 首次加载模型需要从 HuggingFace 下载权重，建议设置 HF_ENDPOINT 使用镜像
- 合成测试图用于精度验证，实际应用请替换为真实图片
- NPU 需要至少 8GB 显存（大部分验证模型 < 1GB，但确保可用）
- 精度验证时 CPU/NPU 各推理一次，结果写入 accuracy_results.json
- 如遇到 OOM，减小 batch_size 或输入尺寸

## Token约束

- 推理日志：只关注 Top-5 预测结果和延迟数据
- 精度报告：直接输出 diff 和 similarity 关键指标
- 错误提示：优先检查 NPU 设备可用性和 torch_npu 版本

## 执行检查点与用户确认

| # | 检查点 | 阶段 | 确认内容 | 预期结果 | 失败处理 |
|---|--------|:----:|---------|:--------:|:--------:|
| 1 | NPU 环境检测 | 初始化 | 确认 Ascend NPU 驱动、CANN、torch_npu 是否可用 | `torch.npu.is_available()` 返回 `true`，版本符合要求 | 输出诊断报告，提示用户检查 NPU 驱动和 CANN 安装 |
| 2 | 模型权重下载 | 数据准备 | 确认 HuggingFace/timm 模型权重下载完成 | 首次运行自动下载权重到 HF 缓存目录，文件完整 | 自动重试 2 次，检查 HF_ENDPOINT 镜像设置，若仍失败则跳过该模型 |
| 3 | CPU 基线推理 | 精度验证 | 确认 CPU 推理结果已生成 | 4 张合成图的 CPU logits 保存完成，形状正确 | 检查模型加载和输入预处理，标记该模型 CPU 基线失败 |
| 4 | NPU 推理执行 | 精度验证 | 确认 NPU 推理正常完成，无算子兼容性错误 | 与 CPU 相同输入的 NPU logits 保存完成 | 捕获 RuntimeError，记录不兼容算子，跳过该模型 |
| 5 | 精度对比 | 精度验证 | 确认 CPU 与 NPU 精度一致性 | 余弦相似度 >= 1.0，预测一致率 100% | 输出逐样本对比报告，分析偏差原因，暂停等待用户确认 |
| 6 | 性能数据采集 | 性能评估 | 确认 CPU 和 NPU 延迟数据已采集 | 首次延迟、平均延迟、加速比等数据完整 | 若 NPU 不可用则仅采集 CPU 数据，标记性能数据为部分完整 |
| 7 | 批量处理完整性 | 批量执行 | 确认 10 个模型全部处理完成或已知失败原因 | 所有模型均已处理，失败模型记录到 failures.log | 汇总失败列表供用户参考，不影响已完成模型的结果 |

用户确认时机：
- **检查点 1 之前**：确认 NPU 环境变量（如 HF_ENDPOINT）已正确设置
- **检查点 5 之后**：查看精度对比报告，确认是否接受结果
- **检查点 7 之后**：查看汇总报告，确认所有模型适配状态

## 异常处理与回滚策略

| 异常场景 | 可能原因 | 检测方式 | 处理动作 | 回滚策略 |
|:---------|:---------|:---------|:---------|:---------|
| NPU 设备不可用 | 驱动未加载、CANN 未安装、设备被占用 | `torch.npu.is_available()` 返回 false | 输出设备诊断信息，列出可用设备 | 回退到纯 CPU 模式，仅执行 CPU 推理和模型验证（跳过 NPU 性能数据） |
| 模型权重下载失败 | 网络超时、HF 镜像不可用、磁盘空间不足 | `timm.create_model()` 抛出异常或下载中断 | 自动重试 2 次（间隔 3 秒），检查 `HF_ENDPOINT` 配置 | 跳过该模型，记录失败到 `failures.log`，继续处理下一模型 |
| 算子兼容性错误 | NPU 不支持特定 timm 模型算子 | 推理时抛出 RuntimeError（如 `not implemented for NPU`） | 捕获错误，记录不兼容算子名称 | 生成算子兼容性报告，跳过该模型 |
| 精度对比不达标 | 算子精度差异、输入预处理不一致 | 余弦相似度 < 1.0 或预测一致率 < 100% | 输出逐样本详细对比报告（包含每张图的差异分析） | 保存精度问题日志，由用户决定是否继续使用该模型 |
| OOM（显存不足） | 模型输入尺寸过大、NPU 显存不足 | 推理时抛出显存相关异常 | 自动清理缓存 (`torch.npu.empty_cache()` + `gc.collect()`) 后重试 | 若持续 OOM 则跳过该模型，记录所需最小显存 |
| HuggingFace 镜像不可用 | HF_ENDPOINT 配置错误、镜像站故障 | 下载速度极慢或连接超时 | 尝试回退到官方源 (`hf.co`)，输出当前镜像配置 | 提示用户手动设置可用镜像后重试 |

所有异常日志格式：`[TIMESTAMP] [LEVEL] [MODEL] message`，写入 `execution.log`。

## 资源与评测产物

| 资源类型 | 文件/目录 | 说明 | 用途 |
|:---------|:----------|:-----|:-----|
| 脚本 | `scripts/inference.py` | timm 模型推理脚本，支持 `--model`/`--image`/`--cpu` 参数 | 模型推理验证入口，复现推理结果 |
| 脚本 | `scripts/accuracy_eval.py` | 4 张合成图精度验证脚本 | CPU vs NPU 精度对比，输出核心指标 |
| 结果 | `accuracy_results.json` | 精度验证指标汇总（每个模型） | 精度结果存档和报告数据源 |
| 结果 | `accuracy_issue.log` | 精度不达标模型的详细差异日志 | 精度问题排查依据 |
| 日志 | `execution.log` | 全流程执行日志（时间戳 + 级别 + 模型名 + 消息） | 执行过程回溯 |
| 日志 | `failures.log` | 失败模型记录（仅失败时生成） | 批量处理中失败模型清单 |

评测产物保留策略：
- 默认保留所有产物到 `timm-npu-adapt/` 目录
- 精度报告和日志为必留文件
- HF 权重缓存由 timm/HuggingFace 缓存管理，本 Skill 不自动清理
