---
name: efficientnet-npu-skill
description: EfficientNet 系列 68 个 timm 图像分类模型在昇腾 Ascend NPU 上的部署 Skill，用于 NPU 推理、CPU/NPU 精度验证、批量运行、README 生成和 GitCode 模型仓库发布。用户提出 EfficientNet、Ascend 验证、NPU 推理、CPU/NPU 精度对比或批量适配时必须使用。
---

# EfficientNet NPU Deployment Skill

本 Skill 面向 `tf_efficientnet_*`、`tf_efficientnetv2_*`、EfficientNet Lite、CondConv、NoisyStudent 和 test EfficientNet 变体。已有详细模型清单、命令和参考结果保存在同目录 `README.md`，执行脚本位于 `scripts/`。

## Step 1: 输入和模型确认

1. 输入 `model_name`、`device`、`image_path`、`output_dir` 和 `topk`。
2. 在 `skill.json` 的 `models` 列表中确认模型存在。
3. 根据模型名判断输入尺寸，B0-B8 默认 224/300/380/456/528/600，V2 和 L2 按 README 表格处理。
4. 输出确认信息：模型名、权重来源、设备、输入图片、输出目录。

## Step 2: Ascend 环境检查

1. 执行 `npu-smi info`，确认 NPU 可见。
2. 执行 `python3 -c "import torch; import torch_npu; import timm; print('OK')"`。
3. 检查 CANN 环境变量、Python 版本、PyTorch 版本和 `torch_npu` 版本。
4. 将检查结果写入 `evals.json`。

```bash
npu-smi info
python3 -c "import torch; import torch_npu; import timm; print(torch.__version__)"
```

## Step 3: 下载权重和准备输入

1. 优先从 ModelScope 下载 `timm/{model_name}`。
2. 如果网络失败，fallback 到本地缓存或用户提供的权重目录。
3. 如果输入图片不存在，使用 README 中约定的测试图片。
4. 将权重路径、输入尺寸和 fallback 结果写入 `results.tsv`。

```python
from modelscope.hub.snapshot_download import snapshot_download
model_path = snapshot_download(f"timm/{model_name}")
```

## Step 4: CPU 与 NPU 推理

1. 使用 `scripts/inference.py` 运行 CPU 推理并保存 `output_cpu.pt`。
2. 使用 `scripts/inference.py` 运行 NPU 推理并保存 `output_npu.pt`。
3. 每个模型执行后释放资源，避免 Ascend NPU 显存累积。
4. 推理失败时 retry 一次，仍失败则记录错误并进入用户确认。

```bash
python3 scripts/inference.py tf_efficientnet_b0.in1k --device cpu
python3 scripts/inference.py tf_efficientnet_b0.in1k --device npu
python3 -c "import gc, torch; gc.collect(); torch.npu.empty_cache() if hasattr(torch, 'npu') else None"
```

## Step 5: 精度验证和评测产物

1. 使用 `scripts/compare_cpu_npu.py` 对比 CPU/NPU logits、概率分布和 Top-K。
2. 验证 MAE、相对误差、余弦相似度、Top-1 和 Top-5。
3. 当误差小于 1% 时，在 README 中写入明确精度结论。
4. 输出 `results.tsv`、`evals.json` 和推理成功日志。

```bash
python3 scripts/compare_cpu_npu.py tf_efficientnet_b0.in1k
```

## Step 6: README、截图和发布

1. 生成 README，包含环境、推理命令、CPU/NPU 精度表和小于 1% 结论。
2. 生成终端截图，展示模型加载、NPU 推理、输出保存和精度通过。
3. 发布前执行用户确认和 dry-run push。
4. 发布后刷新 GitCode 模型页，验证 NPU、Ascend、推理证据和精度结论可见。

```bash
bash scripts/run.sh tf_efficientnet_b0.in1k npu
python3 scripts/batch_run.py
git diff --check
git push --dry-run origin HEAD:main
```

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 失败处理 |
|---|---|---|---|
| CP-1 环境检查点 | `npu-smi info`、CANN 和 `torch_npu` 检查后 | 用户确认当前环境可用于真实 Ascend NPU 验证 | 暂停确认，fallback 到 CPU dry-run |
| CP-2 权重检查点 | 下载或复用权重前 | 用户确认模型名、来源和缓存路径 | retry 下载，失败则切换缓存 |
| CP-3 精度检查点 | `compare_cpu_npu.py` 完成后 | 用户确认误差小于 1% 且 Top-K 一致 | 标记 failed，禁止发布通过结论 |
| CP-4 发布检查点 | GitCode repo 创建或 push 前 | 用户确认目标仓库、分支和 README 内容 | 停止发布，保留本地产物 |
| CP-5 页面验收 | push 成功后 | 用户确认模型页展示 NPU 标签和精度验证信息 | 追加 README 修复提交 |

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 输出 |
|---|---|---|---|
| 不支持的模型 | `model_name` 不在 `skill.json` | 输出支持列表，等待用户确认替换 | `evals.json` |
| NPU 不可用 | `npu-smi info` 失败 | fallback 到 CPU dry-run，不写真实 NPU 结论 | 环境日志 |
| CANN 缺失 | `torch_npu` 导入失败 | 提示加载 CANN，retry 一次 | 检查日志 |
| 网络失败 | ModelScope 下载超时 | retry 2 次，切换缓存路径 | `results.tsv` |
| 磁盘不足 | 权重或输出写入失败 | 清理临时目录，暂停确认 | 错误日志 |
| 端口冲突 | 辅助服务启动失败 | 更换端口或跳过服务模式 | `evals.json` |
| NPU 显存不足 | 推理 OOM | 串行执行、清理缓存、retry 当前模型 | retry 记录 |
| 精度不达标 | 相对误差 >= 1% | 保留 CPU/NPU 输出，标记 failed | 精度表 |
| Git 推送失败 | commit 或 push 报错 | 保留本地分支，用户确认后 retry | git 日志 |
| 页面验证失败 | 模型页缺少 NPU 或精度句子 | 追加 README 修复提交 | 验证记录 |

## 资源与评测产物

| 路径 | 用途 |
|---|---|
| `README.md` | 68 个 EfficientNet 模型列表、命令和参考结果 |
| `skill.json` | 模型清单、输入输出和仓库元数据 |
| `scripts/inference.py` | CPU/NPU 推理脚本 |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本 |
| `scripts/batch_run.py` | 批量串行运行入口 |
| `scripts/run.sh` | 单模型命令封装 |
| `examples/inference_example.py` | 推理示例 |
| `test-prompts.json` | 重复评估本 Skill 的测试提示 |
| `results.tsv` | 模型级推理、精度和状态汇总 |
| `evals.json` | 环境、异常、检查点和验证结果 |

## 验收标准

1. `npu-smi info` 与 `torch_npu` 检查通过时，README 才能写真实 NPU 验证结论。
2. `scripts/compare_cpu_npu.py` 输出误差小于 1% 时，才能发布精度通过结论。
3. `results.tsv`、`evals.json` 和 README 必须同时包含推理成功、精度验证和异常处理结果。
4. 发布前必须经过用户确认、检查点和 dry-run push。
5. 页面验证失败时，保留回滚记录并追加修复提交。
