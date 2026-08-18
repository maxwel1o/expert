---
name: timm-se-npu-batch24
description: timm SE 系列图像分类模型（ResNet、SENet、HaloNet、BoTNet）昇腾 NPU 适配、推理测试、精度对比、文档生成与模型仓库发布的统一部署 Skill。支持的模型包括 seresnet50、seresnet33ts、seresnet152d、senet154、sehalonet33ts、sebotnet33ts 等 9 个 SE 架构变体。在适配 timm SE 系列模型时，必须触发此 Skill。
---

# timm SE 系列模型 NPU 部署 Skill

本 Skill 用于 timm 库中 SE（Squeeze-and-Excitation）系列图像分类模型在 **昇腾 Ascend910 NPU** 上的端到端部署、推理、精度测试、文档生成和模型仓库发布。

## 支持的模型列表

| 序号 | 模型名称 | 架构 | 输入尺寸 | 参数量 | 模型仓库地址 |
|------|---------|------|---------|-------|-------------|
| 1 | `seresnet50.ra2_in1k` | SE-ResNet50 + RA2 | 224×224 | ~28M | [链接](https://gitcode.com/m0_74196153/seresnet50.ra2_in1k-npu) |
| 2 | `seresnet50.a1_in1k` | SE-ResNet50 + AugMix | 224×224 | ~28M | [链接](https://gitcode.com/m0_74196153/seresnet50.a1_in1k-npu) |
| 3 | `seresnet50.a2_in1k` | SE-ResNet50 + AugMix² | 224×224 | ~28M | [链接](https://gitcode.com/m0_74196153/seresnet50.a2_in1k-npu) |
| 4 | `seresnet50.a3_in1k` | SE-ResNet50 + AugMix³ | 224×224 | ~28M | [链接](https://gitcode.com/m0_74196153/seresnet50.a3_in1k-npu) |
| 5 | `seresnet33ts.ra2_in1k` | SE-ResNet33ts + RA2 | 224×224 | ~14M | [链接](https://gitcode.com/m0_74196153/seresnet33ts.ra2_in1k-npu) |
| 6 | `seresnet152d.ra2_in1k` | SE-ResNet152d + RA2 | 224×224 | ~60M | [链接](https://gitcode.com/m0_74196153/seresnet152d.ra2_in1k-npu) |
| 7 | `senet154.gluon_in1k` | SENet-154 (Gluon) | 224×224 | ~115M | [链接](https://gitcode.com/m0_74196153/senet154.gluon_in1k-npu) |
| 8 | `sehalonet33ts.ra2_in1k` | SE-HaloNet33ts + RA2 | 256×256 | ~14M | [链接](https://gitcode.com/m0_74196153/sehalonet33ts.ra2_in1k-npu) |
| 9 | `sebotnet33ts_256.a1h_in1k` | SE-BoTNet33ts + AugMix | 256×256 | ~14M | [链接](https://gitcode.com/m0_74196153/sebotnet33ts_256.a1h_in1k-npu) |

## Skill 用途

- 自动完成 timm SE 系列模型在昇腾 NPU 上的部署和推理
- 执行 CPU 与 NPU 推理结果精度对比
- 生成详细的中文 README 文档
- 生成模拟终端截图
- 串行执行多个模型，防止 NPU 显存溢出
- 调用 GitCode API 创建模型仓库并推送

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 是 | timm 模型名称（如 `seresnet50.ra2_in1k`） |
| `device` | string | 否 | 推理设备（`cpu` / `npu`，默认 `npu`） |
| `image_path` | string | 否 | 测试图片路径（默认 `test_input.jpg`） |
| `topk` | int | 否 | Top-K 结果数（默认 5） |
| `output_dir` | string | 否 | 输出目录（默认 `outputs/{model_name}/`） |

## Skill 输出结果

- `output_cpu.pt` — CPU 推理输出 logits
- `output_npu.pt` — NPU 推理输出 logits
- 精度对比数据（最大绝对误差、余弦相似度、相对误差等）
- `readme.md` — 中文模型文档
- `screenshot.png` — 模拟终端截图

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.1.0 |
| torch-npu | >= 2.1.0 |
| timm | 1.0.27 |
| Pillow | >= 10.0.0 |
| Ascend CANN | 8.5.1 |
| NPU | Ascend910 (32GB) |

## 工作流程

## 执行总览

1. 读取 `model_name`、`device`、`image_path`、`topk` 和 `output_dir`，确认模型属于支持列表。
2. 执行 `npu-smi info`、CANN 环境变量和 `torch_npu` 导入检查，输出环境检查结果。
3. 下载或复用 ModelScope/timm 权重，记录缓存路径和模型输入尺寸。
4. 分别运行 CPU 与 NPU 推理，保存 `output_cpu.pt`、`output_npu.pt` 和 Top-K 结果。
5. 运行 `scripts/compare_cpu_npu.py`，生成精度验证表和小于 1% 的结论。
6. 调用 `scripts/gen_readme.py` 和终端截图脚本，生成 README、`results.tsv`、`evals.json` 和 `screenshot.png`。
7. 在用户确认后创建 GitCode 模型仓库，推送前执行 dry-run 检查。
8. 发布后回读模型仓库 README，确认 NPU、Ascend、精度验证和推理成功信息可见。

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|---|---|---|---|
| CP-1 环境检查点 | `npu-smi info` 或 `torch_npu` 检查完成后 | 用户确认继续使用当前 Ascend/CANN/NPU 环境 | 暂停确认，记录为 dry-run，不写入真实 NPU 结论 |
| CP-2 下载检查点 | 首次下载 ModelScope 或 timm 权重前 | 用户确认模型名、权重来源和缓存目录 | 切换镜像源或复用本地缓存，失败则跳过该模型 |
| CP-3 精度检查点 | CPU/NPU 对比生成后 | 用户确认误差小于 1% 且 Top-K 结果合理 | 标记验证失败，保留日志，禁止发布通过结论 |
| CP-4 发布检查点 | 创建 GitCode 模型仓库或推送前 | 用户确认仓库名、目标分支和 README 内容 | 停止发布，仅保留本地 `results.tsv`、`evals.json` 和 README 草稿 |
| CP-5 最终验收 | 推送完成后 | 用户确认模型页可见 NPU 标签、推理证据和精度结论 | 回滚本次 README 草稿或追加修复提交 |

### 1. NPU 推理

每个模型使用 `inference.py` 进行推理。推理脚本支持 CPU 和 NPU 两种设备：

```bash
# CPU 推理
python3 inference.py --model {model_name} --device cpu --image test_input.jpg

# NPU 推理
python3 inference.py --model {model_name} --device npu --image test_input.jpg
```

**注意**：对于 HaloNet/BoTNet 架构的模型（`sehalonet33ts.ra2_in1k`、`sebotnet33ts_256.a1h_in1k`），输入图片尺寸必须为 256×256，使用 `test_input_256.jpg`。

### 2. CPU/NPU 精度对比

精度对比使用 `compare_cpu_npu.py` 脚本，会计算以下指标：

- 最大绝对误差 (Max Abs Error)
- 平均绝对误差 (Mean Abs Error)
- 均方误差 (MSE)
- 余弦相似度 (Cosine Similarity)
- 相对误差 (Relative Error)
- 最大概率差异 (Max Prob Diff)
- Top-1 标签一致性
- Top-5 重叠率

```bash
python3 compare_cpu_npu.py
```

精度要求：NPU 与 CPU 推理结果相对误差 < 1%。

### 3. 串行执行多个模型

为防止 NPU 显存爆炸，多个模型必须串行执行。每个模型推理完成后，释放资源再处理下一个：

```bash
for model in seresnet50.ra2_in1k seresnet50.a1_in1k ...; do
    python3 inference.py --model $model --device npu --image test_input.jpg
    python3 compare_cpu_npu.py
    # 释放资源
    python3 -c "import gc; gc.collect(); import torch; torch.npu.empty_cache()"
done
```

### 4. 生成 README 文档

使用 `gen_readme.py` 脚本自动生成标准化的中文 README 文档：

```bash
python3 gen_readme.py
```

README 包含：
- 模型介绍与基本信息
- 环境要求与安装步骤
- CPU/NPU 推理命令与结果
- 精度对比数据表格
- 性能对比（推理耗时）
- 模拟终端截图
- 模型标签（#+NPU、#+CV、#+昇腾等）

### 5. 生成终端截图

使用 `terminal_screenshot.py` 生成模拟终端输出截图：

```bash
python3 /opt/atomgit/terminal_screenshot.py \
    --input screenshot.txt \
    --output screenshot.png
```

### 6. 发布模型仓库到 GitCode

每个模型需要创建独立的 GitCode 模型仓库，仓库名为 `{model_name}-npu`：

```bash
# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
    --header "private-token: ${ATOMGIT_USER_TOKEN}" \
    --header "Content-Type: application/json" \
    --data '{
        "name": "{model_name}-npu",
        "path": "{model_name}-npu",
        "private": false,
        "repository_type": "model"
    }'

# 推送代码
git init
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/{username}/{model_name}-npu.git"
git add -A
git commit -m "Add {model_name} NPU inference"
git branch -M main
git push -u origin main
```

## 文件结构

```
timm-se-npu-batch24/
├── SKILL.md                    # 本 Skill 文档
├── test-prompts.json           # 结构化测试提示
├── scripts/
│   ├── inference.py            # 推理脚本
│   ├── compare_cpu_npu.py      # 精度对比脚本
│   ├── gen_readme.py           # README 生成脚本
│   └── requirements.txt        # 依赖清单
└── references/
    └── models.md               # 模型仓库地址参考
```

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|---|---|---|---|
| 不支持的模型名 | `model_name` 不在 9 个 SE 变体中 | 输出支持列表，要求用户确认替换模型名 | `evals.json` 记录 `unsupported_model` |
| NPU 不可用 | `npu-smi info` 失败或无 Ascend910 | fallback 到 CPU dry-run，禁止写真实 NPU 通过结论 | README 标记 `dry_run` |
| CANN 环境缺失 | `ASCEND_HOME`、`LD_LIBRARY_PATH` 或 `torch_npu` 缺失 | 提示加载 CANN 环境脚本，retry 一次环境检查 | 环境检查日志 |
| 权重下载失败 | ModelScope 网络超时或模型不存在 | retry 2 次，随后切换 hf-mirror 或本地缓存 | `results.tsv` 记录权重来源 |
| 输入尺寸错误 | HaloNet/BoTNet 未使用 256x256 图片 | 自动切换 `test_input_256.jpg` 并重新推理 | 推理日志含输入尺寸 |
| NPU 显存不足 | 推理抛出 OOM 或 device busy | 回收 `gc.collect()` 和 `torch.npu.empty_cache()`，串行 retry 当前模型 | `evals.json` 记录 retry 次数 |
| 精度不达标 | 相对误差 >= 1% 或 Top-1 不一致 | 保留 CPU/NPU logits，标记 failed，不生成通过结论 | 精度表显示失败原因 |
| GitCode 发布失败 | 创建仓库、commit 或 push 失败 | 停止后续发布，保留本地目录，用户确认后 retry | git 日志和远端 URL |
| README 验证失败 | 模型页缺失 NPU、Ascend 或精度句子 | 追加 README 修复提交并刷新页面验证 | 页面验证截图或日志 |

## 资源与评测产物

| 路径 | 用途 |
|---|---|
| `scripts/inference.py` | CPU/NPU 单模型推理入口，输出 logits 和 Top-K 分类结果 |
| `scripts/compare_cpu_npu.py` | 生成 CPU 与 NPU 精度对比，输出误差和一致性 |
| `scripts/gen_readme.py` | 根据结果生成模型 README |
| `references/models.md` | 维护 9 个 SE 变体的模型来源和目标仓库 |
| `test-prompts.json` | 提供重复评估本 Skill 的测试提示 |
| `results.tsv` | 每个模型的推理、精度、状态和错误原因汇总 |
| `evals.json` | 结构化保存环境、重试、验证和发布结果 |

## 注意事项

1. **串行执行**：所有模型必须串行推理，严禁并行执行，防止 NPU 显存溢出（每个模型推理后需释放资源）。
2. **HaloNet 输入尺寸**：`sehalonet33ts` 和 `sebotnet33ts_256` 需要 256×256 输入，其他模型使用 224×224。
3. **权重下载**：优先使用 ModelScope 下载，必要时使用 hf-mirror.com 镜像。
4. **精度标准**：余弦相似度 > 0.9999 或相对误差 < 1% 即判断为通过。
5. **README 要求**：必须包含真实推理数据和精度对比表格，不得编写无数据的空文档。
6. **模型标签**：推送到 GitCode 的 README 前端 YAML 标签中不要使用 `#+` 前缀（YAML 语法限制），将 `#+NPU`、`#+昇腾` 等标签放在 README 正文中。
