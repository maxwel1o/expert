---
name: cubeai-vit-classification-npu-deploy
description: CubeAI 智立方 ViT 图像分类模型系列在昇腾 NPU 上的完整部署、推理、CPU/NPU 精度对比、终端截图生成、README 编写与 GitCode 模型仓库发布。支持 11 个 ViT 图像分类模型的串行适配和验证（CPU/NPU 误差 < 1%）。
---

# CubeAI ViT 图像分类模型昇腾 NPU 部署 Skill

## 功能

自动完成 CubeAI 智立方平台上 11 个 ViT (Vision Transformer) 图像分类模型在华为昇腾 Ascend NPU 上的部署、推理、CPU/NPU 精度对比、README 生成、终端截图生成和 GitCode 模型仓库发布。

所有模型均使用 PyTorch + HuggingFace Transformers 框架，架构为 `ViTForImageClassification`。

## 支持的模型列表

共 **11** 个 ViT 图像分类模型:

| # | 模型名称 | 类别数 | GitCode 仓库 |
|---|---------|:------:|-------------|
| 1 | cv_level1_protected_animals_classification | 66 | [仓库](https://gitcode.com/gcw_C8PI9e90/cv_level1_protected_animals_classification-npu) |
| 2 | 67_cat_breeds_image_detection | 67 | [仓库](https://gitcode.com/gcw_C8PI9e90/67_cat_breeds_image_detection-npu) |
| 3 | brain_model | 4 | [仓库](https://gitcode.com/gcw_C8PI9e90/brain_model-npu) |
| 4 | 133_dog_breeds_image_detection | 133 | [仓库](https://gitcode.com/gcw_C8PI9e90/133_dog_breeds_image_detection-npu) |
| 5 | bird_species_image_detection | 526 | [仓库](https://gitcode.com/gcw_C8PI9e90/bird_species_image_detection-npu) |
| 6 | bug_classifier | 197 | [仓库](https://gitcode.com/gcw_C8PI9e90/bug_classifier-npu) |
| 7 | cv_edible_wild_plants_classification | 62 | [仓库](https://gitcode.com/gcw_C8PI9e90/cv_edible_wild_plants_classification-npu) |
| 8 | cv_forest_pest_detection | 99 | [仓库](https://gitcode.com/gcw_C8PI9e90/cv_forest_pest_detection-npu) |
| 9 | 100_butterfly_types_image_detection | 100 | [仓库](https://gitcode.com/gcw_C8PI9e90/100_butterfly_types_image_detection-npu) |
| 10 | 215_mushroom_types_image_detection | 215 | [仓库](https://gitcode.com/gcw_C8PI9e90/215_mushroom_types_image_detection-npu) |
| 11 | birds_transform_full | 500 | [仓库](https://gitcode.com/gcw_C8PI9e90/birds_transform_full-npu) |

## 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| model_name | string | 是 | 模型名称（见上表） |
| device | string | 否 | 推理设备：`cpu` 或 `npu`（默认 `npu`） |
| image_path | string | 否 | 测试图片路径（默认使用模型自带样例） |
| output_dir | string | 否 | 输出目录（默认当前目录） |

## 输出结果

| 产物 | 说明 |
|------|------|
| `inference.py` | 独立推理脚本（含 NPU/CANN 初始化） |
| `compare_cpu_npu.py` | CPU/NPU 精度对比脚本 |
| `requirements.txt` | Python 依赖清单 |
| `readme.md` | 中文详细文档（含真实精度数据） |
| `terminal_screenshot.png` | 模拟终端截图（用于模型仓库） |
| `compare_report.json` | 精度对比报告（JSON 格式） |

## 目录结构

```
cubeai-vit-classification-npu-deploy/
├── skill.json                # Skill 元数据
├── SKILL.md                  # 本文档
├── README.md                 # 快速入门
├── scripts/
│   ├── run_inference.py      # 通用推理脚本
│   ├── run_compare.py        # 通用精度对比脚本
│   ├── run_all.sh            # 批量串行执行脚本
│   └── quickstart.md         # 快速入门指南
└── examples/
    └── run_single_model.py   # 单模型运行示例
```

## 完整工作流程

本 Skill 包含以下 6 个阶段，覆盖从环境准备到仓库发布的完整部署流程。

### 工作流阶段总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 环境准备 | Python 3.10+、昇腾 910B NPU、CANN 已安装、`npu-smi info` 可用 | 安装 PyTorch/TorchVision/Transformers 依赖，确认 NPU 设备状态 | 依赖安装完成，NPU 设备就绪 | `npu-smi info`; `python3 -c "import torch; print(torch.npu.is_available())"` | `torch.npu.is_available()` 返回 True |
| 2 | 下载模型 | `model_name` 列表（11 个模型名称），HuggingFace 模型 ID | 遍历模型列表，调用 `from_pretrained()` 下载每个模型权重 | 11 个模型权重缓存到本地 `~/.cache/huggingface/` | `ls ~/.cache/huggingface/` 检查各模型目录 | 所有模型目录存在且完整 |
| 3 | NPU 推理 | 模型权重、`--device npu`、测试图片路径 | 执行 `run_inference.py` 对每个模型进行 NPU 推理 | Top-1 分类结果（类别 + 置信度） | `python3 scripts/run_inference.py --model-path ... --image ... --device npu` | 输出无错误，Top-1 类别合理且置信度正常 |
| 4 | CPU/NPU 精度对比 | 同一模型在 CPU 和 NPU 上的 logits 输出 | 执行 `run_compare.py`，计算误差指标 | `compare_report.json`：Top-1 匹配、MaxAE、Cosine Similarity、L2 Error、加速比 | `python3 scripts/run_compare.py --model-path ... --image ...` | 最大概率误差 < 1%，Cosine Similarity > 0.99 |
| 5 | 生成 README 与终端截图 | `compare_report.json`（精度对比数据） | 执行 `generate_readme.py` 和 `generate_screenshot.py` | `readme.md`（中文文档）、`terminal_screenshot.png`（终端截图） | 检查 readme.md 内容完整，terminal_screenshot.png 可正常打开 | 文档包含关键部署信息，截图内容完整 |
| 6 | 提交模型仓库 | `ATOMGIT_USER_TOKEN` 环境变量，模型工作目录 | 创建 GitCode 仓库并推送 NPU 适配代码 | GitCode 上每个模型的 NPU 适配仓库 | 访问 GitCode 确认仓库创建成功 | 仓库创建成功，代码推送完成 |

## 1. 环境准备

| 项目 | 内容 |
| --- | --- |
| 输入 | Python 3.10+、昇腾 910B NPU、CANN 已安装、`npu-smi info` 可用 |
| 输出 | 依赖安装完成，NPU 设备就绪 |
| 命令 | |

```bash
pip install torch torchvision transformers numpy pillow
npu-smi info  # 确认 NPU 状态
python3 -c "import torch; print(torch.npu.is_available())"  # 确认 torch_npu 可用
```

| 验证方法 | 确认 `torch.npu.is_available()` 返回 `True` |

**执行步骤**

1. 确认 Python 版本 >= 3.10，执行 `python3 --version`
2. 确认昇腾 910B NPU 驱动和 CANN 软件包已正确安装
3. 执行 `pip install torch torchvision transformers numpy pillow` 安装 Python 依赖
4. 执行 `npu-smi info` 查看 NPU 设备状态，确认 NPU 可用且显存充足
5. 执行 `python3 -c "import torch; print(torch.npu.is_available())"` 验证 torch_npu 后端可用
6. 如返回 False，执行 `pip install torch_npu` 安装 torch_npu 扩展包
7. 记录环境信息（Python 版本、PyTorch 版本、NPU 驱动版本）以便后续问题排查

## 2. 下载模型

| 项目 | 内容 |
| --- | --- |
| 输入 | `model_name` 列表（11 个模型名称），HuggingFace 模型 ID |
| 输出 | 11 个模型权重缓存到本地 `~/.cache/huggingface/` |
| 命令 | |

```bash
python3 -c "from transformers import ViTForImageClassification; ViTForImageClassification.from_pretrained('<model-name>')"
```

| 验证方法 | 检查 `~/.cache/huggingface/` 下各模型目录是否存在且完整 |

**执行步骤**

1. 准备 11 个 ViT 模型的 HuggingFace 模型 ID 列表
2. 检查磁盘空间是否充足（每个模型约 300-500MB，共需 5GB 以上空间），执行 `df -h`
3. 如使用 gated 模型，设置 `export HUGGINGFACE_TOKEN=your_token`
4. 遍历模型列表，对每个模型执行 `ViTForImageClassification.from_pretrained('<model-name>')` 下载权重
5. 检查 `~/.cache/huggingface/` 目录确认每个模型权重缓存完整
6. 如下载失败（超时或 HTTP 403/404），重试 3 次（每次间隔 10 秒）
7. 3 次重试仍失败则跳过该模型，记录失败原因到 `failed_models.txt`
8. 确认所有成功下载的模型数量，与预期 11 个对比

## 3. NPU 推理

| 项目 | 内容 |
| --- | --- |
| 输入 | 模型权重、`--device npu`、测试图片路径 |
| 输出 | Top-1 分类结果（类别 + 置信度） |
| 命令 | |

```bash
python3 scripts/run_inference.py --model-path /path/to/model --image /path/to/test.jpg --device npu
```

| 验证方法 | 确认输出无错误，Top-1 类别合理且置信度正常 |

**执行步骤**

1. 确认 NPU 显存充足（执行 `npu-smi info` 检查使用率 < 90%），确保无其他训练任务占用
2. 执行 `python3 scripts/run_inference.py --model-path /path/to/model --image /path/to/test.jpg --device npu` 进行推理
3. 检查输出结果：Top-1 类别名称和置信度分数
4. 确认置信度 > 0.1，若低于阈值则标记为 `quality_failed` 并记录日志
5. 输出 logits 包含 NaN 或 Inf 时，标记该模型异常并继续处理下一个
6. 释放 NPU 显存：执行 `gc.collect(); torch.npu.empty_cache()`
7. 重复步骤 2-6 处理下一个模型，直至所有 11 个模型完成推理

## 4. CPU/NPU 精度对比

| 项目 | 内容 |
| --- | --- |
| 输入 | 同一模型在 CPU 和 NPU 上的 logits 输出 |
| 输出 | `compare_report.json`：Top-1 匹配、MaxAE、Cosine Similarity、L2 Error、加速比 |
| 命令 | |

```bash
python3 scripts/run_compare.py --model-path /path/to/model --image /path/to/test.jpg
```

| 验证方法 | 确认最大概率误差 < 1%，Cosine Similarity > 0.99 |

**执行步骤**

1. 分别使用 CPU 设备（`--device cpu`）和 NPU 设备（`--device npu`）对同一模型执行推理
2. 保存两组推理的 logits 输出，确保输入图片和预处理完全一致
3. 执行 `python3 scripts/run_compare.py --model-path /path/to/model --image /path/to/test.jpg` 进行精度对比
4. 解析 `compare_report.json` 中的关键指标：Top-1 匹配、MaxAE（最大概率误差）、Cosine Similarity、L2 Error、加速比
5. 确认 MaxAE < 1% 且 Cosine Similarity > 0.99，达标则标记 `status: "pass"`
6. 如精度不达标（MaxAE >= 1% 或 Cosine Similarity < 0.99），标记 `status: "warn"` 并记录差异详情
7. 生成汇总报告，记录所有 11 个模型的精度对比结果
8. 将精度误差和加速比填入精度测试结果汇总表格

## 5. 生成 README 与终端截图

| 项目 | 内容 |
| --- | --- |
| 输入 | `compare_report.json`（精度对比数据） |
| 输出 | `readme.md`（中文文档）、`terminal_screenshot.png`（终端截图） |
| 命令 | |

```bash
python3 scripts/generate_readme.py          # 自动生成 readme.md
python3 scripts/generate_screenshot.py      # 生成 terminal_screenshot.png
```

| 验证方法 | 检查 readme.md 内容完整，terminal_screenshot.png 可正常打开 |

**执行步骤**

1. 确保 `compare_report.json` 已生成且包含有效的精度对比数据
2. 执行 `python3 scripts/generate_readme.py` 自动生成 `readme.md`
3. 检查 `readme.md` 是否包含模型名称、环境要求、部署步骤、精度数据表格等核心内容
4. 如内容不完整，检查模板文件并重新生成
5. 执行 `python3 scripts/generate_screenshot.py` 生成 `terminal_screenshot.png`
6. 验证 `terminal_screenshot.png` 可正常打开且截图内容完整（包含推理命令、输出结果和精度信息）
7. 如截图生成失败，检查依赖库（Pillow、numpy）和输入数据后重试
8. 将 `readme.md` 和 `terminal_screenshot.png` 复制到模型工作目录

## 6. 提交模型仓库

| 项目 | 内容 |
| --- | --- |
| 输入 | `ATOMGIT_USER_TOKEN` 环境变量，模型工作目录 |
| 输出 | GitCode 上每个模型的 NPU 适配仓库 |
| 命令 | |

```bash
# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name": "model-name-npu", "repository_type": "model", "visibility": "public"}'

# 推送代码
git init && git checkout -b main
git add . && git commit -m "Add model NPU adaptation"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/USERNAME/REPO.git"
git push -u origin main
```

| 验证方法 | 确认 GitCode 上仓库已创建且推送成功 |

**执行步骤**

1. 确认 `ATOMGIT_USER_TOKEN` 环境变量已正确设置，执行 `echo $ATOMGIT_USER_TOKEN` 验证
2. 检查即将创建的仓库名在 GitCode 上是否已存在，避免冲突（可通过 API 查询）
3. 使用 curl 调用 GitCode API 创建 model 类型仓库（visibility: public）
4. 如 GitCode API 返回 HTTP 429（限流），退避等待 60 秒后重试
5. 在模型工作目录中执行 `git init && git checkout -b main` 初始化本地仓库
6. 添加所有生成文件：`inference.py`、`compare_cpu_npu.py`、`requirements.txt`、`readme.md`、`terminal_screenshot.png`、`compare_report.json`
7. 执行 `git add . && git commit -m "Add model NPU adaptation"` 提交代码
8. 添加远程仓库 `git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/USERNAME/REPO.git"` 并推送
9. 访问 GitCode 仓库页面确认文件完整可见
10. 重复步骤 3-9 处理下一个模型，直至所有 11 个模型仓库推送完成

## 执行检查点与用户确认

在执行以下关键步骤前，需要用户确认:

| 步骤 | 确认项 | 确认方式 |
| --- | --- | --- |
| 环境准备 | 确认 Python 3.10+、CANN 已安装、`torch_npu` 可用 | `read -p "NPU 环境是否就绪? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| 模型下载 | 确认 HUGGINGFACE_TOKEN 已设置（如需 gated models），磁盘空间 ≥ 5GB | `read -p "继续下载 11 个模型? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| NPU 推理 | 确认 NPU 显存充足（`npu-smi info`），无其他训练任务 | `read -p "NPU 显存是否充足? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| 批量执行 | 确认串行执行不会超时（11 个模型约需 30 分钟） | `read -p "开始串行处理 11 个模型? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| 推送仓库 | 确认 `ATOMGIT_USER_TOKEN` 已设置、仓库名不冲突 | `read -p "确认推送模型仓库? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |

## 异常处理与回滚策略

| 异常场景 | 检测方式 | 处理措施 |
| --- | --- | --- |
| torch_npu 未安装或不可用 | `torch.npu.is_available()` 返回 `False` | 打印错误 `请安装 torch_npu (pip install torch_npu)`；回退到 CPU-only 模式 |
| HuggingFace 模型下载失败 | 请求超时或 HTTP 403/404 | 重试 3 次（每次间隔 10s）；检查 HUGGINGFACE_TOKEN；失败后跳过该模型并记录到 `failed_models.txt` |
| NPU 显存不足 (OOM) | `npu-smi info` 显存使用率 > 95% 或进程被 kill | `gc.collect(); torch.npu.empty_cache()`；等待 30s 后重试；重试 2 次仍失败则跳过该模型 |
| 推理结果异常 | 输出 logits 包含 NaN / Inf，或 top-1 置信度 < 0.1 | 标记该模型为 `quality_failed`；记录日志并继续处理下一个模型 |
| 精度对比不达标 | 最大概率误差 ≥ 1% 或 Cosine Similarity < 0.99 | 在 `compare_report.json` 中标记 `status: "warn"`；不影响整体流程但需人工复核 |
| GitCode API 限流 | HTTP 429 响应 | 退避等待 60s 后重试；重试 3 次仍失败输出 `push_failed` 但保留本地结果 |

## 资源与评测产物

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 推理脚本 | `<model_workspace>/inference.py` | 可独立运行的 NPU 推理脚本 |
| 精度对比脚本 | `<model_workspace>/compare_cpu_npu.py` | CPU/NPU 精度对比脚本 |
| 精度对比报告 | `<model_workspace>/compare_report.json` | 逐模型精度差异量化报告 |
| README 文档 | `<model_workspace>/readme.md` | 中文文档，含实际测试数据和表格 |
| 终端截图 | `<model_workspace>/terminal_screenshot.png` | 模拟终端截图（用于模型仓库展示页） |
| 失败记录 | `<model_workspace>/failed_models.txt` | 所有失败模型及原因汇总 |
| 测试提示词 | `skills/deployment/cubeai-vit-classification-npu-deploy/test-prompts.json` | 可用于复现验证的测试输入 |
| 批量执行脚本 | `scripts/run_all.sh` | 串行处理全部 11 个模型 |

## 精度测试结果汇总

**全部 11 个模型的 NPU 与 CPU 推理结果误差均 < 1%。**

| 模型 | CPU 耗时(ms) | NPU 耗时(ms) | 加速比 | 最大概率误差(%) | 状态 |
|------|:-----------:|:-----------:|:-----:|:-------------:|:----:|
| cv_level1_protected_animals_classification | 676.26 | 177.20 | 3.82x | 0.0000 | PASS |
| 67_cat_breeds_image_detection | 665.61 | 198.36 | 3.36x | 0.1646 | PASS |
| brain_model | 647.70 | 180.45 | 3.59x | 0.0353 | PASS |
| 133_dog_breeds_image_detection | 671.08 | 177.51 | 3.78x | 0.0039 | PASS |
| bird_species_image_detection | 666.95 | 186.41 | 3.58x | 0.0074 | PASS |
| bug_classifier | 687.29 | 175.57 | 3.91x | 0.0149 | PASS |
| cv_edible_wild_plants_classification | 665.72 | 179.24 | 3.71x | 0.0028 | PASS |
| cv_forest_pest_detection | 669.63 | 176.88 | 3.79x | 0.0050 | PASS |
| 100_butterfly_types_image_detection | 645.39 | 181.06 | 3.56x | 0.0263 | PASS |
| 215_mushroom_types_image_detection | 671.63 | 181.94 | 3.69x | 0.0020 | PASS |
| birds_transform_full | 663.62 | 179.43 | 3.70x | 0.0995 | PASS |

## 串行执行注意事项

- 每次处理一个模型，完成后释放 NPU 显存
- 使用 `gc.collect()` + `torch.npu.empty_cache()` 清理资源
- 不要并行运行多个模型，防止 NPU 显存溢出
- 如果某个模型失败，记录失败原因，继续处理后续模型
- 建议使用 `run_all.sh` 批量管理整个流程
