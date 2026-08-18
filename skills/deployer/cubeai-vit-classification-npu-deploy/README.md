# CubeAI ViT 图像分类模型昇腾 NPU 部署 Skill

## 概述

本 Skill 用于自动完成 CubeAI 智立方平台上 11 个 ViT (Vision Transformer) 图像分类模型在华为昇腾 Ascend NPU 上的部署、推理、CPU/NPU 精度对比、README 生成、终端截图生成和 GitCode 模型仓库发布。

所有模型均使用 PyTorch + HuggingFace Transformers 框架，架构为 `ViTForImageClassification`。

## 支持的模型列表

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

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| model_name | string | 是 | 模型名称（见上表） |
| device | string | 否 | 推理设备：`cpu` 或 `npu`（默认 `npu`） |
| image_path | string | 否 | 测试图片路径（默认使用模型自带样例） |
| output_dir | string | 否 | 输出目录（默认当前目录） |

## Skill 输出结果

- `inference.py` — 独立推理脚本
- `compare_cpu_npu.py` — CPU/NPU 精度对比脚本
- `requirements.txt` — Python 依赖
- `readme.md` — 中文详细文档（含真实数据）
- `terminal_screenshot.png` — 模拟终端截图
- `compare_report.json` — 精度对比报告（JSON 格式）

## 目录结构

```
cubeai-vit-classification-npu-deploy/
├── skill.json                # Skill 元数据
├── README.md                 # 本文件
├── scripts/
│   ├── run_inference.py      # 通用推理脚本
│   ├── run_compare.py        # 通用精度对比脚本
│   ├── run_all.sh            # 批量串行执行脚本
│   └── quickstart.md         # 快速入门指南
└── examples/
    └── run_single_model.py   # 单模型运行示例
```

## 如何执行 NPU 推理

```bash
# 单模型推理
python3 scripts/run_inference.py \
  --model-path /path/to/model \
  --image /path/to/test.jpg \
  --device npu

# 使用独立的模型推理脚本
python3 /path/to/model_workspace/inference.py --device npu
```

## 如何执行 CPU/NPU 精度对比

```bash
python3 scripts/run_compare.py \
  --model-path /path/to/model \
  --image /path/to/test.jpg

# 或使用独立的模型对比脚本
python3 /path/to/model_workspace/compare_cpu_npu.py
```

精度指标包括：
- Top-1 类别匹配
- Logits 最大绝对误差 (MaxAE)
- Probability 最大绝对误差
- Cosine Similarity
- L2 Error
- NPU 加速比

## 如何生成 README

README 使用 Python 脚本根据 `compare_report.json` 自动生成，包含真实推理结果和数据表格。

## 如何生成终端截图

使用 `terminal_screenshot.py` 工具：

```python
from terminal_screenshot import render_terminal_screenshot
text = "推理输出文本..."
render_terminal_screenshot(text, "terminal_screenshot.png")
```

## 如何串行执行多个模型避免显存爆炸

使用 `run_all.sh` 脚本，每个模型处理完成后自动释放 NPU 显存：

```bash
bash scripts/run_all.sh
```

关键释放代码：
```python
import gc
gc.collect()
torch.npu.empty_cache()
```

## 如何调用 GitCode API 提交模型仓库

```bash
# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name": "model-name-npu", "repository_type": "model", "visibility": "public"}'

# 推送代码
git init
git checkout -b main
git add .
git commit -m "Add model NPU adaptation"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/USERNAME/REPO.git"
git push -u origin main
```

## 精度测试结果汇总

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

**全部 11 个模型的 NPU 与 CPU 推理结果误差均 < 1%。**
