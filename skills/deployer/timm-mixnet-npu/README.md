# timm-mixnet-npu Skill

## 概述

本 Skill 用于完成 **timm MixNet** 系列图像分类模型在华为昇腾 NPU 上的端到端适配与验证。支持模型的 NPU 推理、CPU/NPU 精度对比、测试结果整理、终端截图生成和模型仓库发布。

## 支持的模型列表

| 模型名称 | 架构 | 参数量 | 输入尺寸 | GitCode 仓库 |
| --- | --- | ---: | --- | --- |
| `tf_mixnet_s.in1k` | MixNet Small | ~2.0M | 3×224×224 | [tf_mixnet_s.in1k-npu](https://gitcode.com/m0_74196153/tf_mixnet_s.in1k-npu) |
| `tf_mixnet_m.in1k` | MixNet Medium | ~3.4M | 3×224×224 | [tf_mixnet_m.in1k-npu](https://gitcode.com/m0_74196153/tf_mixnet_m.in1k-npu) |
| `tf_mixnet_l.in1k` | MixNet Large | ~5.0M | 3×224×224 | [tf_mixnet_l.in1k-npu](https://gitcode.com/m0_74196153/tf_mixnet_l.in1k-npu) |

## Skill 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model_name` | string | 是 | - | 模型名称: `tf_mixnet_s.in1k`, `tf_mixnet_m.in1k`, `tf_mixnet_l.in1k` |
| `checkpoint_dir` | string | 是 | - | 模型权重文件所在目录（包含 model.safetensors） |
| `device` | string | 否 | `npu` | 推理设备: `cpu` 或 `npu` |
| `seed` | int | 否 | 42 | 随机种子 |

## Skill 输出结果

| 输出文件 | 说明 |
| --- | --- |
| `logits_{device}.pt` | 模型输出的全量 logits（PyTorch 张量） |
| `output_{device}.json` | Top-5 预测结果（类别 ID 和概率） |
| `comparison_results.json` | CPU/NPU 精度对比指标 |
| `terminal_screenshot.png` | 模拟终端输出截图 |

## 执行 NPU 推理

```bash
# 单个模型推理
python3 scripts/inference.py \
  --model tf_mixnet_s.in1k \
  --checkpoint-dir /path/to/weights \
  --device npu
```

## 执行 CPU/NPU 精度对比

先分别运行 CPU 和 NPU 推理保存 logits，然后执行对比：

```bash
# 1. CPU 推理
python3 scripts/inference.py --model tf_mixnet_s.in1k --checkpoint-dir /path/to/weights --device cpu

# 2. NPU 推理
python3 scripts/inference.py --model tf_mixnet_s.in1k --checkpoint-dir /path/to/weights --device npu

# 3. 精度对比
python3 scripts/compare_cpu_npu.py
```

### 精度指标说明

| 指标 | 说明 | 通过标准 |
| --- | --- | --- |
| MAE/Mean(\|CPU\|) | 平均绝对误差与 CPU 输出均值的比率 | < 1% |
| Cosine Similarity | CPU 与 NPU 输出余弦相似度 | > 0.999 |
| Top-1 Match | CPU 和 NPU 的 Top-1 分类是否一致 | 一致 |
| Top-5 Overlap | CPU 和 NPU 的 Top-5 分类是否一致 | 一致 |

## 串行执行多个模型

为防止 NPU 显存溢出，多个模型必须串行执行。可使用 `run_all.sh` 脚本：

```bash
bash scripts/run_all.sh /path/to/models/dir
```

该脚本按顺序处理每个模型：CPU 推理 → NPU 推理 → 精度对比 → 结果保存 → 释放显存。

## 生成 README

每个模型的 README 已在模型仓库中提供，包含：
- 模型介绍和原始地址
- 任务类型和框架信息
- 输入输出格式说明
- 环境依赖和 NPU 适配说明
- 推理命令和结果
- CPU/NPU 精度测试数据和结论（含表格）
- 推理性能对比
- 终端截图
- 模型标签

## 生成终端截图

```bash
python3 /path/to/terminal_screenshot.py \
  --input terminal_output.txt \
  --output terminal_screenshot.png
```

## 调用 GitCode API 提交模型仓库

```bash
# 创建仓库
curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "name": "tf_mixnet_s.in1k-npu",
    "description": "tf_mixnet_s.in1k adapted for Ascend NPU",
    "visibility": "public",
    "repository_type": "model"
  }'

# 推送代码
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/<repo-name>.git
git branch -M main
git push -u origin main
```

## 验证结果摘要

| 模型 | NPU 推理 | CPU/NPU 误差 | Top-1 匹配 | NPU 加速比 |
| --- | :---: | :---: | :---: | :---: |
| tf_mixnet_s.in1k | ✅ 通过 | 0.158% | ✅ 一致 | 4.12x |
| tf_mixnet_m.in1k | ✅ 通过 | 0.054% | ✅ 一致 | 4.48x |
| tf_mixnet_l.in1k | ✅ 通过 | 0.053% | ✅ 一致 | 5.51x |
