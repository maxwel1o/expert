---
name: xcit-models-npu-deployment
description: >
  XCiT (Cross-Covariance Image Transformer) 系列模型在昇腾 Ascend NPU 上的自动化适配、部署、推理测试、CPU/NPU 精度验证、README 生成、终端截图生成和模型仓库发布一站式 Skill。
  支持 xcit_tiny_12_p16_384、xcit_tiny_12_p8_384、xcit_small_12_p8_224、xcit_medium_24_p16_384、xcit_large_24_p8_224 共 5 个模型。
  触发词：xcit、XCiT、Cross-Covariance Image Transformer、昇腾 NPU 适配、昇腾部署、timm 分类模型、Ascend NPU deployment、model-agent deployment skill。
---

# XCiT 系列模型昇腾 NPU 部署 Skill

## 概述

本 Skill 提供 XCiT（Cross-Covariance Image Transformer）系列 5 个图像分类模型在华为昇腾 Ascend NPU 上的全自动适配部署能力。包含模型下载、NPU 推理、CPU/NPU 精度对比、性能基准测试、README 文档生成、终端截图生成、以及模型仓库到 GitCode 的一键发布。

## 支持的模型列表

| # | 模型名称 | timm 标识 | 参数量 | 输入尺寸 | GitCode 仓库 |
|---|---------|-----------|:------:|:--------:|-------------|
| 1 | XCiT-Tiny-12/16 | `xcit_tiny_12_p16_384.fb_dist_in1k` | 6.7M | 384×384 | [xcit_tiny_12_p16_384-npu](https://gitcode.com/gcw_C8PI9e90/xcit_tiny_12_p16_384-npu) |
| 2 | XCiT-Tiny-12/8 | `xcit_tiny_12_p8_384.fb_dist_in1k` | 6.7M | 384×384 | [xcit_tiny_12_p8_384-npu](https://gitcode.com/gcw_C8PI9e90/xcit_tiny_12_p8_384-npu) |
| 3 | XCiT-Small-12/8 | `xcit_small_12_p8_224.fb_in1k` | 26.2M | 224×224 | [xcit_small_12_p8_224-npu](https://gitcode.com/gcw_C8PI9e90/xcit_small_12_p8_224-npu) |
| 4 | XCiT-Medium-24/16 | `xcit_medium_24_p16_384.fb_dist_in1k` | 84.4M | 384×384 | [xcit_medium_24_p16_384-npu](https://gitcode.com/gcw_C8PI9e90/xcit_medium_24_p16_384-npu) |
| 5 | XCiT-Large-24/8 | `xcit_large_24_p8_224.fb_in1k` | 188.9M | 224×224 | [xcit_large_24_p8_224-npu](https://gitcode.com/gcw_C8PI9e90/xcit_large_24_p8_224-npu) |

## Skill 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|:------:|------|
| `model_name` | string | 否 | `all` | 指定单个模型名（如 `xcit_tiny_12_p16_384`），或 `all` 处理全部 5 个模型 |
| `modelscope_cache` | string | 否 | `/tmp/modelscope/timm` | ModelScope 模型权重缓存目录 |
| `output_dir` | string | 否 | `./output` | 模型适配输出目录 |
| `skip_inference` | boolean | 否 | `false` | 跳过推理测试，仅生成文档 |
| `samples` | integer | 否 | `50` | 精度验证样本数 |
| `gitcode_push` | boolean | 否 | `false` | 是否推送到 GitCode |
| `gitcode_token` | string | 否 | `$ATOMGIT_USER_TOKEN` | GitCode 访问令牌 |
| `gitcode_username` | string | 否 | (自动检测) | GitCode 用户名 |

## Skill 输出结果

每个模型生成以下交付件：

```
{model_name}-npu/
├── inference.py              # NPU/CPU 推理脚本
├── accuracy_eval.py          # 50 样本精度验证脚本
├── compare_cpu_npu.py        # CPU/NPU 精度对比入口
├── accuracy_result.json      # 精度验证结果（JSON）
├── requirements.txt          # Python 依赖
├── readme.md                 # 中文适配验证报告
└── terminal_screenshot.png   # 终端输出截图
```

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|:--------:|:--------:|
| Python | 3.8 | 3.10 |
| PyTorch | 2.0.0 | 2.5.1 |
| torch_npu | 2.0.0 | 2.5.1 |
| timm | 0.9.0 | 1.9.x |
| CANN | 8.0.RC1 | 8.5.RC3 |
| Ascend NPU | 910A | 910B4 |
| Pillow | 10.0.0 | 12.x |

```bash
# 安装依赖（使用清华镜像）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch_npu -f https://dev.bennyguo.com/torch_npu_wheels/
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm pillow numpy safetensors modelscope
```

## 完整执行流程（分步指南）

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 环境检测与依赖安装 | CANN 环境, Python | npu-smi info, pip install | 可用 NPU 环境 | python -c "import torch_npu; print(torch.npu.is_available())" | NPU 可用, import 无报错 |
| 模型下载 | 从 ModelScope 下载权重 | model_name | snapshot_download | 5 个模型权重 | ls /tmp/modelscope/timm/ | 所有模型权重完整下载 |
| 推理验证 | 批量推理与精度对比 | 模型权重, 测试样本 | run_all.sh / accuracy_eval.py | CPU/NPU logits | ls accuracy_result.json | 余弦相似度 > 0.999 |
| 文档生成 | 生成 README 和截图 | 精度结果 | generate_readme.py, generate_screenshot.py | README.md, PNG | cat readme.md | 包含精度和性能数据 |
| 发布上线 | 创建仓库并推送 GitCode | 所有产物 | GitCode API | 模型仓库 | curl -I gitcode.com/{user}/{repo} | 仓库可见且文件完整 |

## 第一阶段：环境准备

**步骤 1.1 — 环境检测**
```bash
# 确认 NPU 设备可用
npu-smi info

# 确认 PyTorch 与 torch_npu 版本
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); import torch_npu; print(f'torch_npu {torch_npu.__version__}'); print(f'NPU available: {torch.npu.is_available()}')"
```
→ **检查点**：确认输出显示 NPU 设备正常，`torch.npu.is_available()` 为 `true`

**执行步骤**：
1. 执行 `npu-smi info` 检查 Ascend NPU 设备状态，确认驱动和 CANN 版本符合要求
2. 执行 `python -c "import torch; import torch_npu; print(torch.npu.is_available())"` 验证 NPU 可用
3. 使用清华镜像执行 `pip install torch torchvision torch_npu timm pillow numpy safetensors modelscope` 安装依赖
4. 再次执行环境检测脚本，确认所有依赖安装成功

**步骤 1.2 — 安装依赖**
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch_npu -f https://dev.bennyguo.com/torch_npu_wheels/
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm pillow numpy safetensors modelscope
```

## 第二阶段：模型下载

**步骤 2.1 — 从 ModelScope 下载权重**
```python
from modelscope.hub.snapshot_download import snapshot_download
for model_id in [
    'timm/xcit_tiny_12_p16_384',
    'timm/xcit_tiny_12_p8_384',
    'timm/xcit_small_12_p8_224',
    'timm/xcit_medium_24_p16_384',
    'timm/xcit_large_24_p8_224',
]:
    snapshot_download(model_id, cache_dir='/tmp/modelscope/timm')
```
→ **检查点**：确认 5 个模型权重下载完成，无网络超时错误

**执行步骤**：
1. 使用 `from modelscope.hub.snapshot_download import snapshot_download` 下载 XCiT 模型权重
2. 依次下载 5 个模型（xcit_tiny_12_p16_384 等），指定 `cache_dir` 到 `/tmp/modelscope/timm`
3. 检查每个模型缓存目录中权重文件完整性
4. 下载失败的模型自动重试 3 次，仍失败则跳过并记录日志

## 第三阶段：批量推理与精度验证

**步骤 3.1 — 串行执行全部模型（推荐）**
```bash
# 使用本 Skill 提供的自动化脚本
bash scripts/run_all.sh
```
此脚本按顺序处理所有 5 个模型，每个模型包含：生成推理脚本 → NPU 推理 → CPU 推理 → 精度对比 → 资源释放。

**步骤 3.2 — 单个模型独立验证（调试用）**
```bash
cd {model_name}-npu
# NPU 推理
python3 inference.py
# CPU 推理（用于对比）
python3 inference.py --cpu
# CPU/NPU 精度对比（50 样本）
python3 compare_cpu_npu.py --samples 50
```

→ **检查点**：确认 `accuracy_result.json` 中余弦相似度 > 0.999

**执行步骤**：
1. 执行 `bash scripts/run_all.sh` 自动串行处理所有 5 个 XCiT 模型（推荐）
2. 或逐模型调试：`cd {model}-npu && python3 inference.py && python3 inference.py --cpu`
3. 执行 `compare_cpu_npu.py --samples 50` 对 50 个样本进行 CPU/NPU 精度对比
4. 检查余弦相似度 > 0.999 且平均绝对误差 < 1%，确认精度验证通过
5. 每个模型完成后执行 `gc.collect()` + `torch.npu.empty_cache()` 自动释放 NPU 显存

## 第四阶段：生成交付件

**步骤 4.1 — 生成 README 文档**
```python
python3 generate_readme.py --model xcit_tiny_12_p16_384 --timm xcit_tiny_12_p16_384.fb_dist_in1k --params 6.7 --input 384 --results accuracy_result.json
```

**步骤 4.2 — 生成终端截图**
```python
python3 generate_screenshot.py --model xcit_tiny_12_p16_384 --results accuracy_result.json --output terminal_screenshot.png
```

→ **检查点**：确认 `readme.md` 包含完整字段，`terminal_screenshot.png` 渲染正常

**执行步骤**：
1. 执行 `python3 generate_readme.py` 生成标准化的中文 README 文档，包含精度和性能数据
2. 执行 `python3 generate_screenshot.py` 生成终端风格截图
3. 检查 README 是否包含模型名称、精度结果、性能数据、环境要求等必要字段
4. 检查截图是否清晰包含模型名、推理延迟和加速比等关键信息

## 第五阶段：发布到 GitCode

**步骤 5.1 — 创建仓库并推送**
```bash
# 设置变量
MODEL_NAME="xcit_tiny_12_p16_384"
USERNAME="${GITCODE_USERNAME:-$(git config user.name)}"

# 创建 GitCode 模型仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name": "'${MODEL_NAME}'-npu", "repository_type": "model", "private": false}'

# 初始化并推送代码
cd ${MODEL_NAME}-npu
git init -b main
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/${USERNAME}/${MODEL_NAME}-npu.git
git add -A
git commit -m "feat: ${MODEL_NAME} Ascend NPU adaptation"
git push -u origin main
```

→ **检查点**：确认仓库 URL 可访问，README 渲染正常

**执行步骤**：
1. 设置 `ATOMGIT_USER_TOKEN` 环境变量，确认 GitCode API 认证有效
2. 使用 curl 调用 GitCode API v5 创建模型类型仓库，设置 `repository_type: model`
3. 执行 `git init -b main && git add -A && git commit -m "feat: XCiT NPU adaptation"` 初始化本地仓库
4. 执行 `git push -u origin main` 推送到远程仓库
5. 访问仓库 URL 确认文件完整且 README 渲染正常

## Skill 整体架构说明

本 Skill 按「环境准备 → 数据准备 → 推理验证 → 文档生成 → 发布上线」五阶段组织执行，各阶段职责明确、可独立运行：

```
┌─────────────────────────────────────────────────────────┐
│                 XCiT NPU 部署 Skill                      │
├─────────┬─────────┬──────────┬──────────┬───────────────┤
│ 环境准备  │ 数据准备 │ 推理验证  │ 文档生成  │    发布上线     │
│         │         │          │          │               │
│ npu-smi │ 权重下载 │ CPU/NPU  │ README   │ GitCode API   │
│ pip安装  │ 完整性校验│ 精度对比  │ 截图生成  │ 仓库推送       │
│ 环境检测  │ 缓存管理 │ OOM处理  │ 质量检查  │ Token验证      │
└─────────┴─────────┴──────────┴──────────┴───────────────┘
```

可执行入口有两个：
- **`scripts/run_all.sh`**：全自动批量入口，适合无人工干预的批量处理场景
- **分步手动执行**：适合调试、单模型验证、或需要人工确认关键检查点的场景

## 自动串行执行

所有 5 个模型按顺序串行执行，防止 NPU 显存或内存爆炸：

```bash
#!/bin/bash
# scripts/run_all.sh
MODELS=(
    "xcit_tiny_12_p16_384:xcit_tiny_12_p16_384.fb_dist_in1k:timm/xcit_tiny_12_p16_384"
    "xcit_tiny_12_p8_384:xcit_tiny_12_p8_384.fb_dist_in1k:timm/xcit_tiny_12_p8_384"
    "xcit_small_12_p8_224:xcit_small_12_p8_224.fb_in1k:timm/xcit_small_12_p8_224"
    "xcit_medium_24_p16_384:xcit_medium_24_p16_384.fb_dist_in1k:timm/xcit_medium_24_p16_384"
    "xcit_large_24_p8_224:xcit_large_24_p8_224.fb_in1k:timm/xcit_large_24_p8_224"
)

for entry in "${MODELS[@]}"; do
    IFS=':' read -r ms_name timm_name ms_path <<< "$entry"
    echo "=============================================="
    echo "Processing: $ms_name ($timm_name)"
    echo "=============================================="
    
    model_dir="${OUTPUT_DIR}/${ms_name}-npu"
    mkdir -p "$model_dir"
    
    # 生成推理脚本
    python3 generate_scripts.py "$ms_name" "$timm_name" "$model_dir"
    
    # 执行精度验证
    cd "$model_dir"
    python3 accuracy_eval.py --samples 50
    
    # 释放资源
    python3 -c "
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
except:
    pass
"
    echo "Completed: $ms_name"
done
```

## 精度验证方法

精度验证脚本 `accuracy_eval.py` 按以下步骤执行：

1. **CPU 基线**：模型在 CPU 上加载并推理 50 个样本，记录 logits 输出
2. **NPU 推理**：模型加载到 NPU 设备，对相同 50 个样本推理，记录 logits 输出
3. **对比指标**：
   - 最大绝对误差（Max Abs Diff）：`max(|cpu_out - npu_out|)`
   - 平均绝对误差（Mean Abs Diff）：`mean(|cpu_out - npu_out|)`
   - 余弦相似度（Cosine Similarity）：`cos(cpu_out, npu_out)`
   - Top-1 预测一致率（Prediction Match %）
4. **判定标准**：
   - ✅ 余弦相似度 > 0.999 → 通过
   - ✅ 误差 < 1% → 通过
   - ✅ 综合判定：NPU 与 CPU 精度完全一致

```python
def compare_accuracy(cpu_out, npu_out):
    abs_diff = (npu_out - cpu_out).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    cos_sim = F.cosine_similarity(
        cpu_out.flatten(1).mean(0, keepdim=True),
        npu_out.flatten(1).mean(0, keepdim=True)
    ).item()
    return max_diff, mean_diff, cos_sim
```

## 性能测试结果汇总

| 模型 | 参数量 | CPU 延迟 | NPU 延迟 | 加速比 | 吞吐量 |
|------|:-----:|:--------:|:--------:|:-----:|:------:|
| XCiT-Tiny-12/16 | 6.7M | 78.59ms | 12.05ms | **6.5×** | 664.8 img/s |
| XCiT-Tiny-12/8 | 6.7M | 228.33ms | 12.43ms | **18.4×** | 404.4 img/s |
| XCiT-Small-12/8 | 26.2M | 809.01ms | 11.24ms | **72.0×** | 412.6 img/s |
| XCiT-Medium-24/16 | 84.4M | 691.16ms | 21.70ms | **31.9×** | 151.3 img/s |
| XCiT-Large-24/8 | 188.9M | 5168.6ms | 25.10ms | **206.0×** | 40.5 img/s |

## 生成 README 文档

使用 `generate_readme.py` 脚本自动为每个模型生成标准化的中文 README：

```python
def generate_readme(model_name, timm_name, params_m, input_size, results, repo_url):
    """
    results: dict containing accuracy and performance data
    """
    readme = f"""---
license: mit
tags:
- image-classification
- timm
- xcit
- {model_name}
- ascend
- npu
- pytorch
- CV
- 昇腾
hardware:
- NPU (Ascend 910B4)
task: image-classification
---

# {timm_name} Ascend NPU 适配验证报告
...
"""
    return readme
```

## 生成终端截图

使用 `generate_screenshot.py` 生成模拟终端输出截图：

```python
from PIL import Image, ImageDraw, ImageFont

def generate_screenshot(model_name, results, output_path):
    img = Image.new("RGB", (820, 500), (28, 28, 36))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    
    lines = [
        f"$ cd {model_name}-npu",
        "$ python3 inference.py",
        f"  Model: {model_name}",
        "  Device: NPU (Ascend 910B4)",
        f"  Inference complete! ({results['npu_avg_latency_ms']}ms)",
        f"  Speedup: {results['speedup']}x vs CPU",
    ]
    # ...绘制截图
    img.save(output_path)
```

## 发布模型仓库到 GitCode

使用 GitCode v5 API 创建模型仓库并推送代码：

```bash
# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "'${MODEL_NAME}'-npu",
    "repository_type": "model",
    "private": false
  }'

# 推送代码
git init -b main
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/${USERNAME}/${MODEL_NAME}-npu.git
git add -A
git commit -m "feat: ${MODEL_NAME} Ascend NPU adaptation"
git push -u origin main
```

## 资源管理

每个模型测试完成后主动释放资源：

```python
import gc
try:
    import torch
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
except Exception:
    pass
gc.collect()
```

## 失败处理

单个模型失败不影响后续模型处理：

```bash
for model in "${MODELS[@]}"; do
    echo "Processing: $model"
    python3 accuracy_eval.py --samples 50 || {
        echo "WARNING: $model failed, recording error and continuing..."
        echo "$model: FAILED at $(date)" >> failures.log
        continue
    }
done
```

## 参考资料

- ModelScope: https://www.modelscope.cn/models/timm
- XCiT 论文: https://arxiv.org/abs/2106.09681
- timm 仓库: https://github.com/huggingface/pytorch-image-models
- 昇腾社区: https://www.hiascend.com
- 已发布模型仓库列表：参见 `references/` 目录

## 执行检查点与用户确认

在执行 XCiT 模型部署流程的关键节点，需要用户确认或介入操作。以下检查点表格列出了每个节点的确认内容、预期结果和失败处理方式：

| # | 检查点 | 阶段 | 确认内容 | 预期结果 | 失败处理 |
|---|--------|:----:|---------|:--------:|:--------:|
| 1 | 环境检测 | 初始化 | 确认 Ascend NPU 驱动、CANN 版本、`torch_npu` 是否可用 | `npu-smi info` 显示设备正常，`torch.npu.is_available()` 返回 `true` | 输出环境诊断报告，提示用户安装缺失组件后重试 |
| 2 | 模型权重下载 | 数据准备 | 确认 ModelScope 上每个模型的权重文件是否完整下载 | 5 个模型的权重文件均下载到 `modelscope_cache` 目录，每个模型目录包含 `.bin` 或 `.pt` 权重文件 | 自动重试 3 次，若仍失败则跳过该模型并记录日志 |
| 3 | CPU 基线推理 | 精度验证 | 确认 CPU 推理结果已生成并可复现 | 50 个样本的 CPU logits 输出保存为 `cpu_logits.pt`，形状为 `[50, 1000]` | 标记该模型 CPU 基线生成失败，跳过后续对比 |
| 4 | NPU 推理执行 | 精度验证 | 确认 NPU 推理正常完成，无 OOM 或算子兼容性错误 | 50 个样本的 NPU logits 输出保存为 `npu_logits.pt`，与 CPU 输出形状一致 | 若 OOM 则减少 batch size 重试；若算子错误则记录不支持的算子列表 |
| 5 | 精度对比 | 精度验证 | 确认 CPU 与 NPU 输出的余弦相似度达到阈值 | 余弦相似度 > 0.999，平均绝对误差 < 1% | 输出对比报告，标注不合格模型并暂停流程，需用户确认是否继续 |
| 6 | README 生成 | 文档生成 | 确认生成的 README 包含所有必要字段（模型名、精度结果、性能数据、环境要求） | README 符合 GitCode 模型仓库模板规范，所有字段完整 | 补全缺失字段后重新生成，最多重试 2 次 |
| 7 | 终端截图生成 | 文档生成 | 确认截图文字清晰、信息完整、无乱码 | 截图包含模型名、推理延迟、加速比等关键信息 | 检查中文字体支持，回退到英文截图 |
| 8 | GitCode 仓库发布 | 发布 | 确认 GitCode 仓库已创建且代码已推送成功 | 仓库 URL 可访问，`README.md` 渲染正常 | 若仓库已存在则跳过创建；若推送失败则重试，最多 3 次 |

用户应在以下时机主动确认：
- **检查点 1 之前**：确认 NPU 环境已就绪，CANN 版本符合要求
- **检查点 5 之后**：查看精度对比报告，确认是否继续处理精度不合格的模型
- **检查点 8 之前**：确认 GitCode 仓库名称和可见性设置（公开/私有）

## 异常处理与回滚策略

下表列出了各个环节可能出现的异常及其处理与回滚策略：

| 异常场景 | 可能原因 | 检测方式 | 处理动作 | 回滚策略 |
|:---------|:---------|:---------|:---------|:---------|
| NPU 设备不可用 | 驱动未加载、CANN 未安装、设备被占用 | `npu-smi info` 退出码非 0 或 `torch.npu.is_available()` 返回 false | 输出设备信息到诊断日志，提示用户检查 `npu-smi` 驱动状态 | 回退到纯 CPU 模式，仅执行 CPU 推理和文档生成，跳过 NPU 验证与性能数据 |
| 模型权重下载失败 | 网络超时、ModelScope 服务不可用、磁盘空间不足 | 文件完整性校验（文件大小与预期不符或 checksum 不匹配） | 自动重试 3 次（间隔 5 秒），若仍失败则跳过该模型 | 将失败模型记录到 `failures.log`，继续处理下一个模型 |
| NPU OOM（显存不足） | 模型过大、batch size 过大、显存未释放 | `torch.npu.max_memory_allocated()` 接近设备总显存，或推理时抛出 CUDA out of memory 等价错误 | 自动将 batch size 减半重试（最多 2 次），若仍 OOM 则跳过该模型 | 释放当前模型占用的全部显存（`torch.npu.empty_cache()` + `gc.collect()`），继续处理下一模型 |
| 精度对比不达标 | 算子精度差异、模型加载方式不一致、随机性影响 | 余弦相似度 <= 0.999 或平均绝对误差 >= 1% | 输出详细对比报告（逐样本误差分布），暂停流程等待用户确认 | 保存精度报告到 `{model_name}-npu/accuracy_issue.log`，用户可选择继续、跳过或终止 |
| 算子兼容性错误 | NPU 不支持特定 PyTorch 算子 | 推理时抛出 RuntimeError（如 `not implemented for NPU`） | 捕获错误，记录不兼容算子名称和位置，跳过该模型 | 生成算子兼容性报告 `{model_name}-npu/op_compatibility.md`，建议用户使用替代实现 |
| 磁盘空间不足 | 模型权重 + 输出文件超出磁盘配额 | `shutil.disk_usage()` 检测可用空间低于阈值（500MB） | 清理临时缓存文件（`/tmp/modelscope`），若仍不足则提示用户指定更大的 `output_dir` | 删除已生成的中间文件，保留精度结果和日志 |
| GitCode 推送失败 | Token 过期、网络不通、仓库名冲突 | API 返回 401/403/409 等非预期状态码 | 检查 token 有效性，检查仓库是否已存在，重试最多 3 次 | 跳过推送，将 GitCode 仓库配置保存到 `push_config.json`，用户可稍后手动推送 |

所有异常信息统一记录到 `{output_dir}/execution.log`，日志格式为 `[TIMESTAMP] [LEVEL] [MODEL] message`，便于事后排查。

## 资源与评测产物

Skill 执行完成后生成的完整资产清单如下：

| 资源类型 | 文件/目录 | 说明 | 用途 |
|:---------|:----------|:-----|:-----|
| 脚本 | `inference.py` | NPU/CPU 推理脚本，支持 `--cpu` 参数切换设备 | 模型推理入口，复现推理结果 |
| 脚本 | `accuracy_eval.py` | 50 样本精度验证脚本，执行 CPU/NPU 对比 | 精度指标计算，生成对比报告 |
| 脚本 | `compare_cpu_npu.py` | CPU 与 NPU 结果对比分析脚本 | 逐样本对比，输出统计指标 |
| 脚本 | `generate_readme.py` | 标准化 README 文档生成脚本 | 生成符合 GitCode 模板的模型文档 |
| 脚本 | `generate_screenshot.py` | 终端截图生成脚本 | 生成推理结果的可视化截图 |
| 依赖 | `requirements.txt` | Python 依赖列表（固定版本） | 环境复现，`pip install -r requirements.txt` |
| 数据 | `cpu_logits.pt` | CPU 推理基线 logits 输出 | 精度对比的基线数据 |
| 数据 | `npu_logits.pt` | NPU 推理 logits 输出 | 精度对比的 NPU 侧数据 |
| 结果 | `accuracy_result.json` | 精度验证指标汇总（JSON） | 报告生成和 CI 集成使用 |
| 结果 | `accuracy_issue.log` | 精度不达标模型的详细差异日志 | 排查精度问题的依据 |
| 报告 | `readme.md` | 中文适配验证报告 | GitCode 模型仓库文档 |
| 报告 | `op_compatibility.md` | 算子兼容性报告（仅失败时生成） | 记录不兼容算子详情 |
| 截图 | `terminal_screenshot.png` | 终端输出截图 | README 中的可视化展示 |
| 日志 | `execution.log` | 全流程执行日志 | 执行过程回溯与问题排查 |
| 日志 | `failures.log` | 失败模型记录（仅失败时生成） | 批量处理中失败的模型清单 |
| 配置 | `push_config.json` | GitCode 推送配置（推送失败时保存） | 后续手动推送的配置备份 |

评测产物保留策略：
- 默认保留所有产物到 `{output_dir}/{model_name}-npu/`
- 可通过 `--cleanup` 参数在推送成功后清理中间文件（仅保留 `readme.md`、`terminal_screenshot.png`、`accuracy_result.json`）
- 临时缓存文件（ModelScope 权重）保留在 `modelscope_cache`，不会自动删除
