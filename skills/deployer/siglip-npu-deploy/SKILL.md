---
name: siglip-npu-deploy
description: >
  SigLIP/SigLIP2 视觉模型（15 个 ViT-B/ViT-L 变体）在昇腾 NPU 上的完整部署、
  推理验证、CPU/NPU 精度对比与模型仓库发布 Skill。
  支持串行处理多模型，避免显存爆炸。适用于 open_clip SigLIP/SigLIP2 系列在
  Ascend910 上的批量适配与验证。当用户提到 SigLIP、SigLIP2、CLIP、open_clip、
  图像特征提取 NPU 部署时触发。
metadata:
  short-description: SigLIP/SigLIP2 系列昇腾 NPU 批量部署与验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, siglip, siglip2, open_clip, vit, image-feature-extraction, pytorch, inference]
---

# SigLIP/SigLIP2 昇腾 NPU 部署与精度验证 Skill

本 Skill 提供 15 个 SigLIP/SigLIP2 视觉模型在华为昇腾 NPU 上的完整部署、
推理验证和 CPU/NPU 精度对比的标准化可复现流程。

## 支持的模型列表

| # | 模型名称 | 分辨率 | 嵌入维度 | 架构 | CPU 耗时(s) | NPU 耗时(s) | 加速比 | 最大相对误差 | 余弦相似度 |
|---|---------|--------|----------|------|------------|------------|--------|-------------|-----------|
| 1 | ViT-B-16-SigLIP | 224 | 768 | ViT-B/16 | 0.66 | 0.2262 | 2.93x | 0.079% | 0.999995 |
| 2 | ViT-B-16-SigLIP-256 | 256 | 768 | ViT-B/16 | 0.86 | 0.0072 | 120.52x | 0.064% | 0.999997 |
| 3 | ViT-B-16-SigLIP-384 | 384 | 768 | ViT-B/16 | 2.12 | 0.0069 | 304.86x | 0.128% | 0.999994 |
| 4 | ViT-B-16-SigLIP-512 | 512 | 768 | ViT-B/16 | 4.27 | 0.0241 | 177.31x | 0.108% | 0.999994 |
| 5 | ViT-B-16-SigLIP-i18n-256 | 256 | 768 | ViT-B/16 | 0.85 | 0.0060 | 140.66x | 0.084% | 0.999997 |
| 6 | ViT-B-16-SigLIP2 | 224 | 768 | ViT-B/16 | 0.65 | 0.0060 | 108.06x | 0.094% | 0.999996 |
| 7 | ViT-B-16-SigLIP2-256 | 256 | 768 | ViT-B/16 | 0.89 | 0.2051 | 4.32x | 0.064% | 0.999996 |
| 8 | ViT-B-16-SigLIP2-384 | 384 | 768 | ViT-B/16 | 2.09 | 0.0059 | 352.62x | 0.089% | 0.999994 |
| 9 | ViT-B-16-SigLIP2-512 | 512 | 768 | ViT-B/16 | 4.36 | 0.0078 | 559.22x | 0.078% | 0.999995 |
| 10 | ViT-B-32-SigLIP2-256 | 256 | 768 | ViT-B/32 | 0.25 | 0.0070 | 35.49x | 0.005% | 1.000000 |
| 11 | ViT-L-16-SigLIP-256 | 256 | 1024 | ViT-L/16 | 3.05 | 0.2116 | 14.42x | 0.188% | 0.999988 |
| 12 | ViT-L-16-SigLIP-384 | 384 | 1024 | ViT-L/16 | 7.73 | 0.2170 | 35.64x | 0.489% | 0.999863 |
| 13 | ViT-L-16-SigLIP2-256 | 256 | 1024 | ViT-L/16 | 3.14 | 0.2076 | 15.11x | 0.217% | 0.999974 |
| 14 | ViT-L-16-SigLIP2-384 | 384 | 1024 | ViT-L/16 | 7.87 | 0.0131 | 600.01x | 0.157% | 0.999974 |
| 15 | ViT-L-16-SigLIP2-512 | 512 | 1024 | ViT-L/16 | 13.89 | 0.0338 | 410.49x | 0.005% | 1.000000 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS |
| CANN | >= 8.0 |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网从 ModelScope 下载模型权重 |
| 磁盘 | 至少 20GB 可用空间（缓存 15 个模型权重约 12GB） |

## 完整工作流程

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 0. 环境初始化 | NPU 服务器, CANN 包 | 加载 CANN 环境, 选择空闲 NPU, 配置镜像 | 就绪的 NPU 环境, ASCEND_RT_VISIBLE_DEVICES | `npu-smi info` 显示设备列表 | 至少 1 张 NPU 卡, CANN >= 8.0 |
| 环境准备 | 1. 安装依赖 | Python 3.9+, pip | pip 安装 torch, torch_npu, open_clip, modelscope | 完整的 Python 依赖环境 | `python -c "import torch_npu; print(torch_npu.npu.is_available())"` | 返回 True |
| 环境准备 | 2. NPU 基础验证 | torch_npu 已安装 | 运行 Tensor 创建与 NPU 计算测试 | 验证 NPU 计算链路正常 | `python -c "import torch; import torch_npu; a=torch.randn(3,4).npu(); print(a+a)"` | 输出包含 `device='npu:0'` 的 Tensor 且无报错 |
| 模型处理 | 3.1 权重下载 | 模型名称列表 | 通过 ModelScope 下载 15 个模型权重 | 权重文件在 ~/.cache/modelscope/ | `ls ~/.cache/modelscope/timm/*/open_clip_model.safetensors` | 文件存在且大小 > 0 |
| 模型处理 | 3.2 CPU 推理 | 权重文件, 测试图像 | 运行 open_clip 推理脚本, --device cpu | CPU 推理结果 (特征向量) | `python scripts/infer_siglip.py --model NAME --image test.jpg --device cpu` | 输出维度正确 (ViT-B: 768, ViT-L: 1024) |
| 模型处理 | 3.3 NPU 推理 | 权重文件, 测试图像 | 运行 open_clip 推理脚本, --device npu | NPU 推理结果 (特征向量) | `python scripts/infer_siglip.py --model NAME --image test.jpg --device npu` | 输出维度正确, 无报错 |
| 模型处理 | 3.4 精度对比 | CPU/NPU 推理结果 | 计算最大相对误差, 余弦相似度 | 精度指标数据 results.json | 查看 results.json 中的指标字段 | 最大相对误差 < 1%, 余弦相似度 > 0.999 |
| 发布 | 4. 推送到 GitCode | 模型产物 (脚本, README, 结果) | 创建 GitCode 模型仓库, 推送代码 | GitCode 远程仓库 | `git push -u origin main` | 推送成功, 仓库公开可访问 |
| 发布 | 5. 验收确认 | 全部 15 个模型结果 | 逐项检查精度、产物完整性 | 验收通过确认 | 对照检查清单逐项核对 | 所有项目通过 |

按以下各节顺序执行，每步完成后再进入下一步。

---

## 执行检查点与用户确认

在以下关键节点暂停执行，向用户报告当前状态并等待确认后再继续：

| 检查点 | 触发时机 | 确认内容 | 不通过处理 |
|--------|---------|---------|-----------|
| ✅ CP-1 环境就绪 | 步骤 0 完成后 | `npu-smi info` 显示至少 1 张 NPU 卡，CANN >= 8.0 | 检查硬件在位，重新加载 CANN 环境 |
| ✅ CP-2 依赖安装 | 步骤 1 完成后 | `python -c "import torch_npu; print(torch_npu.npu.is_available())"` 返回 True | 检查 pip 源与 torch-npu/CANN 版本匹配 |
| ✅ CP-3 NPU 基础验证 | 步骤 2 完成后 | Tensor 输出包含 `device='npu:0'` 且无报错 | 重新设置 `ASCEND_RT_VISIBLE_DEVICES` |
| ✅ CP-4 模型权重就绪 | 步骤 3.1 完成后 | 确认每个模型权重文件存在且大小 > 0 | 重试 ModelScope 下载或更换源 |
| ✅ CP-5 CPU 推理 | 首个模型 CPU 推理后 | 输出维度正确（ViT-B: 768, ViT-L: 1024） | 检查模型加载参数与权重路径 |
| ✅ CP-6 精度对比 | 每个模型精度对比后 | 最大相对误差 < 1%，余弦相似度 > 0.999 | 记录差异模型，判断是否重跑 |
| ✅ CP-7 产物完整性 | 全部模型处理完成后 | 每个模型生成了 README、results.json、推理脚本 | 回到对应步骤修复 |
| ✅ CP-8 验收确认 | 推送前 | 所有 15 个模型精度达标，仓库推送就绪 | 修复未达标的模型 |

用户确认指令：`通过` / `继续` 进入下一步，`重试 cp-N` 重新执行对应检查点的步骤。

---

## 0. 环境初始化

**输入**: NPU 服务器（至少 1 张 Ascend910 卡）, CANN 安装包 >= 8.0
**输出**: 就绪的 NPU 环境变量, 已选的空闲 NPU 卡

**执行步骤**:
1. 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 CANN 环境变量
2. 运行 `npu-smi info` 确认设备状态，记录可用 NPU 卡编号
3. 执行 `export ASCEND_RT_VISIBLE_DEVICES=0` 选择空闲 NPU 卡
4. 执行 `export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` 配置阿里云 PyPI 镜像
5. 确认环境变量生效：`echo $ASCEND_RT_VISIBLE_DEVICES`

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0

# 配置阿里云 PyPI 镜像
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
```

---

## 1. 安装依赖

**输入**: Python 3.9-3.13, pip 包管理器, 已加载 CANN 环境
**输出**: 完整的 Python 依赖环境（torch, torch_npu, open_clip_torch 等）

**执行步骤**:
1. 确认 Python 版本满足要求：`python3 --version`（需 >= 3.9）
2. 运行 `pip install torch torch_npu open_clip_torch safetensors Pillow modelscope` 安装全部依赖
3. 验证 torch_npu 可导入：`python -c "import torch_npu; print(torch_npu.__version__)"`
4. 验证 NPU 可用性：`python -c "import torch_npu; print(torch_npu.npu.is_available())"` 应返回 True

```bash
pip install torch torch_npu open_clip_torch safetensors Pillow modelscope
```

---

## 2. NPU 基础验证

**输入**: torch_npu 已正确安装, CANN 环境已加载
**输出**: NPU 计算链路验证通过, Tensor 可在 NPU 上正常计算

**执行步骤**:
1. 执行 `python3 -c "import torch; import torch_npu; print(torch.npu.is_available())"` 验证 NPU 可用
2. 确认 NPU 设备数量：`python3 -c "print(torch.npu.device_count())"` 应 >= 1
3. 运行 Tensor NPU 计算测试：`python3 -c "import torch; import torch_npu; a=torch.randn(3,4).npu(); print(a+a)"`
4. 确认输出包含 `device='npu:0'` 的 Tensor 且无任何报错信息

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
a = torch.randn(3, 4).npu()
print(a + a)
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错。

---

## 3. 串行处理所有模型

**输入**: 15 个 SigLIP/SigLIP2 模型名称列表, 测试图像文件 test.jpg, 已安装的 open_clip_torch
**输出**: 每个模型的 CPU/NPU 推理结果, 精度对比数据, 模型仓库产物

**执行步骤**:
1. 确认 scripts/infer_siglip.py 存在于当前目录且可执行
2. 准备测试图像 test.jpg（建议使用标准 512x512 自然图像）
3. 对每个模型依次执行: 权重下载 → CPU 推理 → NPU 推理 → 精度对比 → 记录结果
4. 每个模型处理完成后调用 `torch.npu.empty_cache()` 和 `gc.collect()` 释放显存
5. 所有 15 个模型处理完成后汇总结果至 results/summary.json

使用统一推理脚本 `scripts/infer_siglip.py` 处理单个模型：

```bash
python3 scripts/infer_siglip.py --model ViT-B-16-SigLIP --image test.jpg --device npu
python3 scripts/infer_siglip.py --model ViT-L-16-SigLIP2-384 --image test.jpg --device cpu
```

支持的模型名称（`--model` 参数）：

```
ViT-B-16-SigLIP, ViT-B-16-SigLIP-256, ViT-B-16-SigLIP-384, ViT-B-16-SigLIP-512,
ViT-B-16-SigLIP-i18n-256,
ViT-B-16-SigLIP2, ViT-B-16-SigLIP2-256, ViT-B-16-SigLIP2-384, ViT-B-16-SigLIP2-512,
ViT-B-32-SigLIP2-256,
ViT-L-16-SigLIP-256, ViT-L-16-SigLIP-384,
ViT-L-16-SigLIP2-256, ViT-L-16-SigLIP2-384, ViT-L-16-SigLIP2-512
```

## 3.1 模型权重下载

**输入**: 模型名称（如 ViT-B-16-SigLIP）, ModelScope 镜像可访问
**输出**: 权重文件保存至 ~/.cache/modelscope/timm/{model_name}/open_clip_model.safetensors

**执行步骤**:
1. 运行 `pip install modelscope` 确认 modelscope 已安装
2. 执行 `python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('timm/{model_name}')"` 下载指定模型权重
3. 验证权重文件完整性：`ls -lh ~/.cache/modelscope/timm/{model_name}/open_clip_model.safetensors`
4. 确认文件大小 > 0，若下载失败则重试最多 3 次，间隔 5 秒

```bash
# 通过 ModelScope 下载
pip install modelscope
python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('timm/ViT-B-16-SigLIP')"
```

权重自动下载到 `~/.cache/modelscope/timm/{model_name}/open_clip_model.safetensors`。

## 3.2 推理脚本

<!-- EMBED:scripts/infer_siglip.py -->

---

## 4. 精度指标说明

**输入**: CPU 推理结果特征向量, NPU 推理结果特征向量
**输出**: 精度对比指标（最大相对误差, 余弦相似度等）, 通过/不通过判定

**执行步骤**:
1. 对每个模型加载 CPU 和 NPU 两组推理输出的特征向量
2. 计算最大绝对误差 (Max Abs Diff) 和平均绝对误差 (Mean Abs Diff)
3. 计算余弦相似度 (Cosine Similarity), 确认 > 0.999
4. 计算最大相对误差百分比 (Max Rel Diff), 确认 < 1%
5. 将精度指标写入 results/{model_name}_compare.json 并记录到 results/summary.json

对于每个模型，计算以下精度指标：

| 指标 | 说明 | 通过标准 |
|------|------|---------|
| Max Abs Diff | 输出特征的最大绝对误差 | - |
| Mean Abs Diff | 输出特征的平均绝对误差 | - |
| Cosine Similarity | 输出特征的余弦相似度 | > 0.999 |
| 最大相对误差 (Max Rel Diff) | 最大相对误差百分比 | **< 1%** |
| 平均相对误差 (Mean Rel Diff) | 平均相对误差百分比 | - |

**核心标准：NPU 与 CPU 推理结果相对误差 < 1%。**

实测结果：所有 15 个模型均通过精度验证，最大相对误差 < 0.5%，余弦相似度 > 0.99986。

---

## 5. 模型输出说明

**输入**: 模型推理返回的原始输出元组
**输出**: 图像特征向量，shape 为 [1, 768]（ViT-B）或 [1, 1024]（ViT-L）

**执行步骤**:
1. 调用 `output = model(input_tensor)` 获取模型原始输出
2. 提取第一个元素 `features = output[0]` 获取图像特征向量
3. 使用 `features.shape` 确认输出维度：ViT-B 系列应为 768, ViT-L 系列应为 1024
4. 将特征向量保存至 results/{model_name}_features.npy 供精度对比使用

SigLIP/SigLIP2 模型的输出格式：

```python
output = model(input_tensor)  # 返回 (image_features, ..., ..., ...)
features = output[0]          # 图像特征向量，shape [1, embed_dim]
```

- ViT-B 系列：embed_dim = 768
- ViT-L 系列：embed_dim = 1024

---

## 6. 推送到 GitCode 模型仓库

**输入**: 模型产物（README.md, inference.py, compare_cpu_npu.py, requirements.txt, results.json）, ATOMGIT_USER_TOKEN
**输出**: GitCode 远程模型仓库, 推送成功的 commit

**执行步骤**:
1. 设置 ATOMGIT_USER_TOKEN 环境变量为有效的 GitCode 个人访问令牌
2. 为每个模型创建 GitCode 模型仓库
3. 将本地模型产物推送至对应远程仓库
4. 验证推送结果：仓库公开可访问且包含全部必需文件

## 6.1 使用 GitCode API 创建仓库

```bash
# 设置 token
export ATOMGIT_USER_TOKEN=<your_token>

# 创建仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "Authorization: Bearer ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "name": "ViT-B-16-SigLIP",
    "description": "ViT-B-16-SigLIP - NPU adapted model",
    "visibility": "public",
    "repository_type": "model"
  }'
```

## 6.2 推送代码

**输入**: 模型仓库目录（包含 README.md, inference.py, compare_cpu_npu.py, requirements.txt, results.json）, GitCode 仓库已创建
**输出**: 远程仓库代码推送完成, 全部文件已提交

**执行步骤**:
1. 切换到模型目录：`cd models/{model_name}`
2. 初始化 git 仓库并切换到 main 分支：`git init && git checkout -b main`
3. 添加全部产物文件：`git add README.md inference.py compare_cpu_npu.py requirements.txt results.json`
4. 提交并推送：`git commit -m "Add {model_name} NPU adapted model" && git remote add origin <remote-url> && git push -u origin main`
5. 验证推送结果：浏览器访问仓库 URL 确认文件完整

```bash
cd models/ViT-B-16-SigLIP
git init
git checkout -b main
git add README.md inference.py compare_cpu_npu.py requirements.txt results.json
git commit -m "Add ViT-B-16-SigLIP NPU adapted model"
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/ViT-B-16-SigLIP.git
git push -u origin main
```

## 6.3 仓库命名规则

**输入**: 模型名称列表
**输出**: 符合规范的 GitCode 仓库 URL

**执行步骤**:
1. 确认模型名称与上表完全一致（含连字符, 大小写敏感）
2. 构造仓库 URL：`https://gitcode.com/{username}/{model_name}`
3. 创建仓库时在 API 请求中设置 `"name": "{model_name}"`
4. 验证仓库 URL 可访问

模型仓库名称与模型名称一致，使用原始模型名（含连字符）：
- `ViT-B-16-SigLIP` → `https://gitcode.com/<user>/ViT-B-16-SigLIP`
- `ViT-L-16-SigLIP2-384` → `https://gitcode.com/<user>/ViT-L-16-SigLIP2-384`

---

## 7. 模型仓库列表

**输入**: 已完成全部 15 个模型的精度验证和产物生成
**输出**: 完整的模型仓库地址列表（每个模型对应一个 GitCode 仓库）

**执行步骤**:
1. 确认每个模型已生成完整的 README, inference.py, compare_cpu_npu.py, requirements.txt, results.json
2. 按照上表依次为每个模型创建 GitCode 仓库并推送代码
3. 逐一验证仓库 URL 可公开访问，产物文件完整
4. 在 results/summary.json 中记录每个模型的仓库地址

所有 15 个模型的 GitCode 仓库地址：

| 模型 | 仓库地址 |
|------|---------|
| ViT-B-16-SigLIP | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP |
| ViT-B-16-SigLIP-256 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP-256 |
| ViT-B-16-SigLIP-384 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP-384 |
| ViT-B-16-SigLIP-512 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP-512 |
| ViT-B-16-SigLIP-i18n-256 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP-i18n-256 |
| ViT-B-16-SigLIP2 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP2 |
| ViT-B-16-SigLIP2-256 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP2-256 |
| ViT-B-16-SigLIP2-384 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP2-384 |
| ViT-B-16-SigLIP2-512 | https://gitcode.com/gcw_C8PI9e90/ViT-B-16-SigLIP2-512 |
| ViT-B-32-SigLIP2-256 | https://gitcode.com/gcw_C8PI9e90/ViT-B-32-SigLIP2-256 |
| ViT-L-16-SigLIP-256 | https://gitcode.com/gcw_C8PI9e90/ViT-L-16-SigLIP-256 |
| ViT-L-16-SigLIP-384 | https://gitcode.com/gcw_C8PI9e90/ViT-L-16-SigLIP-384 |
| ViT-L-16-SigLIP2-256 | https://gitcode.com/gcw_C8PI9e90/ViT-L-16-SigLIP2-256 |
| ViT-L-16-SigLIP2-384 | https://gitcode.com/gcw_C8PI9e90/ViT-L-16-SigLIP2-384 |
| ViT-L-16-SigLIP2-512 | https://gitcode.com/gcw_C8PI9e90/ViT-L-16-SigLIP2-512 |

---

## 8. 验收确认

**输入**: 全部 15 个模型已完成推理验证、精度对比和仓库推送
**输出**: 验收通过确认（全部 8 项检查通过）

**执行步骤**:
1. 逐一核对检查清单中的 8 项验收标准
2. 对未通过项记录具体原因并修复对应步骤
3. 所有项目通过后向用户报告最终验收结果
4. 输出验收报告至 results/acceptance_report.md

完成以下检查清单即为部署成功：

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] 所有模型在 CPU 上成功运行推理
- [ ] 所有模型在 NPU 上成功运行推理
- [ ] 所有模型的 CPU/NPU 相对误差 < 1%
- [ ] 所有模型的输出维度正确（ViT-B: 768, ViT-L: 1024）
- [ ] 每个模型都生成了完整的 README 和脚本
- [ ] 每个模型都推送到了 GitCode 模型仓库

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `No module named 'torch_npu'` | 未安装或 CANN 环境未加载 | `source set_env.sh` 后重装 torch_npu |
| `FileNotFoundError: open_clip_model.safetensors` | 权重未下载 | `modelscope.hub.snapshot_download('timm/{model_name}')` |
| 模型下载失败 | 网络问题 | `export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` |
| OOM | 显存不足 | 确保每个模型处理后调用 `torch.npu.empty_cache()` |
| NPU 推理精度差异 > 1% | 算子精度问题 | 检查 NPU 是否为 FP32 模式 |
| `open_clip.create_model_and_transforms` 报错 | 模型名不完整 | 确认模型名称与上表一致 |
| GitCode 推送失败 | token 权限不足 | 检查 ATOMGIT_USER_TOKEN 是否正确 |

## 异常处理与回滚策略

| 异常场景 | 表现 | 处理方式 | 回滚/恢复 |
|---------|------|---------|----------|
| NPU 不可用 | `npu-smi info` 报错或无设备 | 确认硬件在位，重新 `source set_env.sh` | 切换至 CPU 模式：`--device cpu` |
| torch_npu 导入失败 | `ImportError: No module named torch_npu` | 确认 torch-npu 与 CANN 版本匹配 | `pip install torch-npu==<version>` 重装 |
| 模型下载超时 | ModelScope 超时或 404 | 重试最多 3 次，间隔 5s | 切换至 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com` |
| 权重文件缺失 | `FileNotFoundError: safetensors` | 确认模型名编码规则（大小写敏感） | 手动从 ModelScope 下载并指定路径 |
| NPU OOM | `torch.npu.OutOfMemoryError` | `torch.npu.empty_cache()` + `gc.collect()` | 释放其他进程显存后重试 |
| 显存泄漏 | 连续处理多模型后 OOM | 每个模型后执行清理代码 | 重启 Python 进程 |
| CPU/NPU 精度差异大 | 相对误差 > 1% 或余弦相似度 < 0.999 | 检查 FP32/FP16 模式一致性 | 重跑 CPU 推理后对比；仍异常则标记为"精度存疑" |
| 输出维度异常 | 特征维度不是 768 或 1024 | 确认加载的是 SigLIP Vision 模型而非完整 CLIP | 检查 `open_clip.create_model_and_transforms` 参数 |
| 磁盘空间不足 | 下载权重时磁盘满 | `df -h` 检查可用空间 | 清理 `~/.cache/modelscope` 或更换模型缓存目录 |
| GitCode 推送失败 | 401/403/网络错误 | 检查 token 有效期和权限 | 重新生成 token 后重试 |

## 资源与评测产物

本 Skill 在 `skills/deployment/siglip-npu/` 目录下提供以下资源：

| 资源路径 | 用途 |
|---------|------|
| `SKILL.md`（本文件） | 完整执行流程与异常处理指南 |
| `scripts/infer_siglip.py` | 单模型 CPU/NPU 推理脚本 |
| `examples/run_all_models.sh` | 批量执行全部 15 个模型的 Shell 脚本 |
| `examples/run_single_model.sh` | 单模型快速测试 Shell 脚本 |
| `references/test_results.md` | 已完成的 15 个模型精度验证报告 |
| `test-prompts.json` | 评测提示词与预期结果 |

执行结束后确认以下产物已生成：

- [ ] `results/*.json` — 每个模型的详细推理数据和精度指标
- [ ] `results/summary.json` — 全部 15 个模型的汇总结果
- [ ] `models/<model_name>/README.md` — 含真实测试数据的中文文档
- [ ] `models/<model_name>/inference.py` — NPU 推理脚本
- [ ] `models/<model_name>/compare_cpu_npu.py` — CPU/NPU 精度对比脚本

---

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 否 | 要处理的模型名称（默认全部 15 个） |
| image | string | 是 | 测试图像路径 |
| device | string | 否 | 推理设备（cpu/npu，默认 npu） |
| weights | string | 否 | 权重文件路径（默认自动检测） |
| push_to_gitcode | bool | 否 | 是否推送到 GitCode |
| gitcode_token | string | 否 | GitCode 访问令牌（默认使用 ATOMGIT_USER_TOKEN） |

## Skill 输出结果

| 输出 | 说明 |
|------|------|
| results/*.json | 每个模型的详细推理数据和精度指标 |
| results/summary.json | 所有模型的汇总结果 |
| models/*/README.md | 含真实测试数据的中文文档 |
| models/*/inference.py | NPU 推理脚本 |
| models/*/compare_cpu_npu.py | CPU/NPU 精度对比脚本 |
| models/*/results.json | 精度对比结果数据 |

## 资源释放

每个模型测试完成后主动释放资源：

```python
import gc
gc.collect()
torch.npu.empty_cache()
```

串行执行多个模型时，确保在下一个模型开始前调用上述清理代码，避免显存爆炸。
