---
name: xcit-npu
description: >
  XCiT (Cross-Covariance Image Transformer) 系列图像分类模型在
  昇腾 NPU 上的完整部署与精度对比 Skill。涵盖 timm 框架下
  xcit_nano/tiny/small/medium/large 全系列变体的环境搭建、
  NPU 推理、CPU/NPU 精度对比验证、README 生成及模型仓库发布。
  适用于 Ascend910 系列 NPU，支持 in1k 与 fb_dist_in1k 预训练权重。
metadata:
  short-description: XCiT 系列模型昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, xcit, timm, image-classification, pytorch, cv, transformer]
---

# XCiT 系列图像分类模型 昇腾 NPU 部署 Skill

本 Skill 提供 XCiT (Cross-Covariance Image Transformer) 系列图像分类模型在华为昇腾 NPU 上的完整部署、推理验证和精度对比的标准化可复现流程。所有操作需在昇腾服务器上以 NPU 环境执行。

## 执行工作流

1. 检查 NPU 环境状态，使用 npu-smi 确认 NPU 设备可用并选择空闲卡。
2. 安装依赖环境，包括 torch、torch_npu、timm 及 modelscope 等 Python 包。
3. 验证 torch_npu 可用性，确认 NPU 设备可正常调用。
4. 从 ModelScope 下载指定 XCiT 模型的预训练权重。
5. 执行模型推理，分别在 CPU 和 NPU 设备上运行 XCiT 模型并记录推理耗时。
6. 执行 CPU/NPU 精度对比，计算 logits 误差和余弦相似度等指标。
7. 生成终端截图 HTML 和推理结果报告。
8. 生成 README 文档并推送至 GitCode 模型仓库发布。

按以上步骤顺序依次执行，每步完成后需确认 checkpoint 后再进入下一步。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡 NPU） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0 |
| Python | 3.9 – 3.11 |
| 网络 | 需联网下载模型权重，或从 ModelScope 本地加载 |

## 详细步骤

### 1. 检查 NPU 环境状态

使用 npu-smi 命令确认 NPU 卡状态，选择空闲卡号进行后续操作。

```bash
npu-smi info
# 输出示例：
# NPU ID  Chip    Chip Name          Health
# 0       910     Ascend910           OK
# 1       910     Ascend910           OK
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

**checkpoint**: 确认 `npu-smi info` 输出中至少有一张 NPU 卡且 Health 状态为 OK。若 NPU 状态异常则暂停并排查昇腾驱动与 CANN 环境。

### 2. 安装依赖环境

```bash
pip install torch torch_npu timm Pillow numpy requests modelscope \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

详细依赖见 `scripts/requirements.txt`。

**checkpoint**: 确认 pip install 完成无报错，可通过 `python3 -c "import torch; import torch_npu; print(torch.__version__)"` 验证。

### 3. 验证 torch_npu 可用性

```python
import torch
import torch_npu
print('torch:', torch.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('Device name:', torch.npu.get_device_name(0))
```

**checkpoint**: 输出 `NPU available: True` 且无报错。若为 False 则暂停，排查 CANN 与 torch_npu 版本兼容性。

### 4. 下载模型权重

从 ModelScope 下载 XCiT 模型权重（避免 HuggingFace 网络限制）：

```python
from modelscope import snapshot_download
model_dir = snapshot_download('timm/xcit_tiny_24_p8_384.fb_dist_in1k')
print('Model downloaded to:', model_dir)
```

支持的 XCiT 系列模型包括：
- `timm/xcit_nano_12_p8_224.fb_dist_in1k`
- `timm/xcit_tiny_12_p8_224.fb_in1k`
- `timm/xcit_tiny_24_p8_384.fb_dist_in1k`
- `timm/xcit_small_12_p16_224.fb_dist_in1k`
- `timm/xcit_small_24_p16_384.fb_dist_in1k`
- `timm/xcit_medium_24_p8_224.fb_dist_in1k`
- `timm/xcit_large_24_p16_384.fb_dist_in1k`

共计 28 个模型变体，支持 in1k 和 fb_dist_in1k 两种预训练权重。

### 5. 执行模型推理

#### 5.1 单模型推理

使用 `scripts/inference.py` 对指定模型执行 NPU 和 CPU 推理：

```bash
cd /path/to/working/dir
python3 scripts/inference.py test_input.jpg
```

此脚本会依次在 CPU 和 NPU 上运行推理，输出 Top-5 分类结果和推理耗时，结果保存到 `inference_results.json`。

#### 5.2 批量串行推理

多个模型必须串行执行，防止 NPU 显存和系统内存溢出：

```bash
# 为每个模型创建独立目录，串行执行
for model_name in xcit_tiny_12_p8_224 xcit_tiny_24_p8_384 xcit_small_24_p8_384; do
    mkdir -p /path/${model_name}
    cd /path/${model_name}
    python3 scripts/inference.py
    # 释放 NPU 资源
    python3 -c "
import gc, torch
gc.collect()
torch.npu.empty_cache()
"
done
```

### 6. 执行 CPU/NPU 精度对比

```bash
python3 scripts/compare_cpu_npu.py test_input.jpg
```

精度对比脚本使用 `scripts/compare_cpu_npu.py` 计算以下指标：
- Logits 最大/平均绝对误差
- RMSE (均方根误差)
- 余弦相似度
- Softmax 概率最大/平均差异
- Top-1 一致性
- Top-5 重合率

结果保存到 `precision_results.json`。

**精度要求**：NPU 与 CPU 推理结果最大概率差异 < 1%。

**实测结果示例**（基于 xcit_tiny_24_p8_384.fb_dist_in1k 模型验证）：

| 指标 | 数值 |
|------|------|
| 最大绝对值误差（logits） | 4.66e-02 |
| 均方根误差（RMSE） | 1.72e-03 |
| 余弦相似度 | 0.999947 |
| 最大概率差异 | 0.4423% |
| Top-1 一致性 | 100% |
| Top-5 重合率 | 100% |

### 7. 生成终端截图

```bash
python3 scripts/generate_screenshot.py /path/to/results/dir
```

生成 HTML 格式的模拟终端输出截图，可直接在浏览器中打开查看推理和精度对比结果。

### 8. 生成 README 与发布模型仓库

#### 8.1 生成 README

README 需包含：
- 模型介绍与原始模型地址
- 任务类型与依赖环境
- NPU 适配说明
- 推理命令、CPU/NPU 精度测试结果表格
- 性能测试结果与终端截图
- 模型标签（#+NPU, #+CV, #+昇腾 等）

#### 8.2 创建 GitCode 模型仓库

```bash
# 使用 GitCode API 创建模型类型仓库
curl -X POST \
  --header "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "name": "{model_name}-npu",
    "visibility": "public",
    "repository_type": "model"
  }' \
  "https://api.gitcode.com/api/v5/user/repos"
```

#### 8.3 推送代码

```bash
git init
git checkout -b main
git add inference.py compare_cpu_npu.py requirements.txt readme.md \
  generate_screenshot.py terminal_output_*.html
git commit -m "Add {model_name} NPU adaptation"
git remote add origin "https://auth:${GITCODE_TOKEN}@gitcode.com/{username}/{repo_name}.git"
git push -u origin main
```

若 git 操作不可达，使用 GitCode API 上传文件：

```bash
curl -X PUT \
  --header "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "content": "'"$(base64 -w0 file.py)"'",
    "message": "Add file.py"
  }' \
  "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/contents/file.py"
```

## 检查点设计

| 序号 | 检查点 | 确认方法 | 异常处理 |
|------|--------|----------|----------|
| 1 | NPU 设备可用性确认 | 运行 `npu-smi info` 查看设备 Health 状态 | 若 NPU 异常则暂停并排查驱动与 CANN 安装 |
| 2 | 依赖安装确认 | `python3 -c "import torch_npu; print('OK')"` | 安装失败则重新安装 torch_npu 对应版本 |
| 3 | 模型权重下载确认 | 检查 `~/.cache/modelscope/hub/` 目录是否存在权重 | 下载失败则切换 ModelScope 镜像源重试 |
| 4 | CPU 推理 checkpoint | 确认 `inference_results.json` 中 cpu 结果完整 | 推理异常则检查输入图像与模型匹配性 |
| 5 | NPU 推理 checkpoint | 确认 `inference_results.json` 中 npu 结果完整 | 推理异常则释放 NPU 显存后重试 |
| 6 | 精度对比确认 | 确认余弦相似度 > 0.999 且最大概率差异 < 1% | 精度不达标则检查 FP32 模式与模型权重 |
| 7 | 资源释放确认 | 确认 `torch.npu.empty_cache()` 执行后显存归零 | 显存未释放则重启 NPU 进程后重试 |

## 异常处理与边界条件

| 异常场景 | 可能原因 | 解决方案 |
|----------|----------|----------|
| NPU 设备不可用 | CANN 驱动未安装或版本不匹配 | 检查 CANN 安装并执行 `npu-smi info` 确认 |
| torch_npu 导入失败 | torch_npu 版本与 torch 不兼容 | 安装匹配版本：`pip install torch_npu==对应版本` |
| ModelScope 下载超时 | 网络限制导致连接失败 | 切换镜像源或使用 `HF_ENDPOINT` 环境变量 |
| HuggingFace 连接超时 | 境外网络不可达 | 从 ModelScope 下载或配置 HF 镜像 |
| 模型权重路径错误 | ModelScope 缓存路径不匹配 | 使用 `torch.load(local_path)` 指定本地路径 |
| NPU OOM | 模型参数量超出 NPU 显存 | 设置 batch_size=1，单卡串行执行 |
| 余弦相似度偏低 | 精度累积误差或非 FP32 模式 | 确认模型在 FP32 模式下运行 |
| Top-1 不匹配 | NPU 与 CPU 计算结果不一致 | 检查输入预处理是否完全相同 |
| CPU 推理内存不足 | 模型参数过大 | 使用 swap 或分批加载模型 |
| git push 超时 | gitcode.com 网络不可达 | 使用 GitCode API 上传文件替代 git 操作 |
| 截图生成失败 | 结果目录路径错误 | 检查 `precision_results.json` 文件是否存在 |
| pip 安装失败 | 源不可用或版本冲突 | 使用清华镜像源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 预处理参数错误 | 模型与图像尺寸不匹配 | 使用 timm 的 `resolve_model_data_config` 自动获取 |
| 模型组件迁移错误 | 不支持的操作符在 NPU 上 fallback 到 CPU | 检查 torch_npu 兼容算子列表 |
| 权重文件损坏 | 下载过程中断 | 删除缓存后重新下载 `rm -rf ~/.cache/modelscope` |

## 资源清单

| 资源 | 路径 | 说明 |
|------|------|------|
| 推理脚本 | `scripts/inference.py` | CPU 和 NPU 模型推理 |
| 精度对比脚本 | `scripts/compare_cpu_npu.py` | CPU/NPU 精度指标计算 |
| 截图生成脚本 | `scripts/generate_screenshot.py` | 终端截图 HTML 生成 |
| 依赖文件 | `scripts/requirements.txt` | Python 依赖列表 |
| 示例代码 | `examples/xcit_inference_example.py` | XCiT 推理示例 |
| 推理结果 | `results/inference_results.json` | CPU/NPU 推理输出 |
| 精度数据 | `results/precision_results.json` | 精度对比指标数据 |
| 评估报告 | `evals.json` | 批量评估结果 |
| 参考文档 | `references/` | 昇腾部署参考文档 |
| 模型配置 | `skill.json` | Skill 元数据和模型清单 |

## 资源释放

每个模型测试完成后必须释放 NPU 资源：

```python
import gc
gc.collect()
try:
    import torch
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
except Exception:
    pass
```

## 输入参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 模型名称 | XCiT 变体名称，如 xcit_tiny_24_p8_384.fb_dist_in1k | xcit_tiny_24_p8_384.fb_dist_in1k |
| 测试图片 | 任意 RGB 图像路径 | test_input.jpg |
| 推理设备 | cpu / npu:0（自动 CPU + NPU 对比） | cpu + npu:0 |

## 输出结果

| 输出 | 格式 | 说明 |
|------|------|------|
| inference_results.json | JSON | CPU 和 NPU 推理结果及耗时 |
| precision_results.json | JSON | 精度对比指标数据 |
| terminal_output_*.html | HTML | 模拟终端截图 |
| 模型仓库 | GitCode repo | 包含所有交付件的模型仓库 |

## 模型标签

发布仓库时建议包含以下标签：
- `#+NPU`：在昇腾 NPU 上完成适配验证
- `#+CV`：计算机视觉模型
- `#+昇腾`：华为昇腾平台
- `#+图像分类`：图像分类任务
- `#+Transformer`：基于 Transformer 架构

## 测试验证

本 Skill 提供 test-prompts 测试场景用于评估 NPU 部署效果。实测验证包含以下维度：
1. 环境就绪验证：通过 npu-smi 确认 NPU 设备状态
2. 单模型推理 benchmark：验证 XCiT 模型在 NPU 上的推理性能
3. 精度对比 eval：确认 CPU/NPU 推理结果的一致性
4. 异常恢复验证：测试常见错误场景的 fallback 策略

**注意事项**：以上实测数据为结构评估参考（structure_eval），真实 NPU 环境测试需在有昇腾硬件的服务器上执行验证。
