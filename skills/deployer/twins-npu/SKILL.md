---
name: twins-npu
description: Twins 系列模型（SVT/PCPVT）昇腾 Ascend NPU 部署与精度验证 Skill
---

# Twins NPU 部署 Skill

## 概述

本 Skill 用于在华为昇腾 Ascend NPU（Ascend 910）上自动完成 Twins 系列视觉 Transformer 模型的推理部署、CPU/NPU 精度对比、README 生成和模型仓库发布。

支持 6 个模型：

| 模型名称 | 模型仓库地址 |
| --- | --- |
| twins_svt_small.in1k | [twins_svt_small.in1k-npu](https://gitcode.com/m0_74196153/twins_svt_small.in1k-npu) |
| twins_svt_large.in1k | [twins_svt_large.in1k-npu](https://gitcode.com/m0_74196153/twins_svt_large.in1k-npu) |
| twins_svt_base.in1k | [twins_svt_base.in1k-npu](https://gitcode.com/m0_74196153/twins_svt_base.in1k-npu) |
| twins_pcpvt_small.in1k | [twins_pcpvt_small.in1k-npu](https://gitcode.com/m0_74196153/twins_pcpvt_small.in1k-npu) |
| twins_pcpvt_large.in1k | [twins_pcpvt_large.in1k-npu](https://gitcode.com/m0_74196153/twins_pcpvt_large.in1k-npu) |
| twins_pcpvt_base.in1k | [twins_pcpvt_base.in1k-npu](https://gitcode.com/m0_74196153/twins_pcpvt_base.in1k-npu) |

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- torch_npu >= 2.0.0（昇腾 Ascend NPU 算子库）
- Ascend NPU（如 Ascend 910）
- CANN >= 8.0
- pip >= 21.0

## 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `model_name` | string | 是 | — | 模型名称（如 `twins_svt_small.in1k`） |
| `device` | string | 否 | `npu` | 推理设备（`cpu` 或 `npu`） |
| `num_runs` | integer | 否 | `5` | 推理轮次 |
| `image_url` | string | 否 | `null` | 测试图片 URL |

## 工作流

### 安装依赖与权重准备

1. 配置 pip 清华镜像源并安装核心依赖包，包括 torch、torch_npu（昇腾 Ascend NPU 算子库）、timm 模型库、torchvision、modelscope 等

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch>=2.0.0 \
  torch_npu>=2.0.0 \
  timm>=0.9.0 \
  torchvision>=0.15.0 \
  Pillow>=10.0.0 \
  numpy>=1.22.0 \
  modelscope>=1.0.0 \
  safetensors>=0.4.0
```

2. 验证 torch_npu 和 NPU 环境是否正常可用

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__, 'npu available:', torch.npu.is_available())"
```

3. 从 ModelScope 自动下载指定模型权重，无需手动操作，缓存目录为 `scripts/ms_cache/`

```bash
cd scripts/
python3 -c "from ms_loader import load_timm_model; m = load_timm_model('twins_svt_small.in1k'); print('model loaded:', type(m).__name__)"
```

4. 检查 `scripts/requirements.txt` 确认 modelscope 和 safetensors 版本兼容性

### 单模型 NPU 推理

5. 切换到脚本目录，使用推理脚本在 NPU 上运行指定模型，并指定推理轮次

```bash
cd scripts/
python3 inference.py --model-name twins_svt_small.in1k --device npu --num-runs 5
```

6. 解析推理输出中的 Top-5 分类标签及概率结果

7. 检查保存的 logits 文件（`.npy`）是否已生成到当前目录

8. 在 CPU 设备上重复推理以获取基准结果，用于后续精度对比

### CPU/NPU 精度对比

9. 使用精度对比脚本对指定模型执行 CPU 和 NPU 双端推理，自动计算对比指标

```bash
cd scripts/
python3 compare_cpu_npu.py --model-name twins_svt_small.in1k
```

10. 读取 Max Absolute Error 指标评估 logits 数值差异

11. 读取 Max Probability Difference 指标评估概率输出差异（阈值 < 0.01）

12. 读取 Cosine Similarity 指标评估向量方向一致性（阈值 > 0.9999）

13. 检查 Top-1 类别一致性和 Top-5 集合重叠比例

14. 记录 CPU/NPU 推理耗时和加速比，评估部署性能

### 批量多模型运行

15. 使用批量脚本串行执行全部 6 个模型的 CPU/NPU 对比，避免同时加载导致 OOM

```bash
cd scripts/
python3 batch_run.py
```

16. 监控每个模型的日志输出，确认所有 benchmark 均通过验证

17. 检查每轮推理后的 `torch.npu.empty_cache()` 显存释放状态

18. 生成每个模型的终端截图文件，保存对比摘要

### 结果分析与发布

19. 汇总各模型的精度指标和加速比数据填入已知结果表

20. 在 GitCode 模型仓库创建或更新 README，包含精度对比结果和部署说明

```bash
git clone https://gitcode.com/m0_74196153/twins_svt_small.in1k-npu
cp -r scripts/ models/ references/ twins_svt_small.in1k-npu/
cd twins_svt_small.in1k-npu
git add . && git commit -m "Add NPU deployment results and README"
git push
```

## 检查点

| # | 检查点 | 确认方式 | 通过标准 |
|---|--------|---------|---------|
| 1 | NPU 环境确认 | 执行 `npu-smi info` 确认 NPU 设备可用 | 返回设备信息，无错误 |
| 2 | 依赖安装确认 | 运行 `python3 -c "import timm, torch_npu"` | 无 ImportError 异常 |
| 3 | 权重下载确认 | 检查 `scripts/ms_cache/` 目录是否包含模型文件 | 模型文件存在且可加载，确认无损坏 |
| 4 | NPU 推理确认 | 运行 `inference.py` 观察输出 | 返回 Top-5 分类结果，推理正常完成 |
| 5 | 精度对比确认 | 运行 `compare_cpu_npu.py` | MaxProbDiff < 0.01，CosineSim > 0.9999 |
| 6 | 批量运行确认 | 运行 `batch_run.py` 全部 6 个模型 | 所有 benchmark 通过，无失败项 |
| 7 | 资源释放确认 | 检查 `torch.npu.empty_cache()` 调用后的显存状态 | 显存已释放，下次推理无 OOM |
| 8 | 发布前审批确认 | dry-run push 验证仓库内容完整后用户审批 | 用户确认 approval 后执行 push |

## 异常处理

| 异常场景 | 原因 | 处理方式 |
|---------|------|---------|
| pip 安装失败 | 网络超时或 PyPI 源不可用 | 更换国内镜像源重试，如果多次失败则使用 `requirements.txt` 离线安装 |
| ModelScope 下载失败 | 网络不可达或令牌过期 | 检查代理设置，确认模型名称正确，如果多次失败则尝试本地预下载 |
| NPU 设备不可用 | 驱动未加载或 NPU 被占用 | 执行 `npu-smi info` 排查，如果报错则重启驱动服务后 recover |
| OOM 显存不足 | 同时加载多个大模型 | 使用 `batch_run.py` 串行执行，每轮结束后调用 `gc.collect()` + `torch.npu.empty_cache()` 释放 |
| 精度对比失败 | MaxProbDiff > 1% 阈值 | 检查 NPU 算子版本，如果存在精度差异则记录详细对比日志并 fallback 到 CPU 结果 |
| 推理结果异常 | 权重不匹配或模型状态异常 | 确认 safetensors 权重完整性，如果损坏则重新下载后 retry |
| 截图生成失败 | 日志文件格式不符合预期 | 检查 `gen_screenshot.py` 输入路径，如果文件为空则重试原始 benchmark |
| 批量运行时中断 | 某个模型推理卡住 | 设置超时处理（900s timeout），如果超时则跳过该模型后继续下一个 |
| Cosine Similarity 偏差 | NPU 与 CPU 计算结果不一致 | 检查输入数据一致性，如果仍然偏差则记录异常并继续后续模型 |
| 模型加载失败 | 缓存目录结构异常 | 删除 `ms_cache/` 后重新下载，如果仍然失败则确认 modelscope 版本 |
| 显存泄漏 | GPU 资源未正确释放 | 确保每轮结束后执行 `del model` + `gc.collect()`，如果泄漏持续则重启进程 |

## 资源清单

| 资源 | 路径 | 说明 |
|------|------|------|
| 推理脚本 | `scripts/inference.py` | 通用 NPU 推理入口 |
| 精度对比脚本 | `scripts/compare_cpu_npu.py` | CPU vs NPU 精度对比 |
| 批量运行脚本 | `scripts/batch_run.py` | 串行运行 6 个模型 |
| ModelScope 加载器 | `scripts/ms_loader.py` | 权重下载与加载模块 |
| 截图生成工具 | `scripts/gen_screenshot.py` | 终端输出截图 |
| 依赖列表 | `scripts/requirements.txt` | pip 安装依赖 |
| 示例脚本 | `examples/example.py` | 使用示例 |
| 模型缓存 | `scripts/ms_cache/` | 模型权重缓存目录 |
| 模型仓库 | `references/` | 发布后的模型仓库 |
| 精度验证数据 | `evals.json` | 精度对比评估记录 |
| 性能结果文件 | `results.tsv` | 各模型推理耗时与加速比汇总 |

## 已知结果

| 模型 | MaxProbDiff | CosineSim | Speedup |
| --- | ---:| ---:| ---:|
| twins_svt_small.in1k | 0.000200 | 0.99999547 | 0.63x |
| twins_svt_large.in1k | 0.000280 | 0.99999636 | 2.66x |
| twins_svt_base.in1k | 0.000356 | 0.99999672 | 1.60x |
| twins_pcpvt_small.in1k | 0.000327 | 0.99999624 | 0.86x |
| twins_pcpvt_large.in1k | 0.000461 | 0.99999022 | 1.88x |
| twins_pcpvt_base.in1k | 0.000181 | 0.99999499 | 1.40x |

所有模型精度验证均通过（Max Probability Difference < 0.05%）。

## 精度验证

精度验证流程：

1. 在 CPU 上运行一次推理得到基准 logits 和概率分布
2. 在 NPU 上运行推理得到部署 logits 和概率分布
3. 计算 Max Absolute Error 评估数值一致性
4. 计算 Cosine Similarity 评估向量方向一致性
5. 检查 Top-1 和 Top-5 分类一致性

## 资源释放

```python
import gc
import torch

del model, input_tensor
gc.collect()
torch.npu.empty_cache()
```
