---
name: vtp-npu-adapation
description: >
  MiniMax VTP (Visual Tokenizer Pre-training) 全系列模型在华为昇腾 NPU 上的完整适配与部署 Skill。
  覆盖 VTP-Small、VTP-Base、VTP-Large 三个规模（f16d64），支持自编码器重建、
  CLIP 零样本分类、SSL 特征提取三种推理模式。
  含自动设备检测、RoPE dtype 精度修复、NPU vs CPU 精度验证、性能基准测试的全流程。
  当用户提到 VTP 昇腾适配、VTP NPU 推理、视觉分词器 NPU 部署时触发。
metadata:
  short-description: MiniMax VTP 全系列 (S/B/L) 昇腾 NPU 推理适配
  category: NPU-Model-Deploy
  tags: [ascend, npu, vtp, vision-tokenizer, minimax, pytorch, inference, deployment]
---

# VTP 全系列昇腾 NPU 适配与部署 Skill

本 Skill 提供 MiniMax VTP (Visual Tokenizer Pre-training) 全系列模型在华为昇腾 NPU 上的
完整适配、推理验证和精度评测的标准化可复现流程。

## 适配模型

| 模型 | Vision Encoder | 参数量 | HBM 占用 | 推理延迟 (fp16) | 仓库 |
|------|:---:|:---:|:---:|:---:|------|
| VTP-Small-f16d64 | ViT-S (12层, 384维) | ~40M | ~1 GB | ~450 ms | [链接](https://ai.gitcode.com/m0_74196153/VTP-Small-f16d64) |
| VTP-Base-f16d64 | ViT-B (12层, 768维) | 295.6M | ~2.8 GB | 123.9 ms | [链接](https://ai.gitcode.com/m0_74196153/VTP-Base-f16d64) |
| VTP-Large-f16d64 | ViT-L (24层, 1024维) | 731.6M | ~7.0 GB | 144.8 ms | [链接](https://ai.gitcode.com/m0_74196153/VTP-Large-f16d64) |

所有模型均使用 `f16d64` 配置：16x 空间压缩，64 维 Latent 空间，256x256 输入分辨率。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 910 系列（至少 1 卡，>=32GB HBM for Base，>=32GB for Large） |
| OS | openEuler / Ubuntu / KylinOS（aarch64） |
| CANN | >= 8.5.1 |
| Python | 3.10 - 3.13 |
| 依赖 | torch, torch_npu, torchvision, transformers, omegaconf, pillow |
| 网络 | 首次运行需下载模型权重（Base: ~1.2GB, Large: ~2.9GB） |

## 执行总览

1. 确认 NPU 环境状态并选择空闲设备卡号
2. 安装 PyTorch NPU 版本及 VTP 模型依赖项
3. 从 ModelScope 下载对应模型权重文件
4. 克隆 MiniMax VTP 官方源码仓库
5. 应用 RoPE dtype 精度修复适配层
6. 执行基础推理验证（AE / CLIP / SSL 三种模式）
7. 运行 NPU vs CPU 精度对比评测并记录结果
8. 运行 NPU 性能基准测试并保存性能日志
9. 检查各模型推理结果是否通过验收标准
10. 确认交付件清单完整并生成评测报告
11. 清理临时文件并关闭 NPU 内存占用
12. 若以上步骤全部通过则标记适配完成
13. 若出现精度异常则执行回滚至上一版本
14. 若 NPU 不可用则降级至 CPU 调试模式
15. 若模型加载失败则检查权重完整性并重试下载
16. 若推理 OOM 则减小 batch 或启用 fp16 混合精度

> 按以上步骤顺序执行，每步完成后再进入下一步。

---

## 0. 环境初始化

**检查点 1：确认 NPU 环境可用后继续。**

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0

# 华为 pip 镜像（国内加速）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

**边界条件：** 若 `npu-smi info` 未发现可用 NPU，降级至 CPU 模式并输出警告。

```python
import torch_npu
if not torch.npu.is_available():
    print("[WARNING] NPU 不可用，将回退至 CPU 模式。")
    device = torch.device("cpu")
```

---

## 1. 安装依赖

**检查点 2：确认依赖安装成功后继续。**

```bash
pip install torch torch_npu torchvision transformers omegaconf pillow modelscope
```

**边界条件：** 如 network timeout，改用国内镜像源重试：
```bash
pip install -i https://repo.huaweicloud.com/repository/pypi/simple/ torch torch_npu torchvision transformers omegaconf pillow modelscope
```

若继续失败，错误日志保存至 `scripts/install_error.log` 备用排查。

---

## 2. 下载模型权重

根据目标模型规模选择对应命令：

```bash
# VTP-Small (~150MB)
python3 -c "from modelscope import snapshot_download; snapshot_download('MiniMax/VTP-Small-f16d64', cache_dir='./model_weights')"

# VTP-Base (~1.2GB)
python3 -c "from modelscope import snapshot_download; snapshot_download('MiniMax/VTP-Base-f16d64', cache_dir='./model_weights')"

# VTP-Large (~2.9GB)
python3 -c "from modelscope import snapshot_download; snapshot_download('MiniMax/VTP-Large-f16d64', cache_dir='./model_weights')"
```

**边界条件：** 若下载中途中断，清除 `model_weights/` 下不完整文件后重试：
```bash
rm -rf ./model_weights/MiniMax/* && python3 -c "from modelscope import snapshot_download; snapshot_download('MiniMax/VTP-Base-f16d64', cache_dir='./model_weights')"
```

---

## 3. 获取 VTP 源码

```bash
git clone https://github.com/MiniMax-AI/VTP.git vtp-repo
# 或使用镜像
git clone https://gitcode.com/MiniMax-AI/VTP.git vtp-repo
```

**边界条件：** 若 GitHub 不可达自动切换 GitCode 镜像。

---

## 4. NPU 适配：RoPE dtype 精度修复

VTP 模型的 `pixel_decoder.rope_embed` 默认使用 `bfloat16` 存储位置编码，在 NPU 上
与 CPU 存在数值行为差异，会导致重建误差偏高。

**适配方法**：将 RoPE 的 dtype 强制转换为 `float32`：

```python
def adapt_model_for_npu(model):
    """Fix RoPE bf16->fp32 for NPU precision."""
    if hasattr(model, "pixel_decoder") and model.pixel_decoder is not None:
        rope = model.pixel_decoder.rope_embed
        if rope.dtype == torch.bfloat16:
            rope.dtype = torch.float32
            rope.periods = rope.periods.to(torch.float32)
            rope._init_weights()
    return model
```

此修复可将重建误差从 ~1.73% 降低至 < 0.9%（Large）/ < 0.2%（Base）。

完整适配层见 `scripts/npu_compat.py`，包含 `adapt_model_for_npu()`、`get_device()`、`maybe_autocast()` 等辅助函数。

---

## 5. 基础推理验证

**检查点 3：确认三种推理模式均正常运行后继续。**

运行推理脚本（支持三种模式）：

```bash
# 自编码器重建
python3 inference.py --model ./model_weights/MiniMax/VTP-Base-f16d64 --mode ae

# CLIP 零样本分类
python3 inference.py --model ./model_weights/MiniMax/VTP-Base-f16d64 --mode clip

# SSL 特征提取
python3 inference.py --model ./model_weights/MiniMax/VTP-Base-f16d64 --mode feature

# 全部模式
python3 inference.py --model ./model_weights/MiniMax/VTP-Base-f16d64 --mode all
```

验证通过标准：
- 模型成功加载到 NPU (`npu:0`)
- 自编码器输出 shape: `latents=[1,64,16,16], recon=[1,3,256,256]`
- CLIP 图像特征 shape: `[1, 768]`（Base）/ `[1, 1024]`（Large）
- 无运行时错误

**边界条件：** 若模型加载时报 `OutOfMemoryError`，减小 batch size 或切换至 fp16 后重试。若持续失败，错误记录保存至 `results/errors.log`。

---

## 6. 精度对比验证

使用 `evaluate.py` 在 4 种测试图像上与 CPU fp32 参考输出对比：

```bash
python3 evaluate.py \
    --model ./model_weights/MiniMax/VTP-Base-f16d64 \
    --name VTP-Base-f16d64 \
    --output-dir ./eval
```

结果保存至 `eval/results.json` 和 `eval/evaluation.log`。

### 精度标准

| 指标 | 标准 | 说明 |
|------|:---:|------|
| 重建图像余弦相似度 | > 0.9999 | 语义一致性 |
| 重建图像相对误差 | < 1% | 逐像素精度 |
| CLIP 特征余弦相似度 | > 0.9999 | 零样本分类准确性 |

### 各模型精度基准

| 模型 | 重建余弦相似度 | 重建相对误差 | CLIP 余弦相似度 |
|------|:---:|:---:|:---:|
| VTP-Small-f16d64 | > 0.9999 | < 0.8% | > 0.9999 |
| VTP-Base-f16d64 | > 0.99997 | < 0.2% | > 0.99997 |
| VTP-Large-f16d64 | > 0.99997 | < 0.9% | > 0.99999 |

**边界条件：** 若精度不达标，确认 RoPE 修复已应用，并检查 `torch_npu` 版本是否兼容 CANN。若修复仍无法解决，尝试在 `torch.amp.autocast` 中强制 `dtype=torch.float32` 重新评测。

---

## 7. 性能基准测试

测试条件：`batch=1, 256x256 图像, fp16 autocast, 50次取平均`

```bash
python3 benchmark.py --model ./model_weights/MiniMax/VTP-Base-f16d64 --device npu:0
```

结果保存至 `results/perf.json`。

| 指标 | VTP-Small | VTP-Base | VTP-Large |
|------|:---:|:---:|:---:|
| Encode 延迟 | ~220 ms | 60.0 ms | 55.4 ms |
| Decode 延迟 | ~230 ms | 63.9 ms | 89.4 ms |
| 总延迟 | ~450 ms | 123.9 ms | 144.8 ms |
| CLIP 延迟 | ~100 ms | 48.2 ms | 76.1 ms |

> 注：Small 模型数据基于 fp32 推理；Base/Large 基于 fp16 混合精度推理。

**边界条件：** 若性能基准明显低于预期，通过 `npu-smi info` 检查是否有其他进程占用 NPU 资源。

---

## 8. 验收确认

**检查点 4：确认所有验收项通过后标记适配完成。**

完成以下检查项后，适配即告完成：

- [ ] 环境：NPU 可识别，CANN 版本正确
- [ ] 权重：模型文件完整（config.json + model.safetensors）
- [ ] 加载：模型成功加载到 NPU
- [ ] 推理：三种模式（ae/clip/feature）均正常运行
- [ ] 精度：重建相对误差 < 1%
- [ ] 性能：推理延迟在基准范围内
- [ ] 交付：README.md 含 #+NPU 标签，推理脚本完整
- [ ] 报告：`eval/results.json` 和 `results/perf.json` 已生成

若存在未通过项，返回对应步骤排查修复后重新验收。

---

## 异常与边界条件处理

以下预定义各步骤可能出现的异常场景及恢复策略：

| 步骤 | 异常场景 | 触发条件 | 处理动作 |
|:---:|---|:---:|---|
| 0 | NPU 设备不可用 | `npu-smi info` 无输出 | 降级至 CPU 模式，标注 `fallback/cpu` |
| 0 | CANN 环境未加载 | `source set_env.sh` 失败 | 检查 CANN 安装路径，重试或回滚 |
| 1 | 依赖安装网络超时 | pip install 报 timeout | 切换镜像源重试，失败日志存 `scripts/install_error.log` |
| 2 | 模型权重下载中断 | snapshot_download 异常退出 | 清除不完整文件后重试下载 |
| 3 | 源码克隆失败 | git clone 返回非零 | 自动切换镜像源重试 |
| 4 | RoPE 修复后精度仍差 | 重建误差 > 1% | 检查 `torch_npu` 版本并强制 fp32 autocast |
| 5 | 模型加载 OOM | CUDA OutOfMemory | 减小 batch 至 1，切换 fp16，检查其他 NPU 进程 |
| 6 | 精度评测不达标 | 余弦相似度 < 0.9999 | 确认 RoPE 修复已生效后重新评测 |
| 7 | 性能结果异常 | 延迟远超基准值 2x | 检查 NPU 负载，空闲后重测 |
| 8 | 验收项未通过 | 某项 check 为 false | 返回对应步骤重新执行 |
| 任意 | 脚本运行时错误 | Python Traceback | 记录错误栈，检查依赖版本兼容性 |
| 任意 | NPU 进程卡死 | torch_npu 无响应 | `pkill -9 python` 后重启步骤 |

## 资源文件清单

| 文件路径 | 用途 |
|:---|:---|
| `scripts/npu_compat.py` | NPU 适配层：设备检测、RoPE 修复、autocast 辅助 |
| `scripts/inference.py` | 推理脚本（支持 ae/clip/feature 三种模式） |
| `scripts/evaluate.py` | 精度对比评测脚本 |
| `scripts/benchmark.py` | 性能基准测试脚本 |
| `eval/results.json` | 精度评测结构化结果 |
| `eval/evaluation.log` | 评测运行日志（含每张测试图像详情） |
| `results/perf.json` | 性能基准测试结果 |
| `model_weights/` | 模型权重缓存目录（首次运行自动下载） |
| `vtp-repo/` | VTP 官方源码（克隆后开箱即用） |

> 所有脚本位于 skill 目录下的 `scripts/` 子目录。

## 常见问题

### 1. RoPE dtype 导致的精度问题

**现象**：重建图像与 CPU 参考相比有明显色偏或伪影。

**原因**：`pixel_decoder.rope_embed` 使用 bfloat16 存储位置编码，NPU 与 CPU 的 bf16 实现存在微小差异，在深层解码器中逐层累积。

**解决**：参见步骤 4 的 RoPE dtype 修复。确认 `scripts/npu_compat.py` 中 `adapt_model_for_npu()` 已调用。

### 2. 大模型 OOM

**现象**：VTP-Large 加载或推理时 NPU 内存不足。

**解决**：
- 减小 batch size（默认 batch=1）
- 使用 fp16 混合精度（已默认启用）
- 检查是否有其他进程占用 NPU 内存

**错误恢复**：若 OOM 导致进程崩溃，清理 NPU 内存后再重试：
```bash
pkill -9 python
npu-smi info  # 确认内存已释放
```

### 3. transformers 版本兼容性

VTP 模型需要在 `transformers >= 4.55.0` 上运行，且需要 `trust_remote_code=True`（HF 版本）。

```python
# 版本验证
import transformers
assert int(transformers.__version__.split('.')[1]) >= 55, "需要 transformers >= 4.55.0"
```

### 4. 精度回滚策略

若 RoPE 修复导致其他异常行为，可回滚至原始 bf16 行为：
```python
def rollback_rope_dtype(model):
    """回滚 RoPE 精度至原始 bfloat16。"""
    if hasattr(model, "pixel_decoder") and model.pixel_decoder is not None:
        rope = model.pixel_decoder.rope_embed
        rope.dtype = torch.bfloat16
        rope.periods = rope.periods.to(torch.bfloat16)
        rope._init_weights()
    return model
```

---

## 交付件清单

每个模型的 GitCode 仓库包含：

| 文件 | 说明 |
|------|------|
| `README.md` | 模型卡，含 YAML frontmatter + `#+NPU` 标签 |
| `inference.py` | NPU 推理脚本（含精度验证） |
| `evaluate.py` | 完整评测脚本（性能+精度） |
| `eval/evaluation.log` | 评测运行日志 |
| `eval/results.json` | 结构化评测结果 |
| `results/perf.json` | 性能基准测试结果 |
| `scripts/npu_compat.py` | NPU 适配层工具函数 |

---

## 参考

- VTP 论文：https://arxiv.org/abs/2512.13687
- VTP 官方代码：https://github.com/MiniMax-AI/VTP
- VTP ModelScope 合集：https://modelscope.cn/collections/MiniMax/VTP
- VTP-Small 适配仓库：https://ai.gitcode.com/m0_74196153/VTP-Small-f16d64
- VTP-Base 适配仓库：https://ai.gitcode.com/m0_74196153/VTP-Base-f16d64
- VTP-Large 适配仓库：https://ai.gitcode.com/m0_74196153/VTP-Large-f16d64
- ascend-skills-eval 参考：`ascend-skills-eval/skills/skills-eval/evals/evals.json`
- NPU 性能调优指南：`references/npu-performance-tuning.md`
