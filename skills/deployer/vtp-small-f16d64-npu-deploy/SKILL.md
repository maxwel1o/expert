---
name: vtp-small-f16d64-npu-deploy
description: >
  MiniMax VTP-Small-f16d64 视觉 Tokenizer 模型在昇腾 NPU 上的完整部署、推理验证与精度对比 Skill。
  涵盖环境准备、ModelScope/HuggingFace 权重下载、CANN 配置、自动迁移适配、CPU/NPU 精度对比、
  BF16 混合精度推理、批量重建评估、DDP 分布式推理、性能基准测试以及异常处理与回滚的全流程标准化指南。
  当用户提到 VTP NPU 部署、MiniMax 视觉模型 NPU 推理、visual tokenizer NPU 适配时触发。
metadata:
  short-description: VTP-Small-f16d64 昇腾 NPU 部署与推理验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, vtp, minimax, vision-tokenizer, inference, pytorch, torch-npu, cann, ddp]
---

# VTP-Small-f16d64 昇腾 NPU 部署与推理验证 Skill

本 Skill 提供 MiniMax VTP-Small-f16d64 (Visual Tokenizer Pre-training) 视觉 Tokenizer 模型
在华为昇腾 NPU 上的完整端到端部署、自动迁移适配、推理验证和精度对比的标准化可复现流程。

VTP 是一个视觉 tokenizer 预训练框架，支持：
- CLIP/SigLIP 风格的图文对比学习
- DINOv2 风格的自监督学习
- 图像重建 (Reconstruction)

## 工作流总览

```
N.  步骤                       说明                             预期耗时
─────────────────────────────────────────────────────────────────────────
0.   环境初始化                 加载 CANN、检查 NPU 状态            5 min
1.   安装依赖                   Python 包 + torch_npu             10 min
2.   获取代码与权重             克隆 VTP 仓库 + 下载权重            15 min
3.   验证权重完整性              校验模型文件与目录结构              3 min
4.   基础推理验证                单卡冒烟测试 + 输出 shape 检查      5 min
5.   精度对比验证                NPU vs CPU 归一化误差              10 min
6.   BF16 混合精度推理           半精度推理性能与精度验证             5 min
7.   多精度对比测试              FP32/BF16/FP16 三精度横评          8 min
8.   ImageNet 重建评估           可选：批量重建 + FID 指标          30-60 min
9.   DDP 分布式批处理           多卡并行重建评估                    15 min
10.  自动 NPU 适配详解           RoPE 精度修复与适配原理             3 min
11.  性能基准测试                延迟、吞吐与显存基准                10 min
12.  异常恢复与回滚              常见错误处理 + 环境恢复             5 min
13.  常见问题与解决方案          Q&A 场景化排查                     5 min
14.  验收确认                    检查清单 + 用户确认                 3 min
15.  脚本与资源                  脚本清单与参考链接                  —
16.  评估标准                    质量维度与评分体系                 —
```

---

## 0. 环境初始化

### 0.1 加载 CANN 环境

```bash
# 加载昇腾 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

### 0.2 选择空闲 NPU 卡

```bash
# 查看 NPU 状态
npu-smi info

# 设置可用 NPU 卡 (替换为实际空闲卡号)
export ASCEND_RT_VISIBLE_DEVICES=0

# 验证 NPU 是否可用
python3 -c "import torch; import torch_npu; print('NPU count:', torch.npu.device_count()); print('Current device:', torch.npu.get_device_name(0))"
```

### 0.3 国内源加速 (可选)

```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
export HF_ENDPOINT=https://hf-mirror.com
```

> **检查点**：确认 `torch.npu.is_available()` 返回 `True`，否则暂停并排查 CANN 安装。
> **异常处理**：若 `npu-smi` 命令不存在，回滚检查 CANN 安装包是否正确部署。

---

## 1. 安装依赖

### 1.1 安装 Python 依赖

```bash
pip install torch torchvision torch_npu transformers \
    omegaconf timm scipy torchmetrics pytorch-fid tqdm pillow modelscope
```

### 1.2 验证关键依赖

```bash
python3 -c "
import torch; import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
print('NPU device:', torch.npu.get_device_name(0) if torch.npu.is_available() else 'N/A')
"
```

> **异常处理**：若 `torch_npu` 导入失败，检查 CANN 版本与 torch_npu 匹配：
> ```bash
# 查看 CANN 版本与 PyTorch 适配表
# Ascend 官网: https://www.hiascend.com/software/cann
# 安装对应版本
pip install torch_npu==对应CANN版本
```
>
> **失败 fallback**：若 torch_npu 始终无法导入，尝试在 Docker 容器中运行：
> ```bash
docker pull ascendhub.huawei.com/public/ascend-pytorch:latest
docker run -it --rm --device=/dev/davinci0 ascend-pytorch:latest bash
```

---

## 2. 获取代码与权重

### 2.1 克隆 VTP 官方代码

```bash
cd /path/to/vtp-small-f16d64-npu
git clone https://github.com/MiniMax-AI/VTP.git vtp-repo
export PYTHONPATH=$PWD/vtp-repo:$PYTHONPATH
```

### 2.2 从 ModelScope 下载权重 (推荐国内用户)

```bash
python3 -c "
from modelscope import snapshot_download
snapshot_download('MiniMax/VTP-Small-f16d64', cache_dir='./model_weights')
print('Model weights downloaded successfully.')
"
```

### 2.3 从 HuggingFace 下载权重 (国外用户)

```bash
pip install huggingface-hub
huggingface-cli download MiniMaxAI/VTP-Small-f16d64 --local-dir ./model_weights/MiniMax/VTP-Small-f16d64
```

> **异常处理**：下载失败时的 fallback 策略：
> ```bash
# 网络超时切换到国内镜像重试
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download MiniMaxAI/VTP-Small-f16d64 --local-dir ./model_weights/MiniMax/VTP-Small-f16d64 --resume-download

# 若仍失败，使用 ModelScope 替代
python3 -c "from modelscope import snapshot_download; snapshot_download('MiniMax/VTP-Small-f16d64', cache_dir='./model_weights')"
```

---

## 3. 验证权重完整性

```bash
ls -lh ./model_weights/MiniMax/VTP-Small-f16d64/
```

预期文件列表：
```
config.json              # 模型配置 (~1KB)
model.safetensors        # 模型权重 (~638MB)
preprocessor_config.json # 预处理器配置
```

> **检查点**：确认 `model_weights/MiniMax/VTP-Small-f16d64/config.json` 存在且非空。
> **异常处理**：若 config.json 不存在，重新执行下载步骤；仍失败则清理缓存：
> ```bash
rm -rf ./model_weights/
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/modelscope/
# 然后重试步骤 2
```

---

## 4. 基础推理验证 (Quick Start)

### 4.1 快速冒烟测试

```bash
cd /path/to/vtp-small-f16d64-npu
export PYTHONPATH=$PWD/vtp-repo:$PYTHONPATH

python3 scripts/inference.py \
    --model_path ./model_weights/MiniMax/VTP-Small-f16d64 \
    --quick
```

预期输出：

```
VTP-Small-f16d64 NPU Inference
============================================================
Device: npu:0
Precision: fp32
...
[NPU Adapt] pixel_decoder RoPE dtype: bf16 -> fp32
Avg latency: ~450 ms
Image feature shape: torch.Size([1, 768])
Reconstruction latents shape: torch.Size([1, 64, 16, 16])
Reconstructed image shape: torch.Size([1, 3, 256, 256])
Inference completed successfully.
```

### 4.2 自定义 Python 推理

```python
import sys
sys.path.insert(0, "vtp-repo")
sys.path.insert(0, "scripts")

from npu_compat import get_device, adapt_model_for_npu
from vtp.models.vtp_hf import VTPModel

device = get_device()  # 自动检测 NPU
model = VTPModel.from_pretrained("./model_weights/MiniMax/VTP-Small-f16d64")
model = adapt_model_for_npu(model)
model = model.to(device).eval()

# CLIP 图像特征提取
import torch
images = torch.randn(1, 3, 256, 256).to(device)
img_feat = model.get_clip_image_feature(images)
print("Image feature shape:", img_feat.shape)  # [1, 768]

# 图像重建
latents = model.get_reconstruction_latents(images)
recon = model.get_latents_decoded_images(latents)
print("Reconstructed image shape:", recon.shape)  # [1, 3, 256, 256]
```

### 4.3 参数说明

```bash
python3 scripts/inference.py --help
```

```
--model_path    (必填) 模型权重的本地路径
--batch_size    (可选) 批大小，默认 1
--precision     (可选) 推理精度: fp32 / fp16 / bf16，默认 fp32
--quick         (可选) 快速模式，跳过 CPU 对比
--device        (可选) 设备指定: npu / cuda / cpu
--seed          (可选) 随机种子，默认 42
```

> **检查点**：输出 shape 必须与预期一致；若 shape 异常，回滚并检查权重完整性。
> **异常处理**：OOM 时降级批大小或切换卡号：
> ```bash
# OOM fallback: 使用其他空闲卡
export ASCEND_RT_VISIBLE_DEVICES=1
# 或减小 batch_size
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --batch_size 1 --quick
```

---

## 5. 精度对比验证 (NPU vs CPU)

运行 NPU vs CPU 精度对比 (注意：CPU 推理较慢，约需 2-3 分钟)：

```bash
cd /path/to/vtp-small-f16d64-npu
export PYTHONPATH=$PWD/vtp-repo:$PYTHONPATH

python3 scripts/inference.py \
    --model_path ./model_weights/MiniMax/VTP-Small-f16d64 \
    --precision fp32
```

预期精度结果：

```
============================================================
Precision Verification (NPU vs CPU)
============================================================
  img_feat            : PASS | norm_max_err=0.031%
  rec_latents         : PASS | norm_max_err=0.44%
  rec_image           : PASS | norm_max_err=0.77% | PSNR=57.49dB
============================================================
All precision checks PASSED (normalized max error < 1%)
```

### 5.1 精度判据

| 输出张量       | 判据                    | 可接受阈值      |
|----------------|------------------------|-----------------|
| img_feat       | 归一化最大误差 < 1%     | < 1%            |
| rec_latents    | 归一化最大误差 < 1%     | < 1%            |
| rec_image      | 归一化最大误差 < 1% 且 PSNR > 35dB | < 1% & PSNR > 35dB |

> **异常处理**：精度对比 FAIL 时的 recovery 步骤：
> ```bash
# 1. 确认 npu_compat.py 中的适配逻辑已正确执行
# 2. 检查 --precision 参数是否一致
# 3. 回滚：重新下载权重并清除缓存
rm -rf ./model_weights/MiniMax/VTP-Small-f16d64
# 4. 使用 fp32 重新推理验证
```
>
> **检查点**：确认所有精度检查 PASS，`norm_max_err` 均 < 1%。

---

## 6. BF16 混合精度推理

使用 bfloat16 半精度推理以提升吞吐：

```bash
cd /path/to/vtp-small-f16d64-npu
export PYTHONPATH=$PWD/vtp-repo:$PYTHONPATH

python3 scripts/inference.py \
    --model_path ./model_weights/MiniMax/VTP-Small-f16d64 \
    --precision bf16 \
    --quick
```

预期 BF16 输出差异：
- 延迟从 ~450ms (fp32) 降至 ~320ms (bf16)
- 推理输出 shape 与 fp32 一致
- RoPE 自动从 bf16 转换为 fp32 保证数值稳定

```bash
# BF16 完整精度对比
python3 scripts/inference.py \
    --model_path ./model_weights/MiniMax/VTP-Small-f16d64 \
    --precision bf16
```

> **检查点**：确认 BF16 模式下精度对比 `norm_max_err < 1%`。
> **异常处理**：若 BF16 推理报错，fallback 回 fp32 模式：
> ```bash
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --precision fp32 --quick
```

---

## 7. 多精度对比测试

FP32/BF16/FP16 三种精度的横向对比：

```bash
# FP32 精度对比
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --precision fp32

# BF16 精度对比
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --precision bf16

# FP16 精度对比
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --precision fp16
```

预期对比结果：

| 精度 | 延迟 (ms) | img_feat 误差 | rec_image PSNR |
|------|-----------|---------------|----------------|
| fp32 | ~450      | < 0.05%       | > 55 dB        |
| bf16 | ~320      | < 0.5%        | > 50 dB        |
| fp16 | ~310      | < 1.0%        | > 45 dB        |

> **检查点**：所有精度的重建 PSNR 均 > 35 dB。
> **异常处理**：某一精度 FAIL 不阻塞整体流程，记录结果后继续下一精度测试。

---

## 8. ImageNet 重建评估 (可选)

如需在 ImageNet 验证集上评估重建质量：

### 8.1 单 NPU 重建评估

```bash
python3 scripts/test_reconstruction_hf_npu.py \
    --model_path ./model_weights/MiniMax/VTP-Small-f16d64 \
    --data_path /path/to/imagenet/val \
    --precision bf16
```

### 8.2 重建结果评估

```python
# 评估重建质量
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# 加载 real/fake 图像计算 FID
fid = FrechetInceptionDistance(feature=2048)
# fid.update(real_images, real=True)
# fid.update(fake_images, real=False)
# print(f"FID: {fid.compute():.2f}")
```

> **异常处理**：ImageNet 数据路径错误时：
> ```bash
# 验证数据集路径
ls /path/to/imagenet/val/ | head -5
# 若路径错误，重新设置 --data_path
```

---

## 9. DDP 分布式批处理

多卡并行加速重建评估：

```bash
torchrun --nproc_per_node=8 scripts/test_reconstruction_hf_npu.py \
    --model_path ./model_weights/MiniMax/VTP-Small-f16d64 \
    --data_path /path/to/imagenet/val \
    --use_ddp \
    --precision bf16
```

### 9.1 DDP 配置验证

```bash
# 确认可用 NPU 卡数
npu-smi info | grep -c "NPU"

# 设置多卡可见
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

> **检查点**：DDP 模式下确认 `nproc_per_node <= npu-smi info` 显示的可用卡数。
> **异常处理**：DDP 初始化失败时回滚到单卡模式：
> ```bash
# DDP fallback to single NPU
python3 scripts/test_reconstruction_hf_npu.py \
    --model_path ./model_weights/MiniMax/VTP-Small-f16d64 \
    --data_path /path/to/imagenet/val \
    --precision bf16
```

---

## 10. 自动 NPU 适配详解

### 10.1 RoPE 精度修复原理

`pixel_decoder` 中的 `RopePositionEmbedding` 默认使用 `bfloat16`，NPU/CPU 数值行为存在差异。
`scripts/npu_compat.py` 在 NPU 模式下自动将其转换为 `float32`，确保精度达标。

```python
def adapt_model_for_npu(model):
    """Fix pixel_decoder RoPE dtype for NPU compatibility."""
    if hasattr(model, "pixel_decoder") and model.pixel_decoder is not None:
        rope = model.pixel_decoder.rope_embed
        if rope.dtype == torch.bfloat16:
            rope.dtype = torch.float32
            rope.periods = rope.periods.to(torch.float32)
            rope._init_weights()
            print("[NPU Adapt] pixel_decoder RoPE dtype: bf16 -> fp32")
    return model
```

### 10.2 设备自动检测

```python
def get_device(prefer_npu=True):
    """Auto-detect available device: npu > cuda > cpu."""
    if prefer_npu:
        try:
            import torch_npu
            if torch.npu.is_available():
                return torch.device("npu:0")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
```

### 10.3 Autocast 辅助函数

```python
def maybe_autocast(device, precision="fp32"):
    """Return autocast context or no-op context."""
    device_type = "npu" if device.type == "npu" else device.type
    if device_type == "cpu" or precision == "fp32":
        from contextlib import nullcontext
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)
    return torch.amp.autocast(device_type=device_type, dtype=dtype)
```

> **检查点**：验证适配日志中输出 `[NPU Adapt] pixel_decoder RoPE dtype: bf16 -> fp32`。
> **异常处理**：若未输出适配日志，检查 `npu_compat.py` 是否正确导入。

---

## 11. 性能基准测试

### 11.1 延迟基准

```bash
# FP32 延迟
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --quick

# BF16 延迟
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --precision bf16 --quick
```

### 11.2 显存监控

```bash
# 推理前查看显存
npu-smi info | grep -E "NPU|Memory"

# 推理中监控显存占用
watch -n 1 npu-smi info
```

### 11.3 批处理吞吐

```bash
# 测试 batch_size=4 的吞吐
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --batch_size 4 --quick

# 测试 batch_size=8 的吞吐 (需足够显存)
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --batch_size 8 --quick
```

> **异常处理**：批处理 OOM 时减少 batch_size 或切换到更大显存的 Ascend 910B 卡。

---

## 12. 异常恢复与回滚

### 12.1 常见异常及处理

| 异常场景                  | 诊断命令                           | 处理措施                                              |
|--------------------------|-----------------------------------|------------------------------------------------------|
| CANN 未加载               | `which npu-smi`                   | `source /usr/local/Ascend/ascend-toolkit/set_env.sh`  |
| NPU 卡被占用              | `npu-smi info`                    | 切换 `ASCEND_RT_VISIBLE_DEVICES` 到空闲卡号            |
| torch_npu 导入失败        | `pip list \| grep torch_npu`      | `pip install torch_npu=={CANN对应版本}`               |
| 权重下载超时              | `ls ./model_weights/`             | 设置镜像源后 `--resume-download` 重试                   |
| OOM (out of memory)      | `npu-smi info` 查看显存占用       | 减小 batch_size 或切换到更大显存卡                     |
| 精度对比 FAIL             | 检查输出的 norm_max_err           | 确认 npu_compat.py 的 bf16->fp32 适配已生效             |
| 推理输出 shape 异常       | 对比文档中的预期 shape            | 回退权重并清除 `~/.cache/huggingface/` 重新下载          |
| DDP 初始化失败            | `torchrun --nproc_per_node=...`   | 确认 `ASCEND_RT_VISIBLE_DEVICES` 包含足够卡号           |
| Python 依赖冲突           | `pip check`                       | `pip install --upgrade` 相应包或重建虚拟环境            |
| 容器内无 NPU 设备         | `ls /dev/davinci*`                | 添加 `--device=/dev/davinci0` 重新启动容器              |

### 12.2 回滚步骤

```bash
# 1. 卸载当前依赖
pip uninstall -y torch torch_npu torchvision

# 2. 清理权重缓存
rm -rf ./model_weights/
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/modelscope/

# 3. 删除克隆的代码
rm -rf ./vtp-repo/

# 4. 从步骤 0 环境初始化重新执行
```

### 12.3 重试策略

```bash
# 自动重试脚本 (下载失败重试 3 次)
for i in 1 2 3; do
    python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --quick && break
    echo "Retry $i/3 failed, waiting 5s..."
    sleep 5
done
```

> **检查点**：回滚后必须重新进行环境验证 (步骤 0.2)。

---

## 13. 常见问题与解决方案

### 13.1 Q: NPU 推理结果与 CPU 差异较大

**原因**：`pixel_decoder` 的 `RopePositionEmbedding` 在 NPU bf16 模式下数值行为与 CPU fp32 不同。

**解决方案**：
```python
# 在 npu_compat.py 中已自动处理
# 手动验证适配效果
python3 -c "
from npu_compat import adapt_model_for_npu
from vtp.models.vtp_hf import VTPModel
model = VTPModel.from_pretrained('./model_weights/MiniMax/VTP-Small-f16d64')
model = adapt_model_for_npu(model)
print('Adaptation applied successfully')
"
```

### 13.2 Q: 推理报 "out of memory"

**原因**：显存不足，VTP 模型 ~638MB，需约 4-8GB HBM 用于推理。

**解决方案**：
```bash
# 立即减小 batch_size
export ASCEND_RT_VISIBLE_DEVICES=1  # 切换到其他卡
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --batch_size 1 --quick
```

### 13.3 Q: torch_npu 找不到

**原因**：CANN 版本与 torch_npu 版本不匹配。

**解决方案**：
```bash
pip list | grep torch
pip install torch_npu==2.1.0  # 与 CANN 8.0 匹配
```

> **检查点**：确认每个问题的 fallback 方案已记录。
> **异常处理**：若上述方案均失败，切换至 Docker 环境：
> ```bash
docker pull ascendhub.huawei.com/public/ascend-pytorch:latest
```

---

## 14. 验收确认

### 14.1 检查清单

| # | 检查项                               | 确认方法                                      | 状态   |
|---|-------------------------------------|----------------------------------------------|--------|
| 1 | CANN 环境已加载                       | `which npu-smi` 返回路径                      | [ ]    |
| 2 | NPU 卡可用                           | `torch.npu.is_available()` 返回 True          | [ ]    |
| 3 | 依赖安装完成                         | `python3 -c "import torch_npu"` 无报错        | [ ]    |
| 4 | 模型权重下载完成                     | `ls config.json` 存在                        | [ ]    |
| 5 | NPU 冒烟推理通过                     | 输出 shape: `[1,768]`, `[1,64,16,16]`, `[1,3,256,256]` | [ ]    |
| 6 | NPU vs CPU 精度对比 PASS            | `norm_max_err < 1%` 且 PSNR > 35dB            | [ ]    |
| 7 | BF16 推理验证通过                    | BF16 模式下精度对比 PASS                        | [ ]    |
| 8 | 异常处理方案已验证                   | 覆盖 OOM, 超时, 精度 FAIL 等场景               | [ ]    |

### 14.2 用户确认

执行以下命令生成最终验证报告：

```bash
echo "=== VTP-Small-f16d64 NPU 部署验证报告 ==="
echo "日期: $(date)"
npu-smi info | grep "NPU"
python3 -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available())"
python3 scripts/inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --quick
```

> **暂停点**：用户确认所有检查项通过后，方可关闭环境或进行后续操作。

### 14.3 清理环境

```bash
# 释放 NPU 资源
unset ASCEND_RT_VISIBLE_DEVICES

# 清理临时文件
rm -rf __pycache__/
rm -rf .eggs/
```

---

## 15. 脚本与资源

### 15.1 脚本清单

| 脚本                                    | 用途                           | 位置                              |
|----------------------------------------|-------------------------------|-----------------------------------|
| `scripts/inference.py`                 | NPU 推理与精度对比主脚本          | `vtp-small-f16d64-npu/scripts/`   |
| `scripts/npu_compat.py`                | NPU 兼容适配层 (RoPE 转换等)     | `vtp-small-f16d64-npu/scripts/`   |
| `scripts/test_reconstruction_hf_npu.py`| ImageNet 批量重建评估            | `vtp-small-f16d64-npu/scripts/`   |

### 15.2 参考资源

| 资源                               | 链接/路径                                                        |
|-----------------------------------|-----------------------------------------------------------------|
| VTP 官方仓库                       | `https://github.com/MiniMax-AI/VTP`                             |
| ModelScope 模型页                  | `https://modelscope.cn/models/MiniMax/VTP-Small-f16d64`         |
| HuggingFace 模型页                 | `https://huggingface.co/MiniMaxAI/VTP-Small-f16d64`             |
| Ascend CANN 文档                   | `https://www.hiascend.com/document`                             |
| torch_npu 安装指南                 | `https://pypi.org/project/torch_npu/`                           |
| NPU 适配评估结果                   | `ascend-skills-eval/evals/evals.json` (结构评分记录)             |
| ascend-skills-eval 项目            | `ascend-skills-eval/skills/skills-eval/templates/` (模板参考)   |

---

## 16. 评估标准

本 Skill 的质量评估 (ascend-skills-eval) 覆盖以下维度：
- **Frontmatter 质量**: 元数据完整性和可索引性
- **工作流清晰度**: 步骤组织与导航逻辑
- **边界条件覆盖**: 异常、错误与回滚策略
- **检查点设计**: 验证点与暂停确认
- **指令具体性**: 命令、代码块与参数示例
- **资源整合度**: 脚本、结果、评测与引用整合
- **整体架构**: 章节结构与信息层次
- **昇腾适配性**: NPU 与 Ascend 生态适配程度
- **实测表现**: 可执行命令与验证覆盖度

> **注**：当前评估结果记录在 `ascend-skills-eval/evals/evals.json` 中。
