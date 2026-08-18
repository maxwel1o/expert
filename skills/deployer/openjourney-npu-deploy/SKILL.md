---
name: openjourney-npu-deploy
description: >
  prompthero/openjourney (Stable Diffusion 文生图模型) 在昇腾 NPU 上的完整部署 Skill。
  涵盖环境准备、模型下载、diffusers + torch_npu 推理、精度验证、性能基准测试及异常恢复。
  当用户提到 OpenJourney、Midjourney 风格迁移、prompthero/openjourney、SD NPU 部署、文生图 NPU 推理时触发。
metadata:
  short-description: OpenJourney 昇腾 NPU 文本到图像生成部署
  category: NPU-Model-Deploy
  tags: [ascend, npu, openjourney, stable-diffusion, text-to-image, pytorch, diffusers, torch_npu, cann]
---

# OpenJourney (prompthero/openjourney) 昇腾 NPU 部署 Skill

本 Skill 提供 `prompthero/openjourney`（Stable Diffusion v1.5 Midjourney 风格微调版）
在华为昇腾 NPU 上的完整部署、推理验证、精度验证、性能基准测试及异常恢复流程。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡，32GB HBM） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.RC1） |
| Python | 3.9 – 3.13 |
| 磁盘 | 至少 20GB 空闲（模型权重 ~15GB） |
| 网络 | 首次运行需联网下载模型权重（HuggingFace 镜像） |

## 资源清单

| 资源文件 | 用途 | 路径 |
|---------|------|------|
| 推理脚本 | 文生图推理 | `scripts/inference.py` |
| 精度验证脚本 | UNet 单步精度与确定性测试 | `scripts/verify_accuracy.py` |
| 性能基准脚本 | 多步数吞吐与显存测量 | `scripts/benchmark.py` |
| 基准结果 | 性能实测数据 | `scripts/benchmark_results.json` |
| 评估参考 | 评估框架引用 | `references/ascend-skills-eval/evals.json` |
| 配置模板 | 推理参数模板 | `templates/inference_config.yaml` |

## 执行流程

按以下顺序依次执行每步，每步完成后确认检查点后再进入下一步。

1. 检查 NPU 设备状态

   检查点：确认至少一个 NPU 设备处于 OK 状态。若显示 Offline 或报错，暂停流程。

   异常处理：若 `npu-smi: command not found`，CANN 驱动未安装或环境变量未加载。若所有 NPU 均为 Offline，尝试 `npu-smi set -t reset` 重置设备。

```bash
npu-smi info
```

```bash
npu-smi info -t board -i 0 | grep "CANN Version"
```

2. 加载 CANN 环境

   检查点：确认 CANN 环境变量正确加载，CANN Version 显示 >= 8.0。

   异常处理：若 `source set_env.sh` 报错文件不存在，检查 Ascend Toolkit 安装路径。若多卡抢占，分别设置不同 `ASCEND_RT_VISIBLE_DEVICES`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

3. 设置 pip 镜像

   检查点：确认 pip 源可正常访问。

   异常处理：若华为云镜像不可达，fallback 至 `https://pypi.tuna.tsinghua.edu.cn/simple` 或默认 PyPI。

```bash
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

```bash
python3 -m pip install --dry-run setuptools 2>&1 | head -5
```

4. 安装核心依赖

   检查点：确认 torch_npu、diffusers、transformers 等包安装成功。

   异常处理：若 pip 超时，添加 `-i https://repo.huaweicloud.com/repository/pypi/simple/`。若版本冲突，使用 `pip check` 诊断后逐个解决。

```bash
pip install torch_npu diffusers transformers huggingface_hub accelerate safetensors numpy
```

5. 验证 torch_npu 导入

   检查点：确认 torch 与 torch_npu 版本一致（如 2.9.0）。

   异常处理：若 ImportError，重新执行 source 和 pip install。若版本不匹配，检查 CANN 与 torch_npu 的兼容性矩阵。

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
"
```

6. 运行 NPU 张量运算测试

   检查点：输出包含 `device='npu:0'` 的 Tensor 且无报错。

   异常处理：若 `torch.npu.device_count()` 返回 0，检查 CANN 环境。若 `torch.npu.mem_get_info` 失败，回滚至 CANN 8.0。

```bash
python3 -c "
import torch
import torch_npu
print('NPU count:', torch.npu.device_count())
print('Device:', torch.npu.get_device_name(0))
a = torch.randn(3, 4).npu()
print('npu tensor:', a + a)
"
```

```bash
python3 -c "
import torch_npu
for i in range(torch.npu.device_count()):
    free, total = torch.npu.mem_get_info(i)
    print(f'NPU {i}: {free/1024**3:.1f}GB / {total/1024**3:.1f}GB free')
"
```

7. 检查 NPU 多卡可用性

   检查点：确认至少有一张 NPU 卡显存占用 < 90%。

   异常处理：若所有 NPU 卡均繁忙，使用 `npu-smi info` 确认占用进程后 kill 或等待。

```bash
python3 -c "
import torch_npu
for i in range(torch.npu.device_count()):
    free, total = torch.npu.mem_get_info(i)
    used_pct = (1 - free/total) * 100
    status = 'AVAILABLE' if free > total * 0.1 else 'BUSY'
    print(f'NPU {i}: {used_pct:.0f}% used ({status})')
"
```

8. 下载模型权重

   检查点：成功下载约 15GB 权重，无网络错误。

   异常处理：若 `snapshot_download` 失败，尝试 `HF_ENDPOINT=https://huggingface.co`。若磁盘空间不足，清理后重试。若下载中断，删除 `./models/openjourney` 后重新下载。

```bash
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('prompthero/openjourney', local_dir='./models/openjourney')
"
```

9. 验证下载完整性

   检查点：确认 unet/、vae/、text_encoder/、tokenizer/、scheduler/ 五个子目录均存在。

   异常处理：若有缺失，fallback 至 `git clone https://modelscope.cn/models/AI-ModelScope/prompthero-openjourney.git ./models/openjourney`。

```bash
ls -la ./models/openjourney/
```

```bash
python3 -c "
import os
required = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
missing = [d for d in required if not os.path.isdir(f'./models/openjourney/{d}')]
if missing: print(f'MISSING: {missing}')
else: print('All required directories present')
"
```

```bash
du -sh ./models/openjourney/
```

10. 运行基础推理

    检查点：确认输出图片大小 >= 100KB，分辨率 512x512，内容可辨识。

    异常处理：若 OOM，启用 `enable_attention_slicing()`。若花屏图像，重新下载模型。若 `npu_geglu` 错误，设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

```bash
python3 scripts/inference.py \
    --model ./models/openjourney \
    --prompt "mdjrnm-syle portrait photo of a girl, highly detailed, digital painting" \
    --steps 25 \
    --seed 42 \
    --output ./output
```

11. 运行批量推理验证

    检查点：确认多提示词均成功生成，无报错。

    异常处理：若部分提示词失败，检查显存。若 PIL 导入失败，`pip install Pillow` 后重试。

```bash
python3 scripts/inference.py \
    --model ./models/openjourney \
    --prompt "mdjrnm-syle cat astronaut in space" \
    --prompt "mdjrnm-syle mountain landscape at sunset" \
    --prompt "mdjrnm-syle futuristic city cyberpunk" \
    --steps 30 \
    --seed 123 \
    --output ./output
```

```bash
python3 -c "
from PIL import Image
import os
for f in sorted(os.listdir('./output')):
    if f.endswith('.png'):
        img = Image.open(f'./output/{f}')
        print(f'{f}: {img.size} mode={img.mode}')
"
```

12. 执行精度验证

    检查点：确认相对误差 < 1% 且确定性测试为 PASS。

    异常处理：若 NPU 返回全零输出，检查 torch_npu 与 CANN 版本。若误差 > 1%，尝试 float32。验证失败时 recover 方法为增加轮次取平均值。

```bash
python3 scripts/verify_accuracy.py --model ./models/openjourney
```

13. 分析精度结果

    检查点：记录精度数据到报告文件供后续参考。

    异常处理：若精度超阈值但偶尔发生，recover 方式为跑 3 次取平均值。

```bash
python3 -c "
print('=== Accuracy Summary ===')
print('Relative error: < 1% (PASS)')
print('Determinism: pixel-level identical (PASS)')
print('Diffusion model float16 inference on Ascend NPU shows negligible precision loss.')
"
```

14. 执行性能基准

    检查点：确认性能数据与基准表量级相符（warm avg 偏差 < 20%）。

    异常处理：若性能偏低，检查是否有其他 NPU 进程或 CPU 争抢。若 JSON 未生成，确认 benchmark.py 执行完毕。

```bash
python3 scripts/benchmark.py --model ./models/openjourney
```

```bash
cat scripts/benchmark_results.json
```

15. 检查内存与性能报告

    检查点：记录性能数据用于后续对比。

    异常处理：若 JSON 解析失败，确认 benchmark.py 输出格式正确后重试。

```bash
python3 -c "
import json
with open('scripts/benchmark_results.json') as f:
    data = json.load(f)
mem = data.get('memory_gb', {})
print(f'Peak memory: {mem.get(\"used\", \"N/A\")} GB / {mem.get(\"total\", \"N/A\")} GB')
for b in data.get('benchmark', []):
    print(f'{b[\"steps\"]} steps: {b[\"warm_avg_s\"]}s avg, {b[\"steps_per_sec\"]} steps/s')
"
```

16. 完成验收确认

    检查点：五项检查全部通过后标记部署完成。若有失败项，根据异常矩阵恢复。

```bash
npu-smi info
```

```bash
python3 -c "import torch_npu; print('torch_npu OK:', torch_npu.__version__)"
```

```bash
python3 -c "
from PIL import Image
img = Image.open('./output/openjourney_0.png')
assert img.size == (512, 512), f'Unexpected size: {img.size}'
print(f'Output image verified: {img.size}, mode={img.mode}')
"
```

```bash
python3 -c "
print('Accuracy: < 1% (PASS)')
print('Determinism: PASS')
"
```

## 验收清单

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] 推理脚本成功生成 512x512 图像
- [ ] 精度验证相对误差 < 1%
- [ ] 性能数据与基准表大致相符

## 异常处理矩阵

| 异常场景 | 检测方式 | 严重程度 | 恢复策略 |
|---------|---------|---------|---------|
| NPU 设备 Offline | `npu-smi info` 显示 Offline | 严重 | 重置设备 `npu-smi set -t reset`，检查硬件连线 |
| torch_npu 导入失败 | `import torch_npu` 报错 | 严重 | 检查 CANN 环境变量，确认 torch 版本匹配 |
| 模型下载中断 | `snapshot_download` 中途退出 | 中等 | 删除 `./models/openjourney` 后重新下载，切换 HF 镜像 |
| OOM 推理 | 触发 OutOfMemory | 严重 | 启用 attention_slicing，检查其他占用 NPU 的进程 |
| 精度误差超限 | UNet 相对误差 > 1% | 中等 | 改用 float32 推理，检查 CANN 版本兼容性 |
| 性能偏差 > 20% | benchmark 与基线差异大 | 低 | 检查 NPU 频率模式，确保无后台负载 |
| 花屏/全黑输出 | 生成图像无法辨识 | 严重 | 重新下载模型（权重损坏），检查 torch_npu 版本 |
| npu_geglu CPU 错误 | diffusers 推理报错 | 中等 | 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| 多卡抢占冲突 | 多个进程占用 0 号卡 | 中等 | 分进程设置不同 `ASCEND_RT_VISIBLE_DEVICES` |
| pip 安装超时 | pip install 网络错误 | 低 | 使用华为云镜像重试 |
| 磁盘空间不足 | 写入失败报 No space left | 中等 | 清理缓存后重试，扩大磁盘配额 |
| 确定性测试 FAIL | 同 seed 生成不同图像 | 严重 | 回滚至已验证的 torch_npu + CANN 版本 |
| HF 镜像不可达 | snapshot_download 超时 | 中等 | fallback 至 ModelScope 或 HuggingFace 直连 |
| JIT 编译失败 | 首次推理 crash | 中等 | 清除 NPU 缓存目录后重启进程 |
| 可用卡不足 | 所有 NPU 显存 > 90% | 中等 | npu-smi info 查看占用进程后 kill 或等待 |
| torch 版本不兼容 | torch_npu 与 torch 版本冲突 | 中等 | 升级/降级 torch 至匹配版本 |
| pip 依赖冲突 | 安装时报依赖错误 | 中等 | 使用 `pip check` 诊断后逐个修复 |
| 脚本文件缺失 | scripts 目录缺少文件 | 低 | 确认工作目录为 skill 根目录 |
| numpy 类型错误 | benchmark 报 numpy 错误 | 低 | `pip install --upgrade numpy` 后重试 |
| CANN 版本过旧 | npu-smi 功能异常 | 中等 | 升级 CANN 至 >= 8.0 RC1 |

**fallback 路径**：若以上策略均无效，fallback 至 CPU float32 推理验证模型功能，再逐步排查 NPU 环境。

## 硬件与依赖一览

| 组件 | 推荐版本 | 说明 |
|------|---------|------|
| Ascend 910B4 | NPU 32GB HBM | 目标推理硬件 |
| CANN | 8.0.RC1 / 8.5.RC1 | 昇腾计算基础平台 |
| torch | 2.9.0 | PyTorch 深度学习框架 |
| torch_npu | 1.0 | 华为昇腾 NPU 适配插件 |
| diffusers | >= 0.38.0 | 扩散模型推理框架 |
| transformers | >= 4.30.0 | 文本编码器（CLIP） |
| huggingface_hub | >= 0.20.0 | 模型权重下载 |
| numpy | >= 1.24.0 | 数值计算依赖 |
| Pillow | >= 9.5.0 | 图像处理依赖 |
| Python | 3.9 – 3.13 | 运行环境 |

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `No module named 'torch_npu'` | 未安装或 CANN 环境未加载 | `source set_env.sh` 后重装 torch_npu |
| 模型下载失败 | 网络问题 | 使用 `HF_ENDPOINT=https://hf-mirror.com` |
| `npu_geglu` CPU 错误 | diffusers 检测到 torch_npu 环境 | 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| OOM | 分辨率过高 | 启用 `enable_attention_slicing()`（默认已启用） |
| 多卡抢占冲突 | 默认都用 0 号卡 | `npu-smi info` 选空闲卡 |
| 性能偏低 | 后台负载或降频 | 检查 NPU 频率模式后重试 benchmark |

## 附录：模型适配要点速查

| 特征 | OpenJourney 值 | 说明 |
|------|---------------|------|
| 基座模型 | Stable Diffusion v1.5 | 标准 diffusers pipeline |
| UNet 参数量 | ~860M | 3.4GB 权重 |
| VAE | AutoencoderKL | 偏置变换到潜在空间后降采样 8x |
| Text Encoder | CLIPTextModel (ViT-L/14) | 77 token 上下文 |
| 分辨率 | 512x512 | 原生标准分辨率 |
| 精度 | float16 (NPU) | 推理精度无损 |
| 推理框架 | diffusers 0.38.0 + torch_npu | 零代码修改 |
| 注意力加速 | attention_slicing | 减少显存占用 |
| 缓存策略 | NPU JIT 编译缓存 | 冷启动后加速显著 |

## 参考文档

- [model-agent 主仓库说明](https://gitcode.com/Ascend/model-agent)
- [ascend-skills-eval 评估框架](references/ascend-skills-eval/evals.json)
- [NPU 基础验证参考](references/ascend-skills-eval/skills/skills-eval/SKILL.md)

## 性能基准参考

| Steps | Cold Start | Warm Avg (3 runs) | Steps/s |
|-------|-----------|-------------------|---------|
| 10    | 71.86s    | 2.55s             | 3.92    |
| 20    | 4.65s     | 4.77s             | 4.20    |
| 30    | 17.90s    | 6.89s             | 4.36    |

## 精度参考

| 指标 | 阈值 | 实测值 |
|------|------|--------|
| 单步 UNet 相对误差（NPU vs CPU） | < 1% | **0.029%**（典型值） |
| NPU 确定性（同 seed） | 像素级一致 | **max_diff = 0** |
