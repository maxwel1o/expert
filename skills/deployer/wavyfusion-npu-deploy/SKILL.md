---
name: wavyfusion-npu-deploy
description: >
  WavyFusion（wavymulder/wavyfusion）Stable Diffusion 1.5 Fine-tune 模型在
  昇腾 NPU 上的完整部署与推理验证 Skill。涵盖环境准备、模型权重下载、NPU 推理、
  精度验证（fp16 vs fp32 精度对比）、性能基准测试的全流程。可在任意 Ascend910 系列
  服务器上一键复现，并支持多卡并行推理性能评估。
metadata:
  short-description: WavyFusion SD1.5 昇腾 NPU 部署与推理验证与精度对比
  category: NPU-Model-Deploy
  tags: [ascend, npu, stable-diffusion, diffusers, text-to-image, wavyfusion, pytorch, inference, benchmark]
---

# WavyFusion 昇腾 NPU 部署与推理验证 Skill

本 Skill 提供 WavyFusion（wavymulder/wavyfusion）模型在华为昇腾 NPU 上的
完整部署、推理验证和性能基准测试的标准化可复现流程。适用于 Ascend910 系列芯片的
模型部署与昇腾推理验证场景。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡，32GB HBM） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0.RC1 |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（约 5GB） |

## 执行工作流

1. 环境初始化与 NPU 状态确认：加载 CANN 环境、设置设备可见性、使用 npu-smi 确认空闲卡。
2. 安装依赖：通过 pip 安装 torch、torch_npu、diffusers、transformers、accelerate。
3. 模型权重下载：从 HuggingFace 镜像或 ModelScope 下载 wavymulder/wavyfusion。
4. NPU 设备验证：确认 NPU 设备数量、型号、显存容量，确保推理环境就绪。
5. 基础推理验证：运行 inference.py 进行单张图片生成，验证输出为 512×512 RGB 图片。
6. 性能基准测试：启用 --benchmark 模式，采集稳态推理延迟与吞吐量。
7. 精度验证：对比 fp16 与 fp32 输出差异，计算 MAE、RelErr、PSNR。
8. 验收确认：逐项检查 NPU 可用性、模型加载、图片生成、推理延迟、精度误差。
9. 多卡并行推理验证：在多 NPU 卡场景下分别运行推理并比对输出一致性。

## 详细执行步骤

### 1. 环境初始化与 NPU 状态确认

加载 CANN 环境并检查 NPU 设备状态：

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查 NPU 占用，选择空闲卡
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# PyPI 镜像（国内加速）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

### 2. 安装依赖

```bash
pip install torch torch_npu diffusers transformers accelerate
```

验证安装：

```bash
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}, NPU可用: {torch.npu.is_available()}')"
```

预期输出：torch: 2.9.0, NPU可用: True

### 3. 模型权重下载

从 HuggingFace 镜像下载：

```bash
python3 << 'PYEOF'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('wavymulder/wavyfusion', local_dir='./models/wavyfusion')
print('模型下载完成')
PYEOF
```

或使用 ModelScope：

```bash
python3 << 'PYEOF'
from modelscope import snapshot_download
snapshot_download('wavymulder/wavyfusion', local_dir='./models/wavyfusion')
print('模型下载完成')
PYEOF
```

### 4. NPU 设备验证

```bash
npu-smi info
python3 -c "
import torch
import torch_npu
print(f'NPU count: {torch.npu.device_count()}')
print(f'NPU 0: {torch.npu.get_device_name(0)}')
print(f'Memory: {torch.npu.get_device_properties(0).total_memory / 1024**3:.0f} GB')
"
```

### 5. 基础推理验证

```bash
cd skills/deployment/wavyfusion-npu
python3 scripts/inference.py \
  --model-path ../../../models/wavyfusion \
  --prompt "a beautiful landscape in wavy art style, surreal waves of color" \
  --steps 25 \
  --seed 42 \
  --output output.png
```

验证结果：
- 输出文件 output.png 为 512×512 RGB 图片
- 图片内容应与提示词相符

### 6. 性能基准测试

```bash
python3 scripts/inference.py \
  --model-path ../../../models/wavyfusion \
  --prompt "a cat in a wavy dreamlike style" \
  --steps 25 \
  --benchmark
```

预期性能（单卡 Ascend 910B4，warm cache，fp16）：

| 推理步数 | 平均延迟 | 吞吐量 |
|----------|---------|--------|
| 20       | ~1.45s  | 13.8 steps/s |
| 25       | ~1.80s  | 13.9 steps/s |
| 30       | ~2.20s  | 13.6 steps/s |

### 7. 精度验证（fp16 vs fp32 精度对比）

```bash
python3 << 'PYEOF'
import torch, torch_npu, numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_path = '../../../models/wavyfusion'
prompt = 'a cat in a wavy dreamlike style, psychedelic colors'

for dtype, tag in [(torch.float16, 'fp16'), (torch.float32, 'fp32')]:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('npu')
    pipe.set_progress_bar_config(disable=True)

    gen = torch.Generator(device='npu').manual_seed(42)
    with torch.no_grad():
        img = pipe(prompt=prompt, generator=gen,
                   num_inference_steps=20, guidance_scale=7.5,
                   height=512, width=512).images[0]
    torch.save(torch.from_numpy(np.array(img)).float(), f'precision_{tag}.pt')
    print(f'{tag} done')

t_fp16 = torch.load('precision_fp16.pt')
t_fp32 = torch.load('precision_fp32.pt')
mae = torch.mean(torch.abs(t_fp16 - t_fp32)).item()
rmse = torch.sqrt(torch.mean((t_fp16 - t_fp32) ** 2)).item()
rel_err = mae / (torch.mean(t_fp32).item() + 1e-8) * 100
psnr = 20 * np.log10(255.0 / (rmse + 1e-8))
print(f'MAE: {mae:.4f}, RelErr: {rel_err:.4f}%, PSNR: {psnr:.2f}dB')
print(f'精度验证: {"PASS" if rel_err < 1.0 else "FAIL"} (阈值 < 1%)')
PYEOF
```

### 8. 验收确认

| 检查项 | 预期结果 |
|--------|---------|
| NPU 可用 | torch.npu.is_available() = True |
| 模型加载 | pipe.to('npu') 无报错 |
| 图片生成 | 输出 512×512 RGB 图片 |
| 推理延迟 | < 3s @ 25steps |
| 精度误差 | < 1%（fp16 vs fp32 精度对比） |

### 9. 多卡并行推理验证

在多 NPU 卡场景下验证推理一致性：

```bash
# 在多卡环境下依次指定不同卡号运行推理
for card in 0 1 2 3; do
  echo "=== NPU card $card ==="
  ASCEND_RT_VISIBLE_DEVICES=$card python3 scripts/inference.py \
    --model-path ../../../models/wavyfusion \
    --prompt "a cat in a wavy dreamlike style" \
    --steps 25 --seed 42 \
    --output "output_card${card}.png"
done
```

验证结果：各卡输出的图片应保持视觉一致性，且推理延迟差异 < 10%

## 多卡推理验证

在多卡环境中验证 NPU 推理的可扩展性。WavyFusion SD1.5 模型支持在多个 Ascend910
NPU 卡上独立执行推理任务，各卡互不干扰。

单卡推理环境准备：

```bash
# 查看所有 NPU 卡状态
npu-smi info

# 指定单卡运行（示例：卡 0）
export ASCEND_RT_VISIBLE_DEVICES=0
python3 scripts/inference.py --model-path ../../../models/wavyfusion \
  --prompt "multi-card verification test" --steps 25 --seed 42 --output output.png
```

多卡场景注意事项：
- 各卡需独立设置 ASCEND_RT_VISIBLE_DEVICES 环境变量
- 各卡共享模型权重目录，无需重复下载
- 卡间推理结果应在视觉上保持一致
- 若卡间结果不一致，检查 HCCL 通信与卡间同步状态

## 实测验证汇总

| 验证维度 | 验证方法 | 预期结果 | 实测工具 |
|---------|---------|---------|---------|
| 环境就绪 | npu-smi info + torch.npu.is_available() | NPU 卡可见 | npu-smi / python3 |
| 单卡推理 | inference.py 单张图片生成 | 512×512 图片输出 | scripts/inference.py |
| 性能基准 | --benchmark 模式稳态性能 | 延迟 < 2s @ 25 steps | scripts/inference.py --benchmark |
| 精度对比 | fp16 vs fp32 对比 | RelErr < 1% | python3 精度验证脚本 |
| 多卡一致性 | 多卡分别推理并比对 | 输出内容一致 | 多卡推理脚本 |
| 稳定性 | 连续推理 10 次无崩溃 | 稳定运行 | 循环调用 inference.py |

## 执行检查点与用户确认

执行过程中需要用户确认以下检查点：

| 序号 | 检查点 | 确认操作 | 异常处理 |
|------|--------|---------|---------|
| 1 | CANN 环境加载 | 用户确认 set_env.sh 无报错，npu-smi 可输出设备信息 | 若 CANN 未安装，暂停并引导安装 |
| 2 | NPU 设备状态 | 用户确认选定空闲 NPU 卡号，确认显存充足（>= 8GB 可用） | 若显存不足，暂停并选择其他 NPU 卡 |
| 3 | 依赖安装完成 | 用户确认 torch_npu 导入成功，torch.npu.is_available() = True | 若导入失败，检查 CANN 版本与 Python 版本兼容性 |
| 4 | 模型权重下载 | 用户确认 ./models/wavyfusion 目录存在且包含 model_index.json | 若下载失败，切换至 ModelScope 镜像重试 |
| 5 | 基础推理结果 | 用户确认 output.png 已生成且内容合理 | 若推理失败，检查模型路径和 NPU 状态 |
| 6 | 性能基准结果 | 用户确认 benchmark 延迟和吞吐量在合理范围 | 若性能异常，尝试重启推理进程或切换 NPU 卡 |
| 7 | 精度对比结果 | 用户确认 RelErr < 1%，PSNR > 45 dB，结论为 PASS | 若精度超标，检查随机种子一致性，调高计算精度 |
| 8 | 多卡场景确认 | 若使用多卡，用户确认各卡推理结果一致 | 若卡间结果不一致，检查 ASCEND_RT_VISIBLE_DEVICES 配置 |

## 异常处理与回滚策略

| 异常场景 | 表现 | 处理方法 | 回滚策略 |
|---------|------|---------|---------|
| CANN 环境未安装 | set_env.sh 不存在 | 下载并安装对应版本 CANN 工具包 | 安装完成后重新加载环境变量 |
| NPU 驱动异常 | npu-smi 报错或无设备 | 重启 NPU 驱动：npu-smi set -t reset -d 0 | 若重启无效，检查服务器硬件状态 |
| Python 版本不兼容 | torch_npu 导入报错 | 切换至 Python 3.9-3.13 兼容版本 | 使用 conda 创建新 Python 环境 |
| pip 安装超时 | 依赖下载失败 | 设置国内镜像源：PIP_INDEX_URL | 重试 pip install 命令 |
| 模型下载失败 | snapshot_download 网络超时 | 切换 HF_ENDPOINT 至 hf-mirror.com | fallback 到 ModelScope 镜像源 |
| 模型路径错误 | inference.py 报 FileNotFoundError | 检查 --model-path 指向实际权重目录 | 重新下载模型至正确路径 |
| 推理时 OOM | CUDA out of memory / NPU OOM | 减少 batch_size 或使用 fp16 | 切换至空闲 NPU 卡重试 |
| NPU Graph 编译失败 | 首次推理报 Graph 编译错误 | 清空 ACL 缓存目录：rm -rf $HOME/.cache/torch_npu | retry 推理命令 |
| 性能异常偏低 | 推理延迟远超预期 | 检查 NPU 频率与功耗模式，确认无进程抢占 | 重启推理进程并 retry benchmark |
| fp16 vs fp32 精度异常 | RelErr > 1% | 检查随机种子 seed 是否一致，确认 no_grad 上下文 | 切换至 fp32 运行所有推理并重试精度对比 |
| torch_npu 版本不匹配 | torch_npu 与 torch 版本冲突 | 安装匹配版本的 torch_npu（参考官方兼容性矩阵） | 卸载冲突版本后重新安装 |
| ACL 缓存损坏 | 推理过程段错误 | 清理缓存并重新编译 Graph | recover NPU 驱动后重试 |
| 多卡 NCCL 通信异常 | 多卡推理 hang | 关闭多卡模式，回退至单卡推理 | 检查 HCCL 配置后 retry |

## 资源与评测产物

| 资源路径 | 类型 | 说明 |
|---------|------|------|
| skills/deployment/wavyfusion-npu/scripts/inference.py | 推理脚本 | 支持单图生成与 benchmark 模式 |
| skills/deployment/wavyfusion-npu/SKILL.md | 流程文档 | 本 Skill 的完整执行流程定义 |
| skills/deployment/wavyfusion-npu/test-prompts.json | 测试提示词 | 标准化测试 prompt 集合 |
| models/wavyfusion/ | 模型权重 | HuggingFace 或 ModelScope 下载的模型权重目录 |
| output.png | 推理产物 | 基础推理验证输出的 512×512 图片 |
| precision_fp16.pt / precision_fp32.pt | 精度数据 | fp16 和 fp32 推理结果张量 |
| results/benchmark.json | 评测结果 | benchmark 模式输出的延迟与吞吐量数据 |
| references/ | 参考文档 | 模型卡、NPU 部署指南、昇腾推理优化参考材料 |
| evals.json | 评测汇总 | 精度对比与性能评测的结构化 JSON 输出 |
| scripts/validate.sh | 验证脚本 | 环境检查和依赖验证的辅助脚本 |

## 注意事项

1. 首次冷启动：首次推理包含 ACL Graph 编译，约 80-90s，之后稳定在约 1.8s
2. GEGLU 算子：diffusers 0.38.0 在 torch_npu 环境下自动使用 npu_geglu 融合算子，NPU 推理无需额外配置
3. CPU 对比：若需 CPU 精度对比，需在无 torch_npu 的独立 Python 环境中运行
4. 模型权重：建议使用 HuggingFace 镜像下载（hf-mirror.com），避免直连超时
5. 显存：SD 1.5 在 fp16 下约需 4-6GB HBM，910B4 的 32GB 完全满足
6. 实测验证：建议在每个检查点处执行实测验证，确保环境符合预期
7. 精度对比基准：以 fp32 为精度基准，fp16 精度误差不超过 1%
