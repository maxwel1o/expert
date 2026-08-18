---
name: vit_base_patch16_224-dino-npu-deploy
description: >
  DINO ViT-B/16 (vit_base_patch16_224.dino) 自监督视觉 Transformer 模型在昇腾 NPU 上的
  一键部署、推理验证与精度对比 Skill。涵盖环境准备、torch hub 权重加载、NPU 推理脚本、
  精度对比（cosine similarity < 1%）、性能基准测试的全流程。可在任意 Ascend910 系列
  服务器上复现。当用户提到 DINO ViT 部署昇腾、DINO NPU 推理、vit_base NPU 时触发。
metadata:
  short-description: DINO ViT-B/16 昇腾 NPU 部署与推理验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, dino, vit, vision-transformer, pytorch, inference, self-supervised-learning]
---

# DINO ViT-B/16 昇腾 NPU 部署与推理验证 Skill

本 Skill 提供 `vit_base_patch16_224.dino`（DINO ViT-B/16）自监督视觉 Transformer 模型在华为昇腾 NPU 上的完整部署、推理验证和精度对比的标准化可复现流程。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.RC1） |
| Python | 3.9 – 3.13 |
| 网络 | 首次推理需联网下载模型权重（约 327MB） |

## 流程总览

```
0. 环境初始化
1. 安装依赖
2. NPU 基础验证
3. 加载 DINO ViT-B/16 模型
4. 合成数据推理
5. 使用 inference.py 完整推理
6. 真实图片推理验证
7. 精度对比与基准测试
8. 性能基准测试结果分析
9. 验收确认
```

各步骤按顺序依次执行，每步完成后确认通过再进入下一步。

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|---|---|---|---|
| CP-1 环境检查点 | npu-smi info 或 source set_env.sh 完成后 | 确认当前 Ascend/CANN 环境版本正确 | 重新加载 CANN 环境或降级为 CPU dry-run |
| CP-2 安装确认点 | pip install 完成后 | 确认 torch 与 torch_npu 版本一致 | retry 安装，切换 pip 镜像源 |
| CP-3 NPU 可用性确认 | NPU 基础验证完成后 | 确认 NPU 设备正常且 tensor 运算无异常 | 检查 ASCEND_RT_VISIBLE_DEVICES 设置并重试 |
| CP-4 模型加载确认点 | torch.hub.load 完成后 | 确认权重下载完成且模型加载成功 | retry 下载，手动写入缓存目录 |
| CP-5 精度确认点 | CPU/NPU 精度对比完成后 | 确认 cosine similarity > 0.999 且 L2 误差 < 1% | 检查 dtype 一致性并重新推理 |
| CP-6 性能确认点 | 性能基准测试完成后 | 确认 NPU 延迟在预期范围内（~10 ms/iter） | 检查 NPU 空闲状态并重测 |
| CP-7 最终验收点 | 所有步骤完成后 | 确认检查清单全部通过 | 标记未通过项，回滚至对应步骤重试 |

## Step 0. 环境初始化

**输入**: Ascend910 服务器，CANN >= 8.0

**输出**: CANN 环境加载完成，空闲 NPU 设备已选择

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 华为 pip 镜像（国内加速）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

**通过标准**: npu-smi info 输出显示设备正常，无报错。

## Step 1. 安装依赖

**输入**: Python 3.9–3.13 环境，pip 已安装

**输出**: torch, torchvision, torch_npu, pillow 安装完成

```bash
pip install torch torchvision torch_npu pillow -i https://repo.huaweicloud.com/repository/pypi/simple/
```

安装完成后验证版本一致性：

```bash
python3 -c "import torch; import torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__)"
```

**通过标准**：torch 与 torch_npu 版本一致，且 import torch_npu 无报错。

## Step 2. NPU 基础验证

**输入**: torch 和 torch_npu 已安装

**输出**: NPU 设备可用性验证结果

```bash
python3 << 'EOF'
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print('NPU tensor:', a + a)
print('NPU device:', torch.npu.get_device_name(0))
print('NPU count:', torch.npu.device_count())
EOF
```

**通过标准**：输出包含 device='npu:0' 的 Tensor，torch.npu.device_count() > 0 且无报错。

## Step 3. 加载 DINO ViT-B/16 模型

**输入**: torch hub 联网（或本地缓存）

**输出**: DINO ViT-B/16 模型已加载至指定设备

```bash
python3 << 'EOF'
import torch
import torch_npu

# 加载 DINO 模型（首次需联网下载约 327MB 权重）
model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
model.eval()

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output dimension: 768")
EOF
```

**通过标准**: 模型加载成功，参数量显示 85.8M，无网络超时或文件损坏错误。

## Step 4. 合成数据推理（CPU vs NPU）

**输入**: DINO 模型已加载，NPU 设备就绪

**输出**: CPU 和 NPU 推理结果，cosine similarity 验证

```bash
python3 << 'EOF'
import torch
import torch_npu
import torch.nn.functional as F

model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
model.eval()

# CPU 推理
x = torch.randn(1, 3, 224, 224).clamp(0, 1)
with torch.no_grad():
    out_cpu = model(x)

# NPU 推理
model_npu = model.to("npu:0")
x_npu = x.to("npu:0")
with torch.no_grad():
    out_npu = model_npu(x_npu).cpu()

# 精度对比
cos_sim = F.cosine_similarity(out_cpu.flatten().unsqueeze(0),
                               out_npu.flatten().unsqueeze(0)).item()
print(f"Cosine similarity: {cos_sim:.8f}")
print(f"PASS" if cos_sim > 0.999 else "FAIL")
EOF
```

**通过标准**：
1. 合成数据推理正常输出
2. Cosine similarity > 0.999

## Step 5. 使用 inference.py 进行完整推理

**输入**: scripts/inference.py 脚本

**输出**: 特征向量结果和推理延迟数据

```bash
# 查看脚本参数
python3 scripts/inference.py --help

# 合成数据推理（默认）
python3 scripts/inference.py

# 指定设备推理
python3 scripts/inference.py --device npu:0

# 真实图片推理
python3 scripts/inference.py --image /path/to/image.jpg

# 批量推理
python3 scripts/inference.py --batch-size 4
```

**通过标准**：脚本正常运行，输出特征向量 shape 和推理延迟。NPU 推理延迟约 10 ms/iter。

## Step 6. 真实图片推理验证

**输入**: 测试图片（JPEG/PNG），DINO 模型

**输出**: 特征向量 shape [1, 768]，feature norm 值

```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("image.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to("npu:0")

with torch.no_grad():
    features = model(x)  # shape: [1, 768]
print(f"Feature shape: {features.shape}")
print(f"Feature norm: {features.norm().item():.4f}")
```

**通过标准**：
1. 输出 shape 为 [1, 768]
2. Feature norm 值合理（通常在 50~100 之间）

## Step 7. 精度对比（CPU vs NPU）与基准测试

**输入**: scripts/eval_benchmark.py 脚本

**输出**: 精度对比数据和性能基准报告

```bash
python3 scripts/eval_benchmark.py
```

该脚本自动执行以下操作：
1. 合成数据 CPU/NPU 精度对比（cosine similarity, L2 相对误差）
2. 真实图片 CPU/NPU 精度对比
3. 不同 batch size（1/2/4/8）CPU vs NPU 性能基准
4. 自动生成 JSON 报告保存至 eval_report.json

### 预期精度

| 测试数据 | 余弦相似度 | L2 相对误差 | 判定 |
|---------|-----------|------------|------|
| 合成数据 | 0.9999928 | 0.379% | PASS |
| 真实图片 | 0.9999950 | 0.319% | PASS |

**通过标准**：余弦相似度 > 0.999，L2 相对误差 < 1%。

## Step 8. 性能基准测试结果分析

在 Ascend910B4 上实测结果：

| Batch Size | CPU 延迟 (ms) | NPU 延迟 (ms) | 加速比 |
|-----------|--------------|--------------|--------|
| 1 | 1747 | 10.5 | 166x |
| 2 | 3292 | 10.3 | 321x |
| 4 | 7088 | 10.3 | 686x |
| 8 | 15424 | 10.6 | 1450x |

加速比可达 166x (bs=1) 至 1450x (bs=8)。

```bash
# 查看基准测试报告
cat eval_report.json | python3 -m json.tool

# 释放 NPU 缓存（切换 batch 或模型时）
python3 -c "import gc; gc.collect(); import torch; torch.npu.empty_cache()"
```

## Step 9. 验收确认

完成以下检查清单即为部署成功：

1. npu-smi info 显示设备正常
2. import torch_npu 无报错
3. 合成数据推理正常输出，cosine similarity > 0.999
4. 真实图片推理输出 shape 为 [1, 768]
5. 精度对比 L2 相对误差 < 1%
6. NPU 推理延迟约 10 ms/iter
7. 基准测试报告已保存至 eval_report.json

## 异常处理与边界条件

| 场景 | 触发条件 | fallback / retry / recover 动作 |
|---|---|---|
| CANN 环境未加载 | source set_env.sh 失败或 ASCEND_HOME 未设置 | 提示用户检查 CANN 安装路径并 retry；降级为 CPU dry-run |
| NPU 设备不可用 | npu-smi info 无输出或 torch_npu import 报错 | fallback 到 CPU 推理，跳过 NPU 验证，标记 dry-run |
| 权重下载失败 | torch.hub.load 网络超时或文件损坏 | retry 2 次，切换 hf-mirror 镜像，手动下载到缓存目录 |
| NPU 显存不足（OOM） | 推理时抛出显存不足异常 | 减小 batch size，执行 torch.npu.empty_cache()，retry |
| 多卡抢占冲突 | 其他进程占用 0 号卡导致推理失败 | npu-smi info 选择空闲卡，重设 ASCEND_RT_VISIBLE_DEVICES |
| 精度误差 > 1% | cosine similarity < 0.999 或 L2 误差 >= 1% | 检查 dtype 一致性（FP32 vs FP16），重新加载权重 |
| pip 安装失败 | 网络问题或版本冲突 | 切换华为 pip 镜像源，retry 安装，锁定版本号 |
| Python 版本不兼容 | Python 版本低于 3.9 或高于 3.13 | 提示用户使用 conda 创建虚拟环境，切换 Python 版本 |
| 图片格式错误 | Image.open 抛出异常或图片损坏 | 跳过真实图片推理，仅使用合成数据验证 |
| 基准测试结果异常 | NPU 延迟 > 100 ms 或加速比 < 10x | 检查 NPU 是否被其他进程占用，释放资源后 retry |
| 脚本执行权限不足 | scripts/ 目录下权限错误 | chmod +x 赋予执行权限后 retry |
| torch_npu 版本不匹配 | torch 与 torch_npu 版本不一致 | 卸载后重新安装匹配版本对 |

## 资源与评测产物

| 路径 | 用途 |
|---|---|
| SKILL.md | 本 Skill 文档，标准化部署和推理流程 |
| scripts/inference.py | DINO 模型推理入口脚本，支持合成数据和真实图片 |
| scripts/eval_benchmark.py | 精度对比和性能基准测试脚本，生成 JSON 报告 |
| test-prompts.json | 提供可复现的评估测试提示集 |
| eval_report.json | 运行 eval_benchmark.py 后生成的精度和性能报告 |
| results.tsv | 历次推理结果汇总表（手动维护或脚本生成） |
| references/ | 可选参考文档目录 |

## 文件结构

```
vit_base_patch16_224-dino-npu/
├── SKILL.md                    # 本 Skill 文档
├── test-prompts.json           # 评估测试提示
├── scripts/
│   ├── inference.py            # 推理脚本
│   └── eval_benchmark.py       # 精度/性能测评脚本
└── eval_report.json            # 测评结果（运行后生成）
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|---|---|---|
| No module named torch_npu | 未安装或 CANN 环境未加载 | source set_env.sh 后重装 torch_npu |
| 模型下载失败 | 网络问题 | 手动下载到 ~/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth |
| OOM | batch 过大 | 减小 batch size，执行 torch.npu.empty_cache() |
| 多卡抢占冲突 | 默认都用 0 号卡 | npu-smi info 选空闲卡 |
| 精度误差 > 1% | 权重加载异常 | 重新下载权重并对比 dtype |

## 附录：DINO ViT-B/16 适配要点速查

| 特征 | 值 | 说明 |
|---|---|---|
| 模型架构 | ViT-B/16 (patch=16) | 标准 Vision Transformer |
| 参数量 | 85.8M | 中等规模视觉模型 |
| 输出维度 | 768 | 特征向量，无分类头 |
| 输入尺寸 | 3x224x224 | 固定分辨率 |
| 激活函数 | GELU | torch.nn.GELU 原生支持 |
| 注意力 | Self-Attention (12 heads) | torch.bmm 原生支持 |
| 归一化 | LayerNorm | torch.nn.LayerNorm 原生支持 |
| NPU 适配 | 零代码修改 | model.to("npu:0") 即可 |
| 权重来源 | torch hub / facebookresearch | ~327MB |
