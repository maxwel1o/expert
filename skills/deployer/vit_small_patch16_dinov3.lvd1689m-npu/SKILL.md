---
name: vit_small_patch16_dinov3.lvd1689m-npu
description: "DINOv3 ViT-Small (vit_small_patch16_dinov3.lvd1689m) 图像特征提取模型在昇腾 Ascend910B4 NPU 上的推理适配 Skill。覆盖环境准备、依赖安装、模型加载、NPU 推理验证、CPU/NPU 精度对比（cos > 0.999）、性能基准测试（batch size sweep）全流程。当用户提到 vit_small_patch16_dinov3、DINOv3 NPU、ViT-Small 昇腾、timm 模型 NPU 推理时触发。"
---

# vit_small_patch16_dinov3.lvd1689m — Ascend NPU 推理适配 Skill

> 在昇腾 NPU 和 CPU 上自动部署 DINOv3 ViT-Small 图像特征提取模型，完成推理、精度验证和性能基准测试。

## 概述

本 Skill 提供 `vit_small_patch16_dinov3.lvd1689m`（DINOv3 ViT-Small）图像特征提取模型在华为昇腾 Ascend910B4 NPU 上的完整部署、推理验证、CPU/NPU 精度对比（cosine similarity > 0.999）和性能基准测试的标准化可复现流程。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910B4) |
| 模型 | `vit_small_patch16_dinov3.lvd1689m`（DINOv3 ViT-Small） |
| 框架版本 | PyTorch 2.9.0+, torch_npu 2.9.0+, timm 1.0.27+ |
| 精度目标 | CPU 与 NPU 推理余弦相似度 > 0.999，Normalized MAE < 1% |
| CANN 版本 | 8.5.1+ |
| 模型权重 | ~86 MB（首次运行需联网下载） |

## 前置条件

| 项目 | 要求 |
|:---|:---|
| 硬件 | Ascend910 系列（>= 1 卡） |
| CANN | >= 8.5.1 |
| Python | >= 3.10 |
| PyTorch | >= 2.9.0 |
| torch_npu | >= 2.9.0.post1 |
| timm | >= 1.0.27 |
| 网络 | 首次运行需联网下载模型权重（~86 MB） |

## Skill 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 是 | vit_small_patch16_dinov3.lvd1689m | timm 模型名称 |
| `device` | string | 否 | npu | 推理设备: cpu 或 npu |
| `image_path` | string | 否 | 自动下载样例 | 测试图片路径 |
| `random_seed` | int | 否 | 42 | 随机种子（精度测试使用） |
| `batch_sizes` | string | 否 | 1,2,4,8 | 性能测试 batch size 列表 |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.10+ 环境，昇腾 NPU 驱动，CANN 8.5.1+。

**动作**:

1. 检查 Python 版本并确认 CUDA/CANN 环境：

```bash
python3 --version
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 检测 NPU 状态与可用设备：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 选择空闲 NPU 卡：

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0  # 替换为实际空闲卡号
```

> **注意**: `ASCEND_RT_VISIBLE_DEVICES` 必须设为 `0`（NPU 本地索引），而非 npu-smi 显示的物理设备号（如 `7`）。可用 `torch.npu.device_count()` 确认。

**输出**: NPU 可用状态，已设置 ASCEND_RT_VISIBLE_DEVICES。

### Step 2: 安装依赖

**输入**: Python 环境已激活。

**动作**:

4. 安装所需 Python 依赖：

```bash
pip install timm safetensors pillow numpy
```

5. 验证 torch_npu 可导入且版本符合要求：

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU count:', torch.npu.device_count())
print('NPU name:', torch.npu.get_device_name(0))
a = torch.randn(3, 4).npu()
print(a + a)
"
```

**通过标准**: 输出包含 `device='npu:0'` 的 Tensor 且无报错。

**输出**: 依赖安装完成，torch_npu 环境正常。

### Step 3: 模型加载

**输入**: 模型名称 `vit_small_patch16_dinov3.lvd1689m`，设备类型。

**动作**:

6. 使用 timm 创建模型并加载预训练权重：

```python
import timm
import torch_npu

model = timm.create_model(
    "vit_small_patch16_dinov3.lvd1689m",
    pretrained=True,
)
model = model.to("npu:0").eval()
for p in model.parameters():
    p.requires_grad = False
```

7. 确认模型参数数量并记录：

```bash
python3 -c "
import timm
m = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False)
print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')
"
```

8. 如果本地存在缓存的 checkpoint 文件（`vit_small_patch16_dinov3_lvd1689m.pth`），优先加载本地权重以避免网络下载。

**输出**: 模型已加载到 NPU，参数约 21.7M。

### Step 4: CPU 基线推理

**输入**: 模型加载到 CPU，随机输入或图片输入。

**动作**:

9. 创建随机输入 Tensor（`torch.randn(1, 3, 256, 256)`）或加载图片进行 DINOv3 标准预处理（resize 256x256 bicubic + ImageNet 归一化）。

10. 执行 CPU 推理作为精度基线：

```bash
python3 scripts/inference.py --random
```

或指定图片：

```bash
python3 scripts/inference.py --image /path/to/image.jpg
```

11. 记录 CPU 推理的 latency 和输出特征向量（shape `(1, 384)`）。

**输出**: CPU 推理输出特征向量，~384 维。

### Step 5: NPU 推理

**输入**: 模型加载到 NPU，与 CPU 相同的输入。

**动作**:

12. 将输入 Tensor 移至 NPU：

```python
x_npu = x.to("npu:0")
```

13. 执行 NPU 推理并记录 latency：

```python
torch.npu.synchronize(DEVICE)
t0 = time.perf_counter()
out = model(x_npu)
torch.npu.synchronize(DEVICE)
t_ms = (time.perf_counter() - t0) * 1000
```

14. 确认输出 shape 为 `(1, 384)`，打印特征统计信息（mean, std, min, max）。

**输出**: NPU 推理输出特征向量，推理延迟 ~20-30 ms。

### Step 6: CPU/NPU 精度对比

**输入**: CPU 和 NPU 推理输出特征向量。

**动作**:

15. 计算余弦相似度 (Cosine Similarity) 和归一化平均绝对误差 (Normalized MAE)：

```python
out_np = out.float().cpu()
cos = torch.nn.functional.cosine_similarity(out_np, out_cpu, dim=1).item()
mae = (out_np - out_cpu).abs().mean().item()
nmae = mae / (out_cpu.abs().mean().item() + 1e-12)
```

16. 执行多种子精度验证（5 seeds: 42, 123, 256, 512, 1024）：

```bash
python3 scripts/evaluate.py
```

17. 精度通过标准：
    - Cosine similarity > 0.999
    - Normalized MAE < 1%
    - Max absolute error < 0.01

18. 如果精度不达标，记录失败原因并停止后续流程。

**输出**: 精度对比报告（余弦相似度、MAE、NMAE、最大绝对误差、MSE）。

### Step 7: 性能基准测试

**输入**: 模型加载到 NPU，多 batch size 配置。

**动作**:

19. 执行 warmup 推理（10 次），消除初始延迟影响：

```python
for _ in range(10):
    _ = model(torch.randn(1, 3, 256, 256).npu())
torch.npu.synchronize(DEVICE)
```

20. 执行 batch size sweep 性能测试（bs=1, 2, 4, 8），每个配置重复 50 次取平均：

```bash
python3 scripts/evaluate.py
```

21. 记录每个 batch size 的 latency (ms) 和 throughput (samples/s)。

预期结果：

| Batch Size | Latency (ms) | Throughput (samples/s) |
|:---|:---:|:---:|
| 1 | ~27.65 | ~36.17 |
| 2 | ~26.06 | ~76.74 |
| 4 | ~21.49 | ~186.10 |
| 8 | ~25.78 | ~310.36 |

**输出**: 性能基准测试报告。

### Step 8: 验收确认与报告生成

**输入**: 所有步骤完成后的结果。

**动作**:

22. 确认评测报告中 `evaluation_report.json` 的 accuracy.passed 为 true。
23. 生成最终验收总结：
    - [x] `npu-smi info` 显示设备正常
    - [x] `import torch_npu` 无报错
    - [x] NPU 推理输出 shape `(1, 384)` 的特征向量
    - [x] CPU/NPU 余弦相似度 > 0.999
    - [x] Normalized MAE < 1%
    - [x] 性能基准测试结果已记录

**输出**: 验收确认报告，`evaluation_report.json`。

## 执行检查点与用户确认

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|:---|:---|:---|:---|:---|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，CANN 版本是否正确 | 提示加载 CANN 环境脚本或安装 torch_npu，retry 一次 |
| 2 | CP-2: 模型加载检查点 | 模型创建与权重加载后 | 模型名称 `vit_small_patch16_dinov3.lvd1689m` 是否正确，参数数量 ~21.7M | 检查网络连接，切换 HF_ENDPOINT 或 ModelScope 镜像 |
| 3 | CP-3: CPU 基线检查点 | CPU 推理完成后 | CPU 推理输出 shape 是否为 `(1, 384)` 且数值合理 | 检查模型加载方式是否一致 |
| 4 | CP-4: NPU 推理检查点 | NPU 推理完成后 | NPU 推理是否成功，latency 是否正常（~20-30 ms） | 检查 NPU 显存占用，释放资源后重试 |
| 5 | CP-5: 精度确认检查点 | CPU/NPU 精度对比完成后 | 余弦相似度 > 0.999，Normalized MAE < 1% | 记录精度偏差明细，禁止发布通过结论，建议检查推理脚本一致性 |
| 6 | CP-6: 性能确认检查点 | 性能基准测试后 | batch size sweep 结果是否合理 | 调整测试参数后重试 |
| 7 | CP-7: 最终验收 | 评测报告生成后 | 确认 `evaluation_report.json` 中所有检查项通过 | 逐一排查未通过项 |

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|:---|:---|:---|:---|
| NPU 不可用 | `torch.npu.is_available()` 为 False | fallback 到 CPU 推理，标记 dry-run，禁止写入 NPU 通过结论 | 环境检查日志标记 `NPU_AVAILABLE=false` |
| CANN 环境缺失 | `ASCEND_HOME`、`LD_LIBRARY_PATH` 或 `torch_npu` 缺失 | 提示加载 CANN 环境脚本 (`source set_env.sh`)，retry 一次环境检查 | 环境检查日志 |
| 设备号错误 | `SetDevice failed, device id error` | 用 `torch.npu.device_count()` 确认可用设备 ID，建议设为 0 | 日志记录可用设备列表 |
| 模型下载失败 | 无法访问 HuggingFace 或 ModelScope | retry 2 次，设置 `HF_ENDPOINT=https://hf-mirror.com`，切换 ModelScope 镜像 | 日志记录权重来源 |
| 权重加载异常 | `timm.create_model` 报错 | 打印错误堆栈，检查是否使用 `pretrained=True`，确认本地 checkpoint 存在 | 错误日志 |
| NPU 显存 OOM | 推理抛出 OOM 异常 | 执行 `gc.collect()` 和 `torch.npu.empty_cache()` 释放资源后 retry | `evals.json` 记录 retry 次数 |
| 精度不达标 | Cosine similarity <= 0.999 或 NMAE >= 1% | 记录精度偏差明细，标记 failed，不生成通过结论 | 精度表显示失败原因和偏差值 |
| 性能异常 | Latency 远超预期或 throughput 过低 | 检查 NPU 负载和显存占用，释放其他进程后重试 | 性能对比日志 |
| 本地 checkpoint 损坏 | `load_state_dict` 报错 | fallback 到在线预训练权重 | 加载方式日志 |

## 资源与评测产物

| 路径 | 用途 |
|:---|:---|
| `scripts/inference.py` | CPU/NPU 推理入口脚本（支持随机输入/图片输入，含精度对比） |
| `scripts/evaluate.py` | 综合评测脚本（5 seeds 精度测试 + batch size sweep 性能基准） |
| `SKILL.md` | 本 Skill 文档 |
| `test-prompts.json` | 结构化的测试提示词（用于重复评估本 Skill） |
| `evaluation_report.json` | 评测报告（运行后生成，含精度和性能数据） |
| `evaluation.log` | 运行日志（运行后生成） |

## 使用约束

1. **串行执行**：各步骤需串行执行，上一步完成后再进入下一步，防止 NPU 显存竞争。
2. **权重下载**：优先使用 HuggingFace 下载，失败时使用 `HF_ENDPOINT=https://hf-mirror.com` 或 ModelScope 镜像。
3. **精度标准**：Cosine similarity > 0.999 且 Normalized MAE < 1% 方为通过。精度不通过时禁止写入 NPU 通过结论。
4. **设备选择**：`ASCEND_RT_VISIBLE_DEVICES` 应设置为 NPU 本地索引 `0`，而非物理设备号。
5. **输入尺寸**：DINOv3 标准预处理需 resize 到 256x256 (bicubic)，ImageNet 归一化。
6. **NPU 显存管理**：每次推理后释放资源（`gc.collect()` + `torch.npu.empty_cache()`），防止显存泄漏。
