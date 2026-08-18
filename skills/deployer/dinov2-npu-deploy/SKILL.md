---
name: dinov2-npu-deploy
description: >
  DINOv2 (vit_base_patch14_dinov2.lvd142m) 视觉 Transformer 模型在昇腾 Ascend910 NPU 上的
  完整部署与推理验证 Skill。涵盖环境准备、模型权重获取、NPU 迁移推理、精度对比
  （CPU vs NPU fp32/fp16）、性能基准测试的全流程标准化可复现操作。
  可在任意 Ascend910/910B 系列服务器上一键复现。
  当用户提到 DINOv2 部署昇腾、ViT-B/14 NPU 推理、DINOv2 Ascend、dinov2-npu、
  Facebook DINOv2 昇腾、视觉模型 NPU 迁移时触发。
metadata:
  short-description: DINOv2 ViT-B/14 昇腾 NPU 完整部署与推理验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, dinov2, vit, vision-transformer, pytorch, inference, canntoolkit]
---

# DINOv2 昇腾 NPU 部署与推理验证 Skill

本 Skill 提供 DINOv2 (vit_base_patch14_dinov2.lvd142m) 视觉 Transformer 模型在华为昇腾 Ascend910/910B NPU 上的完整部署、推理验证和精度性能评测的标准化可复现流程。

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型架构 | ViT-B/14 (12层, 768 dim, 12头) |
| 参数量 | 86.6M |
| 输入尺寸 | 518×518 (center crop → 224×224) |
| 输出 | CLS embedding [B, 768] |
| 权重来源 | Facebook DINOv2 / HuggingFace / ModelScope |
| 推理精度 | fp32 / fp16 均支持 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910/Ascend910B 系列（至少 1 卡） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.RC1） |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（~330MB） |

## 文件结构

```
dinov2-npu/
├── SKILL.md                        # 本 Skill 文档
├── test-prompts.json               # 结构化测试提示
├── scripts/
│   ├── dinov2_infer.py             # 单张/批量图片推理入口
│   ├── dinov2_npu_adapt.py         # CPU vs NPU 精度对比适配脚本
│   └── eval_dinov2.py              # 精度与性能综合评测
└── references/                     # 预留参考资料目录
```

## 工作流程

### 执行总览

1. **环境初始化与 NPU 检查** — 加载 CANN 环境、选择空闲 NPU 卡、配置国内镜像源。
2. **安装依赖** — 安装 torch_npu、transformers、Pillow 等依赖库。
3. **NPU 基础验证** — 验证 torch_npu 可用、NPU 设备正常通信。
4. **模型权重获取** — 从 ModelScope 或 HuggingFace 下载权重。
5. **基础推理验证** — 使用单张/批量图片验证 NPU 推理正常。
6. **精度对比验证** — CPU (fp32) vs NPU (fp32/fp16) 全精度余弦相似度与绝对误差对比。
7. **性能基准测试** — 多 batch_size、双精度下延迟和吞吐量基准测试。
8. **验收确认** — 逐项检查清单确认部署成功。

### 1. 环境初始化与 NPU 检查

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查 NPU 设备状态，选空闲卡
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 华为 pip 镜像（国内加速）
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

**输入**：NPU 服务器 SSH 登录凭证
**输出**：NPU 环境就绪，`npu-smi info` 显示设备状态正常
**通过标准**：`npu-smi info` 输出中显示 Ascend910 设备且 Temperature/Pressure 正常

### 2. 安装依赖

```bash
# 安装 torch_npu（版本应与 CANN 匹配）
pip install torch_npu -i https://repo.huaweicloud.com/repository/pypi/simple/

# 安装其他依赖
pip install transformers torchvision Pillow -i https://repo.huaweicloud.com/repository/pypi/simple/
```

**输入**：Python 3.9-3.13 + pip
**输出**：依赖安装完成，`torch` 与 `torch_npu` 版本一致
**通过标准**：`pip list | grep -E "torch|transformers|torch_npu"` 显示各组件版本

### 3. NPU 基础验证

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
"
```

**输入**：有效的 torch_npu 安装
**输出**：NPU tensor 运算正常，显示设备名称
**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错

### 4. 模型权重获取

**方式 A：从 ModelScope 下载（推荐，国内加速）**

```bash
pip install modelscope
python3 -c "
from modelscope import snapshot_download
snapshot_download('facebook/dinov2-base', cache_dir='./models/dinov2')
"
```

**方式 B：从 HuggingFace 下载**

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-base', cache_dir='./models/dinov2')
"
```

**输入**：网络可达的服务器
**输出**：模型权重目录，包含 `config.json`、`model.safetensors`（或 `pytorch_model.bin`）、`preprocessor_config.json`
**通过标准**：`ls ./models/dinov2/facebook/dinov2-base/ | grep -E "config|model|preprocessor"` 至少包含 config.json 和 model.safetensors

### 5. 基础推理验证

使用 `scripts/dinov2_infer.py` 进行单张/批量图片推理。此脚本封装了模型加载、预处理和推理全流程。

**单张图片推理**：

```bash
python3 scripts/dinov2_infer.py --image /path/to/test_image.jpg --dtype fp32
```

**批量推理**：

```bash
python3 scripts/dinov2_infer.py --image-dir /path/to/images/ --dtype fp32
```

**NPU 指定**：

```bash
# 自动使用 NPU（如有）
python3 scripts/dinov2_infer.py --image test_image.jpg

# 显式指定模型路径
export DINOV2_MODEL_PATH=./models/dinov2/facebook/dinov2-base
python3 scripts/dinov2_infer.py --image test_image.jpg
```

**输入**：JPEG/PNG/BMP 图片文件或目录
**输出**：CLS token embedding，shape 为 (768,) 或 (N, 768)
**通过标准**：
- 输出 embedding shape 为 `(768,)`（单张）或 `(N, 768)`（批量）
- 无 NPU 相关报错
- fp32 与 fp16 均可正常推理

**核心代码逻辑**：

```python
import torch
import torch_npu
from transformers import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model.eval()
model.to('npu:0')
processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

inputs = processor(images=image, return_tensors='pt')
pixel_values = inputs['pixel_values'].to('npu:0')

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
embedding = outputs.last_hidden_state[:, 0]  # CLS token [B, 768]
```

### 6. 精度对比验证

使用 `scripts/dinov2_npu_adapt.py` 完成 CPU (fp32) vs NPU (fp32/fp16) 的精度对比。

```bash
# fp32 精度验证
python3 scripts/dinov2_npu_adapt.py --dtype fp32 --num-images 10

# fp16 半精度验证
python3 scripts/dinov2_npu_adapt.py --dtype fp16 --num-images 10
```

**预期精度（Ascend910B4 实测）**：

| 指标 | NPU fp32 | NPU fp16 |
|------|----------|----------|
| Cosine Similarity (mean) | 0.999974 | 0.999970 |
| Cosine Similarity (min) | 0.999970 | 0.999965 |
| Mean Absolute Diff | 0.0099 | 0.0108 |
| Max Absolute Diff | 0.0494 | 0.0549 |

**输入**：已加载的模型权重
**输出**：逐样本精度对比报告 + 批量汇总
**通过标准**：余弦相似度 > 0.999 且 mean_abs_diff < 0.015 (fp32) / < 0.02 (fp16)

### 7. 性能基准测试

使用 `scripts/eval_dinov2.py` 完成完整的精度与性能评测。

```bash
python3 scripts/eval_dinov2.py
```

**预期性能（Ascend910B4 实测）**：

| dtype | batch_size | 延迟 (ms) | 吞吐 (samples/s) |
|-------|-----------|-----------|------------------|
| fp32 | 1 | 14.37 | 69.6 |
| fp32 | 4 | 15.79 | 253.3 |
| fp32 | 8 | 15.89 | 503.4 |
| fp16 | 1 | 18.11 | 55.2 |
| fp16 | 4 | 15.41 | 259.6 |
| fp16 | 8 | 15.82 | 505.5 |

**输入**：已完成精度验证的模型
**输出**：eval_report.json（含精度和性能汇总）
**通过标准**：多 batch_size 下延迟和吞吐指标与预期值偏差不超过 20%

### 8. 验收确认

完成以下检查清单即为部署成功：

- [ ] `npu-smi info` 显示设备正常，温度、功耗指标在正常范围
- [ ] `import torch_npu` 无报错，版本与 CANN 匹配
- [ ] 模型权重下载成功（config.json + model.safetensors 存在）
- [ ] `dinov2_infer.py` 输出合理 embedding（shape 正确、数值非 NaN）
- [ ] 精度对比余弦相似度 > 0.999，mean_abs_diff 满足阈值
- [ ] 性能基准测试完成，结果写入 eval_report.json
- [ ] eval_report.json 包含精度和性能两大部分，格式有效

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|--------|----------|-------------|----------------|
| CP-1 环境检查 | 执行 npu-smi info 和 CANN 加载后 | 确认当前 NPU 卡号和 CANN 版本可用 | 若 NPU 不可用，降级为 CPU dry-run 模式，标注 npu_verified=false |
| CP-2 依赖检查 | torch_npu 和 transformers 安装后 | 确认版本兼容性 | 版本不匹配时输出推荐版本号，引导用户手动安装 |
| CP-3 NPU 可用性检查 | NPU tensor 测试后 | 确认 NPU 设备通信正常 | 若 NPU 测试失败，回退检查 CANN 环境和 torch_npu 安装 |
| CP-4 权重下载检查 | 模型权重下载前 | 确认模型名称和缓存目录 | 下载失败自动 retry 2 次，随后切换镜像源或缓存 |
| CP-5 推理验证检查 | dinov2_infer.py 运行后 | 确认 embedding 输出 shape 和数值合理 | 输出异常时检查模型加载和预处理参数 |
| CP-6 精度验证检查 | 精度对比报告生成后 | 确认余弦相似度和 abs_diff 在阈值内 | 精度不达标时保留 logits，标记 FAIL，不写入通过结论 |
| CP-7 性能基准检查 | eval_dinov2.py 运行后 | 确认延迟和吞吐在预期范围内 | 性能异常时检查 NPU 频率、功耗、卡间干扰 |
| CP-8 最终验收 | 所有检查点完成后 | 确认整体部署验收通过 | 任意检查点 FAIL 则整体标记 FAIL，提供排查指引 |

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|------|---------|----------------------------------|----------|
| NPU 设备不可用 | `npu-smi info` 失败或无 Ascend 设备 | fallback 到 CPU dry-run 模式，不在结论中写入 NPU 通过信息 | eval_report.json 标注 `npu_verified=false` |
| CANN 环境未加载 | `source set_env.sh` 报错或 `ASCEND_HOME` 未设置 | 提示用户检查 CANN 安装路径，输出常见安装目录列表 | 环境检查日志 |
| torch_npu 导入失败 | `import torch_npu` ModuleNotFoundError | 输出 torch+CANN 版本对应表，引导用户安装正确版本 | pip 安装日志 |
| ModelScope 下载超时 | 网络连接超时或 SSL 错误 | retry 2 次，然后切换 HuggingFace 镜像（hf-mirror.com） | `cache_dir` 中已有文件列表 |
| HuggingFace 不可用 | HuggingFace 连接被墙 | 切换 ModelScope 或 hf-mirror.com 镜像 | 下载日志中的来源标记 |
| 图片文件不存在 | --image 路径无效 | 检查路径存在性，提示使用绝对路径或生成测试图片 | error 信息 + 退出码 1 |
| 模型权重加载失败 | config.json 不完整或 model.safetensors 损坏 | retry 下载 1 次，仍失败则提示手动检查权重目录完整性 | 加载异常堆栈 |
| NPU OOM | batch_size 过大导致显存不足 | 自动减小 batch_size 至 1，释放显存（gc.collect + empty_cache） | eval_report.json 记录 `oom_retry` 次数 |
| NPU 多卡抢占 | 多进程同时使用 ASCEND_RT_VISIBLE_DEVICES=0 | 提示用户用 npu-smi info 选空闲卡更新环境变量 | 错误日志含占用进程 PID |
| 精度不达标 | cosine_sim < 0.999 或 mean_abs_diff 超阈值 | 保留 CPU/NPU logits 到文件，标记 failed，不生成通过结论 | 精度表显示失败原因 |
| 性能偏离预期 | 延迟/吞吐与参考值偏差 > 20% | 检查 NPU 频率设置、散热、卡间干扰，输出诊断建议 | 性能对比表标记 `outlier` |
| eval_report.json 写入失败 | 磁盘空间不足或 OUTPUT_DIR 权限错误 | 回退写入 /tmp/，提示用户检查输出目录 | 日志中的最终路径 |
| 脚本 import 循环依赖 | dinov2_npu_adapt.py 相对路径 import 失败 | 检查 sys.path 设置，提示在脚本目录下运行 | import 异常堆栈 |

## 资源与评测产物总览

| 路径 | 用途 |
|------|------|
| `scripts/dinov2_infer.py` | 单张/批量图片 NPU/CPU 推理入口，输出 CLS embedding |
| `scripts/dinov2_npu_adapt.py` | CPU vs NPU 精度对比适配主脚本，含逐样本/批量精度报告 |
| `scripts/eval_dinov2.py` | 综合精度+性能评测，输出 eval_report.json |
| `test-prompts.json` | 本 Skill 的结构化测试提示集，用于重复评估 |
| `eval_report.json` | 精度与性能汇总报告（日志输出目录） |
| `eval_output.log` | 评测过程日志（日志输出目录） |

## 附录：DINOv2 NPU 适配要点速查

| 特征 | DINOv2 值 | 说明 |
|------|-----------|------|
| 模型架构 | ViT-B/14 (12层, 768dim, 12头) | 标准 Vision Transformer |
| 参数量 | 86.6M | 轻量级视觉模型 |
| 输入尺寸 | 518×518 (center crop → 224×224) | 预处理由 BitImageProcessor 完成 |
| 输出 | CLS embedding [B, 768] | backbone 模型，无分类头 |
| 注意力类型 | 双向 (Bidirectional) | 非因果注意力 |
| 激活函数 | GELU | PyTorch 原生 F.gelu |
| LayerNorm | torch.nn.LayerNorm | 无需替换 |
| 权重格式 | safetensors / bin | 两者等价，safetensors 优先 |
| 推理精度 | fp32 / fp16 | 均通过验证 |
| NPU 适配度 | 原生支持 | 标准 PyTorch 算子，无需额外编译 |
