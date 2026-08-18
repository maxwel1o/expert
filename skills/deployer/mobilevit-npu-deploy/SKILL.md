---
name: mobilevit-npu-deploy
description: >
  Apple MobileViT (small/x-small/xx-small) 轻量级视觉 Transformer
  模型在昇腾 NPU 上的完整部署与验证 Skill。涵盖环境检查、推理脚本、
  精度验证（CPU vs NPU 余弦相似度 > 0.999）、性能基准测试全流程。
  使用 torch_npu + transfer_to_npu 一键迁移 PyTorch 模型至 Ascend NPU。
  当用户提到 MobileViT 昇腾部署、MobileViT NPU 推理、
  轻量级视觉模型 NPU 适配时触发。
metadata:
  short-description: Apple MobileViT 昇腾 NPU 部署与验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, mobilevit, image-classification, vision, transformers, pytorch, inference]
---

# Apple MobileViT 昇腾 NPU 部署 Skill

本 Skill 提供 Apple MobileViT 系列图像分类模型（MobileViT-Small / X-Small / XX-Small）
在华为昇腾 NPU 上的完整部署、推理验证、精度对比和性能基准测试的标准化可复现流程。

## 支持的模型

| 模型 | 参数量 | ImageNet Top-1 | HF Model ID |
|------|--------|----------------|-------------|
| MobileViT-Small | 5.6M | 78.4% | `apple/mobilevit-small` |
| MobileViT-X-Small | 2.3M | 74.8% | `apple/mobilevit-x-small` |
| MobileViT-XX-Small | 1.3M | 69.0% | `apple/mobilevit-xx-small` |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.RC1） |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重 |

## 流程总览

## 工作流阶段汇总

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 0. 环境初始化 | 加载 CANN 环境、选择空闲 NPU | 无（需 NPU 就绪） | `source set_env.sh`; `npu-smi info`; `export ASCEND_RT_VISIBLE_DEVICES` | NPU 环境就绪 | `npu-smi info` | 至少 1 个 NPU 状态 OK，变量设置正确 |
| 1. 安装依赖 | 安装 torch_npu + transformers + 图像库 | Python 3.9+、CANN ≥ 8.0 | `pip install torch torch_npu transformers pillow requests` | 依赖安装完成 | `python -c "import torch; import torch_npu; print(torch.__version__)"` | 无导入报错 |
| 2. NPU 环境验证 | 验证 torch_npu 基本运算 | 依赖安装完成 | `python3 -c "import torch_npu; print(torch.npu.is_available())"` | NPU 可用确认 | `python3` 执行 Tensor NPU 运算 | `torch.npu.is_available()` = True，Tensor 含 `device='npu:0'` |
| 3. 模型下载 | 从 HuggingFace 下载 3 个模型权重 | 网络连接 | `huggingface-cli download apple/mobilevit-{small,x-small,xx-small}` | 本地模型权重 | `ls ./models/*/pytorch_model.bin` | 3 个模型权重文件均存在 |
| 4. 基础推理验证 | 单张图片分类推理 | 模型权重 + 测试图片 | `python scripts/mobilevit_infer.py --model_path ... --image ...` | Top-5 分类结果 | 检查输出 Top-5 结果 | 无报错，Top-1 置信度 > 0.5 |
| 5. 精度对比验证 | CPU vs NPU 精度对比 | 模型权重 + 测试图片 | `python scripts/mobilevit_accuracy_run.py` | `accuracy_report_*.json` | `cat accuracy_report_*.json` | 余弦相似度 > 0.999，Top-1 一致 |
| 6. 性能基准测试 | 50 次迭代性能基准 | 模型权重 | `python scripts/mobilevit_perf_run.py --model_path ...` | `perf_report_*.json` | `cat perf_report_*.json` | P50 延迟波动 < 2ms |
| 7. 验收确认 | 逐项检查验收清单 | 以上所有阶段的输出 | 逐项检查验收清单 | 验收通过结论 | 确认所有检查项均已通过 | 全部验收项标记为通过 |

每个模型应串行验证（防止显存爆炸），完成一个后再进行下一个。

---

## 0. 环境初始化

**执行步骤**

1. 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 CANN 环境变量
2. 执行 `npu-smi info` 确认 NPU 设备状态为 OK，记下空闲设备号
3. 设置设备可见范围：`export ASCEND_RT_VISIBLE_DEVICES=0`（替换为实际空闲卡号）
4. 配置国内 pip 镜像：`export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/`
5. 执行检查点确认 NPU 状态、设备号和 Python 版本均达标

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号

# 国内 pip 镜像加速
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 检查点

- [ ] `npu-smi info` 确认至少一个 NPU 处于 `OK` 状态且显存可用
- [ ] `echo $ASCEND_RT_VISIBLE_DEVICES` 确认设备号设置正确
- [ ] `which python && python --version` 确认 Python 版本 ≥ 3.9

---

## 1. 安装依赖

**执行步骤**

1. 安装 PyTorch 基础库：`pip install torch transformers pillow requests`
2. 安装 torch_npu NPU 插件：`pip install torch_npu`
3. 若 pip 安装失败，参考昇腾官方文档使用离线 whl 包安装
4. 验证 PyTorch 导入：`python -c "import torch; import torch_npu; print(torch.__version__)"`
5. 确认所有额外依赖已安装：`pip list | grep -E "transformers|pillow|requests"`

```bash
pip install torch transformers pillow requests
```

确保 `torch_npu` 已正确安装：

```bash
pip install torch_npu
```

> 如果 pip 安装失败，参考昇腾官方文档：https://www.hiascend.com/document/

## 检查点

- [ ] `python -c "import torch; import torch_npu; print(torch.__version__)"` 确认 PyTorch 可导入
- [ ] `pip list | grep torch_npu` 确认 torch_npu 已安装
- [ ] 额外依赖：`pip list | grep -E "transformers|pillow|requests"` 确认均已安装

---

## 2. NPU 环境验证

**执行步骤**

1. 执行 `python3` 验证脚本，导入 `torch` 和 `torch_npu`
2. 创建 NPU Tensor：`a = torch.randn(3, 4).npu()` 并执行加法运算
3. 确认输出中 Tensor 的 device 字段包含 `device='npu:0'`
4. 确认 `torch.npu.is_available()` 返回 `True`
5. 记录 NPU 设备名称和数量，确认环境就绪后进入下一阶段

```bash
python3 -c "
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
a = torch.randn(3, 4).npu()
print(a + a)
print('NPU device:', torch.npu.get_device_name(0))
print('NPU available:', torch.npu.is_available())
"
```

**通过标准**：输出包含 `device='npu:0'` 的 Tensor 且无报错。

## 检查点

- [ ] `torch.npu.is_available()` 返回 `True`
- [ ] `torch.npu.device_count()` ≥ 1
- [ ] 输出日志中包含 `device='npu:0'` 或类似信息

---

## 3. 模型下载

**执行步骤**

1. 设置 HuggingFace 镜像加速下载：`export HF_ENDPOINT=https://hf-mirror.com`
2. 使用 `huggingface-cli download` 下载 MobileViT-Small 模型权重
3. 使用 `huggingface-cli download` 下载 MobileViT-X-Small 模型权重
4. 使用 `huggingface-cli download` 下载 MobileViT-XX-Small 模型权重
5. 验证各模型目录下 `pytorch_model.bin` 文件存在且大小合理

```bash
# 方式一：从 HuggingFace 镜像下载
export HF_ENDPOINT=https://hf-mirror.com

# 下载 MobileViT-Small
huggingface-cli download apple/mobilevit-small \
  --local-dir ./models/mobilevit-small --local-dir-use-symlinks False

# 下载 MobileViT-X-Small
huggingface-cli download apple/mobilevit-x-small \
  --local-dir ./models/mobilevit-x-small --local-dir-use-symlinks False

# 下载 MobileViT-XX-Small
huggingface-cli download apple/mobilevit-xx-small \
  --local-dir ./models/mobilevit-xx-small --local-dir-use-symlinks False
```

## 检查点

- [ ] `ls ./models/mobilevit-small/pytorch_model.bin 2>/dev/null` 确认 Small 模型权重文件存在
- [ ] `ls ./models/mobilevit-x-small/pytorch_model.bin 2>/dev/null` 确认 X-Small 模型权重文件存在
- [ ] `ls ./models/mobilevit-xx-small/pytorch_model.bin 2>/dev/null` 确认 XX-Small 模型权重文件存在
- [ ] `du -sh ./models/*/` 检查模型目录大小是否合理

---

## 4. 基础推理验证

**执行步骤**

1. 准备一张 JPEG/PNG 格式的测试图片作为推理输入
2. 对 MobileViT-Small 执行推理：`python scripts/mobilevit_infer.py --model_path ./models/mobilevit-small --image test.jpg --top_k 5`
3. 对 MobileViT-X-Small 执行推理：使用相同图片和对应脚本
4. 对 MobileViT-XX-Small 执行推理：使用对应脚本 `mobilevit_xx_small_infer.py`
5. 检查每个模型的 Top-5 分类结果，确认 Top-1 置信度 > 0.5 且与 ImageNet 类别对应

使用推理脚本对单张图片进行分类：

```bash
# MobileViT-Small
python scripts/mobilevit_infer.py \
  --model_path ./models/mobilevit-small \
  --image /path/to/image.jpg \
  --top_k 5

# MobileViT-X-Small
python scripts/mobilevit_infer.py \
  --model_path ./models/mobilevit-x-small \
  --image /path/to/image.jpg \
  --top_k 5

# MobileViT-XX-Small
python scripts/mobilevit_xx_small_infer.py \
  --model_path ./models/mobilevit-xx-small \
  --image /path/to/image.jpg \
  --top_k 5
```

## 检查点

- [ ] 推理脚本无报错退出
- [ ] 输出 Top-5 分类结果中 Top-1 label 与 ImageNet 类别对应
- [ ] Top-1 置信度 > 0.5（合理范围）

> **故障排除**：若推理失败，检查模型路径是否正确、图片路径是否存在、图片格式是否为 JPEG/PNG。

---

## 5. 精度对比验证

**执行步骤**

1. 串行验证 MobileViT-Small：`python scripts/mobilevit_accuracy_run.py ./models/mobilevit-small accuracy_report_small.json`
2. 串行验证 MobileViT-X-Small：`python scripts/mobilevit_accuracy_run.py ./models/mobilevit-x-small accuracy_report_xsmall.json`
3. 串行验证 MobileViT-XX-Small：`python scripts/mobilevit_accuracy_run.py ./models/mobilevit-xx-small accuracy_report_xxsmall.json`
4. 检查各模型 `accuracy_report_*.json`，确认余弦相似度 > 0.999
5. 确认 CPU vs NPU 的 Top-1 预测标签完全一致，最大概率差异 < 0.01

分别使用 CPU 和 NPU 推理，对比输出 logits 的余弦相似度和 Top-1 标签一致性。

**验证标准**：
- Top-1 标签一致（CPU vs NPU 预测同一类别）
- Logits 余弦相似度 > 0.999
- 最大概率差异 < 0.01（1%）

```bash
# 串行验证每个模型（防止显存溢出）
echo "=== MobileViT-Small ==="
python scripts/mobilevit_accuracy_run.py ./models/mobilevit-small accuracy_report_small.json

echo "=== MobileViT-X-Small ==="
python scripts/mobilevit_accuracy_run.py ./models/mobilevit-x-small accuracy_report_xsmall.json

echo "=== MobileViT-XX-Small ==="
python scripts/mobilevit_accuracy_run.py ./models/mobilevit-xx-small accuracy_report_xxsmall.json
```

**参考结果**（Ascend 910B4实测）：

| 模型 | Logits 余弦相似度 | 最大概率差异 | Top-1 一致性 | 状态 |
|------|-------------------|-------------|-------------|------|
| MobileViT-Small | 0.999997 | 0.000492 | 完全一致 | PASS |
| MobileViT-X-Small | 0.999998 | 0.000768 | 完全一致 | PASS |
| MobileViT-XX-Small | 0.999996 | 0.000418 | 完全一致 | PASS |

## 检查点

- [ ] 精度验证脚本均无报错退出
- [ ] 所有模型的余弦相似度 > 0.999
- [ ] Top-1 预测标签 CPU vs NPU 完全一致
- [ ] `accuracy_report_*.json` 文件已生成且内容完整

> **故障排除**：若余弦相似度低于 0.999，检查模型是否在 CPU 和 NPU 上使用相同输入数据；确认 `torch_npu` 版本与 CANN 版本匹配。

---

## 6. 性能基准测试

**执行步骤**

1. 串行测试 MobileViT-Small：`python scripts/mobilevit_perf_run.py ./models/mobilevit-small 50 perf_report_small.json`（50 次迭代，warmup 10 次）
2. 串行测试 MobileViT-X-Small：`python scripts/mobilevit_perf_run.py ./models/mobilevit-x-small 50 perf_report_xsmall.json`
3. 串行测试 MobileViT-XX-Small：`python scripts/mobilevit_perf_run.py ./models/mobilevit-xx-small 50 perf_report_xxsmall.json`
4. 检查各模型 `perf_report_*.json` 文件是否已生成且包含延迟与吞吐量指标
5. 确认 P50 延迟波动 < 2ms，NPU 显存在测试完成后恢复空闲状态

```bash
# 串行测试每个模型
echo "=== MobileViT-Small ==="
python scripts/mobilevit_perf_run.py ./models/mobilevit-small 50 perf_report_small.json

echo "=== MobileViT-X-Small ==="
python scripts/mobilevit_perf_run.py ./models/mobilevit-x-small 50 perf_report_xsmall.json

echo "=== MobileViT-XX-Small ==="
python scripts/mobilevit_perf_run.py ./models/mobilevit-xx-small 50 perf_report_xxsmall.json
```

**参考性能**（Ascend 910B4, 50次迭代, warmup 10次）：

| 模型 | 平均延迟 | P50 | P90 | 吞吐量 |
|------|---------|-----|-----|--------|
| MobileViT-Small | 17.62 ms | 17.55 ms | 18.16 ms | 56.77 img/s |
| MobileViT-X-Small | 17.59 ms | 17.59 ms | 17.74 ms | 56.84 img/s |
| MobileViT-XX-Small | 17.62 ms | 17.50 ms | 17.77 ms | 56.76 img/s |

> 三个模型推理延迟相近，因为 MobileViT 系列计算瓶颈在 Transformer 自注意力层，
> 受算子调度开销影响，小型模型的延迟优势未能充分体现。

## 检查点

- [ ] 性能测试脚本均无报错退出
- [ ] `perf_report_*.json` 文件已生成且包含延迟与吞吐量指标
- [ ] 三次测试结果中 P50 延迟波动 < 2ms
- [ ] NPU 显存在测试完成后恢复到空闲状态（`npu-smi info` 确认）

> **故障排除**：若性能偏低，确认 NPU 未被其他进程占用；检查 `torch.npu.synchronize()` 是否在计时点正确调用。

---

## 7. 验收确认

完成以下检查清单即为部署成功：

- [ ] `npu-smi info` 显示设备正常
- [ ] `import torch_npu` 无报错
- [ ] 所有 3 个模型的基础推理输出 Top-5 分类结果
- [ ] 精度验证：CPU vs NPU 余弦相似度 > 0.999
- [ ] 精度验证：Top-1 预测标签完全一致
- [ ] 性能测试：稳定输出延迟指标
- [ ] 串行验证：每个模型完成后清理，未发生显存溢出

---

## 8. 异常处理

## 8.1 环境与依赖异常

| 异常现象 | 可能原因 | 诊断命令 | 恢复步骤 |
|---------|---------|---------|---------|
| `No module named 'torch_npu'` | torch_npu 未安装或 CANN 未加载 | `pip list \| grep torch_npu` | `source set_env.sh` 后 `pip install torch_npu` |
| `ModuleNotFoundError: No module named 'torch'` | PyTorch 未安装 | `python -c "import torch"` | `pip install torch` |
| `npu-smi info` 显示无 NPU | 驱动未加载 | `ls /dev/davinci*` | 重新加载驱动或联系运维 |
| `ImportError: libc10_npu.so` | CANN 与 torch_npu 版本不匹配 | `cat /usr/local/Ascend/version.cfg` | 安装匹配版本的 torch_npu |


## 8.2 模型加载与推理异常

| 异常现象 | 可能原因 | 恢复步骤 |
|---------|---------|---------|
| `OutOfMemoryError` / OOM | 多模型同时加载 / 显存碎片 | 确保串行验证，每完成一个模型释放资源；`import torch; torch.npu.empty_cache()` |
| 模型加载时 `KeyError` 或 `OSError` | 模型权重文件损坏或不完整 | 重新下载模型 `rm -rf ./models/<model>` 后重试 |
| 推理输出全为 `unknown_NNN` | id2label 映射键格式问题 | 不影响分类准确性，可忽略；自定义映射可重写 `model.config.id2label` |
| 图片加载失败 / 编码错误 | 图片格式不支持 | 确保图片为 JPEG 或 PNG 格式；检查路径是否正确 |

## 8.3 精度与性能异常

| 异常现象 | 可能原因 | 恢复步骤 |
|---------|---------|---------|
| 余弦相似度 < 0.999 | 输入预处理或数据路径不一致 | 确认 CPU 和 NPU 使用相同的输入图片和预处理流程 |
| CPU vs NPU Top-1 不一致 | 浮点精度差异导致的边界情况 | 检查两个预测的置信度是否非常接近（差异 < 0.01 可接受） |
| 延迟显著高于参考值 | NPU 被其他进程占用 / 降频 | `npu-smi info` 检查 NPU 占用和频率；关闭其他任务后重试 |
| 吞吐量波动较大 | 系统负载不稳定 | 增加 warmup 次数至 20+，增加测试迭代次数至 100+ |

## 8.4 通用恢复流程

当遇到未列出的异常时，按以下顺序排查：

1. **确认环境**：`npu-smi info` 确认 NPU 状态
2. **确认版本**：`pip list | grep -E "torch|torch_npu|transformers"`、`python --version`
3. **查看完整日志**：捕获完整错误堆栈，定位首个 `Traceback` 中的错误类型
4. **清理资源**：`torch.npu.empty_cache()` 释放 NPU 显存
5. **升级依赖**：`pip install --upgrade torch_npu transformers`
6. **搜索解决方案**：在 [Ascend 社区](https://www.hiascend.com/) 或 GitCode Issues 中搜索类似错误

---

## 9. 资源清单

## 模型仓库与权重

| 模型 | HuggingFace 地址 | 本地目录 |
|------|-----------------|---------|
| MobileViT-Small | [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small) | `./models/mobilevit-small` |
| MobileViT-X-Small | [apple/mobilevit-x-small](https://huggingface.co/apple/mobilevit-x-small) | `./models/mobilevit-x-small` |
| MobileViT-XX-Small | [apple/mobilevit-xx-small](https://huggingface.co/apple/mobilevit-xx-small) | `./models/mobilevit-xx-small` |

## 依赖版本参考

| 依赖 | 推荐版本 | 说明 |
|------|---------|------|
| PyTorch | ≥ 2.1.0 | 深度学习框架 |
| torch_npu | 匹配 CANN 版本 | Ascend NPU PyTorch 插件 |
| transformers | ≥ 4.36.0 | HuggingFace 模型库 |
| CANN | ≥ 8.0 | 昇腾 AI 处理器驱动 |

## 参考文档

| 文档 | 链接 |
|------|------|
| Ascend PyTorch 适配指南 | [hiascend.com/document](https://www.hiascend.com/document/) |
| MobileViT 论文 (ICLR 2022) | [arxiv.org/abs/2110.02178](https://arxiv.org/abs/2110.02178) |
| HuggingFace MobileViT 文档 | [huggingface.co/docs/transformers/model_doc/mobilevit](https://huggingface.co/docs/transformers/model_doc/mobilevit) |

---

## 10. 适配要点

- MobileViT 使用 `MobileViTImageProcessor` 进行图片预处理（BGR 通道顺序）
- 输入尺寸：缩放至 288×288，中心裁剪至 256×256
- 使用 `transfer_to_npu` 自动将 `torch.cuda.*` 替换为 `torch.npu.*`
- 精度验证使用余弦相似度而非逐元素相对误差（避免近零值放大效应）
- `torch.npu.synchronize()` 用于精确计时
