---
name: detr-npu-deploy
description: >
  DETR (DEtection TRansformer) 目标检测模型在昇腾 NPU 上的完整部署 Skill。
  涵盖 facebook/detr-resnet-50 和 facebook/detr-resnet-101 两个模型的
  环境准备、依赖安装、模型下载、NPU 推理验证、精度对比测试的全流程。
  当用户提到 DETR 部署昇腾、DETR NPU 推理、目标检测模型 NPU、ResNet-50/101 NPU 时触发。
metadata:
  short-description: DETR 目标检测模型昇腾 NPU 部署
  category: NPU-Model-Deploy
  tags: [ascend, npu, detr, object-detection, transformers, pytorch, vision, coco]
---

# DETR NPU 部署 Skill

## 前置条件

- 硬件：Ascend 910 系列 NPU（至少 1 卡）
- 操作系统：Linux (aarch64)
- Python：3.10+
- 网络：可访问 hf-mirror.com（国内镜像）或 HuggingFace

## 流程总览

## 工作流阶段总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 0 | 昇腾 NPU 服务器 | 加载 CANN 环境、检查 NPU 状态、选择空闲卡 | NPU 环境就绪 | `python3 -c "import torch_npu; print(torch.npu.is_available())"` | `torch.npu.is_available()` 返回 `True` |
| 依赖安装 | 1 | Python 3.10+，CANN 已加载 | 安装 PyTorch、torchvision、transformers、timm | 依赖安装完成 | `python3 -c "import timm; import transformers"` | 无 ImportError |
| 模型下载 | 2 | 模型名称、网络连接 | 下载配置文件和权重 | 模型权重文件 | 检查文件完整性 | `config.json`、`model.safetensors` 存在 |
| 推理验证 | 3 | 模型目录、测试图像 | 运行推理脚本进行目标检测 | 检测结果列表 | `python3 scripts/inference.py --image <url>` | 置信度 > 0.9，检测到合理目标 |
| 精度验证 | 4 | 测试图像 | CPU/NPU 分别推理、对比相对误差 | 精度报告 | `python3 scripts/accuracy_test.py` | Logits 和 Boxes 平均相对误差 < 1% |

---

## 0. 环境初始化与 NPU 预检

**执行步骤**：
1. 加载 CANN 环境变量
2. 检查 NPU 设备状态，确认至少 1 张卡可用
3. 选择空闲 NPU 卡并设置环境变量
4. 验证 NPU 基础环境

| 项目 | 内容 |
|------|------|
| **输入** | 昇腾 NPU 服务器，CANN 已安装 |
| **操作** | 加载 CANN 环境、检查 NPU 状态、选择空闲卡、验证环境 |
| **输出** | NPU 环境就绪，torch.npu.is_available() 返回 True |
| **异常** | CANN 路径不存在 → 检查安装路径；NPU 卡全满 → 等待资源释放 |

## 0.1 加载 CANN 环境

1. 确认 CANN 安装路径
2. 加载环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**如果加载失败**，请检查 CANN 安装路径是否正确。

## 0.2 NPU 状态检查

1. 运行 npu-smi 查看设备状态
2. 确认至少 1 张卡可用

```bash
npu-smi info
```

**用户确认**：确保 `npu-smi info` 输出显示至少 1 张卡状态为 `OK`。

## 0.3 选择空闲 NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

## 0.4 基础环境验证

```bash
python3 -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available()); print('Device count:', torch.npu.device_count())"
```

**通过标准**：`torch.npu.is_available()` 返回 `True`。

---

## 1. 安装依赖

**执行步骤**：
1. 安装 PyTorch、torchvision、transformers、timm 等依赖
2. 验证所有依赖导入正常
3. 设置 HuggingFace 国内镜像

| 项目 | 内容 |
|------|------|
| **输入** | Python 3.10+，昇腾 CANN 已加载 |
| **操作** | 安装 PyTorch、torchvision、transformers、timm 等依赖 |
| **输出** | 依赖安装完成，NPU 环境验证通过 |
| **异常** | timm 报错 → 单独执行 pip install timm；torch_npu 报错 → 回退到 0.1 重新加载 CANN 环境 |

1. 安装基础 Python 依赖包
2. 验证依赖安装
3. 设置 HuggingFace 国内镜像

```bash
pip install torch torchvision transformers timm Pillow requests -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1.1 验证安装

```bash
python3 -c "import torch_npu; import timm; import transformers; print('All dependencies OK')"
```

**如果报错 `ImportError: timm not found`**，说明 DETR 依赖 timm 未安装，请执行 `pip install timm`。

> **注意**：DETR 模型依赖 `timm` 库作为 CNN 骨干网络后端，必须安装。

设置 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 2. 下载模型

**执行步骤**：
1. 下载 DETR-ResNet-50 配置文件和权重
2. 下载 DETR-ResNet-101 配置文件和权重
3. 验证文件完整性

DETR 模型权重约 160MB~240MB，包含单个 `model.safetensors` 文件。

## 2.1 DETR-ResNet-50

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /opt/atomgit/detr-resnet-50

# 下载配置文件
python3 -c "
import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('facebook/detr-resnet-50',
    allow_patterns=['config.json', '*.md', 'preprocessor_config.json'],
    local_dir='/opt/atomgit/detr-resnet-50')
"

# 下载权重文件
wget -c "https://hf-mirror.com/facebook/detr-resnet-50/resolve/main/model.safetensors" \
  -P /opt/atomgit/detr-resnet-50/
```

## 2.2 DETR-ResNet-101

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /opt/atomgit/detr-resnet-101

# 下载配置文件
python3 -c "
import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('facebook/detr-resnet-101',
    allow_patterns=['config.json', '*.md', 'preprocessor_config.json'],
    local_dir='/opt/atomgit/detr-resnet-101')
"

# 下载权重文件
wget -c "https://hf-mirror.com/facebook/detr-resnet-101/resolve/main/model.safetensors" \
  -P /opt/atomgit/detr-resnet-101/
```

**输入/输出定义**：
- 输入：模型名称、网络连接
- 输出：`config.json`、`preprocessor_config.json`、`model.safetensors`
- 异常：如果下载超时，请检查 `HF_ENDPOINT` 是否已设置，或尝试 `wget -c` 断点续传。

---

## 3. 模型推理验证

**执行步骤**：
1. 运行 DETR-ResNet-50 推理脚本
2. 运行 DETR-ResNet-101 推理脚本
3. 检查检测结果的类别、置信度和边界框坐标

使用推理脚本对 COCO 测试图片进行目标检测。

<!-- EMBED:scripts/inference.py -->

```bash
# DETR-ResNet-50 NPU 推理
python3 scripts/inference.py \
  --image http://images.cocodataset.org/val2017/000000039769.jpg

# DETR-ResNet-101 NPU 推理
python3 scripts/inference.py \
  --image http://images.cocodataset.org/val2017/000000039769.jpg
```

预期输出（DETR-ResNet-50）：

```
[INFO] 使用 NPU 设备: Ascend910B4
[INFO] 模型参数: 41.5M
[INFO] 图片尺寸: (640, 480)
[RESULTS] 检测到 5 个目标:
  cat             置信度=0.999  框=[330, 70, 370, 210]
  cat             置信度=0.999  框=[15, 55, 330, 360]
  couch           置信度=0.995  框=[0, 1, 640, 370]
  remote          置信度=0.998  框=[35, 75, 190, 175]
  remote          置信度=0.996  框=[330, 75, 370, 130]
```

**输入/输出定义**：
- 输入：图像 URL 或本地路径
- 输出：检测目标列表（类别、置信度、边界框坐标）
- 异常：如果检测结果为空，请检查图像是否成功下载，或尝试更换测试图片。

**通过标准**：
- 输出包含合理的检测目标（如 cat、couch、remote 等）
- 置信度 > 0.9
- 无 `torch_npu` 相关报错

---

## 4. 精度对比测试

**执行步骤**：
1. 运行 DETR-ResNet-50 精度测试脚本
2. 运行 DETR-ResNet-101 精度测试脚本
3. 检查 Logits 和 Boxes 平均相对误差均 < 1%

使用精度测试脚本对比 CPU 和 NPU 的输出差异，验证误差 < 1%。

<!-- EMBED:scripts/accuracy_test.py -->

```bash
# DETR-ResNet-50 精度测试
python3 scripts/accuracy_test.py \
  --model_path /opt/atomgit/detr-resnet-50 \
  --output results/detr50_eval.json

# DETR-ResNet-101 精度测试
python3 scripts/accuracy_test.py \
  --model_path /opt/atomgit/detr-resnet-101 \
  --output results/detr101_eval.json
```

## 4.1 精度测试数据

## DETR-ResNet-50

| 指标 | 数值 |
|---|---|
| Logits 平均相对误差 | **0.027%** |
| Boxes 平均相对误差 | **< 0.001%** |
| 精度要求 | < 1% |
| 结果 | **PASS ✅** |

## DETR-ResNet-101

| 指标 | 数值 |
|---|---|
| Logits 平均相对误差 | **0.040%** |
| Boxes 平均相对误差 | **< 0.001%** |
| 精度要求 | < 1% |
| 结果 | **PASS ✅** |

**通过标准**：Logits 和 Boxes 的平均相对误差均 < 1%。

---

## 5. 异常边界与 Fallback 策略

| 异常场景 | 触发条件 | Fallback 策略 | 恢复操作 |
|---|---|---|---|
| `ImportError: timm not found` | DETR 依赖 timm 未安装 | 暂停执行，安装依赖 | `pip install timm` |
| NPU 推理速度慢于预期 | 首次调用包含编译开销 | 正常行为 | 第二次调用性能明显提升 |
| 模型下载超时 | 网络无法访问 HuggingFace | 重试，切换镜像 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `No module named 'torch_npu'` | CANN 未加载 | 回滚到步骤 0.1 | `source set_env.sh` 后重装 torch_npu |
| OOM | 输入图像过大 | 回滚，减小输入 | 降低图像分辨率或单张推理 |
| 检测结果异常 | 图像损坏或格式不支持 | 失败，需用户确认 | 更换测试图片 |

---

## 6. 检查点与交付确认

完成以下检查清单即为部署成功。每步完成后请**用户确认**再打勾：

- [ ] **Checkpoint 1**：DETR-ResNet-50 NPU 推理可正常运行
- [ ] **Checkpoint 2**：DETR-ResNet-101 NPU 推理可正常运行
- [ ] **Checkpoint 3**：两个模型精度误差均 < 1%
- [ ] **Checkpoint 4**：检测结果与 CPU 推理一致
- [ ] **Checkpoint 5**：精度报告 JSON 已生成（`results/detr50_eval.json`、`results/detr101_eval.json`）

---

## 7. 资源速查表

| 资源类型 | 路径 / 说明 |
|----------|-------------|
| 推理脚本 | `scripts/inference.py` |
| 精度测试脚本 | `scripts/accuracy_test.py` |
| 精度报告 | `results/detr50_eval.json`、`results/detr101_eval.json` |
| DETR-ResNet-50 权重 | `/opt/atomgit/detr-resnet-50/model.safetensors`（~160MB） |
| DETR-ResNet-101 权重 | `/opt/atomgit/detr-resnet-101/model.safetensors`（~240MB） |
| 参考文档 | https://huggingface.co/docs/transformers/model_doc/detr |
| HuggingFace 镜像 | https://hf-mirror.com |

---

## 附录：模型仓库链接

- DETR-ResNet-50 NPU: https://gitcode.com/gcw_C8PI9e90/detr-resnet-50-npu
- DETR-ResNet-101 NPU: https://gitcode.com/gcw_C8PI9e90/detr-resnet-101-npu
- 原始权重: https://huggingface.co/facebook/detr-resnet-50 / detr-resnet-101
- 参考文档: https://huggingface.co/docs/transformers/model_doc/detr
