---
name: plant-classify-npu-deploy
description: >
  植物分类模型在昇腾 NPU 上的完整部署与精度验证 Skill。
  涵盖 DenseNet121、ResNet18、ResNet50 三种模型架构在 Ascend910 NPU 上的
  环境准备、推理脚本生成、CPU/NPU 精度对比、README 文档生成、终端截图生成
  和 GitCode 模型仓库发布的端到端流程。
metadata:
  short-description: 植物分类模型昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, plant-classification, densenet121, resnet18, resnet50, cv, image-classification, pytorch]
---

# 植物分类模型昇腾 NPU 部署与精度验证 Skill

本 Skill 提供一组植物分类模型（DenseNet121、ResNet18、ResNet50）在华为昇腾 Ascend910 NPU 上的完整部署、推理验证和 CPU/NPU 精度对比的标准化可复现流程。

## 支持的模型列表

| 模型名称 | 架构 | ModelScope 源地址 | NPU 适配仓库 |
|---|---|---|---|
| plant-classify-densenet121 | DenseNet121 | [ModelScope](https://www.modelscope.cn/models/flowscolors/plant-classify-densenet121) | [GitCode Repo](https://gitcode.com/gcw_C8PI9e90/plant-classify-densenet121-npu) |
| plant-classify-resnet18 | ResNet18 | [ModelScope](https://www.modelscope.cn/models/flowscolors/plant-classify-resnet18) | [GitCode Repo](https://gitcode.com/gcw_C8PI9e90/plant-classify-resnet18-npu) |
| plant-classify-resnet50 | ResNet50 | [ModelScope](https://www.modelscope.cn/models/flowscolors/plant-classify-resnet50) | [GitCode Repo](https://gitcode.com/gcw_C8PI9e90/plant-classify-resnet50-npu) |

## 前置条件

| 项目 | 要求 |
|---|---|
| 硬件 | Ascend910 系列（至少 1 卡，64GB HBM） |
| OS | openEuler / Ubuntu（aarch64 或 x86_64） |
| CANN | >= 25.5 |
| Python | 3.10 – 3.12 |
| 网络 | 首次运行需联网下载模型权重 |

## 流程总览

```mermaid
flowchart TD
    A[0. 环境初始化] --> B[1. 下载模型权重]
    B --> C[2. 准备测试图片]
    C --> D{选择推理模式}
    D --> E[3. CPU 推理验证]
    D --> F[4. NPU 推理验证]
    E --> G[5. CPU/NPU 精度对比]
    F --> G
    G --> H{误差 < 1%?}
    H -->|是| I[6. 生成 README 和截图]
    H -->|否| J[排查预处理差异]
    J --> C
    I --> K[7. 发布到 GitCode 模型仓库]
    K --> L{还有下一个模型?}
    L -->|是| M[8. 释放 NPU 显存]
    M --> B
    L -->|否| N[完成]

    style A fill:#e1f5fe,stroke:#01579b
    style D fill:#fff3e0,stroke:#e65100
    style H fill:#fff3e0,stroke:#e65100
    style L fill:#fff3e0,stroke:#e65100
    style N fill:#e8f5e9,stroke:#1b5e20
```

---

## 完整工作流程

按以下步骤顺序执行，每步需确认通过后方可进入下一步。单模型约 5-10 分钟，3 模型批量约 20-30 分钟。

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 1. 环境初始化 | 昇腾服务器 + CANN | source set_env.sh, npu-smi info, pip install | 就绪的 Python 环境 | `npu-smi info` | 显示可用 NPU 设备 |
| 模型处理 | 2. 下载模型并推理 | 模型 ID + 测试图片 | snapshot_download, inference.py | CPU/NPU 推理结果 | `python3 inference.py ... cpu && npu:0` | 推理成功无报错 |
| 精度验证 | 2.4 CPU/NPU 精度对比 | CPU + NPU 推理结果 | compare_cpu_npu.py | 精度对比报告 | `python3 compare_cpu_npu.py` | 最大相对误差 < 1% |
| 批量执行 | 3. 串行执行多个模型 | 3 个模型 | run_all_serial.py | 全部模型结果 | `python3 scripts/run_all_serial.py` | 全部模型误差 < 1% |
| 产物生成 | 4. 生成终端截图 | 推理输出 | generate_screenshot.py | PNG 截图 | 检查文件存在 | 截图清晰可读 |
| 发布 | 5. 发布到 GitCode | 访问令牌 | curl 创建仓库, git push | 模型仓库 | 访问仓库页面 | 文件完整可访问 |

## 1. 环境初始化

| 项目 | 内容 |
|---|---|
| 输入 | 已安装 CANN 和 NPU 驱动的昇腾服务器 |
| 输出 | 就绪的 Python 环境，ASCEND_RT_VISIBLE_DEVICES 已设置 |
| 关键验证 | `npu-smi info` 显示可用 NPU 设备 |
| 失败处理 | CANN 路径不存在则检查安装；NPU 被占用则切换卡号 |

**执行步骤**：

1. 加载 CANN 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
2. 检查 NPU 可用性：`npu-smi info`，确认显示 Ascend910 设备且显存可用。
3. 选择空闲 NPU 卡：`export ASCEND_RT_VISIBLE_DEVICES=0`（被占用则依次尝试 1/2/3）。
4. 安装依赖：`pip install torch torchvision numpy Pillow tqdm matplotlib modelscope torch_npu`。
5. 验证安装：`python -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available())"`。

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0

# 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision numpy Pillow tqdm matplotlib modelscope
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch_npu
```

## 2. 下载模型并推理

对每个模型，按以下步骤串行执行：

### 2.1 下载模型

| 项目 | 内容 |
|---|---|
| 输入 | ModelScope 模型 ID: `flowscolors/plant-classify-{model_name}` |
| 输出 | 本地缓存目录 `~/.cache/modelscope/hub/` 下的模型文件 |
| 关键验证 | 下载完成后目录包含 `.pth` 或 `.bin` 权重文件 |
| 失败处理 | 网络超时则切换镜像源或重试 |

```python
from modelscope import snapshot_download
model_path = snapshot_download("flowscolors/plant-classify-densenet121")
```

### 2.2 CPU 推理

| 项目 | 内容 |
|---|---|
| 输入 | 模型权重 + 测试图片 `test_inputs/test_plant.jpg` |
| 输出 | Top-1/Top-5 分类标签与置信度 |
| 命令格式 | `inference.py <image_path> cpu` |
| 失败处理 | 图片格式不支持则转换为 RGB 后重试 |

```bash
python3 inference.py test_inputs/test_plant.jpg cpu
```

### 2.3 NPU 推理

| 项目 | 内容 |
|---|---|
| 输入 | 模型权重 + 测试图片 `test_inputs/test_plant.jpg` |
| 输出 | Top-1/Top-5 分类标签与置信度 |
| 命令格式 | `inference.py <image_path> npu:<device_id>` |
| 失败处理 | OOM 则释放显存或切换到其他 NPU 卡 |

```bash
python3 inference.py test_inputs/test_plant.jpg npu:0
```

### 2.4 CPU/NPU 精度对比

| 项目 | 内容 |
|---|---|
| 输入 | CPU 推理结果 + NPU 推理结果 |
| 输出 | 精度对比报告（绝对误差、相对误差、余弦相似度、Top-1 一致率） |
| 验收标准 | 最大相对误差 < 1% |
| 失败处理 | 误差超标则检查输入预处理一致性 |

```bash
python3 compare_cpu_npu.py
```

## 3. 串行执行多个模型

使用 `scripts/run_all_serial.py` 串行执行所有模型，防止 NPU 显存爆炸：

**执行步骤**：

1. 确认前序步骤全部通过（环境初始化 + 模型下载 + 精度对比）。
2. 执行串行脚本：`python3 scripts/run_all_serial.py`。
3. 监控输出，确认每个模型完成后显存被正确释放。
4. 检查全部模型的精度对比结果，确认误差 < 1%。

```bash
python3 scripts/run_all_serial.py
```

每个模型完成后主动释放资源：

```python
import gc
gc.collect()
if hasattr(torch, "npu"):
    torch.npu.empty_cache()
```

## 4. 生成终端截图

**执行步骤**：

1. 准备截图内容文件：包含推理命令和输出结果。
2. 执行截图生成脚本：`python3 scripts/generate_screenshot.py --title "plant-classify-densenet121@npu" --output "screenshot_inference.png" --lines-file "screenshot_lines.txt"`。
3. 验证截图文件存在：`ls -la screenshot_inference.png`。
4. 确认截图内容清晰、包含关键推理信息。

```bash
python3 scripts/generate_screenshot.py \
  --title "plant-classify-densenet121@npu" \
  --output "screenshot_inference.png" \
  --lines-file "screenshot_lines.txt"
```

## 5. 发布到 GitCode 模型仓库

```bash
# 创建模型仓库
curl --request POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{
    "name": "plant-classify-densenet121-npu",
    "repository_type": "model",
    "visibility": "public"
  }'

# 推送代码
git clone https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/plant-classify-densenet121-npu.git
cd plant-classify-densenet121-npu
git branch -M main
cp -r /path/to/model/files/* .
git add -A
git commit -m "Add NPU adaptation"
git push -u origin main
```

## 执行检查点与用户确认

在流程的关键节点需要用户进行确认，确保状态正确后再继续执行。

| 步骤 | 检查点 | 预期结果 | 用户确认操作 |
|---|---|---|---|
| 0. 环境初始化 | 执行 `npu-smi info` 查看 NPU 状态 | 显示 Ascend910 设备且显存可用 | 确认输出中有可用 NPU 设备 |
| 0. 环境初始化 | 执行 `python -c "import torch; import torch_npu"` | 无报错，NPU 可用 | 确认 import 成功 |
| 1. 下载模型权重 | 下载完成后检查模型文件完整性 | 模型目录包含 `.pth` 或 `.bin` 文件 | 确认目录结构与文档一致 |
| 2. 准备测试图片 | 执行 `python3 generate_test_image.py` | 生成 `test_plant.jpg` | 确认文件存在且可读 |
| 3. CPU 推理验证 | 运行 CPU 推理脚本 | 输出分类结果与置信度 | 确认输出格式正确、无报错 |
| 4. NPU 推理验证 | 运行 NPU 推理脚本 | 输出分类结果与置信度 | 确认输出格式正确、无报错 |
| 5. CPU/NPU 精度对比 | 运行精度对比脚本 | 输出误差指标和结论 | 确认误差小于 1%，结论为"通过" |
| 6. 生成 README 和截图 | 检查生成的截图文件 | `screenshot_inference.png` 内容清晰 | 确认截图包含关键信息 |
| 7. 发布到 GitCode | 访问仓库页面验证 | 仓库可见、文件完整 | 确认推送成功 |

## 异常处理与回滚策略

| 异常场景 | 触发条件 | 处理方式 | 回滚策略 |
|---|---|---|---|
| CANN 环境未加载 | `source set_env.sh` 失败 | 检查 CANN 安装路径，确认 `/usr/local/Ascend/ascend-toolkit/` 目录是否存在 | 重新安装 CANN 或修正环境变量 |
| NPU 设备被占用 | `npu-smi info` 显示显存不足 | 使用 `export ASCEND_RT_VISIBLE_DEVICES=N` 切换到其他空闲 NPU 卡 | 依次尝试不同的 NPU 卡号 |
| 模型下载失败 | `snapshot_download` 网络超时 | 检查网络连接，重试下载或使用镜像源 | 清除已下载的临时目录，重新下载 |
| 内存溢出 | 模型在 NPU 上推理时 OOM | 减少 batch size 或切换到更大的 NPU 卡 | 释放显存缓存 `torch.npu.empty_cache()` |
| CPU/NPU 精度差异过大 | 误差 > 1% | 检查输入预处理一致性，确认数据类型对齐 | 调整预处理逻辑，保持 CPU/NPU 输入一致 |
| 截图生成失败 | 缺少中文字体或 PIL 依赖 | 安装 `fonts-noto-cjk` 等字体包 | 跳过截图步骤，手动截图替代 |
| GitCode 发布失败 | Token 无效或无权限 | 检查 `ATOMGIT_USER_TOKEN` 环境变量 | 重新生成 Token 后重试 |
| 串行脚本中断 | 某个模型推理异常退出 | 记录当前进度，从失败模型继续 | 释放显存后跳过已完成模型 |

## 资源与评测产物

| 类别 | 资源/产物 | 说明 | 路径 |
|---|---|---|---|
| 模型权重 | DenseNet121 权重 | ModelScope 下载 | `~/.cache/modelscope/hub/flowscolors/plant-classify-densenet121/` |
| 模型权重 | ResNet18 权重 | ModelScope 下载 | `~/.cache/modelscope/hub/flowscolors/plant-classify-resnet18/` |
| 模型权重 | ResNet50 权重 | ModelScope 下载 | `~/.cache/modelscope/hub/flowscolors/plant-classify-resnet50/` |
| 测试数据 | 测试图片 | 自动生成或用户提供 | `test_inputs/test_plant.jpg` |
| CPU 推理结果 | CPU 输出日志 | 包含分类标签与置信度 | `outputs/cpu_results.json` |
| NPU 推理结果 | NPU 输出日志 | 包含分类标签与置信度 | `outputs/npu_results.json` |
| 精度对比报告 | CPU vs NPU 对比表 | 包含绝对误差、相对误差、余弦相似度 | `outputs/comparison_report.json` |
| 截图 | 推理终端截图 | PNG 格式，包含命令与输出 | `screenshot_inference.png` |
| README 文档 | 模型仓库 README | 中文部署文档 | `README.md` |
| 评测产物 | 最终比对报告 | 汇总所有模型的精度与性能数据 | `outputs/final_report.md` |

## 验证结果

### DenseNet121

| 指标 | 值 |
|---|---|
| 最大绝对误差 (概率) | 5.08 × 10⁻⁵ |
| 最大相对误差 | 1.22 × 10⁻³ (0.12%) |
| 余弦相似度 | 1.0000001192 |
| Top-1 一致率 | 100.00% |
| CPU 推理耗时 (10 images) | 1367.35 ms |
| NPU 推理耗时 (10 images) | 13.29 ms |
| 加速比 | 102.91× |
| **结论** | **误差 < 1%，通过** |

### ResNet18

| 指标 | 值 |
|---|---|
| 最大绝对误差 (概率) | 9.97 × 10⁻⁵ |
| 最大相对误差 | 1.04 × 10⁻³ (0.10%) |
| 余弦相似度 | 1.0000001192 |
| Top-1 一致率 | 100.00% |
| CPU 推理耗时 | 789.78 ms |
| NPU 推理耗时 | 2.86 ms |
| 加速比 | 276.21× |
| **结论** | **误差 < 1%，通过** |

### ResNet50

| 指标 | 值 |
|---|---|
| 最大绝对误差 (概率) | 1.06 × 10⁻⁴ |
| 最大相对误差 | 8.16 × 10⁻⁴ (0.08%) |
| 余弦相似度 | 0.9999998212 |
| Top-1 一致率 | 100.00% |
| CPU 推理耗时 | 1914.59 ms |
| NPU 推理耗时 | 6.79 ms |
| 加速比 | 281.81× |
| **结论** | **误差 < 1%，通过** |

## 文件结构

```
skills/deployment/plant-classify-npu/
├── SKILL.md                    # Skill 文档
└── scripts/
    ├── run_all_serial.py       # 串行执行多个模型的脚本
    └── generate_screenshot.py  # 终端截图生成脚本
```

## 模型仓库交付件（每个模型独立）

每个模型的 GitCode 仓库包含以下文件：

| 文件 | 说明 |
|---|---|
| `inference.py` | CPU/NPU 推理脚本 |
| `compare_cpu_npu.py` | CPU vs NPU 精度对比脚本 |
| `requirements.txt` | Python 依赖 |
| `generate_test_image.py` | 测试图片生成脚本 |
| `README.md` | 中文文档 |
| `screenshot_inference.png` | 推理结果截图 |

## 标签

- #+NPU
- #+昇腾
- #+Ascend910
- #+CV
- #+图像分类
- #+植物分类
- #+部署
- #+模型适配
- #+PyTorch
