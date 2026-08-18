# Sequencer2D NPU Deployment Skill

## 简介

一站式完成 Sequencer2D 系列模型（s/m/l 三个变体）在昇腾 NPU 上的部署、推理、CPU/NPU 精度测试、README 生成、终端截图生成和 GitCode 模型仓库发布。

## 支持的模型列表

| 模型名称 | 原始地址 | GitCode 仓库 |
| --- | --- | --- |
| sequencer2d_s.in1k | [ModelScope](https://www.modelscope.cn/models/timm/sequencer2d_s.in1k) | [sequencer2d_s.in1k-npu](https://gitcode.com/m0_74196153/sequencer2d_s.in1k-npu) |
| sequencer2d_m.in1k | [ModelScope](https://www.modelscope.cn/models/timm/sequencer2d_m.in1k) | [sequencer2d_m.in1k-npu](https://gitcode.com/m0_74196153/sequencer2d_m.in1k-npu) |
| sequencer2d_l.in1k | [ModelScope](https://www.modelscope.cn/models/timm/sequencer2d_l.in1k) | [sequencer2d_l.in1k-npu](https://gitcode.com/m0_74196153/sequencer2d_l.in1k-npu) |

## Skill 输入参数

通过 `skill.json` 定义，支持以下参数：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| model_name | string | 否 | "all" | 指定单个模型名称，不指定则执行全部3个 |
| skip_inference | boolean | 否 | false | 跳过推理步骤，只生成文档和提交仓库 |
| skip_push | boolean | 否 | false | 跳过 GitCode 仓库推送 |

## Skill 输出结果

每个模型处理完成后输出：

- `inference.py` - 推理脚本
- `compare_cpu_npu.py` - CPU/NPU 精度对比脚本
- `requirements.txt` - 依赖列表
- `readme.md` - 中文 README 文档
- `terminal_*.png` - 模拟终端输出截图
- GitCode 模型仓库（自动创建并推送）

## 环境要求

| 组件 | 版本 |
| --- | --- |
| Python | >= 3.10 |
| torch + torch_npu | >= 2.1.0 |
| timm | >= 1.0.0 |
| Pillow | >= 10.0.0 |
| modelscope | >= 1.0.0 |

## 如何执行 NPU 推理

```python
import torch
import torch_npu
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

model = timm.create_model("sequencer2d_s.in1k", pretrained=False)
state_dict = torch.load("model.safetensors", map_location="cpu")
model.load_state_dict(state_dict)
model.eval().to(torch.device("npu"))

img = Image.new("RGB", (224, 224), color=(128, 128, 128))
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
input_tensor = transform(img).unsqueeze(0).to(torch.device("npu"))

with torch.no_grad():
    output = model(input_tensor)
    print(f"Top-1: {torch.argmax(torch.softmax(output, dim=1), dim=1).item()}")
```

## 如何执行 CPU/NPU 精度对比

1. 使用 `inference.py` 分别在 CPU 和 NPU 上推理，保存 logits
2. 使用 `compare_cpu_npu.py` 对比两者差异

```bash
python3 scripts/compare_cpu_npu.py --cpu logits_cpu.npy --npu logits_npu.npy --model sequencer2d_s
```

## 如何生成 README

每个模型在完成推理和精度对比后，Skill 会根据实际测试数据自动生成中文 README，包含：
- 模型介绍和原始地址
- 任务类型和框架信息
- NPU 适配说明
- CPU/NPU 推理结果对比表格
- 精度测试数据（误差率 < 1%）
- 性能测试数据（加速比）
- 部署和推理方法

## 如何生成终端截图

使用 `terminal_screenshot.py` 工具生成模拟终端输出截图：

```bash
python3 /opt/atomgit/terminal_screenshot.py --text "推理结果文本" --output terminal.png
```

## 串行执行多个模型

`batch_runner.py` 自动串行执行所有模型，每个模型完成后释放 NPU 显存和 CPU 内存，避免显存爆炸：

```bash
python3 scripts/batch_runner.py
```

## 如何调用 GitCode API 提交模型仓库

Skill 使用 GitCode API 自动创建模型仓库并推送代码：

```bash
# 创建仓库
curl -X POST -H "Authorization: Bearer ${ATOMGIT_USER_TOKEN}" \
  -d '{"name":"model-name-npu","repository_type":"model"}' \
  https://api.gitcode.com/api/v5/user/repos

# 推送代码
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/user/repo.git
git push -u origin main
```

## 精度测试要求

- 必须进行 CPU 与 NPU 的对比
- 误差率 < 1% 为通过
- 误差率 = max(|CPU_logits - NPU_logits|) / (max(CPU_logits) - min(CPU_logits))
- 额外验证 Top-1/Top-5 类别一致性、余弦相似度
