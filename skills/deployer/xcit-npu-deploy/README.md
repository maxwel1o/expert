---
name: xcit-npu-deploy
description: >
  XCiT 系列图像分类模型在昇腾 NPU 上的完整部署与精度验证 Skill。
  涵盖环境准备、模型下载（ModelScope）、NPU 推理、CPU/NPU 精度对比、
  终端截图生成、README 编写和模型仓库发布的全流程。
  可在任意 Ascend910 系列服务器上一键复现。
  当用户提到 XCiT 部署昇腾、timm XCiT NPU 推理、XCiT 模型 NPU 时触发。
metadata:
  short-description: XCiT 昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, xcit, timm, pytorch, image-classification, inference]
---

# XCiT 昇腾 NPU 部署与精度验证 Skill

本 Skill 提供 5 个 XCiT (Cross-Covariance Image Transformer) 图像分类模型在华为昇腾 NPU 上的完整部署、推理验证和 CPU/NPU 精度对比的标准化可复现流程。

## 支持的模型列表

| 模型名称 | 输入尺寸 | 参数量 | ModelScope 地址 |
|---------|---------|--------|----------------|
| xcit_tiny_12_p16_384 | 384×384 | ~12M | [链接](https://www.modelscope.cn/models/timm/xcit_tiny_12_p16_384.fb_dist_in1k) |
| xcit_large_24_p8_224 | 224×224 | ~189M | [链接](https://www.modelscope.cn/models/timm/xcit_large_24_p8_224.fb_in1k) |
| xcit_medium_24_p16_384 | 384×384 | ~84M | [链接](https://www.modelscope.cn/models/timm/xcit_medium_24_p16_384.fb_dist_in1k) |
| xcit_tiny_12_p8_384 | 384×384 | ~12M | [链接](https://www.modelscope.cn/models/timm/xcit_tiny_12_p8_384.fb_dist_in1k) |
| xcit_small_12_p8_224 | 224×224 | ~42M | [链接](https://www.modelscope.cn/models/timm/xcit_small_12_p8_224.fb_in1k) |

## 已提交的模型仓库

| 模型 | GitCode 仓库地址 |
|------|----------------|
| xcit_tiny_12_p16_384-npu | [仓库](https://gitcode.com/gcw_C8PI9e90/xcit_tiny_12_p16_384-npu) |
| xcit_large_24_p8_224-npu | [仓库](https://gitcode.com/gcw_C8PI9e90/xcit_large_24_p8_224-npu) |
| xcit_medium_24_p16_384-npu | [仓库](https://gitcode.com/gcw_C8PI9e90/xcit_medium_24_p16_384-npu) |
| xcit_tiny_12_p8_384-npu | [仓库](https://gitcode.com/gcw_C8PI9e90/xcit_tiny_12_p8_384-npu) |
| xcit_small_12_p8_224-npu | [仓库](https://gitcode.com/gcw_C8PI9e90/xcit_small_12_p8_224-npu) |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| CANN | >= 8.0 |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重 |

## Skill 输入参数

- `--model`: 模型名称（必须），可选值：`xcit_tiny_12_p16_384`, `xcit_large_24_p8_224`, `xcit_medium_24_p16_384`, `xcit_tiny_12_p8_384`, `xcit_small_12_p8_224`
- `--device`: 推理设备，可选值：`auto`, `cpu`, `npu:0`
- `--batch-size`: 批量大小，默认 1
- `--warmup`: 预热迭代次数，默认 3
- `--iters`: 基准测试迭代次数，默认 10

## Skill 输出结果

- 推理耗时（ms）和吞吐量（samples/sec）
- Top-5 预测类别和概率
- CPU/NPU 精度对比结果（logits 误差、概率误差、Top-1 一致性）
- 输出文件：`inference_output.pt`

## 环境准备

```bash
# 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torch_npu timm modelscope safetensors numpy Pillow

# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查 NPU
npu-smi info
```

## 执行 NPU 推理

```bash
# 单模型推理
python3 scripts/run_inference.py --model xcit_tiny_12_p16_384 --device npu:0

# 所有模型串行推理
bash scripts/run_all.sh
```

## 执行 CPU/NPU 精度对比

```bash
# 单模型精度对比
python3 scripts/run_compare.py --model xcit_tiny_12_p16_384

# 所有模型串行精度对比 + 推理
bash scripts/run_all.sh
```

## 串行执行（避免显存爆炸）

使用 `scripts/run_all.sh` 脚本，每个模型完成后自动释放 NPU 显存：

```bash
bash scripts/run_all.sh
```

该脚本在每两个模型之间插入 5 秒延迟，并在每个模型完成后执行：

```python
import gc
gc.collect()
torch.npu.empty_cache()
```

## 生成 README

每个模型仓库已包含完整的 README，格式如下：
- 模型介绍和原始地址
- 任务类型和框架
- 环境依赖
- 推理命令和结果（含 CPU/NPU 耗时对比表格）
- CPU/NPU 精度测试方法
- 精度测试结果表格（logits 误差、概率误差、Top-1 一致性）
- 明确结论：NPU 与 CPU 推理误差 < 1%
- 模型标签（#+NPU、#+CV、#+图像分类、#+昇腾、#+XCiT）
- 运行截图

## 生成终端截图

使用 `terminal_screenshot.py` 工具生成模拟终端输出截图：

```bash
python3 terminal_screenshot.py --input output.txt --output terminal_screenshot.png
```

该工具自动渲染为深色终端风格，已放入各模型仓库的 README 中。

## 提交模型仓库

使用 GitCode API v5 创建和推送模型仓库：

```bash
# 创建仓库
curl --location "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "<model>-npu", "repository_type": "model", "private": false}'

# 推送代码
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/<model>-npu.git
git branch -M main
git push -u origin main
```

## 模型标签

所有模型仓库均包含以下标签：
- #+NPU
- #+CV
- #+图像分类
- #+昇腾
- #+XCiT

## 目录结构

```
skills/deployment/xcit-npu-deploy/
├── skill.json                 # Skill 元数据
├── README.md                  # 本说明文档
├── scripts/
│   ├── run_all.sh            # 串行执行所有模型
│   ├── run_inference.py      # 统一推理脚本
│   └── run_compare.py        # 统一精度对比脚本
└── examples/
    └── quickstart.md          # 快速上手示例
```
