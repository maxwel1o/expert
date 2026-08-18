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

## 流程总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 加载 CANN 并安装依赖 | CANN Toolkit, Python 环境 | source set_env.sh, pip install | 可用 NPU 环境 | npu-smi info, python import | NPU 设备可见且 import 无报错 |
| 模型下载 | 下载 ModelScope 或 timm 模型 | model_name | snapshot_download / timm.create_model | 模型权重 | python -c "import timm; m = timm.create_model('xcit...')" | 模型加载成功 |
| NPU 推理 | 执行 NPU 推理 | 模型名称, NPU 设备 | run_inference.py --device npu:0 | inference_output.pt | ls inference_output.pt | 输出文件存在 |
| CPU 推理 | 执行 CPU 推理 | 模型名称 | run_inference.py --device cpu | inference_output.pt | ls inference_output.pt | 输出文件存在 |
| 精度对比 | CPU/NPU 对比 | CPU/NPU 结果 | run_compare.py | 对比报告 | 终端输出误差指标 | 概率误差 < 1% |
| README 生成 | 生成部署文档 | 测试结果 | 按模板填写 README | README.md | cat README.md | 包含精度对比和截图 |
| 截图生成 | 生成终端截图 | 推理输出 | terminal_screenshot.py | PNG 截图 | ls terminal_screenshot.png | 截图清晰可读 |
| 仓库发布 | 创建并推送仓库 | 所有产物 | GitCode API | GitCode 仓库 | curl -I gitcode.com/{user}/{repo} | 仓库可见且文件完整 |

## 执行检查点与用户确认

在流程的关键节点需要用户确认状态正确后再继续执行。

| 步骤 | 检查点 | 预期结果 | 用户确认操作 |
|---|---|---|---|
| 1. 环境准备 | 执行 `npu-smi info` 查看 NPU 状态 | 显示 Ascend910 设备且显存可用 | 确认输出中有可用 NPU 设备 |
| 1. 环境准备 | 执行依赖安装 | Python 包安装成功 | 确认 import 无报错 |
| 2. 模型下载 | 检查模型缓存目录 | 模型权重文件完整 | 确认下载完成 |
| 3. NPU 推理 | 运行 `run_inference.py --device npu:0` | 输出推理结果和 `inference_output.pt` | 确认输出文件存在 |
| 4. CPU 推理 | 运行 `run_inference.py --device cpu` | 输出推理结果和 `inference_output.pt` | 确认输出文件存在 |
| 5. 精度对比 | 运行 `run_compare.py` | 输出误差指标和结论 | 确认误差 < 1% |
| 6. README 生成 | 检查生成的文档 | 包含精度对比表格和截图 | 确认内容完整 |
| 7. 仓库发布 | 访问 GitCode 仓库 URL | 仓库可见、文件完整 | 确认推送成功 |

## 异常处理与回滚策略

| 异常场景 | 触发条件 | 处理方式 | 回滚策略 |
|---|---|---|---|
| CANN 环境未加载 | `source set_env.sh` 失败 | 检查 CANN 安装路径 | 重新安装 CANN 或修正环境变量 |
| NPU 设备不可用 | `torch.npu.is_available()` 返回 False | 执行 `npu-smi info` 检查设备 | 切换到 CPU 模式或重启驱动 |
| 模型下载失败 | 网络超时 | 配置国内镜像源重试 | 清除不完整下载后重试 |
| NPU 推理 OOM | 大模型（large_24_p8）在单卡上显存不足 | 减少 batch size 或增加 warmup 次数 | 释放显存后重试 |
| CPU/NPU 精度超标 | 概率误差 >= 1% | 检查预处理一致性、数据类型对齐 | 对齐预处理管道后重新推理 |
| 截图生成失败 | 缺少字体 | 安装字体包 | 跳过截图，手动截图 |
| GitCode 发布失败 | Token 无效 | 检查 `ATOMGIT_USER_TOKEN` | 重新生成 Token 后重试 |

## 资源与评测产物

| 类别 | 资源/产物 | 说明 | 路径 |
|---|---|---|---|
| 模型权重 | XCiT 模型 | ModelScope/timm 预训练权重 | `~/.cache/modelscope/hub/` |
| 推理结果 | NPU/CPU 推理输出 | 包含 logits、耗时 | `inference_output.pt` |
| 精度对比报告 | 对比结果 | logits 误差、概率误差、Top-1 一致性 | 终端输出 + 报告文件 |
| 终端截图 | PNG | 推理结果终端风格截图 | `terminal_screenshot.png` |
| README 文档 | GitCode README | 中文部署文档 | 各模型仓库 README.md |

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

| 项目 | 内容 |
|---|---|
| 输入 | 已安装 CANN 和 NPU 驱动的昇腾服务器 |
| 输出 | 就绪的 Python 环境，依赖已安装 |
| 关键验证 | `npu-smi info` 显示可用 NPU 设备 |
| 失败处理 | 依赖安装失败则使用国内镜像重试 |

```bash
# 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torch_npu timm modelscope safetensors numpy Pillow

# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查 NPU
npu-smi info
```

**执行步骤**：
1. 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 CANN 环境变量
2. 执行 `npu-smi info` 确认 Ascend910 设备可用且显存充足
3. 执行 `pip install torch torchvision torch_npu timm modelscope safetensors numpy Pillow` 安装依赖
4. 执行 `python -c "import torch; import torch_npu; print(torch.npu.is_available())"` 验证 NPU 可用

## 执行 NPU 推理

| 项目 | 内容 |
|---|---|
| 输入 | 模型名称 + 推理设备 |
| 输出 | 推理结果文件 `inference_output.pt` |
| 命令格式 | `run_inference.py --model {name} --device {device}` |
| 失败处理 | OOM 则释放显存或调整 batch size |

```bash
# 单模型推理
python3 scripts/run_inference.py --model xcit_tiny_12_p16_384 --device npu:0

# 所有模型串行推理
bash scripts/run_all.sh
```

**执行步骤**：
1. 选择模型执行 `python3 scripts/run_inference.py --model xcit_tiny_12_p16_384 --device npu:0`
2. 检查是否正确输出 `inference_output.pt` 结果文件
3. 如需运行所有 5 个模型串行推理，执行 `bash scripts/run_all.sh`
4. 每个模型完成后脚本自动释放 NPU 显存以避免 OOM

## 执行 CPU/NPU 精度对比

| 项目 | 内容 |
|---|---|
| 输入 | NPU 推理结果 + CPU 推理结果 |
| 输出 | 精度对比（logits 误差、概率误差、Top-1 一致性） |
| 验收标准 | NPU 与 CPU 推理误差 < 1% |
| 失败处理 | 误差超标则检查预处理一致性 |

```bash
# 单模型精度对比
python3 scripts/run_compare.py --model xcit_tiny_12_p16_384

# 所有模型串行精度对比 + 推理
bash scripts/run_all.sh
```

**执行步骤**：
1. 执行 `python3 scripts/run_inference.py --model {name} --device cpu` 运行 CPU 推理
2. 检查 CPU 推理结果文件 `inference_output.pt` 是否正常生成
3. 确认 CPU 和 NPU 推理均完成后再进入精度对比步骤

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

**执行步骤**：
1. 确认 CPU 和 NPU 推理结果均已生成，检查 `inference_output.pt` 文件存在
2. 执行 `python3 scripts/run_compare.py --model xcit_tiny_12_p16_384` 运行精度对比
3. 检查终端输出的 logits 误差、概率误差、Top-1 一致性
4. 确认概率误差 < 1%，结论为 PASS

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

**执行步骤**：
1. 按模板编写 README.md，填入模型名称和原始 ModelScope 地址
2. 在文档中包含 CPU/NPU 精度对比表格和推理性能数据
3. 添加明确的精度结论：NPU 与 CPU 推理误差 < 1%
4. 添加模型标签（#+NPU、#+CV、#+图像分类、#+昇腾、#+XCiT）
5. 将生成的终端截图嵌入 README 文档中

## 生成终端截图

使用 `terminal_screenshot.py` 工具生成模拟终端输出截图：

```bash
python3 terminal_screenshot.py --input output.txt --output terminal_screenshot.png
```

该工具自动渲染为深色终端风格，已放入各模型仓库的 README 中。

**执行步骤**：
1. 执行 `python3 terminal_screenshot.py --input output.txt --output terminal_screenshot.png` 生成截图
2. 检查生成的 PNG 文件内容是否清晰包含推理命令与结果
3. 如缺少字体导致截图生成失败，安装 `fonts-noto-cjk` 后重试

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

**执行步骤**：
1. 确认 `ATOMGIT_USER_TOKEN` 环境变量已配置且有效
2. 使用 curl 调用 GitCode API v5 创建模型类型仓库（`repository_type: model`）
3. 将所有产物文件（README.md、inference_output.pt、terminal_screenshot.png）添加到仓库
4. 执行 `git init && git add -A && git commit -m "Initial commit"` 初始化本地仓库
5. 执行 `git push -u origin main` 推送到远程仓库并验证仓库可见

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
├── SKILL.md                   # Skill 文档
├── README.md                  # Skill 说明文档
├── skill.json                 # Skill 元数据
├── scripts/
│   ├── run_all.sh            # 串行执行所有模型
│   ├── run_inference.py      # 统一推理脚本
│   └── run_compare.py        # 统一精度对比脚本
└── examples/
    └── quickstart.md          # 快速上手示例
```
