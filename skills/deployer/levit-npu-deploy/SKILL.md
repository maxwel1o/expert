---
name: levit-npu-deploy
description: >
  LeViT (LeViT-128/128S/192/256/384) 系列轻量级视觉 Transformer 模型在昇腾 NPU 上的
  完整部署、推理测试、CPU/NPU 精度对比、README 生成和模型仓库发布 Skill。
  可串行执行多个模型的 NPU 推理、精度对比、终端截图生成和 GitCode 发布全流程。
  当用户提到 LeViT NPU、LeViT 昇腾部署、levit 模型适配时触发。
metadata:
  short-description: LeViT 系列模型昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, levit, vit, image-classification, pytorch, inference]
---

# LeViT 系列模型昇腾 NPU 部署 Skill

本 Skill 提供 Facebook LeViT 系列轻量级视觉 Transformer 模型在华为昇腾 NPU 上的
标准化部署、推理验证、CPU/NPU 精度对比、README 生成和模型仓库发布全流程。

## 支持的模型列表

| 模型名称 | 模型仓库 | 隐藏维度 | 参数量 |
|----------|----------|---------|--------|
| LeViT-128 | [levit-128-npu](https://gitcode.com/gcw_C8PI9e90/levit-128-npu) | [128, 256, 384] | 36MB |
| LeViT-128S | [levit-128S-npu](https://gitcode.com/gcw_C8PI9e90/levit-128S-npu) | [128, 256, 384] | 30MB |
| LeViT-192 | [levit-192-npu](https://gitcode.com/gcw_C8PI9e90/levit-192-npu) | [192, 288, 384] | 43MB |
| LeViT-256 | [levit-256-npu](https://gitcode.com/gcw_C8PI9e90/levit-256-npu) | [256, 384, 512] | 73MB |
| LeViT-384 | [levit-384-npu](https://gitcode.com/gcw_C8PI9e90/levit-384-npu) | [384, 512, 768] | 150MB |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910 系列（至少 1 卡） |
| Python | 3.9 – 3.13 |
| torch_npu | >= 2.0 |
| PyTorch | >= 2.0.0 |
| Transformers | >= 4.30.0 |
| 网络 | 首次运行需联网下载模型权重 |

## 流程总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 环境准备 | 安装依赖、配置 token | CANN 环境, Python | pip install, export token | 可用 NPU 环境 | python -c "import torch; import torch_npu" | import 无报错 |
| NPU 推理 | 执行 NPU 推理 | 模型名称, 测试图片 | run_inference.py --device npu | 推理结果 JSON | ls output/*_npu_results.json | 输出文件存在且格式正确 |
| CPU 推理 | 执行 CPU 推理 | 模型名称, 测试图片 | run_inference.py --device cpu | 推理结果 JSON | ls output/*_cpu_results.json | 输出文件存在且格式正确 |
| 精度对比 | CPU/NPU 精度对比 | CPU/NPU 结果 JSON | compare.py | 对比报告 | python compare.py --model {name} | 概率 MaxAE < 1% |
| README 生成 | 生成部署文档 | 测试结果 | generate_readme.py | README.md | ls output/{model}/README.md | 包含精度对比和性能数据 |
| 截图生成 | 生成终端截图 | 推理输出 | generate_screenshot.py | PNG 截图 | ls output/{model}/screenshot.png | 截图清晰可读 |
| 仓库发布 | 创建并推送仓库 | 所有产物 | publish_repo.py | GitCode 仓库 | curl -I https://gitcode.com/{user}/{repo} | 仓库可见且文件完整 |

## Skill 输入参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models` | 要处理的模型列表（逗号分隔），可选 `128,128s,192,256,384` 或 `all` | `all` |
| `--image` | 测试图像路径 | `test.jpg` |
| `--skip-cpu` | 跳过 CPU 推理（仅运行 NPU） | `false` |
| `--skip-push` | 跳过模型仓库发布 | `false` |
| `--output-dir` | 输出目录 | `./output` |
| `--token` | GitCode API token（默认使用 ATOMGIT_USER_TOKEN 环境变量） | 环境变量 |

## Skill 输出结果

- 每个模型的推理结果 JSON（logits、概率、耗时）
- CPU vs NPU 精度对比报告（JSON + 终端输出）
- 模拟终端截图 PNG
- 模型仓库 README 文档
- 推送到 GitCode 的模型仓库

## 串行执行说明

为避免 NPU 显存溢出，多个模型必须串行执行：

```bash
# 执行所有模型（串行）
python3 scripts/batch_run.py --models all

# 执行指定模型（串行）
python3 scripts/batch_run.py --models 128,192
```

每个模型完成后自动释放 NPU 显存：
```python
import gc, torch
gc.collect()
if hasattr(torch, "npu"):
    torch.npu.empty_cache()
```

## 环境准备

| 项目 | 内容 |
|---|---|
| 输入 | 已安装 CANN 和 NPU 驱动的昇腾服务器 |
| 输出 | 就绪的 Python 环境，ATOMGIT_USER_TOKEN 已配置 |
| 关键验证 | `import torch; import torch_npu` 无报错 |
| 失败处理 | 安装失败则使用国内镜像重试；NPU 不可用则检查驱动 |

```bash
# 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch torchvision transformers Pillow numpy requests

# 配置 GitCode token
export ATOMGIT_USER_TOKEN="your_token_here"
```

**执行步骤**：
1. 确认 CANN 环境已加载，执行 `python -c "import torch; import torch_npu"` 验证 NPU 驱动正常
2. 执行 `pip install torch torchvision transformers Pillow numpy requests` 安装依赖
3. 配置 `ATOMGIT_USER_TOKEN` 环境变量用于 GitCode API 认证
4. 准备测试图片 `test.jpg` 用于后续推理验证

## NPU 推理

| 项目 | 内容 |
|---|---|
| 输入 | 模型名称 + 测试图片路径 |
| 输出 | 推理结果 JSON（logits、probabilities、time_ms） |
| 命令格式 | `run_inference.py --model {name} --device {cpu\|npu} --image {path}` |
| 失败处理 | OOM 则释放显存或切换到其他 NPU 卡 |

```bash
# 运行单个模型
python3 scripts/run_inference.py --model levit-128 --device npu --image test.jpg

# 串行运行所有模型
python3 scripts/batch_run.py --models all --device npu
```

**执行步骤**：
1. 选择要推理的模型（levit-128/128s/192/256/384），执行 `run_inference.py --model {name} --device npu --image test.jpg`
2. 检查输出文件 `output/{model}_npu_results.json` 中 logits、probabilities、time_ms 字段完整
3. 如运行多个模型，使用 `batch_run.py --models all` 串行执行避免 OOM

## CPU 推理

| 项目 | 内容 |
|---|---|
| 输入 | 模型名称 + 测试图片路径 |
| 输出 | 推理结果 JSON（logits、probabilities、time_ms） |
| 命令格式 | `run_inference.py --model {name} --device cpu --image {path}` |
| 失败处理 | OOM 则释放显存或切换到其他 NPU 卡 |

```bash
# 运行 CPU 推理
python3 scripts/run_inference.py --model levit-128 --device cpu --image test.jpg
```

**执行步骤**：
1. 执行 `run_inference.py --model {name} --device cpu --image test.jpg` 运行 CPU 推理
2. 检查输出文件 `output/{model}_cpu_results.json` 是否与 NPU 结果文件具有相同字段结构
3. 记录 CPU 推理耗时，与预期值对比验证无明显异常

## CPU/NPU 精度对比

| 项目 | 内容 |
|---|---|
| 输入 | CPU 推理结果 JSON + NPU 推理结果 JSON |
| 输出 | 精度对比报告（Logits MAE/MaxAE、Probs MaxAE、余弦相似度、Top-1 匹配） |
| 验收标准 | Probs MaxAE < 1% |
| 失败处理 | 误差超标则检查输入预处理和数据类型对齐 |

```bash
# 对比精度
python3 scripts/compare.py --model levit-128
```

输出示例：
```
Logits MAE:     0.00356148
Logits MaxAE:   0.01721168
Probs MaxAE:    0.246301%
余弦相似度:      0.99999537
Top-1 匹配:      YES
结论: PASS (误差 < 1%)
```

**执行步骤**：
1. 确认 CPU 和 NPU 结果文件均已生成后执行 `python3 scripts/compare.py --model {name}`
2. 检查 Probs MaxAE 是否 < 1%，确认结论为 PASS
3. 如误差超标，检查输入预处理（resize、crop、normalize）是否对齐
4. 记录对比结果到对比报告文件

## 生成 README

```bash
python3 scripts/generate_readme.py --model levit-128 --output ./output/levit-128
```

根据真实测试结果自动生成中文 README，包含：
- 模型介绍和原始地址
- CPU/NPU 精度对比表格
- 推理性能和加速比
- 模型标签

**执行步骤**：
1. 执行 `generate_readme.py --model {name} --output ./output/{model}` 生成中文 README
2. 检查生成的 README 文件是否包含精度对比表和推理性能数据
3. 确认模型标签和描述信息填写正确

## 生成终端截图

```bash
python3 scripts/generate_screenshot.py --model levit-128 --output screenshot.png
```

生成的截图包含：
- NPU 推理输出
- CPU/NPU 精度对比结果
- 仿真终端界面

**执行步骤**：
1. 执行 `generate_screenshot.py --model {name} --output screenshot.png` 生成终端截图
2. 检查生成的 PNG 文件是否包含推理命令与结果输出
3. 如截图生成失败，安装字体包后重试或手动截图替代

## 发布模型仓库

```bash
# 通过 GitCode API 创建并推送模型仓库
python3 scripts/publish_repo.py --model levit-128 --token ${ATOMGIT_USER_TOKEN}
```

或使用 `gitcode-publish` Skill 辅助完成。

**执行步骤**：
1. 确认 `ATOMGIT_USER_TOKEN` 环境变量已正确配置
2. 执行 `publish_repo.py --model {name} --token ${ATOMGIT_USER_TOKEN}` 创建并推送模型仓库
3. 访问 GitCode 仓库 URL 确认仓库可见、文件完整、README 渲染正常
4. 如有多个模型，重复上述步骤逐个发布

## 已知测试结果

所有模型均已通过 CPU/NPU 精度验证：

| 模型 | CPU 耗时 | NPU 耗时 | 加速比 | 概率误差 |
|------|---------|---------|--------|---------|
| LeViT-128 | 40.17ms | 9.24ms | 4.3× | 0.246% |
| LeViT-128S | 31.17ms | 7.30ms | 4.3× | 0.054% |
| LeViT-192 | 48.83ms | 9.76ms | 5.0× | 0.017% |
| LeViT-256 | 75.04ms | 9.78ms | 7.7× | 0.245% |
| LeViT-384 | 124.94ms | 9.42ms | 13.3× | 0.109% |

所有模型的 NPU 与 CPU 推理概率误差均 < 1%。

## 执行检查点与用户确认

在流程的关键节点需要用户确认状态正确后再继续执行。

| 步骤 | 检查点 | 预期结果 | 用户确认操作 |
|---|---|---|---|
| 1. 环境准备 | 安装依赖后执行 `python -c "import torch; import torch_npu"` | 无报错 | 确认 import 成功 |
| 2. NPU 推理 | 运行 `batch_run.py --models all` | 所有模型推理完成，输出 JSON 结果 | 确认输出文件存在 |
| 3. CPU 推理 | 运行 `run_inference.py --device cpu` | 输出 CPU 推理结果 JSON | 确认文件格式正确 |
| 4. 精度对比 | 运行 `compare.py` | 输出误差指标和 PASS/FAIL 结论 | 确认误差 < 1%，结论为 PASS |
| 5. README 生成 | 检查生成的 README 文件 | 包含精度对比表格和性能数据 | 确认内容完整 |
| 6. 截图生成 | 检查 PNG 截图 | 包含推理输出和对比结果 | 确认截图清晰可读 |
| 7. 仓库发布 | 访问 GitCode 仓库 URL | 仓库可见，文件完整 | 确认推送成功 |

## 异常处理与回滚策略

| 异常场景 | 触发条件 | 处理方式 | 回滚策略 |
|---|---|---|---|
| torch_npu 导入失败 | CANN 环境未加载或版本不匹配 | 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` | 检查 CANN 安装并重新加载环境 |
| NPU 显存不足 | 同时启动多个模型推理 | 使用 `batch_run.py` 串行执行，确保每个模型后释放显存 | 执行 `torch.npu.empty_cache()` 释放缓存 |
| 模型下载失败 | HuggingFace 连接超时 | 使用国内镜像 `--mirror hf-mirror.com` | 清除不完整缓存后重试 |
| 精度误差超标 | 概率 MaxAE >= 1% | 检查输入预处理和数据类型对齐 | 对齐预处理管道后重新推理 |
| 截图生成失败 | 缺少字体或 PIL 依赖 | 安装 `fonts-noto-cjk` 或使用后备字体 | 跳过截图步骤，手动截图 |
| GitCode 发布失败 | Token 无效或无权限 | 检查 `ATOMGIT_USER_TOKEN` 环境变量 | 重新生成 Token 后重试 |
| 权重加载异常 | 模型结构与 checkpoint 不匹配 | 检查 PyTorch 和 transformers 版本兼容性 | 使用 `strict=False` 加载 |

## 资源与评测产物

| 类别 | 资源/产物 | 说明 | 路径 |
|---|---|---|---|
| 模型权重 | LeViT-128/128S/192/256/384 | HuggingFace 预训练权重 | `~/.cache/huggingface/hub/` |
| 测试数据 | 测试图片 | 用于推理验证 | `test.jpg` |
| 推理结果 | CPU/NPU 推理 JSON | logits、probabilities、推理耗时 | `output/{model}_{device}_results.json` |
| 精度对比报告 | 对比 JSON | MAE、MaxAE、余弦相似度、Top-1 匹配 | `output/{model}_comparison.json` |
| README | GitCode 仓库 README | 中文部署文档 | `output/{model}/README.md` |
| 终端截图 | PNG | 推理结果模拟终端截图 | `output/{model}/screenshot.png` |
| 打包仓库 | Git 仓库目录 | 包含所有发布文件的本地仓库 | `output/{model}/repo/` |

## 故障排除

| 问题 | 解决方法 |
|------|---------|
| NPU 显存不足 | 减小 batch_size；确保前一个模型已释放显存 |
| 模型下载失败 | 使用代理：将 huggingface.co 替换为 githubfast.com |
| 仓库推送失败 | 检查 ATOMGIT_USER_TOKEN 是否正确；确认仓库不存在或使用 force push |
| 截图字体问题 | 使用 `PIL.ImageFont.load_default()` 作为后备字体 |
