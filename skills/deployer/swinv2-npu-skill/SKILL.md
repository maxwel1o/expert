---
name: swinv2-npu-skill
description: SwinV2 Transformer models in 昇腾 NPU deployment covering environment setup, model inference, CPU/NPU precision comparison, README generation, screenshot generation, GitCode model repo publishing, and batch processing for all 15 SwinV2 variants from the timm model zoo.
keywords:
  - swinv2
  - npu
  - ascend
  - image-classification
  - timm
  - pytorch
  - transformer
---

# SwinV2 昇腾 NPU 部署 Skill

本 Skill 提供 SwinV2 (timm) 图像分类模型在华为昇腾 Ascend 910 NPU 上的完整部署、推理验证、精度对比、文档生成和 GitCode 模型仓库发布流程。覆盖所有 15 个 SwinV2 变体（Tiny/Small/Base/Large），支持单模型和批量两种运行模式。

## 入口参数

| 参数名 | 含义 | 取值约束 |
|--------|------|---------|
| model_name | 模型名称 | 必须为 supported_models 列表中的值 |
| test_image | 测试图片路径或 URL | 可选，默认使用 PyTorch hub dog.jpg |
| modelscope_cache_dir | ModelScope 缓存目录 | 可选，默认 ./modelscope_cache |
| batch_mode | 批量模式开关 | 可选，默认 false，开启后串行处理全部模型 |

## 工作流

1. 环境初始化与依赖安装

   加载 CANN 环境，安装 torch_npu 及依赖包。

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   npu-smi info
   export ASCEND_RT_VISIBLE_DEVICES=0
   pip install torch torch_npu timm Pillow numpy safetensors modelscope \
     -i https://pypi.tuna.tsinghua.edu.cn/simple
   python3 -c "import torch; print(torch.npu.is_available())"
   ```

   确认：`npu-smi info` 显示 NPU 状态正常，`torch.npu.is_available()` 返回 True。

2. 模型权重下载

   从 ModelScope 下载 SwinV2 预训练权重到本地缓存目录。

   ```bash
   export MODEL_NAME=swinv2_tiny_window8_256.ms_in1k
   export MODELSCOPE_DIR=./modelscope_cache
   python3 scripts/inference.py --download-only --model $MODEL_NAME --cache-dir $MODELSCOPE_DIR
   ```

   确认：`$MODELSCOPE_DIR/timm/` 下包含 `model.safetensors` 或 `pytorch_model.bin` 文件。

3. CPU 推理

   加载模型并在 CPU 上运行推理，保存 top-5 预测结果。

   ```bash
   export MODEL_NAME=swinv2_tiny_window8_256.ms_in1k
   export TEST_IMAGE=/path/to/test/image.jpg
   export MODELSCOPE_DIR=./modelscope_cache
   python3 scripts/inference.py
   ```

   确认：`results/inference_results.json` 包含 cpu 推理的 logits、top-5 索引和概率。

4. NPU 推理

   在昇腾 Ascend 910 NPU 上运行推理，记录 NPU 侧 top-5 预测结果和推理耗时。

   ```bash
   python3 scripts/inference.py
   ```

   确认：`results/inference_results.json` 包含 npu 推理结果，NPU 推理时间明显快于 CPU。

5. CPU/NPU 精度对比验证

   对比 CPU 和 NPU 推理结果，计算余弦相似度、最大绝对误差等关键指标。

   ```bash
   export MODEL_NAME=swinv2_tiny_window8_256.ms_in1k
   python3 scripts/compare_cpu_npu.py
   ```

   指标包括 Cosine Similarity、Max Absolute Error、Mean Absolute Error、Top-5 Agreement。

   确认：余弦相似度 > 0.999 且 Top-5 一致数 >= 4，整体判定为 PASS。

6. README 文档与截图生成

   为每个模型生成带真实推理结果的中文 README 和终端截图。

   ```bash
   python3 scripts/generate_readmes.py
   python3 scripts/generate_screenshots.py
   ```

   确认：每个模型目录下生成 readme.md 和 terminal_screenshot.png 文件。

7. GitCode 模型仓库发布

   自动创建并推送模型仓库到 GitCode，包含推理脚本、README、截图和结果。

   ```bash
   export ATOMGIT_USER_TOKEN=your_token_here
   python3 scripts/gitcode_push.py
   ```

   确认：GitCode 上出现对应模型仓库，所有文件完整可访问。

8. 批量处理与验收确认

   串行处理全部 15 个 SwinV2 模型，逐一完成下载、推理、对比、截图和发布。

   ```bash
   python3 scripts/batch_runner.py
   python3 scripts/generate_readmes.py
   python3 scripts/generate_screenshots.py
   python3 scripts/gitcode_push.py
   ```

   确认：15 个模型全部通过精度验收，cosine similarity 均 > 0.999。

## 检查点

| # | 检查点 | 确认方法 | 异常恢复 / Error Recovery |
|---|--------|---------|--------------------------|
| 1 | 环境就绪确认 | `npu-smi info` 显示 NPU 正常 | 如果 CANN 未加载则执行 source set_env.sh |
| 2 | 依赖安装确认 | `python3 -c "import torch_npu"` 无错误 | 如果报错则检查 pip list，确认 torch_npu 版本匹配 |
| 3 | 模型权重确认 | 缓存目录下存在 safetensors 文件 | 如果文件缺失则重新下载或手动复制到缓存 |
| 4 | CPU 推理确认 | top-5 概率合理且分布正常 | 如果结果异常则检查 test_image 路径和格式 |
| 5 | NPU 推理确认 | 推理结果与 CPU 高度接近 | 如果 NPU OOM 则调用 torch.npu.empty_cache() 后重试 retry |
| 6 | 精度对比确认 | cosine similarity > 0.999 | 如果低于阈值则检查权重加载和 state_dict 完整度 |
| 7 | 仓库发布确认 | GitCode 仓库文件完整可访问 | 如果 403 则检查 token 权限后重新推送 |

## 异常处理

| 异常场景 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `No module named 'torch_npu'` | torch_npu 未安装或 CANN 未加载 | source set_env.sh 后 pip install torch_npu |
| `npu-smi: command not found` | CANN 或驱动未正确安装 | 确认 CANN 安装路径后重新执行 source set_env.sh |
| `torch.npu.is_available()=False` | 驱动问题或 NPU 卡被占用 | npu-smi info 检查卡状态，切换到空闲卡 |
| 模型下载超时 time out | 网络不稳定或代理配置错误 | 设置镜像源 retry 重试，或手动下载到 modelscope_cache |
| NPU OOM RuntimeError | 显存不足或上一轮未释放 | 调用 torch.npu.empty_cache() + gc.collect() 后 retry |
| 模型权重 key 不匹配 | 缓存损坏或模型版本冲突 | 清除 modelscope_cache 后重新下载 |
| 余弦相似度 < 0.999 | 权重加载不完整或数值溢出 | 检查 filtered keys 数量占比，确认加载全部权重 |
| README 生成失败 | 缺少 inference_results.json | 先运行 inference.py，确认 results/ 目录存在 |
| GitCode 推送失败 403 | token 过期或权限不足 | 重新生成 ATOMGIT_USER_TOKEN，确保有写入权限 |
| 截图生成空白 | PIL 渲染字体缺失 | fallback：安装中文字体 fonts-wqy-zenhei |
| batch_runner 中断 | NPU OOM 或脚本异常 crash | 重新运行即可 retry，已完成的模型自动 skip（断点续跑 checkpoint） |
| conda 环境未正确激活 | python3 或 pip 命令找不到 | 执行 conda activate npu_env 后 retry |

## USAGE 约束

1. 硬件约束：必须在 Ascend 910 系列 NPU 上运行，不兼容 GPU 或纯 CPU 环境
2. 串行处理：多模型推理必须串行执行，不允许并行启动以防止 NPU OOM
3. 环境隔离：推荐使用独立 conda 环境，避免 CANN/torch_npu 版本冲突
4. 网络要求：首次运行需联网下载模型权重（每个模型约 100MB-800MB）
5. 磁盘空间：ModelScope 缓存加结果输出至少需要 20GB 可用空间
6. 模型一致性：model_name 必须严格匹配 supported_models 列表中的值
7. Token 安全：ATOMGIT_USER_TOKEN 禁止写入日志或硬编码到脚本中
8. 断点续跑：批处理中断后重新执行，已完成的模型自动跳过

## 资源

| 类型 | 路径 | 说明 |
|------|------|------|
| 脚本 | `scripts/` | 核心脚本，含 inference.py、compare_cpu_npu.py 等 6 个脚本 |
| 结果 | `results.tsv` | 推理结果和精度对比汇总表格 |
| 评测 | `evals.json` | 技能评测结果记录文件 |
| 参考 | `references/` | 技术参考文档目录（可扩展） |
| 示例 | `examples/` | 示例脚本 run_single_model.sh |
| 缓存 | `modelscope_cache/` | 模型权重缓存目录 |
| 模板 | `templates/` | README 和截图模板目录（可扩展） |

## 验证与测试

```bash
# 环境验证
python3 -c "
import torch
print('NPU available:', torch.npu.is_available())
print('NPU device:', torch.npu.get_device_name(0))
print('torch version:', torch.__version__)
"

# benchmark 单模型推理测试
python3 scripts/inference.py

# 精度对比验证
python3 scripts/compare_cpu_npu.py

# eval 测试验证
# test-prompts.json 中定义了测试用例，参考执行
```

## 精度结果汇总

| 指标 | 范围 |
|------|------|
| Cosine Similarity | 0.99998580 - 0.99999848 |
| Max Absolute Error | 0.006 - 0.021 |
| Top-5 Agreement | 5/5 (100%) |
| NPU Speedup | 18x - 96x over CPU |

## 支持模型列表

| # | 模型 | 参数量 | 输入尺寸 |
|---|------|--------|---------|
| 1 | swinv2_tiny_window8_256.ms_in1k | 28M | 256x256 |
| 2 | swinv2_tiny_window16_256.ms_in1k | 28M | 256x256 |
| 3 | swinv2_small_window8_256.ms_in1k | 50M | 256x256 |
| 4 | swinv2_small_window16_256.ms_in1k | 50M | 256x256 |
| 5 | swinv2_large_window12to24_192to384.ms_in22k_ft_in1k | 197M | 192x192→384x384 |
| 6 | swinv2_large_window12to16_192to256.ms_in22k_ft_in1k | 197M | 192x192→256x256 |
| 7 | swinv2_large_window12_192.ms_in22k | 197M | 192x192 |
| 8 | swinv2_cr_tiny_ns_224.sw_in1k | 28M | 224x224 |
| 9 | swinv2_cr_small_ns_224.sw_in1k | 50M | 224x224 |
| 10 | swinv2_cr_small_224.sw_in1k | 50M | 224x224 |
| 11 | swinv2_base_window8_256.ms_in1k | 88M | 256x256 |
| 12 | swinv2_base_window16_256.ms_in1k | 88M | 256x256 |
| 13 | swinv2_base_window12to24_192to384.ms_in22k_ft_in1k | 88M | 192x192→384x384 |
| 14 | swinv2_base_window12to16_192to256.ms_in22k_ft_in1k | 88M | 192x192→256x256 |
| 15 | swinv2_base_window12_192.ms_in22k | 88M | 192x192 |

## 文件结构

```
swinv2-npu-skill/
├── SKILL.md              # 本文档
├── skill.json            # 技能元数据
├── test-prompts.json     # 测试提示词
├── scripts/
│   ├── inference.py        # 推理脚本
│   ├── compare_cpu_npu.py  # 精度对比
│   ├── generate_readmes.py # README 生成
│   ├── generate_screenshots.py  # 截图生成
│   ├── batch_runner.py     # 批量处理
│   └── gitcode_push.py     # 仓库发布
├── examples/
│   └── run_single_model.sh # 单模型示例
├── results/               # 结果输出目录
├── references/            # 参考文档目录
└── evals.json             # 评测记录
```
