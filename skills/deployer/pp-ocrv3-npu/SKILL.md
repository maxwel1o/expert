---
name: pp-ocrv3-npu
description: PP-OCRv3 多语言 OCR 模型昇腾 NPU 批量部署与验证
---

# PP-OCRv3 NPU Deployment Skill

## 功能

自动完成 PP-OCRv3 多语言 OCR 模型在昇腾 NPU 上的批量部署、推理、CPU/NPU 精度对比、README 生成和模型仓库发布。

## 支持的模型

共 **16** 个 PP-OCRv3 模型，涵盖:

**文本检测模型 (4个):**
| 模型名称 | 仓库地址 |
| --- | --- |
| Multilingual_PP-OCRv3_det_infer | https://gitcode.com/gcw_C8PI9e90/Multilingual_PP-OCRv3_det_infer-npu |
| ch_PP-OCRv3_det_infer | https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv3_det_infer-npu |
| en_PP-OCRv3_det_infer | https://gitcode.com/gcw_C8PI9e90/en_PP-OCRv3_det_infer-npu |
| en_PP-OCRv3_det_infer_somohk | https://gitcode.com/gcw_C8PI9e90/en_PP-OCRv3_det_infer_somohk-npu |

**文本识别模型 (12个):**
| 模型名称 | 仓库地址 |
| --- | --- |
| arabic_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/arabic_PP-OCRv3_rec_infer-npu |
| ch_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv3_rec_infer-npu |
| chinese_cht_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/chinese_cht_PP-OCRv3_rec_infer-npu |
| cyrillic_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/cyrillic_PP-OCRv3_rec_infer-npu |
| devanagari_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/devanagari_PP-OCRv3_rec_infer-npu |
| en_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/en_PP-OCRv3_rec_infer-npu |
| japan_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/japan_PP-OCRv3_rec_infer-npu |
| ka_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/ka_PP-OCRv3_rec_infer-npu |
| korean_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/korean_PP-OCRv3_rec_infer-npu |
| latin_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/latin_PP-OCRv3_rec_infer-npu |
| ta_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/ta_PP-OCRv3_rec_infer-npu |
| te_PP-OCRv3_rec_infer | https://gitcode.com/gcw_C8PI9e90/te_PP-OCRv3_rec_infer-npu |

## 输入参数

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `models` | array | 是 | 模型配置列表，每个模型需指定 model_id, task, model_file, has_dict 等 |
| `test_image` | string | 是 | 测试图像路径 (检测模型用) |
| `test_rec_image` | string | 是 | 测试文本图像路径 (识别模型用) |
| `cache_dir` | string | 否 | 模型缓存目录，默认 `/opt/atomgit/ocr_models` |
| `output_dir` | string | 否 | 输出目录，默认 `/opt/atomgit/ocr_npu_adapt` |
| `push_repos` | boolean | 否 | 是否推送模型仓库，默认 `true` |

## 输出结果

每个模型独立输出:
- `compare_result.json` — CPU/NPU 精度对比结果
- `readme.md` — 中文 README 文档，含真实测试数据
- GitCode 模型仓库地址

## 完整工作流程

### 工作流阶段总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
| --- | --- | --- | --- | --- | --- | --- |
| 环境准备 | 1. 环境准备 | Python 3.10+, 昇腾910B NPU, CANN已安装 | 安装 onnxruntime-cann 等依赖，验证 NPU 状态 | Python 虚拟环境激活，CANN 提供者可用 | `npu-smi info`; `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` | CANNExecutionProvider 在运行时提供者列表中 |
| 模型下载 | 2. 下载模型 | `cache_dir` 路径，≥10GB磁盘空间，网络连通 | 运行 download_models.py 下载全部16个模型权重 | 16个模型权重文件下载到 `cache_dir` | `ls <cache_dir>/*/model.onnx` 或等価命令 | 所有 `.onnx` 文件存在且 size > 0 |
| NPU推理 | 3. NPU推理 | 检测/识别模型路径，test_image/test_rec_image | 在 CANNExecutionProvider 上运行 NPU 推理 | 检测边界框坐标+置信度；识别文本+置信度 | 检查输出日志和 `compare_result.json` | 日志无错误，结果文件写入成功 |
| CPU/NPU对比 | 4. CPU/NPU精度对比 | 同一模型在 CPU 和 NPU 上的推理输出 | 运行 compare_cpu_npu.py 计算逐模型差异 | `<model_dir>/compare_result.json` — 差异量化报告 | `cat <model_dir>/compare_result.json` | 精度误差 < 1%，检测框匹配率 ≥ 90% |
| 生成README | 5. 生成README | 每个模型的 `compare_result.json` | 运行 generate_readmes.py 生成中文文档 | `<model_dir>/readme.md` — 完整中文 README | 检查每个模型目录下的 readme.md | 内容完整，含模型信息/测试结果/示例命令 |
| 提交仓库 | 6. 提交模型仓库 | ATOMGIT_USER_TOKEN，模型目录下的 README 和权重文件 | 使用 GitCode API 创建模型仓库并推送代码 | GitCode 上每个模型的 NPU 适配仓库 | 确认 GitCode 仓库存在且包含文件 | 每个 `<model_name>-npu` 仓库已创建并包含 README 和模型文件 |

### 1. 环境准备

**执行步骤:**

1. 检查 Python 版本: 执行 `python --version` 确认版本 ≥ 3.10
2. 验证 NPU 驱动状态: 执行 `npu-smi info` 确认昇腾 910B 芯片可用且输出正常
3. 安装依赖包: 执行 pip 安装 onnxruntime-cann、numpy、opencv-python-headless、onnx、modelscope、paddle2onnx
4. 验证 CANN ExecutionProvider: 执行 `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` 确认输出包含 `CANNExecutionProvider`
5. 确认虚拟环境已激活: 执行 `which python` 确认使用正确的 Python 解释器

| 项目 | 内容 |
| --- | --- |
| 输入 | Python 3.10+、昇腾 910B NPU、CANN 已安装、`npu-smi info` 可用 |
| 输出 | Python 虚拟环境激活，onnxruntime-cann 安装完成 |
| 命令 | |

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  onnxruntime-cann numpy opencv-python-headless onnx modelscope paddle2onnx
```

| 验证方法 | `npu-smi info` 确认 NPU 状态；`python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` 确认包含 CANN 提供者 |

### 2. 下载模型

**执行步骤:**

1. 确认磁盘空间: 执行 `df -h /opt/atomgit/` 确保可用空间 ≥ 10GB
2. 确认网络连通: 执行 `curl -I https://gitcode.com` 确认可访问 GitCode 和 modelscope
3. 运行下载脚本: 执行 `python scripts/download_models.py --cache_dir /opt/atomgit/ocr_models`
4. 验证下载完整性: 检查 `cache_dir` 下每个模型目录是否包含 `.onnx` 文件且文件大小大于 0
5. 记录失败模型: 如果部分模型下载失败，记录到 `failed_models.txt` 并标记跳过

| 项目 | 内容 |
| --- | --- |
| 输入 | `cache_dir` 路径（默认 `/opt/atomgit/ocr_models`），需 ≥ 10GB 磁盘空间 |
| 输出 | 16 个模型权重文件下载到 `cache_dir`（4 检测 + 12 识别） |
| 命令 | |

```bash
python scripts/download_models.py --cache_dir /opt/atomgit/ocr_models
```

| 验证方法 | 检查 `cache_dir` 下每个模型目录是否包含 `.onnx` 文件且文件大小 > 0 |

### 3. NPU 推理

**执行步骤:**

1. 确认 NPU 显存充足: 执行 `npu-smi info` 查看显存使用率，确保空闲显存 > 95%
2. 单模型推理验证: 执行 `python scripts/inference.py --model_path <model.onnx> --provider CANNExecutionProvider --image test.png` 验证单个模型推理正常
3. 批量推理全部模型: 执行 `python scripts/process_all_models.py` 串行处理全部 16 个模型
4. 清理 NPU 显存: 每个模型推理完成后调用 `gc.collect()` 和 `torch.npu.empty_cache()` 防止 OOM
5. 检查推理结果: 确认所有模型输出日志无错误，推理结果已写入 `compare_result.json`

| 项目 | 内容 |
| --- | --- |
| 输入 | 检测模型: `test_image` 路径；识别模型: `test_rec_image` 路径；模型配置列表 `models` |
| 输出 | 检测模型输出边界框坐标 + 置信度；识别模型输出文本 + 置信度 |
| 命令 | |

```bash
# 单模型推理
python scripts/inference.py --model_path <model.onnx> --provider CANNExecutionProvider --image test.png

# 批量处理（全部 16 个模型，串行执行）
python scripts/process_all_models.py
```

| 验证方法 | 检查输出日志无错误、NPU 推理结果写入 `compare_result.json` |

### 4. CPU/NPU 精度对比

**执行步骤:**

1. 准备 CPU baseline: 确保同一模型在 CPUExecutionProvider 下的推理结果已保存为参考基准
2. 运行精度对比: 执行 `cd <model_dir> && python compare_cpu_npu.py` 计算 CPU 与 NPU 推理结果的逐元素差异
3. 查看量化报告: 执行 `cat <model_dir>/compare_result.json` 读取差异量化统计
4. 验证精度达标: 确认所有模型相对差异 < 1%，检测模型框匹配率 ≥ 90%
5. 标记不达标项: 对精度不达标的模型标记 `status: "warn"` 并记录差异详情，不影响整体流程但需人工复核

| 项目 | 内容 |
| --- | --- |
| 输入 | 同一模型在 CPU（CPUExecutionProvider）和 NPU（CANNExecutionProvider）上的推理输出 |
| 输出 | `<model_dir>/compare_result.json` — 逐模型差异量化报告 |
| 命令 | |

```bash
# 单模型对比
cd <model_dir> && python compare_cpu_npu.py

# 查看结果
cat <model_dir>/compare_result.json
```

| 验证方法 | 确认所有模型精度误差 < 1%；检测模型框匹配率 ≥ 90% |

### 5. 生成 README

**执行步骤:**

1. 确认输入就绪: 检查每个模型的 `compare_result.json` 已生成且包含完整 CPU/NPU 对比数据
2. 运行生成脚本: 执行 `python scripts/generate_readmes.py` 自动生成中文 README 文档
3. 验证文档完整性: 检查每个模型目录下的 `readme.md` 是否包含模型信息、精度对比结果、测试截图和用法示例
4. 抽样复核: 随机选取 2-3 个模型目录，核对 README 中的测试数据与 `compare_result.json` 的实际输出一致

| 项目 | 内容 |
| --- | --- |
| 输入 | 每个模型的 `compare_result.json`（含 CPU/NPU 对比数据和测试图片） |
| 输出 | `<model_dir>/readme.md` — 中文 README 文档，含真实测试数据和示例命令 |
| 命令 | |

```bash
python scripts/generate_readmes.py
```

| 验证方法 | 检查每个模型目录下的 `readme.md` 内容完整，含模型信息、测试结果和用法示例 |

### 6. 提交模型仓库

**执行步骤:**

1. 确认令牌可用: 执行 `echo ${ATOMGIT_USER_TOKEN:0:4}****` 确认 `ATOMGIT_USER_TOKEN` 环境变量已设置
2. 检查仓库名冲突: 通过 GitCode API 查询仓库名是否存在，避免与现有仓库冲突
3. 创建模型仓库: 执行 curl 命令调用 GitCode API 创建每个模型的 NPU 适配仓库
4. 初始化并推送: 进入模型目录，执行 `git init && git remote add origin && git push` 推送模型权重和 README
5. 验证推送结果: 确认 GitCode 上每个 `<model_name>-npu` 仓库已成功创建且包含 README 和模型权重文件

| 项目 | 内容 |
| --- | --- |
| 输入 | `ATOMGIT_USER_TOKEN` 环境变量（GitCode 个人访问令牌）；模型目录下的 README 和权重文件 |
| 输出 | GitCode 上每个模型的 NPU 适配仓库 |
| 命令 | |

```bash
# 使用 GitCode API 创建模型仓库
curl --header "private-token: ${ATOMGIT_USER_TOKEN}" \
  -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -d '{"name":"<model_name>-npu","repository_type":"model","default_branch":"main"}'

# 推送代码
git init && git branch -M main
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/<username>/<model_name>-npu.git
git push -u origin main
```

| 验证方法 | 确认 GitCode 上每个 `<model_name>-npu` 仓库已创建且包含 README 和模型文件 |

## 执行检查点与用户确认

在执行以下关键步骤前，需要用户确认：

| 步骤 | 确认项 | 确认方式 |
| --- | --- | --- |
| 环境准备 | 确认 Python 3.10+、昇腾驱动 (CANN) 已安装、`npu-smi info` 可正常输出 | `read -p "CANN 环境是否就绪? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| 模型下载 | 确认 `cache_dir` 磁盘空间 ≥ 10GB，网络可访问 GitCode/modelscope | `read -p "继续下载模型? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| NPU 推理 | 确认 NPU 显存充足（`npu-smi info` 查看），无其他训练任务占用 | `read -p "NPU 显存是否充足? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| CPU/NPU 对比 | 确认已获取 CPU 环境下的 baseline 结果文件 | `read -p "CPU baseline 是否已生成? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |
| 推送模型仓库 | 确认 `ATOMGIT_USER_TOKEN` 环境变量已设置、仓库名不与现有仓库冲突 | `read -p "确认推送模型仓库? [y/N] " -r; if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi` |

## 异常处理与回滚策略

| 异常场景 | 检测方式 | 处理措施 |
| --- | --- | --- |
| CANN / onnxruntime-cann 未安装 | 运行 `import onnxruntime` 时 `CANNExecutionProvider` 不可用 | 打印错误 `请安装 onnxruntime-cann`；跳过 NPU 推理，回退到 CPU-only 模式 |
| 模型文件下载失败 | 模型文件不存在或大小为 0 | 重试 3 次；如果仍失败则跳过该模型并记录到 `failed_models.txt` |
| NPU 显存不足 (OOM) | `npu-smi info` 显存使用率 > 95% 或推理进程被 kill | `gc.collect(); torch.npu.empty_cache()`；等待 30s 后重试；重试 2 次仍失败则跳过该模型 |
| 推理结果异常 | 输出张量包含 NaN / Inf | 标记该模型为 `quality_failed`；记录日志并继续处理下一个模型 |
| 精度对比不达标 | 相对差异 ≥ 1% 或框匹配率 < 90% | 在 `compare_result.json` 中标记 `status: "warn"`；不影响整体流程但需人工复核 |
| GitCode API 限流 | HTTP 429 响应 | 退避等待 60s 后重试；重试 3 次仍失败输出 `push_failed` 但保留本地结果 |

## 资源与评测产物

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 精度对比结果 | `<model_dir>/compare_result.json` | CPU vs NPU 逐模型精度差异量化报告 |
| README | `<model_dir>/readme.md` | 完整中文文档，含实际测试数据和示例命令 |
| 失败记录 | `<model_dir>/failed_models.txt` | 所有失败模型及原因汇总 |
| 测试提示词 | `skills/deployment/pp-ocrv3-npu/test-prompts.json` | 可用于复现验证的测试输入 |
| GitCode 仓库 URL | — | 每个成功推送的模型仓库链接 |

## 串行执行注意事项

- 每个模型处理完成后释放 NPU 显存
- 使用 `gc.collect()` + `torch.npu.empty_cache()` 清理资源
- 不要并行运行多个模型，防止 NPU 显存溢出
- 如果某个模型失败，记录失败原因，继续处理后续模型

## 精度要求

- NPU 与 CPU 推理结果误差 < 1%
- 检测模型: 框匹配率 ≥ 90%，输出概率图相对差异 < 1%
- 识别模型: 文本完全一致，输出 logits 相对差异 < 1%

## 项目结构

```
skills/deployment/pp-ocrv3-npu/
├── skill.json                     # Skill 元数据
├── SKILL.md                       # 本文档
├── scripts/
│   ├── process_all_models.py      # 批量处理主脚本
│   ├── generate_readmes.py        # README 生成脚本
│   ├── ocr_utils.py               # 通用 OCR 预处理/后处理
│   └── model_configs.py           # 模型配置
└── examples/
    └── usage_example.sh           # 使用示例
```
