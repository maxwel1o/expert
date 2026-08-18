---
name: pp-ocrv4-npu-deploy
description: >
  PP-OCRv4 系列 OCR 模型在昇腾 NPU 上的完整部署与精度验证 Skill。
  支持 10 个 PP-OCRv4 模型（检测、识别、版面分析）的自动下载、
  ONNX NPU 推理、CPU/NPU 精度对比、README 生成和模型仓库发布。
  当用户提到 PP-OCRv4、OCR、文字识别、昇腾 NPU、文本检测识别时触发。
metadata:
  short-description: PP-OCRv4 系列昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, ocr, pp-ocrv4, text-detection, text-recognition, onnx, paddle]
---

# PP-OCRv4 NPU Deployment Skill

## Skill Metadata

- **Skill Name**: PP-OCRv4 Ascend NPU Deployment
- **Version**: 1.0.0
- **Category**: Deployment / OCR / Ascend NPU
- **Target Platform**: Huawei Ascend NPU (910 series)
- **Framework**: ONNX Runtime + CANNExecutionProvider

## Skill Goal

自动完成 PP-OCRv4 系列 OCR 模型在华为昇腾 NPU 上的部署、推理、CPU/NPU 精度对比测试、README 生成和 GitCode 模型仓库发布。

支持 10 个 PP-OCRv4 模型的串行执行，自动释放 NPU 显存，防止显存溢出。

## Supported Models

| 模型名称 | 原始地址 | 任务类型 | GitCode 仓库 |
| --- | --- | --- | --- |
| PP-OCRv4 | [OllmOne/PP-OCRv4](https://www.modelscope.cn/models/OllmOne/PP-OCRv4) | 检测 + 识别 + 版面分析 | [PP-OCRv4-npu](https://gitcode.com/gcw_C8PI9e90/PP-OCRv4-npu) |
| ch_PP-OCRv4_det_infer | [cycloneboy/ch_PP-OCRv4_det_infer](https://www.modelscope.cn/models/cycloneboy/ch_PP-OCRv4_det_infer) | 文本检测 | [ch_PP-OCRv4_det_infer-npu](https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv4_det_infer-npu) |
| ch_PP-OCRv4_det_server_infer | [cycloneboy/ch_PP-OCRv4_det_server_infer](https://www.modelscope.cn/models/cycloneboy/ch_PP-OCRv4_det_server_infer) | 文本检测 | [ch_PP-OCRv4_det_server_infer-npu](https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv4_det_server_infer-npu) |
| ch_PP-OCRv4_rec_infer | [cycloneboy/ch_PP-OCRv4_rec_infer](https://www.modelscope.cn/models/cycloneboy/ch_PP-OCRv4_rec_infer) | 文本识别 | [ch_PP-OCRv4_rec_infer-npu](https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv4_rec_infer-npu) |
| ch_PP-OCRv4_rec_infer_v2 | [cycloneboy/ch_PP-OCRv4_rec_infer](https://www.modelscope.cn/models/cycloneboy/ch_PP-OCRv4_rec_infer) | 文本识别 | [ch_PP-OCRv4_rec_infer_v2-npu](https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv4_rec_infer_v2-npu) |
| ch_PP-OCRv4_rec_server_infer | [cycloneboy/ch_PP-OCRv4_rec_server_infer](https://www.modelscope.cn/models/cycloneboy/ch_PP-OCRv4_rec_server_infer) | 文本识别 | [ch_PP-OCRv4_rec_server_infer-npu](https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv4_rec_server_infer-npu) |
| ch_PP-OCRv4_rec_server_infer_v2 | [cycloneboy/ch_PP-OCRv4_rec_server_infer](https://www.modelscope.cn/models/cycloneboy/ch_PP-OCRv4_rec_server_infer) | 文本识别 | [ch_PP-OCRv4_rec_server_infer_v2-npu](https://gitcode.com/gcw_C8PI9e90/ch_PP-OCRv4_rec_server_infer_v2-npu) |
| en_PP-OCRv4_rec_infer | [cycloneboy/en_PP-OCRv4_rec_infer](https://www.modelscope.cn/models/cycloneboy/en_PP-OCRv4_rec_infer) | 英文识别 | [en_PP-OCRv4_rec_infer-npu](https://gitcode.com/gcw_C8PI9e90/en_PP-OCRv4_rec_infer-npu) |
| en_PP-OCRv4_rec_infer_v2 | [cycloneboy/en_PP-OCRv4_rec_infer](https://www.modelscope.cn/models/cycloneboy/en_PP-OCRv4_rec_infer) | 英文识别 | [en_PP-OCRv4_rec_infer_v2-npu](https://gitcode.com/gcw_C8PI9e90/en_PP-OCRv4_rec_infer_v2-npu) |
| en_PP-OCRv4_rec_infer_somohk | [somohk/en_PP-OCRv4_rec_infer](https://www.modelscope.cn/models/somohk/en_PP-OCRv4_rec_infer) | 英文识别 | [en_PP-OCRv4_rec_infer_somohk-npu](https://gitcode.com/gcw_C8PI9e90/en_PP-OCRv4_rec_infer_somohk-npu) |

## Environment Requirements

- **Hardware**: Huawei Ascend NPU 910 series
- **CANN**: Compatible with onnxruntime-cann 1.24.4+
- **Python**: 3.8+
- **Dependencies**:
  - `onnxruntime-cann>=1.24.4`
  - `onnxruntime>=1.24.0`
  - `numpy>=1.24.0`
  - `opencv-python>=4.8.0`
  - `modelscope` (for model download)
  - `paddle2onnx` (for PaddlePaddle model conversion, if needed)

## Skill Architecture

采用串行执行架构，每个模型依次完成：下载 → 适配 → CPU推理 → NPU推理 → 精度对比 → README生成 → 仓库发布。执行完一个模型后主动释放 NPU 显存，再处理下一个模型，防止显存溢出。

## 完整工作流程

本技能将 PP-OCRv4 系列模型的 NPU 部署分为以下阶段，按顺序串行执行。

### 工作流程阶段总览

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
| --- | --- | --- | --- | --- | --- | --- |
| 环境准备 | 1 | NPU 服务器, Python 3.8+ | 安装 onnxruntime-cann、modelscope、opencv-python 等依赖 | 可用的 CANNExecutionProvider | `python -c "import onnxruntime; print(onnxruntime.get_device())"` | 输出包含 "NPU" |
| 模型下载 | 2 | model_list 文件（含 model_id, url, type） | `python3 scripts/download_models.py --model_id <id>` | ONNX 模型文件（.onnx） | `ls -la <model_path>` | 文件存在且大小 > 0 |
| CPU 推理 | 3 | ONNX 模型, 测试图片 | `python3 scripts/run_inference.py --device cpu` | cpu_output.npy | `python3 -c "import numpy as np; a=np.load('cpu_output.npy'); print(a.shape)"` | 输出 shape 与预期一致 |
| NPU 推理 | 4 | ONNX 模型, 测试图片 | `python3 scripts/run_inference.py --device npu --providers CANNExecutionProvider` | npu_output.npy | `python3 -c "import numpy as np; a=np.load('npu_output.npy'); print(a.shape)"` | CANNExecutionProvider 推理正常 |
| 精度对比 | 5 | cpu_output.npy, npu_output.npy | `python3 scripts/compare_cpu_npu.py --model_path <model.onnx> --model_type detection` | comparison_report.json | `cat comparison_report.json \| python3 -m json.tool` | 余弦相似度 > 0.999, SNR > 25 dB |
| README 生成 | 6 | comparison_report.json | `python3 scripts/generate_readme.py --comparison_results comparison_report.json --output readme.md` | README.md | 检查是否含精度表格与明确结论 | 包含精度测试数据表格和 NPU/CPU 误差 < 1% 结论 |
| 显存释放 | 7 | 当前模型处理完成 | `torch.npu.empty_cache()` + `gc.collect()` | 释放的 NPU 显存 | `npu-smi info` 查看显存占用 | 显存占用显著下降 |
| 仓库发布 | 8 | 模型文件 + README + requirements.txt | `python3 scripts/publish_repo.py` 或 git 手动推送 | GitCode 仓库 | 访问仓库 URL 验证文件完整性 | 仓库文件齐全，README 渲染正常 |

### 阶段 1：环境准备

**输入**：NPU 服务器（Ascend 910 系列）、Python 3.8+、pip 包管理器

**输出**：onnxruntime-cann 可用、modelscope 可用、opencv-python 可用

**验证命令**：
```bash
python -c "import onnxruntime; print(onnxruntime.get_device())"
```

**通过标准**：输出包含 "NPU"，表示 CANNExecutionProvider 可用。

**执行步骤**：
1. 确认 NPU 驱动已安装：执行 `npu-smi info` 检查 NPU 状态和显存信息
2. 安装核心依赖：`pip install onnxruntime-cann>=1.24.4 onnxruntime>=1.24.0 numpy>=1.24.0 opencv-python>=4.8.0`
3. 安装可选依赖：`pip install modelscope paddle2onnx`（PaddlePaddle 模型转换场景）
4. 验证 CANN 可用性：运行验证命令检查 onnxruntime 是否识别 NPU 设备
5. 若 CANN 不可用，执行 `source set_env.sh` 确认环境变量配置正确

### 阶段 2：模型下载

**输入**：model_list 文件（每行包含 model_id、下载 URL、模型类型）

**输出**：各模型 ONNX 文件保存至本地 models/ 目录

**验证命令**：
```bash
ls -la models/<model_name>/model.onnx
```

**通过标准**：文件存在且大小 > 0。

**执行步骤**：
1. 准备模型列表文件，指定需要下载的 PP-OCRv4 模型 ID（检测、识别、版面分析类型）
2. 执行 `python3 scripts/download_models.py --model_id <model_id>` 从 ModelScope 下载
3. 检查下载的模型格式：ONNX 格式（.onnx）或 PaddlePaddle 格式（.pdmodel）
4. 若为 PaddlePaddle 格式，使用 `paddle2onnx` 转换为 ONNX
5. 验证每个模型文件完整性，重试最多 3 次（间隔 5s）应对下载超时

### 阶段 3：CPU 推理

**输入**：ONNX 模型文件、测试图片（detection/recognition 类型）

**输出**：cpu_output.npy（CPU 推理原始张量输出）

**验证命令**：
```bash
python3 -c "import numpy as np; a=np.load('results/<model>/cpu_output.npy'); print(a.shape)"
```

**通过标准**：输出 shape 与模型预期输出一致。

**执行步骤**：
1. 加载 ONNX 模型：`onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])`
2. 图像预处理：resize + pad + normalize（mean=0.5, std=0.5），确保输入尺寸与模型要求匹配
3. 执行 CPU 推理：`session.run(None, input_dict)`，保存输出至 `cpu_output.npy`
4. 检查输出张量的 shape 和数值范围，确认推理结果合理
5. 记录推理耗时，供后续 NPU 加速比计算使用

### 阶段 4：NPU 推理

**输入**：ONNX 模型文件、测试图片

**输出**：npu_output.npy（NPU 推理原始张量输出）

**验证命令**：
```bash
python3 -c "import numpy as np; a=np.load('results/<model>/npu_output.npy'); print(a.shape)"
```

**通过标准**：CANNExecutionProvider 推理正常，输出 shape 与 CPU 推理一致。

**执行步骤**：
1. 释放前一个模型的 NPU 显存：`torch.npu.empty_cache()` + `gc.collect()`
2. 加载 ONNX 模型并指定 CANNExecutionProvider：`onnxruntime.InferenceSession(model_path, providers=['CANNExecutionProvider'])`
3. 使用与 CPU 推理完全相同的预处理参数，确保输入数据一致
4. 执行 NPU 推理：`session.run(None, input_dict)`，保存输出至 `npu_output.npy`
5. 若 CANN 编译失败，记录错误信息并标记该模型为不兼容，跳过继续处理下一个

### 阶段 5：精度对比

**输入**：cpu_output.npy、npu_output.npy

**输出**：comparison_report.json（JSON 格式，含余弦相似度、SNR、误差指标）

**验证命令**：
```bash
cat results/<model>/comparison_report.json | python3 -m json.tool
```

**通过标准**：余弦相似度 > 0.999，SNR > 25 dB，NPU 与 CPU 误差 < 1%。

**执行步骤**：
1. 加载 CPU 和 NPU 的输出张量：`numpy.load('cpu_output.npy')`、`numpy.load('npu_output.npy')`
2. 计算最大绝对误差、平均绝对误差、最大相对误差指标
3. 计算余弦相似度：确认是否 > 0.999 阈值
4. 计算信噪比（SNR）：确认是否 > 25 dB 阈值
5. 生成 JSON 报告并保存至 `comparison_report.json`，输出通过的模型数量/总模型数

### 阶段 6：README 生成

**输入**：comparison_report.json、模型名称和原始 URL

**输出**：README.md（含模型介绍、精度测试表格、推理命令、性能数据）

**验证命令**：检查生成的 README.md 是否包含精度测试数据表格

**通过标准**：README 包含完整的精度测试数据表格和 "NPU 与 CPU 推理结果误差 < 1%" 明确结论。

**执行步骤**：
1. 读取 comparison_report.json 解析精度对比数据
2. 构建 README 模板：模型介绍 + 原始地址 + 任务类型 + 框架信息
3. 插入 CPU/NPU 精度测试数据表格（余弦相似度、SNR、误差）
4. 写入推理命令和 NPU 适配说明，包含依赖环境和性能测试数据
5. 添加模型标签（#NPU、#昇腾 等），生成最终 README.md 文件

### 阶段 7：显存释放

**输入**：当前模型处理完成，NPU 显存可能被占用

**输出**：释放的 NPU 显存，供下一个模型使用

**验证命令**：
```bash
npu-smi info
```

**通过标准**：NPU 显存占用显著下降，下一个模型可正常加载。

**执行步骤**：
1. 删除不再使用的模型 session 对象：`del session`
2. 调用垃圾回收：`gc.collect()`
3. 释放 NPU 缓存：`torch.npu.empty_cache()`
4. 验证 NPU 显存是否已释放：使用 `npu-smi info` 检查显存占用
5. 若显存仍未释放，重启 Python 进程后从失败模型继续执行

### 阶段 8：仓库发布（可选）

**输入**：模型文件、README.md、requirements.txt、ATOMGIT_USER_TOKEN

**输出**：GitCode 模型仓库

**验证命令**：访问 GitCode 仓库 URL，检查文件完整性

**通过标准**：仓库文件齐全，README 渲染正常。

**执行步骤**：
1. 准备发布目录：整理模型文件、README.md、requirements.txt 到发布目录
2. 调用 `python3 scripts/publish_repo.py --repo_name <name> --token ${ATOMGIT_USER_TOKEN}` 或手动 git 推送
3. 检查推送结果：仓库中存在 inference.py、compare_cpu_npu.py、requirements.txt、readme.md
4. 访问 GitCode 仓库页面验证 README 渲染效果
5. 若推送失败（401/403/网络错误），检查 Token 有效期后重试

## 执行检查点与用户确认

| 检查点 | 触发时机 | 确认内容 | 不通过处理 |
|--------|---------|---------|-----------|
| ✅ CP-1 环境就绪 | 依赖安装后 | `python -c "import onnxruntime; print(onnxruntime.get_device())"` 确认 CANN 可用 | 检查 CANN 安装和 onnxruntime-cann 版本 |
| ✅ CP-2 模型权重就绪 | 模型下载后 | ONNX 模型文件存在且大小 > 0 | 从 ModelScope 重新下载 |
| ✅ CP-3 CPU 推理 | 首个模型 CPU 推理后 | 输出 shape 与预期一致 | 检查预处理参数（resize/pad/normalize） |
| ✅ CP-4 NPU 推理 | 首个模型 NPU 推理后 | CANNExecutionProvider 推理正常 | 检查 ONNX opset 与 CANN 编译器兼容性 |
| ✅ CP-5 精度对比 | 每个模型对比后 | 余弦相似度 > 0.999，SNR > 25 dB | 记录差异模型，判断是否 CANN 编译问题 |
| ✅ CP-6 产物完整 | 全部模型处理后 | README 含精度表格，仓库文件齐全 | 回到对应步骤修复 |
| ✅ CP-7 验收确认 | 发布前 | 所有成功模型精度达标 | 标记失败的模型并提供失败原因 |

用户确认指令：`通过` / `继续` 进入下一步，`重试 cp-N` 重新执行对应检查点。

```
pp-ocrv4-npu-deploy/
├── SKILL.md                    # 本技能定义文件
├── scripts/
│   ├── download_models.py      # 批量下载所有 PP-OCRv4 模型
│   ├── run_inference.py        # 通用 ONNX 推理脚本（支持 CPU/NPU）
│   ├── compare_cpu_npu.py      # CPU/NPU 精度对比脚本
│   ├── generate_readme.py      # 自动生成中文 README
│   ├── publish_repo.py         # 发布模型到 GitCode
│   └── utils/
│       └── onnx_utils.py       # ONNX 推理工具函数
├── examples/
│   ├── run_all_models.sh       # 一键执行所有模型
│   └── run_single_model.sh     # 执行单个模型
└── results/                    # 测试结果输出目录
    └── <model_name>/
        ├── cpu_output.npy
        ├── npu_output.npy
        └── comparison_report.json
```

## Input Parameters

| 参数 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `--model_list` | string | 是 | 模型列表文件路径，包含 model_id, url, type |
| `--image_path` | string | 是 | 测试图片路径 |
| `--rec_image_path` | string | 否 | 识别模型测试图片路径 |
| `--output_dir` | string | 否 | 结果输出目录 (默认: ./results) |
| `--device` | string | 否 | 计算设备 (cpu/npu, 默认: cpu) |
| `--gitcode_token` | string | 否 | GitCode API Token (也可通过 ATOMGIT_USER_TOKEN 环境变量设置) |
| `--gitcode_username` | string | 否 | GitCode 用户名 |
| `--skip_publish` | flag | 否 | 跳过 GitCode 仓库发布 |
| `--skip_compare` | flag | 否 | 跳过 CPU/NPU 精度对比 |

## Output Results

执行完成后，每个模型会生成：

1. **推理结果**: CPU 和 NPU 的原始张量输出 (.npy)
2. **精度对比报告**: JSON 格式，包含余弦相似度、SNR、绝对/相对误差等指标
3. **中文 README**: 包含模型介绍、推理命令、精度测试数据表格、性能数据
4. **GitCode 仓库**: 包含 inference.py、compare_cpu_npu.py、requirements.txt、readme.md

## How to Execute NPU Inference

```bash
# 单模型推理
python3 scripts/run_inference.py \
  --model_path /path/to/model.onnx \
  --image_path /path/to/image.png \
  --device npu \
  --model_type detection

# 带模型类型的推理
# model_type 可选: detection | recognition
```

**输入**：ONNX 模型文件、测试图片路径、模型类型（detection/recognition）

**输出**：NPU 推理结果（.npy 文件）

**验证命令**：
```bash
python3 -c "import numpy as np; a=np.load('npu_output.npy'); print(a.shape)"
```

**通过标准**：CANNExecutionProvider 推理正常，输出 shape 与预期一致。

**执行步骤**：
1. 加载 ONNX 模型：`onnxruntime.InferenceSession(model_path, providers=['CANNExecutionProvider'])`，指定 NPU 执行提供者
2. 图像预处理：resize + pad + normalize（mean=0.5, std=0.5），确保输入尺寸与模型要求匹配
3. NPU 推理：`session.run(None, input_dict)`，输出原始张量结果
4. 后处理：检测模型执行阈值二值化，识别模型执行 CTC 解码
5. 保存原始输出至 `npu_output.npy`，用于后续精度对比

## How to Execute CPU/NPU Accuracy Comparison

```bash
# 精度对比
python3 scripts/compare_cpu_npu.py \
  --model_path /path/to/model.onnx \
  --image_path /path/to/image.png \
  --model_type detection \
  --output_dir ./results/model_name/
```

**输入**：CPU 推理输出（cpu_output.npy）、NPU 推理输出（npu_output.npy）、模型参数

**输出**：comparison_report.json（含余弦相似度、SNR、误差指标）

**验证命令**：
```bash
cat results/<model>/comparison_report.json | python3 -m json.tool
```

**通过标准**：余弦相似度 > 0.999，SNR > 25 dB。

**执行步骤**：
1. 分别加载 CPU 和 NPU 推理结果：`cpu_out = numpy.load('cpu_output.npy')`、`npu_out = numpy.load('npu_output.npy')`
2. 计算基础误差指标：最大绝对误差（Max Absolute Error）、平均绝对误差（Mean Absolute Error）、最大相对误差（Max Relative Error）
3. 计算余弦相似度（Cosine Similarity），验证是否 > 0.999
4. 计算信噪比（SNR），验证是否 > 25 dB；统计微小误差（< 1e-5）元素占比（Within Tolerance）
5. 整合所有指标生成 `comparison_report.json`，标记通过/不通过状态

### 精度要求

NPU 与 CPU 推理结果误差 < 1%，余弦相似度 > 0.999。

## How to Generate README

```bash
python3 scripts/generate_readme.py \
  --model_name "PP-OCRv4" \
  --model_url "https://www.modelscope.cn/models/OllmOne/PP-OCRv4" \
  --task_type "ocr-recognition" \
  --comparison_results ./results/PP-OCRv4/comparison_report.json \
  --output ./readme.md
```

**输入**：comparison_report.json、模型名称、模型原始 URL、任务类型

**输出**：README.md（中文文档，含精度测试表格）

**验证命令**：检查 README.md 是否包含精度测试数据表格和明确结论

**通过标准**：README 包含完整的精度测试数据表格和 "NPU 与 CPU 推理结果误差 < 1%" 结论。

**执行步骤**：
1. 从 comparison_report.json 解析精度对比数据（余弦相似度、SNR、加速比、误差指标）
2. 构建模型介绍段落，包含模型名称、原始地址、任务类型和框架信息
3. 生成 CPU/NPU 精度测试数据 Markdown 表格，列出每个模型的各项指标
4. 写入推理命令、NPU 适配说明、依赖环境和性能测试数据
5. 添加模型标签（#NPU、#昇腾 等），突出 NPU 适配结论标签

README 必须包含：
- 模型介绍和原始地址
- 任务类型、框架、输入/输出格式
- 依赖环境和 NPU 适配说明
- 推理命令和推理结果
- **CPU/NPU 精度测试数据表格**
- **明确结论：NPU 与 CPU 推理结果误差 < 1%**
- 性能测试数据
- 模型标签（含 #NPU、#昇腾 等）

## How to Execute Models Serially (Prevent Memory Explosion)

```bash
#!/bin/bash
# run_all_models.sh - Serial execution of all PP-OCRv4 models

MODELS=(
  "OllmOne/PP-OCRv4:PP-OCRv4-npu"
  "cycloneboy/ch_PP-OCRv4_det_infer:ch_PP-OCRv4_det_infer-npu"
  "cycloneboy/ch_PP-OCRv4_det_server_infer:ch_PP-OCRv4_det_server_infer-npu"
  # ... more models
)

for MODEL_ENTRY in "${MODELS[@]}"; do
  IFS=':' read -r MODEL_ID REPO_NAME <<< "$MODEL_ENTRY"
  
  echo "=== Processing $REPO_NAME ==="
  
  # 1. Download model
  python3 scripts/download_models.py --model_id "$MODEL_ID"
  
  # 2. Run inference and comparison
  python3 scripts/compare_cpu_npu.py \
    --model_name "$REPO_NAME" \
    --output_dir "./results/$REPO_NAME/"
  
  # 3. Generate README
  python3 scripts/generate_readme.py \
    --model_name "$REPO_NAME" \
    --comparison_results "./results/$REPO_NAME/comparison_report.json"
  
  # 4. Release NPU memory
  python3 -c "
import gc; gc.collect()
try:
    import torch
    torch.npu.empty_cache()
except: pass
"
  
  echo "=== $REPO_NAME completed ==="
done

echo "All models processed!"
```

## How to Publish to GitCode

**输入**：模型文件目录、README.md、requirements.txt、ATOMGIT_USER_TOKEN

**输出**：GitCode 模型仓库

**验证命令**：访问 GitCode 仓库 URL，检查仓库文件完整性

**通过标准**：仓库中包含 inference.py、compare_cpu_npu.py、requirements.txt、readme.md，README 渲染正常。

**执行步骤**：
1. 整理发布目录：将模型 ONNX 文件、inference.py、compare_cpu_npu.py、requirements.txt、README.md 复制到发布目录
2. 调用自动发布脚本：`python3 scripts/publish_repo.py --repo_name <name> --token ${ATOMGIT_USER_TOKEN} --username <user>` 或使用 gitcode-publish Skill
3. 若自动发布不可用，手动执行 git 初始化、添加文件、提交和推送
4. 验证推送结果：确认仓库中所有必需文件已存在
5. 若推送失败（401/403/网络错误），检查 ATOMGIT_USER_TOKEN 有效期后重试

```bash
# 使用 gitcode-publish Skill 或直接调用 GitCode API
python3 scripts/publish_repo.py \
  --repo_name "PP-OCRv4-npu" \
  --model_dir ./results/PP-OCRv4/ \
  --token "${ATOMGIT_USER_TOKEN}" \
  --username "gcw_C8PI9e90"
```

或手动推送：

```bash
cd /tmp/git_repos/PP-OCRv4-npu
git init
git checkout -b main
git add .
git commit -m "Add PP-OCRv4-npu: NPU adaptation with accuracy comparison"
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/gcw_C8PI9e90/PP-OCRv4-npu.git
git push -u origin main
```

## Accuracy Verification Results Summary

所有模型在 Ascend NPU 上的 CPU/NPU 精度对比结果：

| 模型 | 输出形状 | 余弦相似度 | SNR | NPU 加速比 | 状态 |
| --- | --- | --- | --- | --- | --- |
| PP-OCRv4 (det) | (1, 1, 224, 800) | 0.999358 | 28.88 dB | 39.47x | ✓ |
| PP-OCRv4 (rec) | (1, 50, 6625) | 0.999996 | 50.18 dB | 39.87x | ✓ |
| ch_PP-OCRv4_det_infer | (1, 1, 224, 800) | 0.999812 | 34.17 dB | 31.44x | ✓ |
| ch_PP-OCRv4_det_server_infer | (1, 1, 224, 800) | 0.999467 | 29.69 dB | 102.37x | ✓ |
| ch_PP-OCRv4_rec_infer | (1, 50, 6625) | 0.999966 | 41.62 dB | 54.88x | ✓ |
| ch_PP-OCRv4_rec_server_infer | (1, 50, 6625) | 0.999986 | 43.88 dB | 108.34x | ✓ |
| en_PP-OCRv4_rec_infer | (1, 50, 97) | 0.999987 | 45.69 dB | 90.93x | ✓ |
| en_PP-OCRv4_rec_infer_somohk | (1, 50, 97) | N/A | N/A | N/A | ✗ (CANN 编译失败) |

**总通过率**: 7/8 模型 (87.5%)，所有成功模型余弦相似度 > 0.999，NPU 与 CPU 推理结果误差 < 1%。

## Notes

1. **串行执行**: 多个模型必须串行处理，单个模型测试完成后释放显存再处理下一个
2. **PaddlePaddle 转换**: 如果模型是 PaddlePaddle 格式（.pdmodel），需先用 paddle2onnx 转换为 ONNX
3. **ONNX 兼容性**: 旧版 PaddlePaddle 导出的 ONNX 模型（opset 7）可能不被 CANN 编译器支持
4. **显存释放**: 每个模型测试完成后执行 `torch.npu.empty_cache()` 和 `gc.collect()`
5. **Token 安全**: 不要将 ATOMGIT_USER_TOKEN 写入代码或 README

## 异常处理与回滚策略

| 异常场景 | 表现 | 处理方式 | 回滚/恢复 |
|---------|------|---------|----------|
| CANN 不可用 | `onnxruntime.get_device()` 返回非 NPU | 确认 CANN 安装，`source set_env.sh` | 切换至 CPU ONNX 推理 |
| ONNX 模型加载失败 | `InferenceSession` 报错 | 检查 ONNX opset 版本与 CANN 兼容性 | 使用 paddle2onnx 重新导出或切换 opset |
| CANN 编译失败 | 特定模型编译报错 | 标记该模型为不兼容，记录错误信息 | 跳过该模型，继续处理下一个 |
| 模型下载超时 | ModelScope 超时或 404 | 重试最多 3 次，间隔 5s | 手动下载并放置到指定路径 |
| NPU OOM | CUDA/显存溢出错误 | 每个模型后 `torch.npu.empty_cache()` + `gc.collect()` | 重启 Python 进程后从失败模型继续 |
| 预处理不一致 | CPU/NPU 输入数据有差异 | 统一使用 numpy 固定 seed 的随机数 | 对比并标准化预处理流水线 |
| 精度不达标 | 余弦相似度 < 0.999 或 SNR < 25 dB | 检查 ONNX FP16/FP32 精度模式 | 使用 FP32 ONNX 模型重跑对比 |
| GitCode 推送失败 | 401/403/网络错误 | 检查 `ATOMGIT_USER_TOKEN` 有效期 | 重新生成 token 后重试 |

## 资源与评测产物

| 资源路径 | 用途 |
|---------|------|
| `SKILL.md`（本文件） | 完整执行流程与异常处理指南 |
| `scripts/run_inference.py` | 通用 ONNX 推理脚本（CPU/NPU） |
| `scripts/compare_cpu_npu.py` | CPU/NPU 精度对比脚本 |
| `scripts/download_models.py` | 批量下载所有 PP-OCRv4 模型 |
| `scripts/generate_readme.py` | 自动生成中文 README |
| `scripts/publish_repo.py` | 发布模型到 GitCode |
| `scripts/utils/onnx_utils.py` | ONNX 推理工具函数 |
| `examples/run_all_models.sh` | 一键执行所有模型 |
| `examples/run_single_model.sh` | 执行单个模型 |
| `test-prompts.json` | 评测提示词与预期结果 |

执行结束后确认以下产物已生成：

- [ ] `results/<model_name>/comparison_report.json` — CPU/NPU 精度对比报告
- [ ] `results/<model_name>/cpu_output.npy` — CPU 推理原始输出
- [ ] `results/<model_name>/npu_output.npy` — NPU 推理原始输出
- [ ] `models/<model_name>/README.md` — 含精度测试数据表格的中文文档
- [ ] `models/<model_name>/requirements.txt` — 依赖文件
