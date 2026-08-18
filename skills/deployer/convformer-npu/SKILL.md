---
name: convformer-npu
description: ConvFormer 系列图像分类模型的昇腾 NPU 统一部署与精度验证 Skill。支持 timm 库中 convformer_b36/m36/s18/s36 共 20 个变体模型的自动下载、NPU 推理、CPU/NPU 精度对比、README 生成和模型仓库发布。适用于华为昇腾 Ascend 910 NPU。
---

# ConvFormer NPU 部署与验证 Skill

本 Skill 用于自动完成 ConvFormer 系列图像分类模型在昇腾 NPU 上的部署、推理、CPU/NPU 精度测试、README 文档生成和模型仓库发布。

## 支持的模型列表

本 Skill 支持以下 20 个 ConvFormer 变体模型（均来自 timm 库）：

| 模型架构 | 预训练数据 | 输入尺寸 | 参数规模 |
|----------|-----------|---------|---------|
| convformer_b36 | ImageNet-1K / ImageNet-22K / 22K→1K Finetune | 224×224 / 384×384 | ~100M |
| convformer_m36 | ImageNet-1K / ImageNet-22K / 22K→1K Finetune | 224×224 / 384×384 | ~60M |
| convformer_s18 | ImageNet-1K / ImageNet-22K / 22K→1K Finetune | 224×224 / 384×384 | ~27M |
| convformer_s36 | ImageNet-1K / ImageNet-22K / 22K→1K Finetune | 224×224 / 384×384 | ~39M |

完整模型列表见 `scripts/model_list.txt`。

## 输入参数

Skill 接受以下配置参数：

```json
{
  "model_name": "convformer_b36.sail_in1k_384",
  "device": "npu",
  "test_image": "test.jpg",
  "repo_username": "gcw_C8PI9e90",
  "publish": false
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | string | - | 模型名称（必须） |
| device | string | "npu" | 推理设备：cpu / npu |
| test_image | string | "test.jpg" | 测试图像路径 |
| repo_username | string | "" | GitCode 用户名（发布时需要） |
| publish | bool | false | 是否发布到 GitCode |

## 完整工作流程

```
1. 环境检查 ──→ 2. 模型下载 ──→ 3. CPU 推理 ──→ 4. NPU 推理 ──→ 5. 精度对比 ──→ 6. README 生成 ──→ 7. 仓库发布(可选)
```

### 工作流程阶段汇总

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| **P1** | 环境准备 | 配置参数、NPU 设备 | 检查 CANN/torch_npu，安装依赖 | 可用的 NPU 推理环境 | `python -c "import torch_npu; print(torch_npu.npu.is_available())"` | 返回 True |
| **P2** | 模型下载 | `model_name` 参数 | ModelScope `snapshot_download` 下载权重 | 模型权重文件（.safetensors） | `ls <model_dir>/model.safetensors` | 文件存在且大小 > 0 |
| **P3** | CPU 推理 | 权重文件、测试图像 | 运行 `inference.py --device cpu` | `cpu_results.json`（Top-5 分类结果） | `python -c "import json; d=json.load(open('cpu_results.json')); print(d['top1_label'])"` | 输出正常分类标签 |
| **P4** | NPU 推理 | 权重文件、测试图像 | 运行 `inference.py --device npu` | `npu_results.json`（Top-5 分类结果） | `python -c "import json; d=json.load(open('npu_results.json')); print(d['top1_label'])"` | NPU 推理正常完成 |
| **P5** | 精度对比 | `cpu_results.json`、`npu_results.json` | 运行 `compare_cpu_npu.py` | `comparison.json`（余弦相似度、Top-1/Top-5 一致率） | `python -c "import json; d=json.load(open('comparison.json')); print(d['cosine_similarity'])"` | 余弦相似度 > 0.999999 |
| **P6** | README 生成 | 精度对比数据、模型元信息 | 运行 `generate_readme.py` | `README.md`（中文文档） | `ls README.md` | 文档已生成 |
| **P7** | 仓库发布（可选） | README、脚本、产物 | 运行 `publish_repo.py` | GitCode 模型仓库 | `git ls-remote <repo_url>` | 仓库发布成功 |

### P1: 环境准备

**执行步骤：**
1. 确认昇腾 NPU 硬件在线：执行 `npu-smi info` 检查设备状态
2. 激活 Python 虚拟环境，确保 CANN 环境变量已生效（`source set_env.sh`）
3. 安装依赖包：`pip install timm torchvision Pillow safetensors modelscope`
4. 验证 torch_npu 可用：`python -c "import torch_npu; print(torch_npu.npu.is_available())"`
5. 设置 `ASCEND_RT_VISIBLE_DEVICES` 环境变量指定使用的 NPU 设备

**输入：** 配置参数（model_name, device, test_image）、NPU 硬件设备

**输出：** 可用的 NPU 推理环境

**验证命令：**
```bash
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

**通过标准：** 返回 `True`

### P2: 模型下载

**执行步骤：**
1. 根据 `model_name` 参数构造 ModelScope 模型 ID：`timm/<model_name>`
2. 调用 `modelscope.hub.snapshot_download.snapshot_download` 下载模型权重
3. 验证下载完成的权重文件存在且大小大于 0
4. 若下载失败或文件异常，最多重试 3 次，每次间隔 5 秒
5. 记录模型权重路径，供后续推理步骤使用

**输入：** `model_name` 参数（如 `convformer_b36.sail_in1k_384`）

**输出：** 模型权重文件（`.safetensors` 格式），存放于 ModelScope 缓存目录

**验证命令：**
```bash
ls <model_dir>/model.safetensors
```

**通过标准：** 文件存在且大小 > 0

### P3: CPU 推理

**执行步骤：**
1. 加载模型权重到 CPU：`model = timm.create_model(model_name, pretrained=True).cpu()`
2. 读取测试图像 `test.jpg`，执行预处理（resize 到模型输入尺寸、归一化、添加 batch 维度）
3. 执行 CPU 推理：`with torch.no_grad(): output = model(input_tensor)`
4. 对输出进行 Softmax 处理，提取 Top-5 分类标签与置信度
5. 将推理结果保存为 `cpu_results.json`（包含 top1_label, top1_prob, top5_labels, top5_probs）

**输入：** 模型权重文件、测试图像 `test.jpg`

**输出：** `cpu_results.json`

**验证命令：**
```bash
python -c "import json; d=json.load(open('cpu_results.json')); print(d['top1_label'], d['top1_prob'])"
```

**通过标准：** Top-5 分类结果正常，置信度分布合理

### P4: NPU 推理

**执行步骤：**
1. 检查 NPU 设备可用性，确认 `ASCEND_RT_VISIBLE_DEVICES` 正确设置
2. 加载模型权重并转移到 NPU：`model = model.to('npu')`
3. 读取测试图像并执行与 CPU 推理一致的预处理
4. 执行 NPU 推理：`with torch.no_grad(): output = model(input_tensor.to('npu'))`
5. 将输出结果从 NPU 拷贝到 CPU：`output = output.cpu()`
6. Softmax + Top-5 提取，保存为 `npu_results.json`

**输入：** 模型权重文件、测试图像 `test.jpg`

**输出：** `npu_results.json`

**验证命令：**
```bash
python -c "import json; d=json.load(open('npu_results.json')); print(d['top1_label'], d['top1_prob'])"
```

**通过标准：** NPU 推理正常完成，无 OOM 或 NaN 异常

### P5: 精度对比

**执行步骤：**
1. 读取 `cpu_results.json` 和 `npu_results.json`
2. 对比 CPU 与 NPU 输出的 logits 向量，计算余弦相似度
3. 对比 Top-1 分类标签是否一致
4. 对比 Top-5 分类标签是否完全一致
5. 计算平均相对误差：`|npu_logits - cpu_logits| / |cpu_logits|`
6. 将对比结果写入 `comparison.json`

**输入：** `cpu_results.json`、`npu_results.json`

**输出：** `comparison.json`（包含 cosine_similarity, avg_relative_error, top1_match, top5_match）

**验证命令：**
```bash
python -c "import json; d=json.load(open('comparison.json')); print('cos:', d['cosine_similarity'], 'err:', d['avg_relative_error'])"
```

**通过标准：** 余弦相似度 > 0.999999，Top-1 一致，平均相对误差 < 1%

### P6: README 生成

**执行步骤：**
1. 读取精度对比结果和模型元信息
2. 渲染 README 模板，包含模型介绍、精度数据、性能数据、使用方法
3. 将生成的 `README.md` 写入 `models/<model_name>/` 目录
4. 若指定 `--all` 参数，为所有 20 个模型批量生成 README

**输入：** `comparison.json`、模型元信息（名称、参数量、输入尺寸等）

**输出：** `README.md`（中文文档，含精度与性能数据表格）

**验证命令：**
```bash
ls models/<model_name>/README.md
```

**通过标准：** README 已生成，内容完整无误

### P7: 仓库发布（可选）

**执行步骤：**
1. 确认 `ATOMGIT_USER_TOKEN` 环境变量已设置
2. 初始化 GitCode 模型仓库（若不存在则创建）
3. 将推理脚本、README、精度数据推送到仓库
4. 验证仓库已成功创建并可访问

**输入：** `repo_username`、README、推理脚本、精度数据

**输出：** GitCode 模型仓库（如 `https://gitcode.com/<username>/<model-name>-npu`）

**验证命令：**
```bash
git ls-remote https://gitcode.com/<username>/<model-name>-npu.git HEAD
```

**通过标准：** 仓库发布成功，可通过浏览器访问

## 执行检查点与用户确认

| 检查点 | 触发时机 | 确认内容 | 不通过处理 |
|--------|---------|---------|-----------|
| ✅ CP-1 环境就绪 | 依赖安装后 | `python -c "import torch_npu; print(torch_npu.npu.is_available())"` 返回 True | 检查 CANN 环境与 torch-npu 版本 |
| ✅ CP-2 模型权重就绪 | 模型下载后 | 权重文件存在且大小 > 0 | 重试 ModelScope 下载，最多 3 次 |
| ✅ CP-3 CPU 推理 | 首个模型 CPU 推理后 | 输出 top-5 分类结果正常 | 检查输入图片格式和权重路径 |
| ✅ CP-4 NPU 推理 | 首个模型 NPU 推理后 | NPU 推理正常完成 | 检查 `ASCEND_RT_VISIBLE_DEVICES` |
| ✅ CP-5 精度对比 | 每个模型精度对比后 | 余弦相似度 > 0.999999，Top-1 一致 | 记录差异，判断是否重跑 |
| ✅ CP-6 产物完整 | 全部模型处理后 | README 已生成，comparison.json 存在 | 回到对应步骤修复 |
| ✅ CP-7 验收确认 | 发布前 | 全部 20 个模型精度误差 < 1% | 修复未达标模型 |

用户确认指令：`通过` / `继续` 进入下一步，`重试 cp-N` 重新执行对应检查点。

## 使用方法

## 1. 安装依赖

```bash
pip install timm torchvision Pillow safetensors modelscope
```

## 2. 下载模型

```python
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download("timm/<model_name>")
```

## 3. 执行 NPU 推理

参见 `scripts/inference.py`：

```bash
python scripts/inference.py --model convformer_b36.sail_in1k_384 --device npu --image test.jpg
```

## 4. CPU/NPU 精度对比

参见 `scripts/compare_cpu_npu.py`：

```bash
python scripts/compare_cpu_npu.py --model convformer_b36.sail_in1k_384 --image test.jpg
```

## 5. 串行执行多个模型

参见 `scripts/run_all_models.sh`：

```bash
bash scripts/run_all_models.sh
```

该脚本会自动串行处理所有 20 个模型，每个模型完成后释放显存，防止 OOM。

## 6. 生成 README

```bash
python scripts/generate_readme.py --model convformer_b36.sail_in1k_384
python scripts/generate_readme.py --all
```

## 7. 发布到 GitCode

```bash
python scripts/publish_repo.py --model convformer_b36.sail_in1k_384
python scripts/publish_repo.py --all
```

## 输出结果

每个模型的输出目录结构：

```
models/<model_name>/
├── inference.py            # 推理脚本
├── compare_cpu_npu.py      # CPU/NPU 精度对比脚本
├── requirements.txt        # 依赖文件
├── README.md               # 中文文档（含精度数据）
├── screenshot.png          # 运行截图
├── cpu_results.json        # CPU 推理结果
├── npu_results.json        # NPU 推理结果
└── comparison.json         # 精度对比结果
```

## 精度验证

所有模型已完成 CPU vs NPU 精度验证，验证结果如下：

| 模型 | CPU/NPU 误差 | 余弦相似度 | Top-1 一致 | Top-5 一致 |
|------|-------------|-----------|-----------|-----------|
| convformer_b36.* | < 0.33% | > 0.999999 | 是 | 5/5 |
| convformer_m36.* | < 0.76% | > 0.999999 | 是 | 5/5 |
| convformer_s18.* | < 0.46% | > 0.999999 | 是 | 5/5 |
| convformer_s36.* | < 0.89% | > 0.999999 | 是 | 5/5 |

**结论：所有 20 个模型的 NPU 与 CPU 推理结果误差均 < 1%，精度完全满足要求。**

## 性能数据

| 模型类型 | CPU 耗时 | NPU 耗时 | 加速比 |
|---------|---------|---------|-------|
| convformer_b36 (224×224) | ~1.2s | ~0.029s | ~40× |
| convformer_b36 (384×384) | ~3.2s | ~0.032s | ~98× |
| convformer_m36 (224×224) | ~0.7s | ~0.025s | ~28× |
| convformer_m36 (384×384) | ~1.9s | ~0.027s | ~70× |
| convformer_s18 (224×224) | ~0.24s | ~0.013s | ~19× |
| convformer_s18 (384×384) | ~0.63s | ~0.014s | ~45× |
| convformer_s36 (224×224) | ~0.45s | ~0.023s | ~19× |
| convformer_s36 (384×384) | ~1.2s | ~0.024s | ~50× |

## 模型仓库地址

每个模型对应一个 GitCode 模型仓库：

| 模型 | 仓库地址 |
|------|---------|
| convformer_b36.sail_in1k_384 | https://gitcode.com/gcw_C8PI9e90/convformer-b36-sail-in1k-384-npu |
| convformer_b36.sail_in22k | https://gitcode.com/gcw_C8PI9e90/convformer-b36-sail-in22k-npu |
| ... | ... |

## 注意事项

1. **串行执行**：多个模型必须串行执行，防止 NPU 显存爆炸。使用 `run_all_models.sh` 自动管理。
2. **显存释放**：每个模型测试完后执行 `torch.npu.empty_cache()` + `gc.collect()`。
3. **网络要求**：模型权重从 ModelScope 下载，需要网络连接。必要时可切换到 hf-mirror。
4. **API Token**：发布到 GitCode 需要设置 `ATOMGIT_USER_TOKEN` 环境变量。
5. **测试图像**：需准备至少一张 JPEG 测试图像用于推理验证。

## 异常处理与回滚策略

| 异常场景 | 表现 | 处理方式 | 回滚/恢复 |
|---------|------|---------|----------|
| NPU 不可用 | `npu-smi info` 报错或无设备 | 确认硬件在位，重新 `source set_env.sh` | 切换至 CPU 模式：`--device cpu` |
| torch_npu 导入失败 | `ImportError: No module named torch_npu` | 确认 torch-npu 与 CANN 版本匹配 | `pip install torch-npu==<version>` 重装 |
| 模型下载超时 | ModelScope 超时或 404 | 重试最多 3 次，间隔 5s | 切换至 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com` |
| NPU OOM | `torch.npu.OutOfMemoryError` | `torch.npu.empty_cache()` + `gc.collect()` | 释放其他进程显存；减少 batch size 为 1 |
| 显存泄漏 | 连续处理多模型后 OOM | 每个模型后执行清理代码 | 重启 Python 进程 |
| CPU/NPU 精度差异大 | 余弦相似度 < 0.999 | 检查输入预处理一致性（归一化、resize） | 重跑 CPU 推理后对比；仍异常则标记"精度存疑" |
| 权重文件缺失 | `FileNotFoundError: model.safetensors` | 确认 ModelScope 编码规则（`.` → `___`） | 手动指定模型路径 |
| 磁盘空间不足 | 下载权重时磁盘满 | `df -h` 检查磁盘 | 清理 `modelscope_cache` 或更换路径 |
| 推理结果 NaN | NPU 输出全为 NaN | 检查 `ASCEND_RT_VISIBLE_DEVICES` 是否正确 | 重新设置环境变量并重启 Python |

## 资源与评测产物

| 资源路径 | 用途 |
|---------|------|
| `SKILL.md`（本文件） | 完整执行流程与异常处理指南 |
| `scripts/inference.py` | 单模型 CPU/NPU 推理脚本 |
| `scripts/compare_cpu_npu.py` | CPU 与 NPU 精度对比脚本 |
| `scripts/run_model.sh` | 单模型串行执行 Shell |
| `scripts/run_all_models.sh` | 串行执行全部 20 个模型 |
| `scripts/generate_readme.py` | 模型 README 自动生成脚本 |
| `scripts/model_list.txt` | 完整 20 个模型名称清单 |
| `examples/example_inference.py` | 快速入门推理示例 |
| `examples/example_batch.sh` | 批量示例脚本 |
| `test-prompts.json` | 评测提示词与预期结果 |

执行结束后确认以下产物已生成：

- [ ] `models/<model_name>/comparison.json` — CPU/NPU 精度对比数据
- [ ] `models/<model_name>/README.md` — 中文模型文档
- [ ] `models/<model_name>/cpu_results.json` — CPU 推理结果
- [ ] `models/<model_name>/npu_results.json` — NPU 推理结果
