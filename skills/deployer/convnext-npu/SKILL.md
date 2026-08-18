---
name: convnext-npu
description: >
  ConvNeXt 系列图像分类模型在昇腾 NPU 上的统一部署与精度验证 Skill。
  覆盖 timm 库中 convnext_nano/pico/small/tiny 共 24 个变体模型的
  自动下载、NPU 推理、CPU/NPU 精度对比验证的全流程。
  当用户提到 ConvNeXt NPU 部署、ConvNeXt 昇腾、ConvNeXt 精度验证时触发。
metadata:
  short-description: ConvNeXt 系列昇腾 NPU 部署与精度验证
  category: NPU-Model-Deploy
  tags: [ascend, npu, convnext, image-classification, timm, pytorch, inference]
---

# ConvNeXt NPU 部署与验证 Skill

本 Skill 用于自动完成 ConvNeXt 系列图像分类模型在昇腾 NPU 上的部署、推理、CPU/NPU 精度测试、README 文档生成和模型仓库发布。

## 支持的模型列表

本 Skill 支持以下 24 个 ConvNeXt 变体模型（均来自 timm 库）：

| 模型架构 | 变体数量 | 预训练数据 | 输入尺寸 | 参数规模 |
|----------|---------|-----------|---------|---------|
| convnext_nano | 6 | ImageNet-1K / ImageNet-12K | 224×224 / 384×384 | ~15M |
| convnext_nano_ols | 1 | ImageNet-1K | 224×224 | ~15M |
| convnext_pico | 1 | ImageNet-1K | 224×224 | ~9M |
| convnext_pico_ols | 1 | ImageNet-1K | 224×224 | ~9M |
| convnext_small | 7 | ImageNet-1K / ImageNet-12K / ImageNet-22K | 224×224 / 384×384 | ~50M |
| convnext_tiny | 7 | ImageNet-1K / ImageNet-12K / ImageNet-22K | 224×224 / 384×384 | ~28M |
| convnext_tiny_hnf | 1 | ImageNet-1K | 224×224 | ~28M |

完整模型列表见 `scripts/model_list.txt`。

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 910 系列（至少 1 卡，64GB HBM） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.5.1 |
| Python | 3.9 – 3.13 |
| 网络 | 首次运行需联网下载模型权重（ModelScope） |

## 输入参数

Skill 接受以下配置参数：

```json
{
  "model_name": "convnext_nano.in12k_ft_in1k",
  "device": "npu",
  "test_image": "test.jpg",
  "publish": false
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | string | - | 模型名称（必须） |
| device | string | "npu" | 推理设备：cpu / npu |
| test_image | string | "test.jpg" | 测试图像路径 |
| publish | bool | false | 是否发布到 GitCode |

## 完整工作流程

下表概括从环境初始化到模型仓库发布的完整工作流阶段：

| 阶段 | 步骤 | 输入 | 执行动作 | 输出 | 验证命令 | 通过标准 |
|------|------|------|---------|------|---------|---------|
| 初始化 | 0. 环境初始化 | CANN 安装包、NPU 硬件 | 加载 CANN 环境，设置 NPU 设备号 | 就绪的 NPU 运行环境 | `npu-smi info` | 至少 1 张空闲 NPU 卡，CANN >= 8.5.1 |
| 初始化 | 1. 安装依赖 | Python 3.9–3.13，CANN 已加载 | pip 安装 torch、torch-npu、timm 等依赖 | Python 依赖环境 | `python -c "import torch_npu; print(torch_npu.npu.is_available())"` | 返回 True |
| 数据准备 | 2. 模型下载 | model_name，网络连接 | 从 ModelScope 下载模型权重 | model.safetensors 权重文件 | `ls -lh <model_dir>/model.safetensors` | 文件存在且大小 > 0 |
| 推理验证 | 3. CPU 推理 | 权重文件、测试图像 | 运行 inference.py，指定 device=cpu | CPU Top-5 分类结果 | 检查终端输出的 top-5 结果 | 推理正常完成，输出合理类别 |
| 推理验证 | 4. NPU 推理 | 权重文件、测试图像、NPU 环境 | 运行 inference.py，指定 device=npu | NPU Top-5 分类结果 | 检查终端输出的 top-5 结果 | 无 device 相关错误 |
| 精度验证 | 5. CPU/NPU 精度对比 | CPU 与 NPU 推理结果 | 运行 compare_cpu_npu.py，计算精度指标 | comparison_results.json | `cat comparison_results.json \| python -m json.tool` | 余弦相似度 > 0.999，Top-1 一致 |
| 发布准备 | 6. README 生成 | 推理结果、精度数据、性能数据 | 运行 generate_readme.py 生成模型文档 | models/\<model_name\>/README.md | 检查 README.md 文件完整性 | 包含推理结果、精度表格、性能数据 |
| 发布 | 7. 模型仓库发布（可选） | 模型权重、README | 运行 publish_repo.py 发布到 GitCode | GitCode 模型仓库 | 检查 GitCode 仓库页面 | 仓库创建成功，文件完整 |
| 汇总 | 8. 串行执行全部模型 | model_list.txt | 串行运行所有 24 个模型，自动管理显存 | 全部模型的精度与性能数据 | 检查终端输出汇总报告 | 全部模型通过精度验证 |

按以下各节顺序执行，每步完成后确认产物再进入下一步。

---

## 执行检查点与用户确认

在以下关键节点暂停执行，向用户报告当前状态并等待确认后再继续：

| 检查点 | 触发时机 | 确认内容 | 不通过处理 |
|--------|---------|---------|-----------|
| ✅ CP-1 环境就绪 | 步骤 0 完成后 | `npu-smi info` 输出显示至少 1 张空闲 NPU 卡，CANN 版本 >= 8.5.1 | 检查硬件连接，重新加载 CANN 环境，或切换到有 NPU 的节点 |
| ✅ CP-2 依赖安装 | 步骤 1 完成后 | `python -c "import torch_npu; print(torch_npu.npu.is_available())"` 返回 True | 检查 pip 源，确认 torch-npu 版本与 CANN 版本匹配 |
| ✅ CP-3 模型下载 | 步骤 2 完成后 | 确认模型权重文件路径存在且大小 > 0 | 检查 ModelScope 网络连接，切换下载源或手动下载 |
| ✅ CP-4 CPU 推理 | 步骤 3 完成后 | CPU 推理正常完成并输出 top-5 分类结果 | 检查输入图片格式和模型权重路径 |
| ✅ CP-5 NPU 推理 | 步骤 4 完成后 | NPU 推理正常完成，无 device 相关错误 | 检查 `ASCEND_RT_VISIBLE_DEVICES` 环境变量 |
| ✅ CP-6 精度对比 | 步骤 5 完成后 | 余弦相似度 > 0.999，Top-1 分类结果一致 | 记录差异模型，判断是否需重跑或报告异常 |
| ✅ CP-7 验收确认 | 全流程完成后 | 精度结果达标、README 已生成、产物完整 | 回到对应步骤修复问题 |

用户确认指令示例：
- "通过" / "继续" — 进入下一步
- "重试 cp-3" — 重新执行检查点 3 对应的步骤
- "跳过 cp-5" — 跳过 NPU 推理（仅做 CPU 验证）

---

## 0. 环境初始化

**输入**：CANN 安装包（/usr/local/Ascend/ascend-toolkit），NPU 硬件（Ascend 910 系列）

**输出**：已加载的 CANN 环境变量，已设置的 NPU 设备号（ASCEND_RT_VISIBLE_DEVICES）

**执行步骤**：

1. **加载 CANN 环境** — source `/usr/local/Ascend/ascend-toolkit/set_env.sh` 确保 CANN 工具链可用
2. **检查 NPU 状态** — `npu-smi info` 查看所有 NPU 卡的状态和空闲情况
3. **设置目标设备** — 选择一张空闲 NPU 卡，设置 `ASCEND_RT_VISIBLE_DEVICES=<卡号>`
4. **验证环境** — 确认 `npu-smi info` 显示设备状态为 "Normal"，CANN 版本满足 >= 8.5.1

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 选择空闲 NPU
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0   # 替换为实际空闲卡号
```

---

## 1. 安装依赖

**输入**：Python 3.9–3.13，已加载的 CANN 环境

**输出**：Python 依赖包已安装，torch_npu 可正常导入

**执行步骤**：

1. **安装核心依赖** — `pip install torch torch-npu timm Pillow safetensors modelscope`
2. **验证 torch_npu** — `python -c "import torch_npu; print(torch_npu.npu.is_available())"` 确认返回 True
3. **检查版本兼容** — 确认 torch-npu 版本与 CANN 版本匹配（如 torch-npu 2.1.0.post8 对应 CANN 8.5.1）

```bash
pip install torch torch-npu timm Pillow safetensors modelscope
```

---

## 2. 模型下载

**输入**：模型名称（model_name），可用的网络连接（用于访问 ModelScope）

**输出**：model.safetensors 权重文件下载到本地缓存目录

**执行步骤**：

1. **确定模型名称** — 从支持列表中找到完整的模型名（如 `convnext_nano.in12k_ft_in1k`）
2. **下载模型权重** — 调用 `snapshot_download("timm/<model_name>")` 从 ModelScope 下载
3. **适配路径编码** — ModelScope 会将 `.` 编码为 `___`，下载后权重路径为 `modelscope_cache/timm/<model_name___with___triple_underscore>/model.safetensors`
4. **验证下载** — 确认权重文件存在且大小 > 0

```python
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download("timm/<model_name>")
# 或指定缓存目录
snapshot_download("timm/<model_name>", cache_dir="./modelscope_cache")
```

ModelScope 会对模型名中的 `.` 编码为 `___`（三个下划线），下载后的权重路径为：

```
modelscope_cache/timm/<model_name___with___triple_underscore>/model.safetensors
```

---

## 3. CPU 推理

**输入**：模型权重文件（model.safetensors），测试图像（test.jpg），模型名称

**输出**：CPU 推理 Top-5 分类结果（终端输出）

**执行步骤**：

1. **准备输入参数** — 指定模型名称、CPU 设备、测试图像路径和权重文件路径
2. **运行推理脚本** — 执行 `python3 scripts/inference.py` 并传入对应参数
3. **验证输出** — 检查终端输出的 top-5 分类结果是否合理（类别名称和对应置信度）

```bash
python3 scripts/inference.py \
    --model convnext_nano.in12k_ft_in1k \
    --device cpu \
    --image test.jpg \
    --model-path /path/to/model.safetensors
```

## 4. NPU 推理

**输入**：模型权重文件（model.safetensors），测试图像（test.jpg），模型名称，已就绪的 NPU 环境

**输出**：NPU 推理 Top-5 分类结果（终端输出）

**执行步骤**：

1. **确认 NPU 就绪** — 确保 `ASCEND_RT_VISIBLE_DEVICES` 已正确设置且 `npu-smi info` 显示设备正常
2. **运行推理脚本** — 执行 `python3 scripts/inference.py`，device 参数设为 `npu`
3. **验证输出** — 检查终端输出无 device 相关错误，top-5 分类结果合理

```bash
python3 scripts/inference.py \
    --model convnext_nano.in12k_ft_in1k \
    --device npu \
    --image test.jpg \
    --model-path /path/to/model.safetensors
```

## 5. CPU/NPU 精度对比

**输入**：CPU 推理结果、NPU 推理结果（由步骤 3、4 生成）

**输出**：`comparison_results.json` — 包含 MAE、MSE、余弦相似度等精度指标

**执行步骤**：

1. **运行对比脚本** — 执行 `python3 scripts/compare_cpu_npu.py`，传入模型名称、测试图像和权重路径
2. **检查精度指标** — 确认余弦相似度 > 0.999，Top-1 分类结果一致
3. **保存结果** — 精度对比数据自动写入 `comparison_results.json`
4. **异常处理** — 若相似度 < 0.99，检查输入预处理是否一致（归一化、resize），并重跑 CPU 推理对比

```bash
python3 scripts/compare_cpu_npu.py \
    --model convnext_nano.in12k_ft_in1k \
    --image test.jpg \
    --model-path /path/to/model.safetensors
```

对比输出会计算以下指标并保存到 `comparison_results.json`：

- MAE（平均绝对误差）
- MSE（均方误差）
- 最大绝对误差
- 余弦相似度
- Top-100 平均相对误差
- Top-1 分类一致性
- Top-5 重叠数

[精度测试结果示例] 所有 24 个模型的 NPU 与 CPU 推理误差均 < 1%。

## 6. README 生成

**输入**：模型推理结果、精度对比数据（comparison_results.json）、性能数据

**输出**：`models/<model_name>/README.md` — 包含推理结果、精度对比表格、性能对比数据和运行截图

**执行步骤**：

1. **收集数据** — 确保 comparison_results.json 和推理日志已生成
2. **生成 README** — 运行 `generate_readme.py` 传入模型名称和模型目录路径
3. **验证产物** — 检查生成的 README.md 是否包含推理结果、精度表格和性能图表
4. **保存模型配置** — 确认生成的文件中包含模型名称、输入尺寸、参数规模等元信息

```bash
python3 scripts/generate_readme.py --model convnext_nano.in12k_ft_in1k --model-dir ./models/convnext_nano.in12k_ft_in1k
```

生成的 README 包含推理结果、精度对比表格、性能对比数据和运行截图。

## 7. 模型仓库发布（可选）

**输入**：模型权重文件、README.md（由步骤 6 生成）、GitCode 访问 Token

**输出**：GitCode 模型仓库创建完成，模型文件已上传

**执行步骤**：

1. **配置认证** — 确保 GitCode 个人访问 Token 已设置环境变量
2. **发布单个模型** — 运行 `publish_repo.py --model <model_name>` 发布指定模型
3. **批量发布** — 运行 `publish_repo.py --all` 发布所有模型
4. **验证发布** — 检查 GitCode 仓库页面确认模型文件和 README 已正确上传

```bash
# 发布单个模型
python3 scripts/publish_repo.py --model convnext_nano.in12k_ft_in1k

# 发布所有模型
python3 scripts/publish_repo.py --all
```

## 8. 串行执行全部模型

**输入**：model_list.txt（24 个模型名称清单），每个模型的权重文件和测试图像

**输出**：全部模型的 CPU/NPU 精度结果汇总，性能对比数据

**执行步骤**：

1. **启动批量脚本** — 运行 `bash scripts/run_all_models.sh` 自动串行处理所有 24 个模型
2. **监控显存** — 每个模型完成后自动执行 `torch.npu.empty_cache()` + `gc.collect()` 释放 NPU 显存
3. **检查中间结果** — 每个模型完成后确认 comparison_results.json 已更新
4. **汇总报告** — 全部模型执行完成后，检查终端输出的汇总报告确认所有模型均通过精度验证

```bash
bash scripts/run_all_models.sh
```

该脚本会自动串行处理所有 24 个模型，每个模型完成后释放显存（`torch.npu.empty_cache()` + `gc.collect()`），防止 OOM。

---

## 精度验证结果

所有 24 个模型已完成 CPU vs NPU 精度验证，典型结果如下：

| 模型类型 | CPU/NPU 误差 | 余弦相似度 | Top-1 一致 | Top-5 一致 |
|----------|-------------|-----------|-----------|-----------|
| convnext_nano.* | < 0.5% | > 0.99997 | 是 | 5/5 |
| convnext_pico.* | < 0.5% | > 0.99997 | 是 | 5/5 |
| convnext_small.* | < 0.5% | > 0.99997 | 是 | 5/5 |
| convnext_tiny.* | < 0.5% | > 0.99997 | 是 | 5/5 |

**结论：所有 24 个模型的 NPU 与 CPU 推理结果误差均 < 1%，精度完全满足要求。**

## 性能数据

| 模型类型 | CPU 耗时 | NPU 耗时 | 加速比 |
|---------|---------|---------|-------|
| convnext_nano (224×224) | ~0.18s | ~0.006s | ~30× |
| convnext_nano (384×384) | ~0.3s | ~0.007s | ~43× |
| convnext_pico (224×224) | ~0.10s | ~0.005s | ~20× |
| convnext_small (224×224) | ~0.40s | ~0.008s | ~50× |
| convnext_small (384×384) | ~0.9s | ~0.010s | ~90× |
| convnext_tiny (224×224) | ~0.25s | ~0.007s | ~36× |
| convnext_tiny (384×224) | ~0.55s | ~0.008s | ~69× |

## 模型仓库地址

每个模型对应一个 GitCode 模型仓库，命名格式为 `<sanitized-model-name>-npu`：

| 模型 | 仓库路径 |
|------|---------|
| convnext_nano.in12k_ft_in1k | convnext-nano-in12k-ft-in1k-npu |
| convnext_nano.d1h_in1k | convnext-nano-d1h-in1k-npu |
| convnext_nano.in12k | convnext-nano-in12k-npu |
| ... | ... |

完整仓库列表见 [GitCode 用户页](https://gitcode.com/gcw_C8PI9e90)。

## 注意事项

1. **串行执行**：多个模型必须串行执行，防止 NPU 显存爆炸。
2. **显存释放**：每个模型测试完后执行 `torch.npu.empty_cache()` + `gc.collect()`。
3. **网络要求**：模型权重从 ModelScope 下载，需要网络连接。
4. **测试图像**：需准备至少一张 JPEG 测试图像用于推理验证。
5. **ModelScope 编码**：模型名中的 `.` 在缓存路径中会被替换为 `___`（三个下划线）。

## 异常处理与回滚策略

| 异常场景 | 表现 | 处理方式 | 回滚/恢复 |
|---------|------|---------|----------|
| NPU 不可用 | `npu-smi info` 报错或无设备 | 确认硬件在位，`source /usr/local/Ascend/ascend-toolkit/set_env.sh` | 切换至 CPU 模式：`--device cpu` |
| NPU OOM | `torch.npu.OutOfMemoryError` | 减少 batch size 为 1，执行 `torch.npu.empty_cache()` | 释放其他进程占用的显存 |
| 模型下载失败 | ModelScope 超时或 404 | 检查网络，重试最多 3 次 | 切换至 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com` |
| 权重文件缺失 | `FileNotFoundError: model.safetensors` | 确认模型名编码规则正确（`.` → `___`） | 手动指定 `--model-path` 参数 |
| CPU/NPU 精度差异大 | 余弦相似度 < 0.99 | 检查输入预处理是否一致（归一化、resize） | 重跑 CPU 推理后对比；仍异常则标记该模型为 "精度存疑" |
| torch_npu 导入失败 | `ImportError: No module named torch_npu` | 确认 torch-npu 版本与 CANN 版本匹配 | `pip install torch-npu==<cann-version>` |
| 磁盘空间不足 | 下载权重时磁盘满 | `df -h` 检查磁盘 | 清理 `modelscope_cache` 目录或更换下载路径 |
| 推理结果 NaN | NPU 输出全为 NaN | 检查 `ASCEND_RT_VISIBLE_DEVICES` 是否正确 | 重新设置环境变量并重启 Python 进程 |

## 资源与评测产物

本 Skill 在 `skills/deployment/convnext-npu/` 目录下提供以下资源：

| 资源路径 | 用途 |
|---------|------|
| `SKILL.md`（本文件） | 完整执行流程与异常处理指南 |
| `scripts/inference.py` | 单模型 CPU/NPU 推理脚本 |
| `scripts/compare_cpu_npu.py` | CPU 与 NPU 精度对比脚本 |
| `scripts/run_all_models.sh` | 串行执行全部 24 个模型的 Shell 脚本 |
| `scripts/generate_readme.py` | 模型 README 自动生成脚本 |
| `scripts/model_list.txt` | 完整 24 个模型名称清单 |
| `examples/quickstart.py` | 快速入门示例脚本 |
| `test-prompts.json` | 评测提示词与预期结果 |

执行结束后确认以下产物已生成：

- [ ] `comparison_results.json` — 每个模型的 CPU/NPU 精度对比数据
- [ ] `models/<model_name>/README.md` — 模型 README 文档
- [ ] 终端输出的推理结果与性能数据
