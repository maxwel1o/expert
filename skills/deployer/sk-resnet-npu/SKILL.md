---
name: sk-resnet-npu
description: SK-ResNet models in 昇腾 NPU deployment, including SKResNeXt50, SKResNet34, and SKResNet18. Automates NPU adaptation, CPU/NPU inference, accuracy comparison, terminal screenshot generation, and GitCode model repository publishing for Selective Kernel Network models on Ascend910 NPU.
---

# SK-ResNet 昇腾 NPU 部署 Skill

本 Skill 用于 SKNet（Selective Kernel Networks）系列图像分类模型在 **昇腾 Ascend910 NPU** 上的端到端适配部署、推理验证、CPU/NPU 精度对比、终端截图生成和 GitCode 模型仓库发布。支持 skresnext50_32x4d, skresnet34, skresnet18 三个模型的串行部署。

## 支持的模型列表

| 序号 | 模型名称 | 架构 | 参数量 | 输入尺寸 | 模型仓库地址 |
|------|---------|------|--------|---------|-------------|
| 1 | `skresnext50_32x4d.ra_in1k` | SKResNeXt50 | ~25M | 224×224 | [链接](https://gitcode.com/m0_74196153/skresnext50_32x4d.ra_in1k-npu) |
| 2 | `skresnet34.ra_in1k` | SKResNet34 | ~22M | 224×224 | [链接](https://gitcode.com/m0_74196153/skresnet34.ra_in1k-npu) |
| 3 | `skresnet18.ra_in1k` | SKResNet18 | ~12M | 224×224 | [链接](https://gitcode.com/m0_74196153/skresnet18.ra_in1k-npu) |

## Skill 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 是 | timm 模型名称，支持 skresnext50_32x4d.ra_in1k / skresnet34.ra_in1k / skresnet18.ra_in1k |
| `action` | string | 否 | 执行操作：inference / compare / readme / screenshot / publish / all（默认 all） |

## Skill 输出结果

| 输出 | 格式 | 说明 |
|------|------|------|
| inference_result | JSON | CPU/NPU 推理结果，含 Top-5 预测标签和概率 |
| compare_result | JSON | CPU/NPU 精度对比结果，含最大概率差异和余弦相似度 |
| screenshot | PNG | 模拟终端输出截图 |
| repo_url | URL | GitCode 模型仓库地址 |
| readme | markdown | 标准化的中文模型 README 文档 |

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| NPU | Ascend910 (Atlas 800 A2) |
| CANN | >= 8.5.1 |
| Python | >= 3.11 |
| PyTorch | >= 2.9.0 |
| torch_npu | >= 2.9.0.post1 |
| timm | >= 0.9.0 |
| ModelScope | 最新版（权重下载） |

## 工作流程

### 执行总览

1. **环境初始化与模型验证**：读取 model_name 和 action 参数，确认模型属于支持列表（skresnext50_32x4d / skresnet34 / skresnet18），执行 npu-smi info 检查 NPU 状态，验证 CANN 环境和 torch_npu 导入。
2. **模型权重下载**：通过 ModelScope 或 timm 下载模型预训练权重，设置缓存目录，记录下载来源和缓存路径。
3. **CPU 推理**：使用 timm 创建模型并加载权重，在 CPU 设备上执行推理，保存 Top-K 分类结果和 logits 到文件。
4. **NPU 推理**：将模型迁移至 NPU 设备（npu:0），执行推理，保存 NPU 推理结果和 logits。
5. **CPU/NPU 精度对比**：运行 CPU 和 NPU 推理结果对比，计算最大概率差异、余弦相似度、Top-1 标签一致性等指标，生成精度对比表和结论。
6. **终端截图生成**：收集推理和对比日志，使用 terminal_screenshot.py 生成模拟终端截图。
7. **README 文档与结果导出**：基于推理结果和精度数据，生成标准化的中文 README，导出 results.tsv 和 evals.json。
8. **模型仓库发布**：调用 GitCode API 创建模型仓库，配置 remote，提交代码并推送，回读验证页面 NPU 标签和精度信息可见。

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 拒绝或失败处理 |
|--------|---------|-------------|---------------|
| CP-1 环境检查点 | npu-smi info 执行后 | 用户确认当前 Ascend/CANN/NPU 环境正常 | 标记为 dry-run 模式，不写入真实 NPU 结论 |
| CP-2 权重下载检查点 | ModelScope 下载前 | 用户确认模型名称和权重来源 | 切换至 hf-mirror 镜像或复用本地缓存，失败则跳过该模型 |
| CP-3 CPU 推理检查点 | CPU 推理完成后 | 用户确认 Top-5 结果合理 | 回退检查输入图片和预处理参数，重新推理 |
| CP-4 NPU 推理检查点 | NPU 推理完成后 | 用户确认 NPU 推理无异常 | 释放 NPU 显存后重试推理 |
| CP-5 精度验证检查点 | CPU/NPU 对比完成后 | 用户确认最大概率差异 < 1% | 标记精度验证失败，保留 logits 日志，禁止发布通过结论 |
| CP-6 发布准备检查点 | 创建 GitCode 仓库前 | 用户确认仓库名和 README 内容 | 停止发布，仅保留本地 results.tsv、evals.json 和 README 草稿 |
| CP-7 最终验收检查点 | GitCode 推送完成后 | 用户确认模型页可见 NPU 标签和精度结论 | 回滚 README 或追加修复提交 |

## 执行步骤详解

### 1. NPU 环境检查

```bash
# 检查 NPU 设备状态
npu-smi info

# 检查 torch_npu 导入
python3 -c "import torch; print('NPU available:', torch.npu.is_available()); print('Device:', torch.npu.get_device_name(0))"
```

### 2. 安装依赖

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm torchvision Pillow numpy safetensors modelscope
```

### 3. 下载模型权重

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('timm/skresnext50_32x4d.ra_in1k', cache_dir='/opt/atomgit/.cache/modelscope')
snapshot_download('timm/skresnet34.ra_in1k', cache_dir='/opt/atomgit/.cache/modelscope')
snapshot_download('timm/skresnet18.ra_in1k', cache_dir='/opt/atomgit/.cache/modelscope')
```

### 4. CPU 推理

```bash
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action inference
```

使用 Python API 进行单模型推理：

```python
import torch
import timm
from PIL import Image
from torchvision import transforms

model = timm.create_model("skresnext50_32x4d.ra_in1k", pretrained=True).to("cpu")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open("test.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
probs = torch.nn.functional.softmax(output[0], dim=0)
top5 = torch.topk(probs, k=5)
print(f"Top-5 predictions: {top5}")
```

### 5. NPU 推理

```bash
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action inference
```

NPU 推理需将模型迁移至 NPU 设备：

```python
model = model.to("npu:0")
input_tensor = input_tensor.to("npu:0")

with torch.no_grad():
    output = model(input_tensor)
probs = torch.nn.functional.softmax(output[0], dim=0)
top5 = torch.topk(probs, k=5)
print(f"NPU Top-5 predictions: {top5}")
```

### 6. CPU/NPU 精度对比

```bash
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action compare
```

对比指标包括：最大绝对误差、余弦相似度、相对误差、最大概率差异、Top-1 标签一致性。精度要求：最大概率差异 < 1%。

### 7. 串行执行多个模型

为防止 NPU 显存爆炸，多个模型必须串行执行：

```bash
# 方式一：Python 串行模式
python3 scripts/deploy.py --serial --action all

# 方式二：手动串行
for model in skresnext50_32x4d.ra_in1k skresnet34.ra_in1k skresnet18.ra_in1k; do
    python3 scripts/deploy.py --model $model --action all
    python3 -c "import gc; gc.collect(); import torch; torch.npu.empty_cache()"
done

# 方式三：Shell 脚本
chmod +x scripts/run.sh
./scripts/run.sh skresnext50_32x4d.ra_in1k all
```

### 8. 生成 README、截图并发布

```bash
# 生成 README 文档
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action readme

# 生成终端截图
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action screenshot

# 发布模型仓库到 GitCode
python3 scripts/deploy.py --model skresnext50_32x4d.ra_in1k --action publish
```

发布操作通过 GitCode API 创建仓库并推送代码：

```bash
# 创建 GitCode 模型仓库
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "private-token: ${ATOMGIT_USER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name": "{model_name}-npu", "description": "SKNet {model_name} NPU adapted model", "visibility": "public"}'

# 推送代码到仓库
git init
git checkout -b main
git remote add origin https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/m0_74196153/{model_name}-npu.git
git add -A
git commit -m "Add {model_name} NPU adapted model"
git push -u origin main
```

## 异常处理与回滚策略

| 场景 | 触发条件 | fallback / retry / recover 动作 | 验证输出 |
|------|---------|-------------------------------|---------|
| 不支持的模型名 | model_name 不在支持列表中 | 输出可用模型列表，要求用户确认替换 | evals.json 记录 unsupported_model |
| NPU 设备不可用 | npu-smi info 失败或无 Ascend910 | fallback 到 CPU dry-run 模式，禁止写入真实 NPU 结论 | README 标记 dry_run |
| CANN 环境缺失 | ASCEND_HOME 或 LD_LIBRARY_PATH 未设置 | 提示加载 CANN 环境脚本，retry 一次环境检查 | 环境检查日志输出 |
| torch_npu 导入失败 | import torch_npu 抛出 ImportError | 提示安装 torch_npu，retry 一次导入 | evals.json 记录环境错误 |
| 权重下载失败 | ModelScope 网络超时或模型不存在 | retry 2 次，切换 hf-mirror 镜像或使用本地缓存 | results.tsv 记录权重来源 |
| NPU 显存溢出 | OOM 或 device busy 异常 | 执行 gc.collect() 和 torch.npu.empty_cache() 后 retry | evals.json 记录 retry 次数 |
| CPU/NPU 推理结果为空 | logits 输出全部为零或 NaN | 检查输入图片路径和预处理参数，重新推理 | 推理日志记录错误原因 |
| 精度不达标 | 最大概率差异 >= 1% | 保留 CPU/NPU logits，标记 failed，禁止发布通过结论 | 精度对比表显示失败指标 |
| GitCode 仓库创建失败 | API 返回 422 或 409 冲突 | 检查仓库名唯一性，若已存在则更新现有仓库 | git 远程操作日志 |
| Git push 失败 | 推送时认证失败或网络中断 | 检查 token 权限，retry 2 次推送 | evals.json 记录发布状态 |
| README 验证失败 | 模型页缺失 NPU 或精度信息 | 追加 README 修复提交，刷新页面验证 | 页面截图验证 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| scripts/deploy.py | Python 部署入口脚本，支持 inference / compare / readme / screenshot / publish 操作 |
| scripts/run.sh | Shell 部署脚本，支持串行执行和参数传递 |
| examples/example_usage.md | 使用示例文档，含推理代码和依赖安装 |
| examples/run_all_serial.sh | 串行运行三个模型的 Shell 脚本 |
| results.tsv | 每个模型的推理、精度、状态和错误原因汇总表格 |
| evals.json | 结构化保存环境检查、重试、验证和发布结果 |
| references/ | 模型仓库地址和来源参考文档目录 |
| test-prompts.json | 提供重复评估本 Skill 的测试提示 |

## 使用约束

1. **串行执行**：所有模型必须串行推理，严禁并行执行。每个模型推理完成后必须释放 NPU 显存（torch.npu.empty_cache() + gc.collect()），防止显存溢出。
2. **权重来源**：优先使用 ModelScope 下载模型权重，缓存目录为 /opt/atomgit/.cache/modelscope。若下载失败，切换至 hf-mirror.com 镜像。
3. **输入图片**：所有 SKNet 模型输入尺寸为 224×224，使用标准 ImageNet 预处理（Resize 256 + CenterCrop 224 + Normalize）。
4. **精度标准**：NPU 与 CPU 推理结果最大概率差异 < 1% 即判断为通过。若差异 >= 1%，保留推理日志并禁止发布。
5. **数据真实性**：README 和精度结论中必须包含真实测试数据，严禁编写无数据支撑的空文档。
6. **环境变量**：GitCode API 调用依赖 ATOMGIT_USER_TOKEN 环境变量，推送前需确认该变量已设置。
7. **发布确认**：创建 GitCode 模型仓库和推送代码前，必须等待用户确认仓库名、目标分支和 README 内容。
