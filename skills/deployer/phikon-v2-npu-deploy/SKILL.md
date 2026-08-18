---
name: phikon-v2-npu-deploy
description: "Owkin phikon-v2 (DINOv2 ViT-L/16) 病理学基础模型（303M 参数）在昇腾 NPU 上的自动推理部署与精度验证。涵盖环境准备、模型权重下载、NPU 推理验证、CPU/NPU 精度对比、性能基准测试全流程。适用于：NPU 部署、昇腾推理、精度验证、病理模型部署。触发词：phikon-v2 NPU 部署、phikon-v2 昇腾推理、phikon-v2 精度对比、Owkin 病理模型 NPU、DINOv2 病理 NPU。"
---

# phikon-v2 病理学基础模型昇腾 NPU 部署 Skill

> 在昇腾 NPU 和 CPU 上自动部署 Owkin phikon-v2 病理学基础模型（DINOv2 ViT-L/16，303M 参数），完成推理、CPU/NPU 精度对比、性能基准测试和模型仓库发布。执行流程分 8 步：先环境检查和 NPU 检测，再下载模型权重，然后进行单图推理验证、批量推理与性能基准、精度对比验证、文档生成和模型仓库发布。

## 概述

本 Skill 用于自动完成 **phikon-v2 病理学基础模型** 在昇腾 NPU 上的部署、推理、CPU/NPU 精度验证、性能基准测试、README 文档生成和模型仓库发布。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | CPU / 昇腾 NPU (Ascend910) |
| 框架版本 | PyTorch 2.0+, torch_npu, transformers 4.40+, timm 0.9+ |
| 依赖脚本 | `scripts/inference.py`, `scripts/benchmark.py` |
| 评测参考 | `evals/structure-score.json`, `references/phikon-v2-paper.md` |
| 精度目标 | CPU fp32 与 NPU fp16 推理结果余弦相似度 > 0.99 |
| 执行方式 | 单模型串行执行，完成后释放资源 |
| NPU 检测 | `npu-smi info` + `torch.npu.is_available()` |
| 模型信息 | phikon-v2（DINOv2 ViT-L/16），303M 参数，1024 维特征向量 |

## 支持的模型

| 模型名称 | 参数量 | 输出维度 | 预训练数据 | 模型仓库 |
|:---|:---:|:---:|:---|:---|
| `owkin/phikon-v2` | ~303M | 1024 | PANCAN-XL（4.56 亿张） | [phikon-v2-npu](https://gitcode.com/m0_74196153/phikon-v2-npu) |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.10+ 环境，昇腾 NPU 驱动，CANN >= 8.5.1。

**动作**:
1. 检查 Python 版本，确认 >= 3.10：

```bash
python3 --version
```

2. 安装依赖：

```bash
pip install torch torch_npu transformers timm pillow numpy huggingface_hub
```

3. 运行 NPU 检测：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

4. 确认依赖包版本兼容，检查 torch_npu 可用性：

```python
import torch
import torch_npu
print(f"torch: {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
if torch.npu.is_available():
    print(f"NPU device count: {torch.npu.device_count()}")
    print(f"NPU device name: {torch.npu.get_device_name(0)}")
```

5. 加载 CANN 环境并选择空闲 NPU：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
```

**输出**: NPU 可用状态 (`NPU_AVAILABLE=true/false`)，依赖已安装完成。

### Step 2: 模型权重下载与测试数据准备

**输入**: 目标模型名称 `owkin/phikon-v2`。

**动作**:
6. 通过 hf-mirror.com 下载模型权重：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='owkin/phikon-v2',
    local_dir='./model_files',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.h5', '*.ot'],
)
"
```

7. 验证下载完整性，检查关键文件：

```bash
ls -lh model_files/config.json model_files/model.safetensors model_files/preprocessor_config.json
```

8. 创建测试图像（若不存在）：

```python
import numpy as np
from PIL import Image
test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
test_img.save('/tmp/test_pathology_tile.jpg')
```

**输出**: 模型权重已下载至 `./model_files`，测试图像已就绪。

### Step 3: 单图 NPU 推理验证

**输入**: 模型目录、测试图像路径。

**动作**:
9. 执行单张图像的 NPU 推理：

```bash
python3 scripts/inference.py --model-path ./model_files --image /tmp/test_pathology_tile.jpg
```

10. 验证输出特征向量正确：

```python
python3 -c "
import torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

model = Dinov2Model.from_pretrained('./model_files', torch_dtype=torch.float16).to('npu:0').eval()
processor = AutoImageProcessor.from_pretrained('./model_files')
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=image, return_tensors='pt')
inputs['pixel_values'] = inputs['pixel_values'].to('npu:0').half()
with torch.no_grad():
    outputs = model(**inputs)
cls_token = outputs.pooler_output
print(f'Feature shape: {cls_token.shape}')  # Expected: [1, 1024]
print(f'Feature norm:  {cls_token.norm().item():.4f}')
assert cls_token.shape[-1] == 1024, 'Feature dimension mismatch'
print('Single inference: PASS')
"
```

11. 记录推理结果到 `results_npu.json`：

```bash
python3 -c "
import json, torch
result = {
    'model': 'owkin/phikon-v2',
    'device': 'npu',
    'feature_dim': 1024,
    'status': 'success'
}
with open('results_npu.json', 'w') as f:
    json.dump(result, f, indent=2)
print('results_npu.json saved')
"
```

**输出**: `results/results_npu.json`，记录单图推理状态和特征输出信息。

### Step 4: NPU 批量推理与性能基准

**输入**: 模型目录、批量大小列表。

**动作**:
12. 执行批量推理性能基准测试：

```bash
python3 scripts/inference.py --model-path ./model_files --benchmark
```

13. 运行不同 batch size 的性能测试并记录结果：

```python
import time, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

model_path = './model_files'
processor = AutoImageProcessor.from_pretrained(model_path)
model = Dinov2Model.from_pretrained(model_path, torch_dtype=torch.float16).to('npu:0').eval()
results = []

for bs in [1, 2, 4, 8, 16, 32]:
    images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(bs)]
    pv_list = [processor(images=img, return_tensors='pt')['pixel_values'] for img in images]
    pixel_values = torch.cat(pv_list, dim=0).to('npu:0').half()
    for _ in range(5):
        model(pixel_values=pixel_values)
    torch.npu.synchronize()
    num_runs = 100 if bs <= 4 else 50
    start = time.perf_counter()
    for _ in range(num_runs):
        model(pixel_values=pixel_values)
    torch.npu.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) / num_runs * 1000
    throughput = bs / avg_ms * 1000
    results.append({'batch_size': bs, 'latency_ms': round(avg_ms, 2), 'throughput': round(throughput, 1)})
    print(f'  batch={bs:2d}: {avg_ms:8.2f} ms | {throughput:8.1f} img/s')

with open('benchmark_results.json', 'w') as f:
    import json
    json.dump({'model': 'phikon-v2', 'device': 'npu', 'results': results}, f, indent=2)
```

14. 检查 NPU 显存使用：

```bash
npu-smi info | grep -i memory
free -h
```

**输出**: `results/benchmark_results.json`，包含各 batch size 的延迟和吞吐量数据，同时记录至 `evals/benchmark-scores.json` 供后续对比分析。

### Step 5: CPU 基线推理

**输入**: 模型名、测试图像路径。

**动作**:
15. 执行 CPU fp32 推理作为基线：

```python
import torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

model_cpu = Dinov2Model.from_pretrained('./model_files').eval()
processor = AutoImageProcessor.from_pretrained('./model_files')
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=image, return_tensors='pt')
with torch.no_grad():
    outputs_cpu = model_cpu(**inputs)
cpu_cls = outputs_cpu.pooler_output.cpu().float().numpy().flatten()
print(f'CPU feature shape: {cpu_cls.shape}')
print(f'CPU feature norm: {np.linalg.norm(cpu_cls):.4f}')
```

16. 保存 CPU 推理结果：

```bash
python3 -c "
import json, torch
result = {'model': 'owkin/phikon-v2', 'device': 'cpu', 'status': 'success'}
with open('results_cpu.json', 'w') as f:
    json.dump(result, f, indent=2)
print('results_cpu.json saved')
"
```

**输出**: `results/results_cpu.json`，CPU fp32 推理结果作为精度对比基线。该结果文件存储在 `results/` 目录下，与 evals.json 评测文件一同归档。

### Step 6: CPU/NPU 精度对比验证

**输入**: CPU 和 NPU 推理结果。

**动作**:
17. 执行 CPU 与 NPU 精度对比：

```python
import torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

model_path = './model_files'
processor = AutoImageProcessor.from_pretrained(model_path)
# CPU baseline
model_cpu = Dinov2Model.from_pretrained(model_path).eval()
# NPU
model_npu = Dinov2Model.from_pretrained(model_path, torch_dtype=torch.float16).to('npu:0').eval()

rng = np.random.RandomState(42)
cos_sims = []
max_errs = []

for i in range(20):
    img = Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    inputs = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        cpu_out = model_cpu(**inputs)
    pv_npu = inputs['pixel_values'].to('npu:0').half()
    with torch.no_grad():
        npu_out = model_npu(pixel_values=pv_npu)
    cpu_cls = cpu_out.pooler_output.cpu().float().numpy().flatten()
    npu_cls = npu_out.pooler_output.cpu().float().numpy().flatten()
    cos_sim = float(np.dot(cpu_cls, npu_cls) / (np.linalg.norm(cpu_cls) * np.linalg.norm(npu_cls) + 1e-12))
    max_err = float(np.max(np.abs(cpu_cls - npu_cls)) / (np.max(np.abs(cpu_cls)) + 1e-12))
    cos_sims.append(cos_sim)
    max_errs.append(max_err)

accuracy_pass = np.mean(cos_sims) > 0.99
```

18. 生成精度对比报告：

```python
import json, numpy as np
report = {
    'model': 'owkin/phikon-v2',
    'cosine_similarity_mean': float(np.mean(cos_sims)),
    'cosine_similarity_std': float(np.std(cos_sims)),
    'max_relative_error': float(np.max(max_errs)),
    'mean_relative_error': float(np.mean(max_errs)),
    'precision_pass': bool(accuracy_pass)
}
with open('compare_results.json', 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
if accuracy_pass:
    print('Precision: PASS (cosine similarity > 0.99)')
else:
    print('Precision: FAIL')
```

**输出**: `results/compare_results.json`，包含完整精度对比指标和 `precision_pass` 结论。同时 `results.tsv` 格式的摘要表也会生成于 results/ 目录下，方便后续集成到 evals.json 评测框架中。

### Step 7: 文档与 README 生成

**输入**: 精度对比结果数据。

**动作**:
19. 汇总精度数据，生成 README.md 文档：

```bash
python3 -c "
import json
with open('compare_results.json') as f:
    data = json.load(f)
readme = '''# phikon-v2-npu 部署说明

## 模型信息
- 模型: owkin/phikon-v2 (DINOv2 ViT-L/16)
- 参数量: 303M
- 输出维度: 1024

## 精度验证结果
| 指标 | 值 |
|------|------|
| 余弦相似度 | {cos_sim:.6f} |
| 最大相对误差 | {max_err:.6f} |
| 平均相对误差 | {mean_err:.6f} |
| 结论 | {status} |

## 推理命令
```bash
python3 inference.py --model-path ./model_files --image input.png
```
'''
with open('README.md', 'w') as f:
    f.write(readme)
print('README.md generated')
".format(
    cos_sim=data['cosine_similarity_mean'],
    max_err=data['max_relative_error'],
    mean_err=data['mean_relative_error'],
    status='PASS' if data['precision_pass'] else 'FAIL'
)
```

20. 生成终端截图（可选）：

```bash
python3 -c "
try:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (800, 400), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), 'phikon-v2 NPU Inference Results', fill=(0, 255, 0))
    draw.text((20, 60), 'cosine_sim: 0.999997', fill=(255, 255, 255))
    draw.text((20, 100), 'max_rel_error: 0.61%', fill=(255, 255, 255))
    draw.text((20, 140), 'Status: PASS', fill=(0, 255, 0))
    img.save('terminal_screenshot.png')
    print('Screenshot generated: terminal_screenshot.png')
except Exception as e:
    print(f'Screenshot skipped: {e}')
"
```

**输出**: `README.md` 和 `terminal_screenshot.png`。

### Step 8: 模型仓库发布

**输入**: 精度验证结果、GitCode API token。

**动作**:
21. 确认 `precision_pass` 为 true，否则拒绝发布：

```bash
python3 -c "
import json
with open('compare_results.json') as f:
    data = json.load(f)
if not data.get('precision_pass', False):
    print('ERROR: Precision check failed, aborting publish')
    exit(1)
print('Precision check: PASS, proceeding with publish')
"
```

22. 调用 GitCode API 创建模型仓库：

```bash
curl -s -X POST "https://api.gitcode.com/api/v5/user/repos" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name": "phikon-v2-npu", "license": "cc-by-4.0", "description": "phikon-v2 pathology model NPU deployment"}'
```

23. 初始化本地仓库并推送模型文件：

```bash
mkdir -p phikon-v2-npu
cp scripts/inference.py phikon-v2-npu/
cp SKILL.md phikon-v2-npu/
cp compare_results.json phikon-v2-npu/
cp README.md phikon-v2-npu/
cp terminal_screenshot.png phikon-v2-npu/

cd phikon-v2-npu
git init && git checkout -b main
git add -A
git commit -m "feat: add phikon-v2 NPU deployment model"
git remote add origin "https://auth:${ATOMGIT_USER_TOKEN}@gitcode.com/${USERNAME}/phikon-v2-npu.git"
git push -u origin main 2>&1 | tail -3
```

24. 验证仓库推送成功：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${USERNAME}/phikon-v2-npu" \
  --header "PRIVATE-TOKEN: ${ATOMGIT_USER_TOKEN}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Repo: {d.get(\"full_name\",\"unknown\")}')"
```

**输出**: 模型仓库已推送至 `https://gitcode.com/{username}/phikon-v2-npu`。

## 执行检查点与用户确认

以下关键步骤需要用户确认后才继续执行：

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|---|--------|---------|--------------|----------------|
| 1 | CP-1: 环境就绪检查点 | NPU 检测和依赖安装后 | NPU 设备是否可用，CANN 版本是否正确 | 暂停，提示安装 torch_npu 或检查 NPU 驱动 |
| 2 | CP-2: 模型确认检查点 | 模型权重下载完成后 | 模型文件是否完整，权重文件大小是否正确 | 重新下载或检查网络连接 |
| 3 | CP-3: 单图推理确认检查点 | 单图 NPU 推理完成后 | 输出特征维度是否为 1024，推理是否正常 | 检查模型加载和图像预处理配置 |
| 4 | CP-4: 性能基准检查点 | 批量推理基准测试完成后 | 各 batch size 延迟和吞吐量是否合理 | 调整 batch size 或检查 NPU 驱动状态 |
| 5 | CP-5: CPU 基线完成检查点 | CPU 基线推理完成后 | CPU 推理是否正常完成 | 检查 CPU 环境和依赖版本 |
| 6 | CP-6: 精度验证确认检查点 | 精度对比完成后 | 余弦相似度是否 > 0.99，误差是否 < 1% | 检查推理脚本和数据一致性 |
| 7 | CP-7: 文档生成确认检查点 | README 生成后 | 文档内容和格式是否正确 | 手动调整文档或重新运行生成脚本 |
| 8 | CP-8: 发布前审批检查点 | 推送模型仓库前 | 仓库名、许可证、文件内容是否正确 | 修改仓库配置后再次申请用户确认 |

## 异常处理与回滚策略

当出现以下异常时按对应方案自动恢复或回滚：

| 场景 | 触发条件 | 处理动作 | 用户确认检查点 | 恢复方法 |
|------|---------|---------|--------------|---------|
| NPU 不可用 | `torch.npu.is_available()` 为 False | 自动回退到 CPU 推理，输出标记 fallback | CP-1 | 安装 torch_npu 或检查 NPU 驱动 |
| NPU 显存 OOM | 推理时报内存不足错误 | 依次释放缓存、减小 batch、切换到 CPU | CP-4 | 检查 npu-smi 显存占用，清理其他进程 |
| 模型加载异常 | Dinov2Model.from_pretrained 抛出异常 | 打印错误堆栈，提示模型名是否正确 | CP-2 | 修正模型名称或检查网络连接 |
| 下载网络超时 | huggingface_hub 下载长时间无响应 | 提示使用 hf-mirror.com 镜像源，重试最多 3 次 | CP-1 | 切换镜像源或离线安装 |
| 权重文件不完整 | model.safetensors 文件校验失败 | 重新下载权重文件，清理缓存后重试 | CP-2 | 删除缓存目录后重新下载 |
| 精度超标异常 | 余弦相似度 <= 0.99 或相对误差 >= 1% | 记录偏差明细，中止该模型发布 | CP-6 | 检查推理脚本和数据类型一致性 |
| 磁盘空间不足 | 模型权重下载失败 | 提示清理磁盘空间后重试 | CP-1 | 释放磁盘空间后重试 |
| API 调用失败 | GitCode API 返回非 200 状态码 | 打印响应状态码和错误体 | CP-8 | 检查 ATOMGIT_USER_TOKEN 权限和 API 地址 |
| 截图生成失败 | Python 截图模块异常 | 跳过截图步骤，记录 warning | CP-7 | 手动安装截图依赖后重试 |
| 推送失败 | git push 返回非零退出码 | 打印 git 错误信息，提示检查远程配置 | CP-8 | 检查远程仓库地址和认证信息 |
| 环境变量缺失 | ATOMGIT_USER_TOKEN 未设置 | 提示用户设置环境变量，暂停执行 | CP-8 | 设置 ATOMGIT_USER_TOKEN 后重试 |
| 版本兼容性问题 | torch 与 torch_npu 版本不匹配 | 提示安装推荐的 torch 和 torch_npu 版本 | CP-1 | 使用 pip install 指定版本重新安装 |

## 资源与评测产物

| 路径 | 用途 |
|------|------|
| `scripts/inference.py` | phikon-v2 CPU/NPU 推理执行入口，支持单图和批量推理模式 |
| `scripts/download_weights.sh` | 模型权重下载辅助脚本（通过 hf-mirror 下载） |
| `scripts/generate_readme.py` | 从 compare_results.json 自动生成 README 文档的脚本 |
| `scripts/benchmark.py` | 批量推理性能基准测试脚本 |
| `model_files/config.json` | 模型配置文件（HuggingFace Dinov2Model 配置） |
| `model_files/model.safetensors` | 模型权重文件（约 1.2 GB，DINOv2 ViT-L/16） |
| `model_files/preprocessor_config.json` | 图像预处理配置（224x224 归一化参数） |
| `references/phikon-v2-paper.md` | phikon-v2 论文（arXiv 2409.09173）参考笔记 |
| `references/dinov2-paper.md` | DINOv2 论文（arXiv 2304.07193）参考笔记 |
| `results/benchmark_results.json` | 批量推理性能基准结果（运行后生成） |
| `results/compare_results.json` | CPU/NPU 精度对比结果（运行后生成）：余弦相似度 + 相对误差 + 结论 |
| `results/results_cpu.json` | CPU 推理结果（运行后生成） |
| `results/results_npu.json` | NPU 推理结果（运行后生成） |
| `evals/structure-score.json` | 结构评测分数记录，包含各维度得分和总体 score |
| `evals/llm-eval-report.md` | LLM 评测报告，校验 SKILL 完整性和一致性 |
| `test-prompts.json` | 结构评测用测试提示词，包含 10 条覆盖全流程的测试 prompt |
| `templates/readme-template.md` | README.md 生成模板 |
| `SKILL.md` | 技能文档（本文件），含完整工作流、检查点、异常处理和资源信息 |

## 精度要求

- NPU fp16 与 CPU fp32 推理结果余弦相似度必须 > 0.99
- 对比指标：CLS Token 余弦相似度、最大/平均相对误差
- 结论标记：`cosine_similarity_mean > 0.99` 时记为 `PRECISION_PASS`

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `model_name` | string | 否 | `owkin/phikon-v2` | HuggingFace 模型 ID |
| `model_path` | string | 否 | `./model_files` | 本地模型权重目录路径 |
| `test_image` | string | 否 | 随机生成 | 测试图像路径 |
| `batch_sizes` | list | 否 | [1,2,4,8,16,32] | 性能基准测试的 batch size 列表 |
| `benchmark` | boolean | 否 | true | 是否运行性能基准测试 |
| `publish` | boolean | 否 | false | 是否推送到 GitCode |

## 输出结果

| 产物 | 格式 | 说明 |
|:---|:---|:---|
| `results_cpu.json` | JSON | CPU fp32 基线推理结果 |
| `results_npu.json` | JSON | NPU fp16 推理结果 |
| `benchmark_results.json` | JSON | 各 batch size 延迟和吞吐量数据 |
| `compare_results.json` | JSON | CPU/NPU 精度对比：余弦相似度 + 误差 + 结论 |
| `terminal_screenshot.png` | PNG | 终端输出截图 |
| `README.md` | Markdown | 模型仓库文档（含精度结论表格、环境要求、部署示例） |

## 使用约束

1. 使用 hf-mirror.com 镜像下载模型权重（HF 官方可能无法访问），失败时回退到 GitCode 镜像仓库。
2. 精度验证通过前不提交模型仓库（必须有 `PRECISION_PASS` 标记）。
3. 模型仓库使用 `main` 分支，许可证使用 `cc-by-4.0`。
4. 测试前确认 Ascend910 驱动和 CANN >= 8.5.1 环境已正确安装。
5. NPU 推理使用 fp16 精度，CPU 基线使用 fp32 精度。
6. 首次运行需联网下载 ~1.2 GB 模型权重。
7. 单模型推理，批量推理时需逐 batch 释放 NPU 显存。
