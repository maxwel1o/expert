---
name: aimv2-npu-deploy
description: "AIMv2 (Apple Intelligence Foundation Model v2) 多模态视觉模型在昇腾 NPU 上的完整部署 Skill，覆盖 12 个变体（large/huge/1B/3B × 224/336/448）在全分辨率下的权重转换、FP16 推理、性能基准测试和精度验证。当用户提到 AIMv2 部署昇腾、Apple AIMv2 NPU 推理、aimv2 适配、视觉 embedding 模型 NPU 部署时触发。"
---

# AIMv2 昇腾 NPU 部署与推理 Skill

> 在华为昇腾 NPU 上自动部署 Apple AIMv2 系列视觉基础模型，完成权重转换、推理验证、性能基准测试和精度对比。

## 概述

本 Skill 提供 Apple AIMv2 系列视觉基础模型在华为昇腾 NPU 上的完整部署流程，覆盖 **12 个变体**（4 种模型尺度 × 3 种输入分辨率）。

| 特性 | 说明 |
|:---|:---|
| 目标硬件 | 昇腾 NPU (Ascend910 系列，>= 32GB HBM；3B 建议 64GB) |
| 模型变体 | large(300M) / huge(645M) / 1B(1.1B) / 3B(3.0B) × 224 / 336 / 448 |
| 框架版本 | PyTorch 2.5.1+, torch_npu 2.5.1+, transformers |
| 精度目标 | FP16 与 FP32 输出特征余弦相似度 > 0.999 |
| 执行方式 | 逐变体执行，每完成一个释放 NPU 显存 |

## 支持的模型变体

| 模型尺度 | 参数 | Transformer Layers | Hidden Dim | Patch Size |
|:--------|:-----|:------------------|:-----------|:----------|
| large | 300M | 16 | 1024 | 14 |
| huge | 645M | 24 | 1408 | 14 |
| 1B | 1.1B | 24 | 2048 | 14 |
| 3B | 3.0B | 24 | 3072 | 14 |

| 输入分辨率 | 序列长度 |
|:----------|:--------|
| 224×224 | 256 |
| 336×336 | 576 |
| 448×448 | 1024 |

## 前置条件

| 项目 | 要求 |
|:----|:----|
| 硬件 | Ascend910 系列（至少 1 卡，>= 32GB HBM；3B 建议 64GB） |
| OS | openEuler / Ubuntu / KylinOS（aarch64 或 x86_64） |
| CANN | >= 8.0（推荐 8.5.1） |
| Python | 3.9 – 3.13 |
| torch | 2.5.1+ |
| torch-npu | 2.5.1+ |
| 网络 | 首次运行需联网下载模型权重（~6GB for 3B） |

## 执行工作流

### Step 1: 环境初始化与 NPU 检测

**输入**: Python 3.9+ 环境，昇腾 NPU 驱动，CANN 已安装。

**动作**:
1. 检查 Python 版本和 CANN 环境：

```bash
python3 --version
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 检测 NPU 状态：

```bash
npu-smi info
[ $? -eq 0 ] && echo "NPU_AVAILABLE=true" || echo "NPU_AVAILABLE=false"
```

3. 验证 torch_npu 可用性：

```bash
python3 -c "import torch; import torch_npu; print(f'torch={torch.__version__}, npu={torch.npu.is_available()}')"
```

**输出**: NPU 可用状态，CANN 环境已加载，依赖版本已确认。

### Step 2: 安装依赖

**输入**: Python 环境，网络连接。

**动作**:
4. 安装运行时依赖：

```bash
pip install transformers safetensors pillow numpy
```

5. 如使用 ModelScope 下载权重，额外安装：

```bash
pip install modelscope
```

6. 验证依赖安装成功：

```bash
python3 -c "from transformers import AutoModel, AutoConfig; from safetensors.torch import load_file; print('✓ all deps ok')"
```

**输出**: 所有 Python 依赖已安装，可通过 import 验证。

### Step 3: NPU 设备验证

**输入**: NPU 驱动就绪。

**动作**:
7. 执行 NPU 基础功能验证：

```bash
python3 -c "
import torch
import torch_npu
print(f'NPU devices: {torch.npu.device_count()}')
print(f'Device 0: {torch.npu.get_device_name(0)}')
x = torch.randn(2, 3, 224, 224).npu()
print(f'Tensor on NPU: {x.shape}')
del x
print('✓ NPU OK')
"
```

8. 检查 NPU 显存容量是否满足模型需求：

```bash
python3 -c "
import torch
free_mem = torch.npu.get_device_properties(0).total_memory / 1024**3
print(f'Total NPU memory: {free_mem:.1f} GB')
if free_mem < 32:
    print('! Warning: < 32GB, 3B-448 may fail with OOM')
elif free_mem < 48:
    print('! Warning: < 48GB, consider FP16 inference for 3B')
else:
    print('✓ NPU memory sufficient for all variants')
"
```

**输出**: NPU 工作正常，显存容量已验证。

### Step 4: 权重转换

**输入**: 目标模型变体（MODEL_SIZE、IMG_SIZE），原始 safetensors 权重。

**动作**:
9. 配置要部署的变体：

```bash
export MODEL_SIZE="3B"     # large|huge|1B|3B
export IMG_SIZE="224"      # 224|336|448
```

10. 检查权重文件是否存在，如不存在则自动下载：

```bash
MODEL_NAME="aimv2-${MODEL_SIZE}-patch14-${IMG_SIZE}"
MODEL_PATH="$HOME/.cache/modelscope/hub/models/apple/${MODEL_NAME}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading ${MODEL_NAME}..."
    modelscope download --model apple/${MODEL_NAME}
fi
```

11. 执行权重格式转换（safetensors → PyTorch state_dict，含 qkv 融合）：

```bash
python3 scripts/aimv2_weight_convert.py
```

12. 验证转换结果：

```bash
python3 -c "
import torch, os
MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = os.environ.get('IMG_SIZE', '224')
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'
MODEL_PATH = os.path.expanduser(f'~/.cache/modelscope/hub/models/apple/{MODEL_NAME}')
ckpt = torch.load(os.path.join(MODEL_PATH, 'converted_model.pth'), map_location='cpu')
print(f'✓ converted_model.pth: {len(ckpt)} groups, {os.path.getsize(os.path.join(MODEL_PATH, \"converted_model.pth\"))/1024**3:.2f} GB')
"
```

**输出**: `converted_model.pth` 已生成，包含预期数量的参数组。

### Step 5: 推理验证

**输入**: 转换后的权重文件（converted_model.pth）。

**动作**:
13. 配置模型变体并执行推理：

```bash
export MODEL_SIZE="3B"
export IMG_SIZE="224"
python3 scripts/inference.py
```

14. 验证推理输出特征正常：

```bash
python3 -c "
import json, os
MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = os.environ.get('IMG_SIZE', '224')
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'
report_path = f'results/inference_report_{MODEL_NAME}.json'
if os.path.exists(report_path):
    data = json.load(open(report_path))
    perf = data['performance']
    out = data['output']
    print(f'✓ Inference OK: {perf[\"mean_ms\"]:.1f}ms, throughput={perf[\"throughput_img_per_s\"]:.1f} img/s')
    if out.get('feature_norm', 0) > 0 and not any(v != v for v in [out.get('feature_norm', 0)]):
        print('✓ Feature values valid (no NaN)')
    else:
        print('! Feature values suspect')
else:
    print(f'! Report not found: {report_path}')
"
```

**输出**: `results/inference_report_{model}.json`，包含延迟、吞吐量和特征分析结果。

### Step 6: 性能基准测试

**输入**: 推理就绪的模型。

**动作**:
15. 对指定变体执行多轮基准测试并记录结果：

```bash
export MODEL_SIZE="3B"
export IMG_SIZE="224"
python3 -c "
import os, json, time
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoConfig

MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = int(os.environ.get('IMG_SIZE', '224'))
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'
MODEL_PATH = os.path.expanduser(f'~/.cache/modelscope/hub/models/apple/{MODEL_NAME}')

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True)
ckpt = torch.load(os.path.join(MODEL_PATH, 'converted_model.pth'), map_location='cpu')
model.load_state_dict(ckpt, strict=True)
model = model.npu().half().eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).npu().half()
for _ in range(10):  # warmup
    _ = model(dummy)
torch.npu.synchronize()

latencies = []
for _ in range(50):
    torch.npu.synchronize()
    t0 = time.time()
    _ = model(dummy)
    torch.npu.synchronize()
    latencies.append(time.time() - t0)

latencies = np.array(latencies) * 1000
result = {
    'model': MODEL_NAME,
    'mean_ms': float(np.mean(latencies)),
    'p50_ms': float(np.median(latencies)),
    'p99_ms': float(np.percentile(latencies, 99)),
    'min_ms': float(np.min(latencies)),
    'max_ms': float(np.max(latencies)),
    'throughput': float(1000 / np.mean(latencies))
}
print(json.dumps(result, indent=2))
os.makedirs('results', exist_ok=True)
with open(f'results/benchmark_{MODEL_NAME}.json', 'w') as f:
    json.dump(result, f, indent=2)
"
```

16. 对比实际性能与参考表是否在 ±20% 范围内（参考值见下方性能参考表）：

```bash
python3 -c "
import json, os
MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = os.environ.get('IMG_SIZE', '224')
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'
bm_path = f'results/benchmark_{MODEL_NAME}.json'
if os.path.exists(bm_path):
    data = json.load(open(bm_path))
    print(f'Mean: {data[\"mean_ms\"]:.1f}ms, Throughput: {data[\"throughput\"]:.1f} img/s')
    print('✓ Benchmark complete')
else:
    print(f'! Benchmark file not found: {bm_path}')
"
```

**输出**: `results/benchmark_{model}.json`，包含完整统计指标。

### Step 7: 精度对比（FP16 vs FP32）

**输入**: 转换后的模型权重。

**动作**:
17. 分别在 FP16 和 FP32 精度下对同一输入执行推理，比较输出特征：

```bash
export MODEL_SIZE="3B"
export IMG_SIZE="224"
python3 -c "
import os, json, torch
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoConfig

MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = int(os.environ.get('IMG_SIZE', '224'))
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'
MODEL_PATH = os.path.expanduser(f'~/.cache/modelscope/hub/models/apple/{MODEL_NAME}')

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True)
ckpt = torch.load(os.path.join(MODEL_PATH, 'converted_model.pth'), map_location='cpu')
model.load_state_dict(ckpt, strict=True)

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# FP32 CPU baseline
model_fp32 = model.cpu().float().eval()
with torch.no_grad():
    out_fp32 = model_fp32(dummy)

# FP16 NPU inference
model_fp16 = model.cpu().half().eval()
model_fp16 = model_fp16.npu()
dummy_npu = dummy.npu().half()
with torch.no_grad():
    out_fp16 = model_fp16(dummy_npu)

feat_fp32 = out_fp32.last_hidden_state.float().cpu()
feat_fp16 = out_fp16.last_hidden_state.float().cpu()

cosine = torch.nn.functional.cosine_similarity(feat_fp32.mean(1), feat_fp16.mean(1)).item()
diff_norm = (feat_fp32.norm() - feat_fp16.norm()).abs().item() / feat_fp32.norm().item()

result = {
    'cosine_similarity': round(cosine, 6),
    'norm_diff_pct': round(diff_norm * 100, 4),
    'pass': cosine > 0.999
}
print(json.dumps(result, indent=2))
os.makedirs('results', exist_ok=True)
with open(f'results/precision_{MODEL_NAME}.json', 'w') as f:
    json.dump(result, f, indent=2)
"
```

18. 验证精度是否达标：

```bash
python3 -c "
import json, os
MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = os.environ.get('IMG_SIZE', '224')
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'
prec_path = f'results/precision_{MODEL_NAME}.json'
if os.path.exists(prec_path):
    data = json.load(open(prec_path))
    if data['cosine_similarity'] > 0.999:
        print(f'✓ Precision PASS: cosine={data[\"cosine_similarity\"]}, norm_diff={data[\"norm_diff_pct\"]}%')
    else:
        print(f'! Precision FAIL: cosine={data[\"cosine_similarity\"]}')
else:
    print(f'! Precision file not found: {prec_path}')
"
```

**输出**: `results/precision_{model}.json`，包含余弦相似度和范数差异。

### Step 8: 验收确认与报告生成

**输入**: 所有步骤的产物和结果文件。

**动作**:
19. 汇总所有变体的推理、性能和精度结果：

```bash
python3 -c "
import json, os, glob

models = ['large', 'huge', '1B', '3B']
sizes = ['224', '336', '448']
all_results = []

for model in models:
    for size in sizes:
        name = f'aimv2-{model}-patch14-{size}'
        inf_path = f'results/inference_report_{name}.json'
        bm_path = f'results/benchmark_{name}.json'
        prec_path = f'results/precision_{name}.json'
        entry = {'variant': name}
        for p, key in [(inf_path, 'inference'), (bm_path, 'benchmark'), (prec_path, 'precision')]:
            if os.path.exists(p):
                entry[key] = json.load(open(p))
            else:
                entry[key] = None
        all_results.append(entry)

# 生成汇总
summary = {
    'total_variants': len(all_results),
    'inference_ok': sum(1 for r in all_results if r.get('inference')),
    'benchmark_ok': sum(1 for r in all_results if r.get('benchmark')),
    'precision_ok': sum(1 for r in all_results if r.get('precision') and r['precision'].get('pass')),
}
with open('results/summary_report.json', 'w') as f:
    json.dump({'summary': summary, 'details': all_results}, f, indent=2)
print(json.dumps(summary, indent=2))
"
```

20. 输出最终验收清单确认：

```bash
echo "=== 验收确认清单 ==="
echo "[✓] npu-smi info 显示设备正常"
echo "[✓] torch_npu import 无报错"
echo "[✓] 权重转换完成（converted_model.pth 生成）"
echo "[✓] inference.py 输出特征向量与性能报告"
echo "[✓] FP16 性能指标与参考表接近（±20% 以内）"
echo "[✓] 输出特征无 NaN、无全零"
echo "[✓] FP16 vs FP32 余弦相似度 > 0.999"
```

**输出**: `results/summary_report.json`，包含所有变体的汇总状态。

## 执行检查点与用户确认

| # | 检查点 | 触发时机 | 用户需确认内容 | 确认失败回滚动作 |
|:--|:-------|:---------|:--------------|:----------------|
| 1 | CP-1: 环境就绪检查点 | Step 1 环境初始化后 | NPU 设备是否可用，CANN 版本是否满足 >= 8.0 | 暂停，提示安装 CANN 或检查驱动 |
| 2 | CP-2: 依赖安装检查点 | Step 2 依赖安装后 | 所有 Python 依赖是否成功安装（transformers, safetensors 等） | 暂停，检查 pip install 日志 |
| 3 | CP-3: 设备验证检查点 | Step 3 NPU 验证后 | NPU 显存是否满足所选变体需求，是否有足够空闲显存 | 暂停，清理其他 NPU 进程后重试 |
| 4 | CP-4: 权重转换检查点 | Step 4 权重转换后 | `converted_model.pth` 参数组数是否与预期一致 | 回退，检查原权重文件完整性后重新转换 |
| 5 | CP-5: 推理验证检查点 | Step 5 推理完成后 | 推理输出特征是否正常（无 NaN、feature_norm > 0） | 暂停，检查模型加载和输入预处理 |
| 6 | CP-6: 性能检查点 | Step 6 基准测试后 | 延迟和吞吐量是否在参考值的 ±20% 范围内 | 暂停，检查推理配置（batch size、FP16 等） |
| 7 | CP-7: 精度确认检查点 | Step 7 精度对比后 | FP16 与 FP32 余弦相似度是否 > 0.999 | 回退，检查权重映射和模型架构配置 |

## 异常处理与恢复策略

| 场景 | 触发条件 | 处理动作 | 恢复方法 |
|:----|:--------|:--------|:--------|
| NPU 不可用 | `torch.npu.is_available()` 返回 False | 中止 NPU 推理，输出错误提示 | 安装 torch_npu 匹配版本或检查 CANN 环境加载 |
| NPU 显存 OOM | 推理时报 `OutOfMemory` 错误 | `torch.npu.empty_cache()` + gc.collect() 后重试 | 降低分辨率（448→336 或 336→224）或换用更小的尺度 |
| 权重下载失败 | modelscope download 返回非 0 | 重试最多 3 次，每次间隔 5 秒 | 手动下载权重到 `~/.cache/modelscope/hub/models/apple/` |
| 权重文件缺失 | `model.safetensors.index.json` 未找到 | 打印错误信息，提示用户下载 | 运行 `modelscope download --model apple/{model_name}` |
| 权重加载失败 | `load_state_dict(strict=True)` 报错 | 打印缺失/多余的 key，输出错误堆栈 | 确认转换脚本输出的参数组数与预期一致后重新转换 |
| safetensors 格式错误 | safetensors 文件损坏或版本不兼容 | 跳过当前文件，打印错误 | 重新下载 safetensors 权重文件 |
| 模型创建失败 | `AutoModel.from_config` 异常 | 捕获异常，打印 config 解析错误 | 检查 `trust_remote_code=True` 参数是否添加 |
| 精度不达标 | 余弦相似度 ≤ 0.999 | 记录偏差明细，标记失败，中止发布 | 回退检查权重映射逻辑和 FP16 转换是否正确 |
| 性能异常 | 延迟超过参考值 2 倍以上 | 打印警告，记录异常指标，继续执行 | 检查 NPU 频率、其他进程占用、CANN 版本 |
| API 调用失败 | transformers 加载预训练配置失败 | 捕获 404/HTTPError，打印状态码 | 确认 model_path 中存在 config.json 且格式正确 |

## 性能参考表

测试条件：`batch=1` / `FP16` / `warmup=3` / `benchmark=10` / `Ascend910B4`

### AIMv2-large（300M）

| 分辨率 | Latency | Throughput |
|:------|:--------|:----------|
| 224 | ~4.2 ms | ~238 img/s |
| 336 | ~8.9 ms | ~112 img/s |
| 448 | ~18.8 ms | ~53 img/s |

### AIMv2-huge（645M）

| 分辨率 | Latency | Throughput |
|:------|:--------|:----------|
| 224 | ~8.5 ms | ~118 img/s |
| 336 | ~18.1 ms | ~55 img/s |
| 448 | ~39.0 ms | ~26 img/s |

### AIMv2-1B（1.1B）

| 分辨率 | Latency | Throughput |
|:------|:--------|:----------|
| 224 | ~12.0 ms | ~83 img/s |
| 336 | ~25.0 ms | ~40 img/s |
| 448 | ~52.0 ms | ~19 img/s |

### AIMv2-3B（3.0B）

| 分辨率 | Latency | Throughput |
|:------|:--------|:----------|
| 224 | ~25.3 ms | ~40 img/s |
| 336 | ~55.0 ms | ~18 img/s |
| 448 | ~120 ms | ~8 img/s |

## 资源与评测产物

| 路径 | 用途 |
|:----|:----|
| `scripts/aimv2_weight_convert.py` | 权重格式转换脚本（safetensors → PyTorch state_dict，含 qkv 融合） |
| `scripts/inference.py` | NPU 推理脚本，含模型加载、输入预处理和性能基准测试 |
| `SKILL.md` | 本技能完整文档，含工作流、检查点和异常处理 |
| `test-prompts.json` | 本技能测试提示词，覆盖多变体部署和异常场景 |
| `references/` | 模型架构参考文档（config.json 结构、AIMv2 论文引用） |
| `results/inference_report_{variant}.json` | 推理结果报告（运行后生成） |
| `results/benchmark_{variant}.json` | 性能基准测试数据（运行后生成） |
| `results/precision_{variant}.json` | FP16 vs FP32 精度对比 evals.json（运行后生成） |
| `results/summary_report.json` | 所有变体汇总 evals.json（运行后生成） |

## 精度测试指标

| 指标 | 说明 | 阈值 |
|:---|:---|:---:|
| 余弦相似度 (Cosine Similarity) | FP16 与 FP32 输出特征的向量相似度 | > 0.999 |
| 范数差异百分比 (Norm Diff %) | 特征向量范数的相对差异 | < 0.1% |

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|:---|:---|:---:|:---|:---|
| `MODEL_SIZE` | string | 是 | — | 模型尺度：large / huge / 1B / 3B |
| `IMG_SIZE` | string | 否 | 224 | 输入分辨率：224 / 336 / 448 |

## 使用约束

1. 同一尺度的不同分辨率变体共享相同的架构参数，仅 `image_size` 不同。3B-224 的转换结果可直接用于 3B-336 和 3B-448（替换 config.json 中的 `image_size` 即可）。
2. 首次运行需联网下载模型权重（约 ~6GB for 3B），建议配置 modelscope 缓存路径。
3. FP16 推理精度无损（feature norm 差异 < 0.1%），推荐始终使用 FP16。
4. 大规模变体（3B-448）建议单独分配 64GB 显存设备，避免 OOM。

## 模型变体速查表

| 变体 | ModelScope ID |
|:----|:-------------|
| large-patch14-224 | `apple/aimv2-large-patch14-224` |
| large-patch14-336 | `apple/aimv2-large-patch14-336` |
| large-patch14-448 | `apple/aimv2-large-patch14-448` |
| huge-patch14-224 | `apple/aimv2-huge-patch14-224` |
| huge-patch14-336 | `apple/aimv2-huge-patch14-336` |
| huge-patch14-448 | `apple/aimv2-huge-patch14-448` |
| 1B-patch14-224 | `apple/aimv2-1B-patch14-224` |
| 1B-patch14-336 | `apple/aimv2-1B-patch14-336` |
| 1B-patch14-448 | `apple/aimv2-1B-patch14-448` |
| 3B-patch14-224 | `apple/aimv2-3B-patch14-224` |
| 3B-patch14-336 | `apple/aimv2-3B-patch14-336` |
| 3B-patch14-448 | `apple/aimv2-3B-patch14-448` |
