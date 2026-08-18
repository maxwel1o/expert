---
name: tiny-vit-npu
description: "TinyViT 系列模型昇腾 Ascend910 NPU 自动化部署、推理与精度验证 Skill"
---

# TinyViT NPU Deployment Skill

## 1. 环境准备与依赖安装

在昇腾 Ascend910 NPU 环境下准备 Python 虚拟环境并安装依赖。

1. 创建 Python 虚拟环境并激活
2. 安装 PyTorch 与 torch_npu 以及 timm 和 modelscope 等依赖
3. 通过 npu-smi info 命令验证 NPU 设备可用性

```bash
# 创建虚拟环境
python3 -m venv tinyvit_env
source tinyvit_env/bin/activate

# 安装 PyTorch 与 torch_npu
pip install torch>=2.0.0 torch_npu>=2.0.0

# 安装 timm 和 modelscope
pip install timm>=1.0.0 modelscope>=1.0.0

# 验证 NPU 设备可用性
python3 -c "import torch_npu; print(torch.npu.device_count()); print(torch.npu.get_device_name(0))"
```

**确认条件**: 执行 `npu-smi info` 确认 NPU 设备状态正常，驱动版本与 CANN 工具箱兼容。

## 2. 模型加载与权重下载

通过 modelscope 下载 TinyViT 预训练权重，并使用 timm 创建模型实例。

1. 通过 modelscope.snapshot_download API 下载预训练权重到本地缓存
2. 使用 timm.create_model 创建匹配的模型实例并设为评估模式
3. 加载 safetensors 或 pytorch_model.bin 格式的 state_dict 完成权重注入

```python
import os
import torch
from timm import create_model
from modelscope import snapshot_download
from safetensors.torch import load_file

def load_model(model_name):
    ms_path = "timm/" + model_name
    local_path = snapshot_download(ms_path)
    model = create_model(model_name, pretrained=False)
    model.eval()
    safetensors_path = os.path.join(local_path, "model.safetensors")
    pytorch_bin_path = os.path.join(local_path, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    else:
        state_dict = torch.load(pytorch_bin_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model
```

## 3. NPU 单机推理

使用加载的模型在 NPU 上进行前向推理，记录推理性能。

1. 将模型参数移至 NPU 设备并转为 float 精度
2. 执行 3 轮 warmup 推理预热 NPU 硬件
3. 运行 10 轮计时推理并通过 torch.npu.synchronize 获取真实延迟

```bash
python3 scripts/inference.py --model-name tiny_vit_5m_224.in1k --device npu --image-size 224
```

```python
@torch.no_grad()
def run_npu_inference(model_name, image_size=224, num_runs=10):
    model = load_model(model_name)
    model = model.to("npu:0").float()
    
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    from PIL import Image
    img = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
    input_tensor = transform(img).unsqueeze(0).float().to("npu:0")
    
    # Warmup
    for _ in range(3):
        _ = model(input_tensor)
    torch.npu.synchronize()
    
    # Benchmark
    import time
    start = time.time()
    for _ in range(num_runs):
        output = model(input_tensor)
    torch.npu.synchronize()
    elapsed = time.time() - start
    
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
    
    print(f"Average time: {elapsed / num_runs * 1000:.2f} ms")
    for i in range(5):
        print(f"  {i+1}. class {top5_indices[0][i].item()} (prob: {top5_probs[0][i].item():.6f})")
    
    # 释放资源
    del model, input_tensor, output
    gc.collect()
    torch.npu.empty_cache()
```

## 4. CPU 基线推理

在 CPU 上运行相同模型获得基线输出，用于后续精度对比。

1. 在 CPU 上加载相同模型和权重文件
2. 运行 10 轮推理并计算平均延迟
3. 验证 CPU 输出结果是否稳定可复现

```bash
python3 scripts/inference.py --model-name tiny_vit_5m_224.in1k --device cpu --image-size 224
```

**确认条件**: CPU 推理结果应稳定可复现，Top-5 概率分布合理。如果概率集中在单一类别且数值接近 1.0，应检查输入预处理是否正常。

## 5. 精度对比验证

运行 CPU 与 NPU 精度对比，计算余弦相似度与概率相对误差。

1. 在 CPU 上执行推理并提取 softmax 概率分布
2. 在 NPU 上对相同输入执行推理并提取概率分布
3. 计算余弦相似度、概率相对误差和 Top-1 类别匹配率

```bash
python3 scripts/compare_cpu_npu.py --model-name tiny_vit_5m_224.in1k --image-size 224
```

```python
def compare_accuracy(model_name, image_size=224):
    """
    对比 CPU 与 NPU 推理结果的精度一致性。
    如果概率相对误差 < 1% 则判定通过 (PASS)，否则标记失败 (FAIL)。
    """
    # CPU 推理
    model_cpu = load_model(model_name)
    input_cpu = get_test_image(image_size)
    for _ in range(10):
        output_cpu = model_cpu(input_cpu)
    cpu_probs = torch.softmax(output_cpu, dim=1)

    # NPU 推理
    model_npu = load_model(model_name)
    model_npu = model_npu.to("npu:0").float().eval()
    input_npu = get_test_image(image_size).float().to("npu:0")
    for _ in range(3):
        _ = model_npu(input_npu)
    torch.npu.synchronize()
    for _ in range(10):
        output_npu = model_npu(input_npu)
    torch.npu.synchronize()
    npu_probs = torch.softmax(output_npu.cpu(), dim=1)

    # 计算指标
    cos_sim = torch.nn.CosineSimilarity(dim=1)(output_cpu, output_npu.cpu()).item()
    prob_diff = torch.abs(cpu_probs - npu_probs)
    prob_rel_error = (torch.norm(prob_diff) / torch.norm(cpu_probs)).item()
    top1_match = (torch.argmax(output_cpu, dim=1) == torch.argmax(output_npu.cpu(), dim=1)).item()

    print(f"Cosine Similarity:       {cos_sim:.8f}")
    print(f"Prob Relative Error:     {prob_rel_error:.4f}%")
    print(f"Top-1 Class Match:       {top1_match}")
    print(f"CONCLUSION: {'PASS' if prob_rel_error < 0.01 else 'FAIL'}")
```

## 6. 批量串行执行

为避免 NPU 显存爆炸，使用串行策略批量处理所有模型变体。

1. 遍历 11 个模型配置并按冒号分隔解析模型名和图像尺寸
2. 每个模型使用 try/except 包裹，单个失败不影响整体批次
3. 每完成一个模型调用 gc.collect 和 torch.npu.empty_cache 后休眠 2 秒

```bash
bash scripts/run_all.sh
```

```python
import gc
import time

models = [
    ("tiny_vit_5m_224.dist_in22k", 224),
    ("tiny_vit_5m_224.in1k", 224),
    ("tiny_vit_5m_224.dist_in22k_ft_in1k", 224),
    ("tiny_vit_21m_512.dist_in22k_ft_in1k", 512),
    ("tiny_vit_21m_384.dist_in22k_ft_in1k", 384),
    ("tiny_vit_21m_224.dist_in22k_ft_in1k", 224),
    ("tiny_vit_21m_224.in1k", 224),
    ("tiny_vit_21m_224.dist_in22k", 224),
    ("tiny_vit_11m_224.dist_in22k", 224),
    ("tiny_vit_11m_224.in1k", 224),
    ("tiny_vit_11m_224.dist_in22k_ft_in1k", 224),
]

for model_name, img_size in models:
    print(f"Processing: {model_name} (img={img_size})")
    try:
        compare_accuracy(model_name, img_size)
    except Exception as e:
        print(f"ERROR: {model_name} failed - {e}")
        # 如果模型失败，继续处理下一个，不中断整个批次
        continue
    finally:
        gc.collect()
        torch.npu.empty_cache()
        time.sleep(2)
```

## 7. 结果归档与报告生成

将精度验证结果归档并生成汇总报告。

1. 创建 results/ 目录用于存放评测输出
2. 运行批量脚本并将标准输出和错误重定向到日志文件
3. 解析日志文件生成 results.tsv 汇总表和 evals.json 结构化结果

```bash
# 创建结果目录
mkdir -p results

# 运行批量验证并将日志重定向到结果文件
bash scripts/run_all.sh 2>&1 | tee results/benchmark_$(date +%Y%m%d).log

# 生成汇总 TSV
python3 -c "
import re, sys
with open('results/benchmark_$(date +%Y%m%d).log') as f:
    content = f.read()
models = re.findall(r'Model: (.*?)\\n', content)
pass_fail = re.findall(r'CONCLUSION: (PASS|FAIL)', content)
with open('results/results.tsv', 'w') as out:
    out.write('model\tstatus\n')
    for m, s in zip(models, pass_fail):
        out.write(f'{m}\t{s}\n')
print(f'Generated results/results.tsv with {len(models)} entries')
"

# 生成 evals.json
python3 -c "
import json
results = {'benchmark_date': '$(date +%Y%m%d)', 'models': []}
with open('results/results.tsv') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            results['models'].append({'name': parts[0], 'status': parts[1]})
with open('results/evals.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'Generated results/evals.json')
"
```

## 8. 模型仓库发布与文档生成

为每个通过精度验证的模型创建独立仓库并推送至 GitCode。

1. 将推理和精度对比脚本复制到模型仓库目录
2. 创建包含所有依赖声明的 requirements.txt
3. 通过 GitCode API 创建模型仓库并推送代码到 main 分支

```bash
# 初始化模型仓库
MODEL_NAME="tiny_vit_5m_224_in1k_npu"
mkdir -p $MODEL_NAME
cp scripts/inference.py $MODEL_NAME/
cp scripts/compare_cpu_npu.py $MODEL_NAME/

# 创建 requirements.txt
cat > $MODEL_NAME/requirements.txt << 'EOF'
torch>=2.0.0
torch_npu>=2.0.0
timm>=1.0.0
modelscope>=1.0.0
safetensors>=0.4.0
Pillow>=9.0.0
torchvision>=0.15.0
EOF

# 生成终端截图
python3 /opt/atomgit/terminal_screenshot.py \
    --input results/benchmark_$(date +%Y%m%d).log \
    --output $MODEL_NAME/terminal_output.png

# 创建 GitCode 仓库并推送
curl -X POST "https://api.gitcode.com/api/v5/user/repos" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"'$MODEL_NAME'","repository_type":"model"}'

cd $MODEL_NAME
git init
git add .
git commit -m "Initial commit: $MODEL_NAME inference scripts"
git remote add origin https://auth:${TOKEN}@gitcode.com/username/$MODEL_NAME.git
git push -u origin main
```

## 检查点设计

在下述检查点处，用户应暂停确认状态无误后再继续执行后续步骤。

| 序号 | 检查点名称 | 检查时机 | 检查内容 | 通过标准 | 失败处理 | 涉及步骤 | 责任人 |
|------|-----------|---------|---------|---------|---------|---------|-------|
| 1 | NPU 设备就绪 | Step 1 之后 | 执行 `npu-smi info` 检查 NPU 数量和状态 | device_count >= 1, 温度正常 | 回退检查驱动和 CANN 安装 | Step 1 | 用户 |
| 2 | 依赖安装完整 | Step 1 之后 | `pip list` 确认 torch, torch_npu, timm 等已安装 | 所有依赖版本满足要求 | 重新执行 `pip install` | Step 1 | 用户 |
| 3 | 模型加载成功 | Step 2 之后 | 检查模型参数量和 state_dict 加载日志 | state_dict 加载无警告或无缺失 key | 重新下载权重或检查模型名 | Step 2 | 用户 |
| 4 | 推理结果合理 | Step 3 之后 | 检查 Top-5 概率分布是否均匀 | Top-1 概率 < 0.5, 无明显 NaN/Inf | 检查输入预处理和数据路径 | Step 3 | 用户 |
| 5 | 精度指标达标 | Step 5 之后 | 检查 Prob Relative Error 和 Cosine Similarity | Prob Rel Error < 1%, Cos Sim > 0.999 | 回退检查 NPU 推理是否数值异常 | Step 5 | 用户 |
| 6 | 批量执行完整 | Step 6 之后 | 检查所有模型是否全部完成 | 11 个模型均输出 PASS/FAIL 结论 | 重新运行失败模型 | Step 6 | 用户 |
| 7 | 结果归档确认 | Step 7 之后 | 检查 results/ 目录下 TSV 和 JSON 文件 | results.tsv 和 evals.json 文件存在且格式正确 | 重新运行归档脚本 | Step 7 | 用户 |

## 异常处理

| 异常场景 | 可能原因 | 检测方式 | 处理策略 | 恢复步骤 | 预防措施 | 预期结果 | 回退方案 | 日志位置 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| NPU 设备不可用 | 驱动未安装、npu-smi 无输出 | `npu-smi info` 返回空 | 回退到 CPU 模式: `--device cpu` | 检查驱动版本: `npu-smi --version` | 执行前通过 `npu-smi info` 确认 | 自动使用 CPU 降级推理 | 硬件修复后重试 | `npu-check.log` |
| 模型下载失败 | 网络超时、modelscope 服务不可用 | `snapshot_download` 抛出网络异常 | 重试 3 次，间隔 5 秒 | 确认网络连通性: `ping gitcode.com` | 设置 `requests` 超时参数 | 重试成功或跳过该模型 | 手动下载权重到缓存 | `download-error.log` |
| NPU 显存不足 OOM | 显存碎片、模型过大 | `torch.npu.out_of_memory` 异常 | 清理缓存后重试 | `torch.npu.empty_cache(); gc.collect()` | 串行执行 + 2 秒间隔 | 释放显存后成功推理 | 拆分模型批次运行 | `oom-recovery.log` |
| 模型权重 key 不匹配 | 模型定义与权重版本不一致 | `load_state_dict` 缺失 key 警告 | 设置 `strict=False` 继续加载 | 检查 timm 模型名和权重来源 | 使用相同 timm 版本 | 加载可用权重跳过缺失层 | 升级/降级 timm 版本 | `weight-warning.log` |
| 输入尺寸不匹配 | image_size 参数与模型不匹配 | `RuntimeError: size mismatch` | 自动调整 image_size | 根据模型规格选择正确尺寸 | 在配置表中预定义尺寸映射 | 调整后推理成功 | 输出错误信息并退出 | `size-error.log` |
| 推理输出 NaN/Inf | 数值溢出、权重未正确加载 | `torch.isnan()/isinf()` 检测结果 | 跳过当前模型继续 | 重新加载权重再次推理 | 推理后加入数值检查断言 | 跳过异常输出并记录 | 标记该模型为 FAIL | `nan-inf-check.log` |
| 精度对比超过阈值 | NPU 推理误差偏大 | Prob Rel Error >= 1% | 标记 FAIL 并记录详细差异 | 检查 NPU 是否开启混合精度 | CPU/NPU 均使用 `float()` 推理 | 记录 FAIL 不影响其他模型 | 尝试 `torch.npu.set_float32_matmul_precision` | `precision-fail.log` |
| 批量执行中间模型失败 | 单模型异常 | `try/except` 捕获 | 记录错误后 `continue` | 跳过失败模型继续下一个 | 加入 `finally` 确保 cleanup | 不影响其余模型结果 | 重跑失败模型 | `batch-error.log` |
| 磁盘空间不足 | 权重缓存过大 | `shutil.disk_usage` 检查 | 清理 modelscope 缓存 | `rm -rf ~/.cache/modelscope` | 设置缓存上限或定期清理 | 释放空间后继续 | 更换更大磁盘 | `disk-cleanup.log` |
| API 调用限流 | GitCode API 频率限制 | HTTP 429 响应 | 退避重试 (指数退避) | `sleep(2^retry)` 后重试 | 控制推送频率，单次推送 | 重试成功完成发布 | 分批推送模型仓库 | `api-rate-limit.log` |

## 资源清单

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| 推理脚本 | `scripts/inference.py` | 单模型 NPU/CPU 推理脚本，支持 --device 参数 |
| 精度对比脚本 | `scripts/compare_cpu_npu.py` | 同一模型 CPU 与 NPU 精度对比，输出余弦相似度与概率相对误差 |
| 批量执行脚本 | `scripts/run_all.sh` | 串行运行所有 11 个模型变体的批量脚本 |
| 使用示例 | `examples/example_inference.sh` | 展示 NPU 推理与精度对比的示例入口脚本 |
| Batch Benchmark 日志 | `results/benchmark_*.log` | 批量执行的全量日志输出 |
| 汇总 TSV | `results/results.tsv` | 各模型 PASS/FAIL 状态汇总表格 |
| Eval JSON | `results/evals.json` | 结构化评估结果，包含 benchmark_date 和 models 数组 |
| Skill 配置 | `skill.json` | 技能元数据，包含模型列表、框架和硬件信息 |
| 测试用例 | `test-prompts.json` | 覆盖各种场景的测试提示词列表 |

## 精度验证标准

| 指标 | 目标值 | 计算方式 | 说明 |
|-----|-------|---------|------|
| Prob Relative Error | < 1% | `||P_cpu - P_npu|| / ||P_cpu||` | 概率相对误差，主通过标准 |
| Cosine Similarity | > 0.999 | `cos(P_cpu, P_npu)` | 余弦相似度，辅助指标 |
| Top-1 Class Match | True | `argmax(P_cpu) == argmax(P_npu)` | Top-1 类别是否一致 |

## 环境依赖

| 组件 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.9 | 运行环境 |
| PyTorch | 2.0.0 | 深度学习框架 |
| torch_npu | 2.0.0 | 昇腾 NPU 适配 |
| timm | 1.0.0 | PyTorch 图像模型库 |
| modelscope | 1.0.0 | 模型权重下载 |
| Ascend CANN | 兼容版本 | NPU 驱动和运行库 |
| Ascend910 | - | NPU 硬件 |

## 注意事项

1. **串行执行**: 多模型推理必须串行，每步间隔 2 秒，避免 NPU 显存爆炸。
2. **资源释放**: 每个模型推理完成后必须执行 `del model; gc.collect(); torch.npu.empty_cache()`。
3. **权重一致性**: CPU 和 NPU 使用同一份权重文件，确保对比公平。
4. **混合精度**: 默认使用 `float()` 推理，如启用 AMP 需记录精度影响。
5. **网络要求**: modelscope 下载权重需要稳定网络，如果失败可使用预缓存权重。
6. **权限确认**: GitCode API Token 需要 `repo` 和 `model` 写入权限，如果 Token 过期需重新申请。
7. **结果复现**: 保存每次 benchmark 日志和 evals.json 以便事后审计和比对。
