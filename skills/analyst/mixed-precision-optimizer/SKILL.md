---
name: mixed-precision-optimizer
description: 混合精度训练与推理优化。流程：精度基线确定 → AMP 策略选择 → Loss Scaling 调优 → 精度敏感算子排查 → 逐层精度验证 → 端到端性能收益确认。支持 Ascend NPU 的 O1/O2 混合精度模式以及自定义精度策略。
keywords:
  - 混合精度
  - AMP
  - loss scaling
  - float16
  - bfloat16
  - 精度优化
  - ascend
  - 显存优化
---

# 混合精度训练与推理优化 Skill

## 重要默认行为

- **精度优先**：所有精度调整必须以模型原始 FP32 精度为基准，核心指标偏差 ≤ 1%。
- **数据驱动**：精度损失必须量化，不能仅凭经验判断"精度够用"。
- **渐进式调优**：每次只调整一个精度策略参数，验证后再调整下一个。
- **逐层排查**：精度异常时采用二分法逐层排查，定位精度敏感算子。
- **文档记录**：每轮精度策略调整必须记录精度指标和性能数据。

## 前置条件

| 检查项 | 说明 |
|--------|------|
| 环境 | CANN ≥ 7.0，torch-npu ≥ 2.0 |
| 模型 | 模型可在 NPU 上以 FP32 正常运行，有可复现的训练/推理脚本 |
| 基线 | FP32 基准精度指标（loss/acc/bleu 等） |
| 数据 | 至少一个标准验证集用于精度对比 |
| 参考 | 了解 Ascend AMP 接口（`torch.npu.amp` 或 `amp` 模块） |

## 工作流总览

```
Phase 1: 精度基线建立
    ├── 输入: FP32 模型 + 验证集
    ├── 活动: FP32 推理/训练采集关键精度指标
    └── 产出: FP32 精度基线报告

Phase 2: AMP 策略选择与基础配置
    ├── 输入: FP32 基线 + 硬件特性
    ├── 活动: 选择 O1/O2 模式 → 配置 Loss Scaling → 启动混合精度训练
    └── 产出: 首版 AMP 训练配置 + 精度对比

Phase 3: 精度异常分析与算子级排查
    ├── 输入: AMP 精度对比报告
    ├── 活动: 精度指标对比 → 若异常则二分法逐层排查热点
    └── 产出: 精度敏感算子清单 + 排查报告

Phase 4: 精度-性能平衡优化
    ├── 输入: 精度敏感算子清单
    ├── 活动: 对敏感算子单独配置 FP32/BF16/FP16 → 微调 Loss Scaling
    └── 产出: 最终精度策略配置

Phase 5: 端到端验证
    ├── 输入: 最终精度策略
    ├── 活动: 全量训练/推理验证 → 性能 Profiling → 精度复测
    └── 产出: 端到端性能 vs 精度对比报告

Phase 6: 迭代决策
    ├── 达标 → 输出最终配置报告
    └── 不达标 → 返回 Phase 2 或 Phase 3 继续优化
```

---

## Phase 1: 精度基线建立

### 1.1 确认硬件支持的精度模式

```bash
# 检查 NPU 支持的数据类型
python -c "
import torch
import torch_npu
print('FP32 support:', True)
print('FP16 support:', torch.npu.is_available())
print('BF16 support:', torch.npu.is_bf16_supported())
print('NPU:', torch.npu.get_device_name(0))
"
```

**Ascend NPU 精度模式对比**：

| 模式 | 计算精度 | 存储精度 | 性能收益 | 适用范围 |
|------|---------|---------|---------|---------|
| FP32 | 32-bit | 32-bit | 1.0x（基线） | 精度敏感层 |
| FP16 (AMP O2) | 16-bit | 16-bit | 1.5-3.0x | 大部分算子 |
| BF16 | 16-bit（高指数精度） | 16-bit | 1.3-2.5x | 梯度易溢出的场景 |
| FP16 + FP32 Master | FP16 计算，FP32 权重 | 32-bit 权重 | 1.3-2.5x | 训练场景推荐 |
| INT8 量化 | 8-bit | 8-bit | 2-4x | 推理场景 |

### 1.2 采集 FP32 基线

```python
# collect_fp32_baseline.py
import torch
import torch_npu

def evaluate(model, dataloader, metric_fn, num_batches=100):
    """采集 FP32 精度基线"""
    model.eval()
    model = model.npu().float()
    all_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batch = {k: v.npu() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch)
            metrics = metric_fn(outputs, batch)
            all_metrics.append(metrics)

    # 汇总指标
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics)
                   for k in all_metrics[0].keys()}
    return avg_metrics

# 运行基线采集
baseline = evaluate(model, dataloader, metric_fn)
print("FP32 Baseline:", baseline)
```

**基线记录模板**：

```json
{
  "model": "model_name",
  "dtype": "fp32",
  "batch_size": 32,
  "num_batches": 100,
  "metrics": {
    "loss": 2.345,
    "accuracy": 0.856,
    "perplexity": 10.45
  },
  "performance": {
    "throughput": 128.5,
    "memory_peak_gb": 23.4
  }
}
```

### 1.3 ⚠️ 检查点：基线确认

- [ ] FP32 基线已采集，指标完整（loss / accuracy / throughput / memory）
- [ ] 验证集和批次大小固定，后续对比可复现
- [ ] 基线数据已归档到 `docs/perf/baseline/fp32_baseline.json`

---

## Phase 2: AMP 策略选择与基础配置

### 2.1 选择 AMP 模式

```python
# Ascend AMP 配置示例
from torch.npu.amp import autocast, GradScaler

# 模式 O1：自动混合精度（推荐首次使用）
# - 自动将大部分算子转为 FP16/BF16
# - 保持精度敏感算子为 FP32
model = Model()
model.npu()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # Loss Scaling 自动管理

for batch in dataloader:
    with autocast(dtype=torch.float16):  # 或 bfloat16
        outputs = model(batch)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**O1 vs O2 选择建议**：

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 首次尝试 | O1 | 自动配置，风险最低 |
| 追求极致性能 | O2 | 更多算子 FP16，但需排查精度 |
| 训练大模型（>7B） | O2 + BF16 | BF16 更高的指数精度减少 loss 溢出 |
| 推理服务 | O2 或 INT8 | 推理场景对精度相对容忍 |
| 微调（Fine-tune） | O1 | 微调精度敏感度高 |

### 2.2 Loss Scaling 配置

```python
# Loss Scaling 参数调优
from torch.npu.amp import GradScaler

# 默认配置（推荐首次使用）
scaler = GradScaler(
    init_scale=2**16,        # 初始缩放因子 65536
    growth_factor=2.0,       # 连续无溢出时翻倍
    backoff_factor=0.5,      # 溢出时减半
    growth_interval=2000,    # 每 2000 步检查一次是否增长
    enabled=True
)

# 激进配置（适合梯度较稳定的模型）
scaler_aggressive = GradScaler(
    init_scale=2**8,         # 较小的初始缩放
    growth_factor=4.0,       # 快速增长
    backoff_factor=0.25,     # 大幅回退
    growth_interval=500      # 频繁检查
)

# 保守配置（适合梯度易溢出的模型）
scaler_conservative = GradScaler(
    init_scale=2**20,        # 较大的初始缩放
    growth_factor=1.5,       # 缓慢增长
    backoff_factor=0.75,     # 小幅回退
    growth_interval=5000     # 较少检查
)
```

**Loss Scaling 关键指标**：

| 指标 | 正常范围 | 问题诊断 |
|------|---------|----------|
| 溢出率 (skip rate) | < 1% | > 5% 说明 Loss Scaling 不足或 FP16 不适合 |
| Scale factor | 稳定在 2^8 ~ 2^16 | 持续下降说明梯度频繁溢出 |
| 有效步数比 | > 99% | < 95% 需要调整策略 |

### 2.3 首轮 AMP 验证

```bash
# 运行 AMP（O1）+ Loss Scaling
python train.py --amp --amp-dtype=float16 --amp-level=O1

# 记录精度指标和性能
# 输出示例：
# Step 100 | Loss: 2.351 (FP32: 2.345) | Throughput: 285.6 samples/s (FP32: 128.5)
```

### 2.4 ⚠️ 检查点：AMP 首轮评估

- [ ] AMP 配置基础（O1/O2 + Loss Scaling）已完成
- [ ] 精度与 FP32 基线对比：loss 差异 < 5%
- [ ] 性能提升初步确认（throughput 提升 > 30%）
- [ ] 记录首轮 AMP 配置到 `docs/perf/round_001/amp_config.json`

---

## Phase 3: 精度异常分析与算子级排查

### 3.1 精度指标偏差判定

| 偏差程度 | Loss 差异 | 核心指标差异 | 行动 |
|----------|----------|-------------|------|
| 🟢 正常 | < 1% | < 0.5% | 可接受，进入 Phase 4 微调 |
| 🟡 轻微 | 1-5% | 0.5-2% | 需排查，可能个别算子精度不足 |
| 🔴 严重 | > 5% | > 2% | 必须逐层排查 |

### 3.2 二分法逐层排查精度敏感算子

```python
# inspect_layer_precision.py — 逐层对比 FP32 vs AMP 输出
def inspect_layer_by_layer(model_fp32, model_amp, input_data):
    """逐层对比 FP32 和 AMP 模型的激活值"""

    # 注册钩子收集中间层输出
    activations_fp32 = {}
    activations_amp = {}

    def hook_fp32(name):
        def fn(module, input, output):
            activations_fp32[name] = output.detach().float().cpu()
        return fn

    def hook_amp(name):
        def fn(module, input, output):
            activations_amp[name] = output.detach().float().cpu()
        return fn

    # 注册所有层
    hooks = []
    for name, module in model_fp32.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            hooks.append(module.register_forward_hook(hook_fp32(name)))
    for name, module in model_amp.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_amp(name)))

    # 前向传播
    model_fp32(input_data.float())
    model_amp(input_data.half() if use_amp else input_data)

    # 对比每层输出的余弦相似度
    results = []
    for name in activations_fp32:
        if name in activations_amp:
            a = activations_fp32[name].flatten()
            b = activations_amp[name].flatten()
            cos_sim = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
            max_err = (a - b).abs().max().item()
            results.append((name, cos_sim, max_err))

    # 按余弦相似度升序排列（最异常的在前）
    results.sort(key=lambda x: x[1])
    return results
```

**排查标准**：

| 余弦相似度 | 结论 | 处理 |
|-----------|------|------|
| ≥ 0.999 | ✅ 精度正常 | 保持 FP16 |
| 0.99 - 0.999 | ⚠️ 轻微偏差 | 可选 BF16 或保持 FP16 |
| 0.95 - 0.99 | 🔴 精度损失明显 | 需切换为该层 FP32 |
| < 0.95 | 🚫 严重精度损失 | 必须切换为 FP32 |

### 3.3 ⚠️ 检查点：精度敏感算子清单

- [ ] 已通过二分法排查完成所有层
- [ ] 已定位精度敏感算子（cos_sim < 0.99 的层）
- [ ] 已生成精度排查报告 `docs/perf/round_NNN/precision_inspection.md`
- [ ] 已与用户确认精度异常算子清单

---

## Phase 4: 精度-性能平衡优化

### 4.1 算子级精度策略配置

```python
# 为不同层级配置独立的精度策略
from torch.npu.amp import autocast

class MixedPrecisionModel(torch.nn.Module):
    def __init__(self, base_model, fp32_modules=None):
        super().__init__()
        self.base_model = base_model
        # 指定需要 FP32 的子模块
        self.fp32_modules = fp32_modules or []

    def forward(self, x):
        # 大部分计算在 FP16/BF16 下
        with autocast(dtype=torch.float16):
            x = self.base_model.embedding(x)
            x = self.base_model.dropout(x)

            for layer in self.base_model.transformer_layers:
                # 精度敏感层走 FP32
                if layer in self.fp32_modules:
                    x = layer.float()(x.float()).half()
                else:
                    x = layer(x)

            # 部分层强制 FP32
            with autocast(dtype=torch.float32):
                x = self.base_model.layer_norm(x)

            return self.base_model.lm_head(x)
```

**典型 FP32 保留层**：

| 层类型 | 推荐精度 | 理由 |
|--------|---------|------|
| LayerNorm / RMSNorm | FP32 | 归一化操作对精度敏感 |
| Softmax | FP32（推理）/ FP16（训练） | 指数运算在 FP16 易溢出 |
| CrossEntropy Loss | FP32 | 损失计算精度直接影响梯度 |
| 首层 Embedding | FP16/BF16 | 通常精度不敏感 |
| 输出层（LM Head） | FP32 | 影响最终预测概率分布 |
| Attention Score | FP32 | softmax 前的大值易溢出 |

### 4.2 BF16 vs FP16 选择

```python
# BF16 vs FP16 性能对比
def benchmark_precision(model, input_data, dtype, num_iter=100):
    model = model.npu()
    if dtype == torch.float16:
        model = model.half()
    elif dtype == torch.bfloat16:
        model = model.bfloat16()

    input_data = input_data.npu().to(dtype)
    torch.npu.synchronize()

    # 预热
    for _ in range(10):
        _ = model(input_data)
    torch.npu.synchronize()

    # 计时
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(num_iter):
        _ = model(input_data)
    end.record()
    torch.npu.synchronize()

    elapsed = start.elapsed_time(end) / num_iter
    return elapsed
```

**选择决策树**：

```
模型训练/推理
├── 模型 < 7B 参数
│   ├── FP16 精度正常 → 使用 FP16（性能最优）
│   └── FP16 精度异常 → 尝试 BF16
│       ├── BF16 精度正常 → 使用 BF16
│       └── BF16 精度异常 → 混合 FP32+FP16
└── 模型 >= 7B 参数
    ├── 推荐 BF16（梯度范围大，FP16 易溢出）
    ├── BF16 精度正常 → 使用 BF16
    └── BF16 不满足 → O1 混合模式
```

### 4.3 ⚠️ 检查点：精度策略确认

- [ ] 精度敏感算子已配置 FP32/BF16 回退
- [ ] 整体精度指标与 FP32 基线差异 < 1%
- [ ] Loss Scaling 参数已调优（溢出率 < 1%）
- [ ] 配置已归档到 `docs/perf/final/amp_policy.yaml`

---

## Phase 5: 端到端验证

### 5.1 全量训练/推理验证

```bash
# 完整训练验证（数百步）
python train.py \
    --amp \
    --amp-level=O2 \
    --amp-dtype=float16 \
    --fp32-modules=layer_norm,softmax,lm_head \
    --max-steps=500 \
    --eval-interval=100

# 完整推理验证
python inference.py \
    --amp \
    --amp-dtype=float16 \
    --benchmark \
    --num-prompts=500
```

### 5.2 性能 vs 精度对比报告

```markdown
## 混合精度优化报告

| 配置 | Loss | Accuracy | Throughput (samples/s) | 显存 (GB) | 加速比 |
|------|------|----------|----------------------|-----------|--------|
| FP32 基线 | 2.345 | 0.856 | 128 | 23.4 | 1.00x |
| AMP O1 FP16 | 2.351 | 0.855 | 285 | 15.2 | 2.23x |
| AMP O2 FP16 | 2.362 | 0.853 | 310 | 14.8 | 2.42x |
| AMP O2 BF16 | 2.348 | 0.856 | 298 | 14.8 | 2.33x |
| AMP O2 + FP32 LN | 2.354 | 0.855 | 305 | 14.9 | 2.38x |
| **最终策略** | **2.353** | **0.855** | **305** | **14.9** | **2.38x** |

**最终精度策略**:
- AMP Level: O2
- 默认精度: float16
- FP32 保留层: [layer_norm, softmax, lm_head]
- Loss Scaling: init_scale=2^16, growth_interval=2000
- 溢出率: 0.3%
```

### 5.3 ⚠️ 检查点：端到端确认

- [ ] 精度指标与 FP32 基线差异 < 1%（核心指标）
- [ ] 性能加速比 ≥ 1.5x（相对 FP32）
- [ ] 显存节省 ≥ 30%
- [ ] 端到端报告已归档

---

## Phase 6: 迭代决策

### 6.1 效果评判

| 等级 | 条件 | 决策 |
|------|------|------|
| 🟢 达标 | 精度差异 < 1% 且加速比 ≥ 1.5x | 输出最终配置，准备部署 |
| 🟡 可接受 | 精度差异 < 2% 且加速比 ≥ 1.3x | 确认用户接受精度损失，归档 |
| 🔴 不达标 | 精度差异 ≥ 2% 或加速比 < 1.3x | 返回 Phase 2 调整策略 |

### 6.2 最终交付物

| 交付物 | 路径 |
|--------|------|
| AMP 配置文件 | `docs/perf/final/amp_policy.yaml` |
| 精度对比报告 | `docs/perf/final/precision_report.md` |
| 性能对比报告 | `docs/perf/final/performance_report.md` |
| 精度敏感算子清单 | `docs/perf/final/sensitive_layers.md` |

### 6.3 ⚠️ 检查点：迭代决策确认

- [ ] 当前策略已满足精度和性能目标
- [ ] 所有配置已归档
- [ ] 已决定下一步（部署 / 进一步尝试 INT8 / 结束）

---

## 异常处理

| 异常场景 | 可能原因 | 排查步骤 | 处理方式 |
|----------|----------|----------|----------|
| Loss 发散（NaN） | FP16 梯度溢出 | 检查 Loss Scaling 是否启用；检查溢出率 | 增大 init_scale；切换 BF16；关键层回退 FP32 |
| 精度损失 > 5% | 大量敏感算子被转为 FP16 | 逐层排查精度敏感算子 | 配置 FP32 白名单 |
| 训练速度无明显提升 | AMP 未生效 / Host 瓶颈 | 检查 autocast 是否正确包裹前向；检查数据加载 | 确认 autocast 作用域；优化数据流水线 |
| Loss Scaling 持续溢出 | 模型梯度范围太宽 | 检查 grad norm 分布；尝试 BF16 | 使用 BF16；增加 FP32 保留层 |
| 推理结果异常 | autocast 未启用 / dtype 不匹配 | 检查推理脚本的精度配置 | 统一训练和推理的精度策略 |
| 显存未显著减少 | 参数未实际转为 FP16 | 检查模型参数 dtype | 确认 model.half() 或 autocast 生效 |

---

## 参考资源

### 关联 Skill

| Skill | 关系 |
|-------|------|
| [msprof-optimizer](../msprof-optimizer/SKILL.md) | **前置** – 提供 Profiling 数据判断是否 FP16 瓶颈 |
| [memory-optimizer](../memory-optimizer/SKILL.md) | **互补** – 混合精度和内存优化常配合使用 |
| [ascendc-operator-optim](../ascendc-operator-optim/SKILL.md) | **互补** – 精度敏感算子可用 AscendC 重写优化 |

### 精度验证清单

- [ ] Phase 1: FP32 基线已建立
- [ ] Phase 2: AMP 首轮配置完成，精度偏差 < 5%
- [ ] Phase 3: 精度敏感算子已排查定位
- [ ] Phase 4: 精度-性能平衡策略已确定
- [ ] Phase 5: 端到端验证精度差异 < 1%，加速比 ≥ 1.5x
- [ ] Phase 6: 配置已交付归档
- [ ] 溢出率 < 1%
- [ ] 逐层对比余弦相似度 ≥ 0.99
