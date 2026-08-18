---
name: memory-optimizer
description: NPU 显存优化。流程：显存 Profiling 采集基线 → 识别显存热点 → 选择优化策略（激活重计算/梯度累积/显存卸载/混合精度/张量压缩）→ 实施优化 → 端到端验证 → 迭代。覆盖训练和推理场景。
keywords:
  - 显存优化
  - activation checkpointing
  - 梯度累积
  - 内存卸载
  - recompute
  - ascend npu
  - 显存分析
  - OOM
---

# NPU 显存优化 Skill

## 重要默认行为

- **不牺牲模型精度**：显存优化不得改变模型输出精度（必须保持端到端精度不变）。
- **渐进式优化**：每次只应用一种策略，验证显存收益和性能影响后再叠加。
- **性能约束**：任何显存优化策略不得导致性能下降 > 10%（除非用户明确允许以显存换速度）。
- **OOM 优先级**：遇到 OOM 时优先使用激活重计算，其次梯度累积，最后才考虑数据卸载。
- **文档记录**：每轮优化的策略、参数、效果必须记录。

## 前置条件

| 检查项 | 说明 |
|--------|------|
| 环境 | CANN ≥ 7.0，`torch_npu` 可用 |
| NPU 显存 | `npu-smi info` 确认可用显存大小 |
| 模型 | 模型可在 NPU 上运行（可小 batch 运行） |
| 基线 | 至少有一个可复现的 batch 可以直接运行 |
| Profiling | msprof 可用，用于采集显存使用曲线 |

## 工作流总览

```
Phase 1: 显存使用基线采集
    ├── 输入: 模型训练/推理脚本
    ├── 活动: 逐段记录显存峰值 → 定位显存热点
    └── 产出: 显存使用分布报告 + 峰值热点定位

Phase 2: 显存优化策略选择
    ├── 输入: 显存热点分布
    ├── 活动: 匹配最优策略（重计算/累积/卸载/混合精度/压缩）
    └── 产出: 优化策略计划

Phase 3: 策略实施
    ├── 输入: 优化策略计划
    ├── 活动: 实施激活重计算 / 配置梯度累积 / 开启显存卸载等
    └── 产出: 修改后的训练/推理配置

Phase 4: 单策略效果验证
    ├── 输入: 修改后的配置
    ├── 活动: 单策略下 Profiling → 显存峰值对比 → 性能影响评估
    └── 产出: 单策略效果报告

Phase 5: 多策略组合验证
    ├── 输入: 各策略效果报告
    ├── 活动: 多策略叠加 → 端到端显存 vs 性能对比
    └── 产出: 最终显存优化报告

Phase 6: 迭代决策
    ├── 达标 → 输出最终配置
    └── 不达标 → 返回 Phase 2 更换策略
```

---

## Phase 1: 显存使用基线采集

### 1.1 显存 Profiling

```python
# memory_profiler.py — 显存使用采集
import torch
import torch_npu

class MemoryProfiler:
    def __init__(self):
        self.records = []
        self.peak_memory = 0

    def reset(self):
        torch.npu.reset_peak_memory_stats()
        self.records = []
        self.peak_memory = 0

    def step(self, tag=""):
        """记录当前显存使用"""
        current = torch.npu.memory_allocated() / (1024**3)  # GB
        reserved = torch.npu.memory_reserved() / (1024**3)
        peak = torch.npu.max_memory_allocated() / (1024**3)
        self.records.append({
            "tag": tag,
            "current_gb": round(current, 2),
            "reserved_gb": round(reserved, 2),
            "peak_gb": round(peak, 2)
        })
        if peak > self.peak_memory:
            self.peak_memory = peak
        return self.records[-1]

    def report(self):
        """生成显存报告"""
        print(f"{'Tag':30s} {'Current(GB)':15s} {'Reserved(GB)':15s} {'Peak(GB)':15s}")
        print("-" * 75)
        for r in self.records:
            print(f"{r['tag']:30s} {r['current_gb']:<15.2f} {r['reserved_gb']:<15.2f} {r['peak_gb']:<15.2f}")
        print(f"\nTotal Peak Memory: {self.peak_memory:.2f} GB")

# 使用示例
profiler = MemoryProfiler()
profiler.reset()

profiler.step("init")
model = Model().npu()
profiler.step("model loaded")

optimizer = torch.optim.AdamW(model.parameters())
profiler.step("optimizer init")

batch = next(iter(dataloader))
batch = {k: v.npu() for k, v in batch.items()}
profiler.step("data loaded")

outputs = model(**batch)
profiler.step("forward pass")

loss = criterion(outputs)
profiler.step("loss")

loss.backward()
profiler.step("backward pass")

optimizer.step()
profiler.step("optimizer step")

profiler.report()
```

### 1.2 显存使用分析

**模型显存组成**：

| 组件 | 说明 | 占比（典型） | 可否优化 |
|------|------|-------------|---------|
| 模型参数 | 权重 + bias | 20-40% | 混合精度、量化 |
| 优化器状态 | Adam 的 momentum + variance | 10-20% | 切换优化器（Adam→AdamW→SGD） |
| 激活值 (Activations) | 前向中间结果 | 30-50% | **激活重计算** |
| 梯度 (Gradients) | 反向传播梯度 | 10-20% | 梯度累积 |
| 临时缓冲区 | 通信/计算临时空间 | 5-10% | 减少 batch size |

**显存瓶颈定位**：

```bash
# 使用 msprof 查看显存时间线
msprof --application="python train.py --batch-size=1" \
       --output=./TIMELINE_MEMORY \
       --trace-level=1 \
       --memory-trace=on

# 查看显存时间线图（找出峰值点对应的算子）
msprof --output=./TIMELINE_MEMORY --level=0 -c timeline
```

### 1.3 ⚠️ 检查点：显存基线确认

- [ ] 显存基线已采集（包括前向/反向/优化器各阶段峰值）
- [ ] 已定位显存峰值点对应的算子/模块
- [ ] 已分类统计（参数/激活/优化器/梯度各占比）
- [ ] 基线报告已归档到 `docs/perf/baseline/memory_baseline.json`

---

## Phase 2: 显存优化策略选择

### 2.1 策略选择矩阵

| 场景 | 推荐策略 | 预期显存节省 | 性能影响 | 实施难度 |
|------|----------|-------------|---------|---------|
| 激活值占比 > 40% | 激活重计算 (Checkpointing) | 30-50%（激活值节省） | -10~20% | ★★ |
| 显存不足且 batch > 1 | 减小 Batch Size + 梯度累积 | 与 batch 缩同比 | +0~5% | ★ |
| 优化器状态占比高 | 切换优化器 / 混合精度 | 15-40% | -0~5% | ★★ |
| 模型参数占比高 | 混合精度 + 量化 | 40-60% | -5~30% | ★★★ |
| 训练 + 推理 OOM | 显存卸载 (Offload) | 20-50%（CPU→NPU搬运） | -20~50% | ★★★★ |
| 推理场景显存不足 | KV Cache 优化 / PagedAttention | 50-80% | -0~10% | ★★★★ |

### 2.2 各策略速览

**激活重计算 (Activation Checkpointing)**：

```python
# PyTorch 原生 checkpoint
from torch.utils.checkpoint import checkpoint

class TransformerWithCheckpoint(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # 每隔一层启用 checkpoint（平衡显存和计算）
            # 注意：checkpoint 内的层在前向时不保存激活值
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**梯度累积 (Gradient Accumulation)**：

```python
# 梯度累积配置
configs = {
    "batch_size": 4,          # 每次前向的 batch
    "gradient_accumulation_steps": 8,  # 累积 8 步
    "effective_batch_size": 32,        # 等效 batch = 4 * 8
}

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs.loss / configs["gradient_accumulation_steps"]
    loss.backward()

    if (i + 1) % configs["gradient_accumulation_steps"] == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**优化器状态优化**：

```python
# 优化器显存占用对比（每个参数）
# Adam: 8 bytes (param) + 8 bytes (momentum) + 8 bytes (variance) = 24 bytes
# AdamW: 8 bytes + 8 bytes + 8 bytes = 24 bytes
# SGD + Momentum: 8 bytes + 8 bytes = 16 bytes
# 无优化器（推理）: 0

# FP16 训练时可用 FP32 优化器状态（即 Master Weights）
# 参数：FP16 (2 bytes) + 优化器状态：FP32 (12 bytes) = 14 bytes per param
# 相比全 FP32: 24→14 bytes, 节省 42%
```

### 2.3 ⚠️ 检查点：策略选择确认

- [ ] 已根据显存热点定位结果选择 1-2 个策略
- [ ] 已评估各策略的显存收益和性能影响
- [ ] 已与用户确认优化策略方向

---

## Phase 3: 策略实施

### 3.1 激活重计算实施

```python
# 选择性重计算 — 只对显存占用最大的层启用
# 规则：每 N 层启用一次 checkpoint，间隔可调

def apply_selective_checkpointing(model, interval=4):
    """间隔 interval 层启用一次 checkpoint"""
    for idx, layer in enumerate(model.transformer_layers):
        if idx % interval == 0:
            # 原始前向
            layer.forward = lambda x, l=layer: l(x)
        else:
            # 带 checkpoint 的前向
            original_forward = layer.forward
            def checkpointed_forward(x, l=layer, of=original_forward):
                return checkpoint(of, x, use_reentrant=False)
            layer.forward = checkpointed_forward
```

**间隔选择参考**：

| 间隔 | 显存节省 | 额外计算开销 | 推荐场景 |
|------|---------|-------------|---------|
| 1（每层） | 50-60% | +30-40% | 显存极端紧张 |
| 2（隔一层） | 35-45% | +15-25% | 平衡推荐 |
| 4（隔三层） | 20-30% | +8-15% | 显存略有不足 |

### 3.2 梯度累积配置

```python
# 梯度累积步数计算
def compute_gradient_accumulation_steps(
    target_batch_size,       # 目标等效 batch size
    device_batch_size,       # NPU 单次前向 batch size
):
    assert target_batch_size % device_batch_size == 0, \
        "target_batch_size 必须能被 device_batch_size 整除"
    return target_batch_size // device_batch_size

# 示例：目标 64，单卡放得下 4
steps = compute_gradient_accumulation_steps(64, 4)  # → 16
```

### 3.3 推理场景 KV Cache 优化

```python
# PagedAttention 风格的 KV Cache 管理（vLLM 等框架已内置）
# 自定义实现可选择：
# 1. 使用 vLLM / MindIE 等推理框架（推荐）
# 2. 手动管理 KV Cache 的分配和释放

# KV Cache 显存计算公式
def estimate_kv_cache_memory(
    num_layers,        # 层数
    num_heads,         # 注意力头数
    head_dim,          # 每头维度
    max_seq_len,       # 最大序列长度
    batch_size,        # 批次大小
    dtype_bytes,       # 数据类型字节数 (FP16=2, FP32=4, INT8=1)
):
    # K 和 V 各一份
    bytes_per_layer = 2 * batch_size * num_heads * max_seq_len * head_dim * dtype_bytes
    total_bytes = num_layers * bytes_per_layer
    return total_bytes / (1024**3)  # GB

# 示例：70B 模型推理，FP16，4096 seq_len
memory = estimate_kv_cache_memory(
    num_layers=80, num_heads=64, head_dim=128,
    max_seq_len=4096, batch_size=1, dtype_bytes=2
)  # ≈ 5.2 GB per seq (with 4096 len)
```

### 3.4 ⚠️ 检查点：实施验证

- [ ] 激活重计算配置已添加（interval/pattern 明确）
- [ ] 梯度累积步数已计算并配置
- [ ] 混合精度已启用（若有）
- [ ] 推理场景 KV Cache 已做限制或启用 Page Attention
- [ ] 代码运行正常，无新错误

---

## Phase 4: 单策略效果验证

### 4.1 单策略显存对比

```bash
# 优化前基线
python train.py --batch-size=4 --no-checkpointing \
    --profiling-memory 2>&1 | tee baseline_memory.log

# 优化后（激活重计算）
python train.py --batch-size=4 --checkpoint-interval=2 \
    --profiling-memory 2>&1 | tee checkpointing_memory.log
```

**对比指标**：

| 指标 | 基线 | 优化后 | 变化 |
|------|------|--------|------|
| 峰值显存 (GB) | 23.4 | 15.8 | **-32.5%** |
| 激活值显存 (GB) | 10.2 | 4.5 | **-55.9%** |
| 单步耗时 (ms) | 125 | 148 | +18.4% |
| 吞吐 (samples/s) | 512 | 432 | -15.6% |

### 4.2 性能影响评估

```markdown
## 显存优化性能影响

| 策略 | 显存节省 | 吞吐下降 | 性价比评分 |
|------|---------|---------|-----------|
| 激活重计算 (interval=2) | 35% | -15% | ★★★★☆ |
| 梯度累积 (4×4→16) | 30% | -2% | ★★★★★ |
| 混合精度 FP16 | 42% | +40% ↑ | ★★★★★ |
| 显存卸载 (CPU offload) | 45% | -35% | ★★☆☆☆ |
| 切换优化器 (Adam→SGD) | 40% | -5% | ★★★☆☆ |
```

### 4.3 ⚠️ 检查点：单策略验证

- [ ] 每个策略的显存节省已量化（≥ 预期值的 80% 视为有效）
- [ ] 性能影响已评估（下降 ≤ 10% 为可接受）
- [ ] 各策略的配置参数已记录

---

## Phase 5: 多策略组合验证

### 5.1 策略叠加

```bash
# 推荐优化组合：混合精度 + 激活重计算 + 梯度累积
python train.py \
    --batch-size=4 \
    --amp --amp-dtype=float16 \
    --checkpoint-interval=2 \
    --gradient-accumulation-steps=4 \
    --profiling-memory
```

### 5.2 端到端对比

```markdown
## 显存优化最终报告

| 配置 | 峰值显存 (GB) | 吞吐 (samples/s) | 单步耗时 (ms) | 显存节省 | 性能影响 |
|------|-------------|-----------------|--------------|---------|---------|
| 基线 | 23.4 | 512 | 125 | — | — |
| +混合精度 FP16 | 13.6 | 712 | 90 | -42% | +39% |
| +激活重计算 (i=2) | 12.1 | 615 | 104 | -48% | -3% |
| **最终组合** | **12.1** | **615** | **104** | **-48.3%** | **-3.1%** |

**最终配置**:
- 混合精度: AMP O2, float16
- 激活重计算 interval: 2
- 梯度累积: steps=4 (batch=4 → effective=16)
- 优化器: AdamW (FP32 master weights)
```

### 5.3 ⚠️ 检查点：组合验证确认

- [ ] 最终显存节省 ≥ 35%（或满足目标 batch size 运行需求）
- [ ] 性能下降 ≤ 10%（相对基线优化前）
- [ ] 端到端精度无变化（与优化前一致的 loss/acc）
- [ ] 无 OOM 风险

---

## Phase 6: 迭代决策

### 6.1 效果评判

| 等级 | 条件 | 决策 |
|------|------|------|
| 🟢 达标 | 显存满足目标 batch + 性能下降 ≤ 10% | 输出最终配置 |
| 🟡 可接受 | 显存满足目标 batch + 性能下降 > 10% | 报告性能影响，询问用户 |
| 🔴 不达标 | 显存仍不满足目标 batch | 返回 Phase 2 尝试其他策略 |

### 6.2 最终交付物

| 交付物 | 路径 |
|--------|------|
| 显存优化配置 | `docs/perf/final/memory_config.yaml` |
| 显存对比报告 | `docs/perf/final/memory_report.md` |
| 各策略效果表 | `docs/perf/final/strategy_comparison.md` |

### 6.3 ⚠️ 检查点：迭代决策确认

- [ ] 显存优化达到预期目标
- [ ] 交付物已归档
- [ ] 配置可复现

---

## 异常处理

| 异常场景 | 可能原因 | 排查步骤 | 处理方式 |
|----------|----------|----------|----------|
| 激活重计算后 OOM | checkpoint 区间内仍有过大张量 | 减小 checkpoint interval；分批重计算 | 缩小 interval 或对部分层关闭 |
| 梯度累积后 loss 震荡 | 等效 batch size 过大，BN 层行为变化 | 检查 BN 层是否改为 sync BN；调整 LR | 同步 BN；按梯度累积步数缩放 LR |
| 显存卸载后训练极慢 | CPU↔NPU 传输频繁 | 增大卸载阈值；减少卸载量 | 优先使用重计算而非卸载 |
| 混合精度 + 重计算精度下降 | 重计算中 FP16 累积误差 | 重计算区域使用 FP32 | checkpoint 内使用 FP32 或个别层 FP32 |
| KV Cache 优化后推理错误 | Cache 索引越界 | 检查 seq_len 管理逻辑 | 调试 cache 分配和释放逻辑 |

---

## 参考资源

### 关联 Skill

| Skill | 关系 |
|-------|------|
| [mixed-precision-optimizer](../mixed-precision-optimizer/SKILL.md) | **互补** – 混合精度是最有效的显存节约手段之一 |
| [msprof-optimizer](../msprof-optimizer/SKILL.md) | **前置** – Profiling 识别显存峰值位置 |
| [ascendc-operator-optim](../ascendc-operator-optim/SKILL.md) | **补充** – 自定义算子可减少中间激活值 |

### 显存优化验证清单

- [ ] Phase 1: 显存基线已采集（各阶段峰值）
- [ ] Phase 2: 优化策略已选择（至少 2 种备选）
- [ ] Phase 3: 策略已实施无错误
- [ ] Phase 4: 单策略效果已验证
- [ ] Phase 5: 组合策略端到端验证通过
- [ ] Phase 6: 交付物已归档
- [ ] 最终显存节省 ≥ 35%
- [ ] 性能影响 ≤ 10%
- [ ] 精度与优化前一致
