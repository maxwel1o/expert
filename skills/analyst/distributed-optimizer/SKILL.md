---
name: distributed-optimizer
description: 多 NPU 分布式训练与推理优化。流程：分析通信模式 → 选择并行策略（数据/模型/流水线/张量并行）→ HCCL 通信调优 → 计算-通信重叠 → 端到端扩展效率验证。覆盖 Ascend HCCL 特性与多卡场景。
keywords:
  - 分布式训练
  - 数据并行
  - 模型并行
  - HCCL
  - allreduce
  - pipeline parallel
  - tensor parallel
  - 扩展效率
---

# 分布式训练与推理优化 Skill

## 重要默认行为

- **扩展效率优先**：多卡训练的加速比是核心指标，目标为线性扩展的 80%+。
- **通信-计算重叠**：优先通过通信与计算重叠来隐藏通信延迟，而非单纯优化通信本身。
- **渐进式扩卡**：1卡→2卡→4卡→8卡，逐步检查扩展效率，定位瓶颈。
- **Profiling 驱动**：所有优化决策基于 HCCL Profiling 数据，避免凭经验猜测。
- **结果记录**：每次扩卡实验必须记录加速比、通信占比、计算效率。

## 前置条件

| 检查项 | 说明 |
|--------|------|
| 环境 | 多卡 NPU 环境（≥2），CANN ≥ 7.0 |
| HCCL | `hccl_tools` 可用，`hccn_tool` 可用 |
| 网络 | 服务器内 NPU 互联（HCCS）状态正常 |
| 模型 | 模型可在单卡 NPU 上正常运行 |
| 通信库 | `torch.distributed` + `torch_npu` 分布式支持 |

## 工作流总览

```
Phase 1: 单卡基线 + 通信拓扑分析
    ├── 输入: 模型 + 单卡 Profiling 数据
    ├── 活动: 单卡性能基线 → HCCL 拓扑探测 → 通信带宽测量
    └── 产出: 单卡基线报告 + 通信拓扑分析

Phase 2: 并行策略选择
    ├── 输入: 模型参数规模 + 显存限制 + 拓扑数据
    ├── 活动: DDP / FSDP / TP / PP 策略评估 → 策略组合设计
    └── 产出: 并行策略方案

Phase 3: HCCL 通信优化
    ├── 输入: 通信 Profiling 数据
    ├── 活动: HCCL 算法选择 → 通信域优化 → 通信量压缩
    └── 产出: 通信优化配置

Phase 4: 计算-通信重叠
    ├── 输入: Timeline + HCCL 事件
    ├── 活动: 通信异步化 → Bucket 调优 → 重叠窗口最大化
    └── 产出: 重叠优化配置

Phase 5: 扩展效率验证
    ├── 输入: 多卡训练配置
    ├── 活动: 1/2/4/8 卡逐级加速比测试 → 扩展效率分析
    └── 产出: 扩展效率报告

Phase 6: 迭代决策
    ├── 达标 → 输出最终配置
    └── 不达标 → 返回 Phase 2 调整策略
```

---

## Phase 1: 单卡基线 + 通信拓扑分析

### 1.1 单卡性能基线

```bash
python train.py --npu=0 --batch-size=4 --max-steps=100 --profiling
```

**关键指标**：

| 指标 | 说明 | 数值示例 |
|------|------|---------|
| 单步耗时 (ms/step) | 单卡一次前向+反向+优化 | 125 ms |
| 吞吐 (samples/s) | 单卡每秒处理样本数 | 32 |
| 计算时间占比 | AI Core 计算时间 / 总时间 | 65% |
| 通信占比（单卡）| 无通信，此处为 baseline | 0% |

### 1.2 HCCL 拓扑探测

```bash
# 查看 HCCL 拓扑
hccn_tool -i 0 -link -g    # 查看 0 号 NPU 的互联拓扑
hccn_tool -i 0 -net -g     # 查看网络配置

# HCCL 带宽测试
# 安装 hccl_tools 后运行带宽测试
hccl_test -n 8 -b 1M -e 1G -f 2
```

**Ascend NPU 互联拓扑速查**：

| 拓扑类型 | 服务器配置 | NPU 互联带宽 | 通信模式建议 |
|----------|-----------|-------------|-------------|
| 单机 8 卡（全互联） | Atlas 800T A2 | HCCS 互联 ~56GB/s | DDP/FSDP 均高效 |
| 双机 16 卡 | 2 × Atlas 800T A2 | 跨机 RDMA ~200Gb/s | 数据并行跨机，模型并行机内 |
| 4 机 32 卡 | 4 × Atlas 800T A2 | 跨机网络 ~200Gb/s | 流水线并行跨机 |

### 1.3 通信 Profiling

```bash
# 使用 msprof 采集通信详情（多卡环境下）
msprof --application="python -m torch.distributed.run --nproc_per_node=8 train.py" \
       --output=./TIMELINE_DIST \
       --trace-level=1 \
       --hccl-trace=on \
       --profile-iterations=50
```

**HCCL Profiling 关键指标解读**：

| msprof 指标 | 说明 | 正常值 | 需优化 |
|-------------|------|--------|--------|
| HCCL Duration | 通信总耗时 | < 计算时间的 30% | > 50% |
| Link Utilization | HCCS 链路利用率 | > 70% | < 40% |
| AllReduce Time | AllReduce 耗时 | < 5ms (大模型 < 20ms) | 持续增长 |
| Communication Ratio | 通信/计算比 | < 0.3 | > 0.5 |

### 1.4 ⚠️ 检查点：基线与拓扑确认

- [ ] 单卡基线已建立
- [ ] HCCL 拓扑已确认（卡间互联拓扑清晰）
- [ ] 通信带宽已测量（HCCS/RDMA）
- [ ] 已记录通信 Profiling 基线

---

## Phase 2: 并行策略选择

### 2.1 策略选择决策树

```
模型规模 < 1B 参数
├── 单卡放得下 → DDP（数据并行）
└── 单卡放不下 → DDP + 混合精度 + 激活重计算

模型规模 1B ~ 10B 参数
├── 单卡放得下 → FSDP（全分片数据并行）
└── 单卡放不下 → FSDP + TP（张量并行）

模型规模 10B ~ 100B 参数
├── FSDP + TP + PP（流水线并行）
└── 显存仍不足 → 增加 PP stage 数

模型规模 > 100B 参数
├── 3D 并行: TP + PP + DP
├── 使用序列并行 (SP) 减少激活值
└── 启用分布式优化器
```

### 2.2 各并行策略对比

| 策略 | 通信量 | 显存节省 | 实现复杂度 | 推荐场景 |
|------|--------|---------|-----------|---------|
| **DDP** (Data Parallel) | 每步 AllReduce 梯度 | 无（每卡完整模型） | ★ | 小模型，大 batch |
| **FSDP** (Fully Sharded DP) | 前向/反向各 gather 参数 | 模型参数量×卡数 | ★★★ | 1B-10B 模型 |
| **TP** (Tensor Parallel) | 每层前向通信 | 每层参数/TP 度 | ★★★ | 10B+ 大模型 |
| **PP** (Pipeline Parallel) | 每 micro batch 边界通信 | 模型参数/PP 度 | ★★ | 100B+ 超大规模 |
| **SP** (Sequence Parallel) | 序列维度通信 | 激活值/SP 度 | ★★★★ | 长序列场景 |
| **Grouped GEMM** | 无额外通信 | 计算融合 | ★★★ | MoE 模型 |

### 2.3 FSDP 配置模板

```python
# FSDP 配置（Ascend NPU）
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)

# 混合精度策略
fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,  # 通信使用 FP16
    buffer_dtype=torch.float16,
)

# FSDP 配置
fsdp_config = dict(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 全分片
    mixed_precision=fp16_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取
    cpu_offload=CPUOffload(offload_params=False),     # 不卸载到 CPU
    limit_all_gathers=True,                           # 限制并行 gather
    use_orig_params=True,
)

model = FSDP(model, **fsdp_config)
```

**FSDP 关键参数调优**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `sharding_strategy` | `FULL_SHARD` | 最高效分片，通信最多 |
| | `SHARD_GRAD_OP` | 只分片梯度+优化器状态 |
| | `NO_SHARD` | 退化到 DDP |
| `backward_prefetch` | `BACKWARD_PRE` | 反向时预取下一层参数 |
| `forward_prefetch` | `True` | 前向时预取参数（需额外显存） |
| `limit_all_gathers` | `True` | 限制同时进行的 all-gather 数量 |
| `mixed_precision` | param=FP16, reduce=FP16 | 通信量减半 |

### 2.4 ⚠️ 检查点：并行策略选择

- [ ] 已根据模型规模和单卡显存确定并行策略
- [ ] 已评估各策略的理论通信量和显存节省
- [ ] 已与用户确认并行方案

---

## Phase 3: HCCL 通信优化

### 3.1 HCCL 算法选择

```python
# HCCL 算法环境变量配置
import os

# AllReduce 算法选择
os.environ["HCCL_ALGO"] = "ring"           # 环形（带宽敏感，小消息适用）
os.environ["HCCL_ALGO"] = "tree"           # 树形（延迟敏感，大消息适用）
os.environ["HCCL_ALGO"] = "ring, tree"     # 自动选择

# 超节点优化（Atlas 800T A2）
os.environ["HCCL_INTRA_NODE_ALGO"] = "fullmesh"  # 单机内全互联
os.environ["HCCL_INTER_NODE_ALGO"] = "ring"      # 跨机使用 ring

# 通信缓冲区
os.environ["HCCL_BUFF_SIZE"] = "256"       # 通信缓冲区大小 (MB)
os.environ["HCCL_NBUFFS"] = "2"            # 双缓冲
```

**HCCL 算法选择指南**：

| 消息大小 | 推荐算法 | 理由 |
|----------|---------|------|
| < 1 MB | Ring | 小消息延迟低 |
| 1 MB - 64 MB | 自动选择 | 根据实测结果 |
| > 64 MB | Tree | 大消息带宽利用率高 |
| 机内 (HCCS) | FullMesh | 全互联拓扑效率最高 |
| 跨机 (RDMA) | Ring | 跨机网络延迟高，ring 更稳定 |

### 3.2 通信量压缩技术

```python
# 梯度压缩
# 1. FP32→FP16 压缩（通信量减半）
os.environ["HCCL_DTYPE"] = "float16"

# 2. 梯度裁剪 + 跳过小梯度
# 在 backward hook 中实现：
def gradient_compression_hook(grad):
    threshold = 1e-6
    # 压缩：只保留大于阈值的梯度
    mask = grad.abs() > threshold
    compressed = grad[mask]
    # ... 发送压缩后的梯度
    return grad

# 3. TopK 梯度压缩（减少通信量 90%+，但有精度损失）
def topk_compression(grad, ratio=0.01):
    """保留 top 1% 的梯度"""
    k = max(1, int(grad.numel() * ratio))
    values, indices = torch.topk(grad.abs().flatten(), k)
    # 只发送 values 和 indices
    return values, indices
```

### 3.3 HCCL 带宽优化检查清单

```bash
# 1. 确认 HCCS 链路状态（应为 Active）
hccn_tool -i 0 -link -g

# 2. 检查 NPU 时钟频率（应 > 1500 MHz）
npu-smi info -t clock -i 0

# 3. 确认网卡绑定（跨机场景）
hccn_tool -i 0 -net -g

# 4. 检查 RoCE 网卡配置
rdma link show
```

### 3.4 ⚠️ 检查点：通信优化确认

- [ ] HCCL 算法已根据消息大小和拓扑选择
- [ ] 通信量压缩已配置（至少 FP16 通信）
- [ ] HCCL 相关环境变量已设置
- [ ] 通信 Profiling 显示带宽利用率 > 60%

---

## Phase 4: 计算-通信重叠

### 4.1 DDP 通信重叠（Bucket 调优）

```python
# DDP 默认在一个 bucket 全部就绪后开始通信
# 优化：减小 bucket 大小，让通信更早开始

import torch

# 默认 bucket 大小：25 MB
# 优化后：5 MB（更早开始通信）
torch.nn.parallel.DistributedDataParallel(
    model,
    bucket_cap_mb=5,           # ← 关键参数：bucket 大小
    gradient_as_bucket_view=True,
    static_graph=True,         # 静态图优化
)
```

**Bucket 大小选择**：

| Bucket 大小 | 通信重叠度 | 适用场景 |
|-------------|-----------|---------|
| 25 MB（默认） | 低（等全部梯度就绪） | 通用场景 |
| 5-10 MB | 中 | 推荐，通信隐藏较好 |
| 1-5 MB | 高 | 通信密集型场景 |
| > 100 MB | 极低 | 计算密集型、小模型 |

### 4.2 FSDP 通信预取优化

```python
# FSDP 通信预取配置
from torch.distributed.fsdp import BackwardPrefetch

# BackwardPrefetch.BACKWARD_PRE: 在当前层的 backward 完成前，
# 预取下一层的参数（默认）
# BackwardPrefetch.BACKWARD_POST: 在当前层 backward 完成后，
# 再发起下一层预取（节省显存，但重叠度低）

fsdp_config = dict(
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,      # 前向时预取下一层参数
    limit_all_gathers=True,     # 限制并行 gather
)
```

### 4.3 通信-计算 Overlap 分析

```bash
# 使用 msprof timeline 分析重叠情况
msprof --output=./TIMELINE_DIST --level=0 -c timeline

# 在 timeline 中查看：
# - HCCL 事件（紫色/蓝色条）是否与 AICore 计算（绿色条）重叠
# - 通信等待时间（HCCL 开始到结束，计算完全空闲）
#
# 理想情况：通信完全隐藏在计算中（纯绿色 + 重叠紫色）
# 不理想：通信独占时间（紫色条前有空白间隔）
```

**重叠度计算**：

```
重叠度 = 通信与计算重叠的时间 / 通信总耗时 × 100%

重叠度 > 80% → 优良（通信几乎被完全隐藏）
重叠度 50-80% → 可接受
重叠度 < 50% → 需要优化（减小 bucket / 增大模型/增大 batch）
```

### 4.4 ⚠️ 检查点：重叠优化确认

- [ ] Bucket 大小已调优（DDP）或预取策略已配置（FSDP）
- [ ] Timeline 分析显示重叠度 > 60%
- [ ] 通信占比已从优化前降低（目标：通信/计算 < 0.3）

---

## Phase 5: 扩展效率验证

### 5.1 逐级扩展测试

```bash
# 1 卡
torchrun --nproc_per_node=1 train.py --batch-size=4
# 2 卡
torchrun --nproc_per_node=2 train.py --batch-size=4
# 4 卡
torchrun --nproc_per_node=4 train.py --batch-size=4
# 8 卡
torchrun --nproc_per_node=8 train.py --batch-size=4
```

### 5.2 扩展效率计算

```markdown
## 扩展效率报告

| 卡数 | 全局 Batch | 吞吐 (samples/s) | 加速比 | 扩展效率 | 通信占比 |
|------|-----------|-----------------|--------|---------|---------|
| 1    | 4          | 32             | 1.0x   | 100%    | 0%      |
| 2    | 8          | 61             | 1.91x  | 95.3%   | 8%      |
| 4    | 16         | 118            | 3.69x  | 92.2%   | 15%     |
| 8    | 32         | 220            | 6.88x  | 85.9%   | 23%     |

**扩展效率 = 实际加速比 / 理论加速比 × 100%**

**分析**:
- 1→2 卡: 效率 95.3%，少量通信开销
- 4→8 卡: 效率下降至 85.9%，通信占比上升
- 瓶颈：AllReduce 通信在大卡数下成为主要开销
- 建议：增大 per-GPU batch size 或启用梯度压缩
```

### 5.3 扩展效率优化要点

| 现象 | 可能原因 | 优化方案 |
|------|----------|----------|
| 1→2 卡效率低 | batch 太小，计算/通信比低 | 增大 per-GPU batch |
| 4→8 卡效率骤降 | 网络带宽不足 | 检查 HCCS 链路带宽 |
| 8 卡后无提升 | 通信完全主导 | 切换到 FSDP 或 TP |
| 显存随卡数线性减少 | 模型参数未分片 | 使用 FSDP |
| 加速比超线性 | 全局 batch 增大，梯度更稳定 | 可尝试进一步扩卡 |

### 5.4 ⚠️ 检查点：扩展效率确认

- [ ] 8 卡扩展效率 ≥ 80%
- [ ] 通信占比 ≤ 30%
- [ ] 显存使用随卡数合理分摊
- [ ] 扩展效率报告已归档

---

## Phase 6: 迭代决策

### 6.1 效果评判

| 等级 | 条件 | 决策 |
|------|------|------|
| 🟢 达标 | 8 卡扩展效率 ≥ 80% + 通信占比 ≤ 30% | 输出最终配置 |
| 🟡 可接受 | 8 卡扩展效率 ≥ 70% + 通信占比 ≤ 40% | 记录瓶颈，考虑 TP/PP |
| 🔴 不达标 | 8 卡扩展效率 < 70% 或 通信占比 > 40% | 返回 Phase 2 调整并行策略 |

### 6.2 最终交付物

| 交付物 | 路径 |
|--------|------|
| 并行策略配置 | `docs/perf/final/distributed_config.yaml` |
| 扩展效率报告 | `docs/perf/final/scaling_efficiency.md` |
| HCCL 优化配置 | `docs/perf/final/hccl_config.sh` |

### 6.3 ⚠️ 检查点：迭代决策确认

- [ ] 并行策略配置已确定，扩展效率达标
- [ ] HCCL 优化参数已记录
- [ ] 交付物已归档

---

## 异常处理

| 异常场景 | 可能原因 | 排查步骤 | 处理方式 |
|----------|----------|----------|----------|
| 多卡 NCCL timeout | HCCL 初始化失败 / 网络不通 | `hccl_tools` 检查链路；`hccn_tool` 检查网络 | 重启 HCCL 服务；检查网卡状态 |
| 某卡显存 OOM | 负载不均 / 分片策略问题 | 检查各卡显存使用 (`npu-smi info`) | FSDP 调整为均匀分片；调整 layout |
| 扩展效率 < 50% | batch 太小 / 通信占比高 | 分析通信占比；增大 per-GPU batch | 增大 batch；使用梯度累积 |
| 通信 Profiling 空的 | msprof 未输出 HCCL 事件 | 检查 `--hccl-trace=on` 是否正确 | 设置 `HCCL_PROFILING=1` 环境变量 |
| FSDP 训练不稳定 | 分片策略 + 混合精度组合问题 | 检查梯度 norm；逐步降级配置 | 先从 NO_SHARD 开始测试精度 |
| 跨机训练无提升 | 跨机网络带宽不足 | 测试跨机 RDMA 带宽 | 减少跨机通信量；调整策略使跨机只传少数据 |

---

## 参考资源

### 关联 Skill

| Skill | 关系 |
|-------|------|
| [msprof-optimizer](../msprof-optimizer/SKILL.md) | **前置** – Profiling 数据是通信优化的输入 |
| [memory-optimizer](../memory-optimizer/SKILL.md) | **互补** – 分布式+显存优化解决大模型训练 |
| [mixed-precision-optimizer](../mixed-precision-optimizer/SKILL.md) | **互补** – 混合精度减少通信量 |

### 分布式优化验证清单

- [ ] Phase 1: 单卡基线 + 通信拓扑已确认
- [ ] Phase 2: 并行策略已选择（DDP/FSDP/TP/PP）
- [ ] Phase 3: HCCL 通信优化已完成
- [ ] Phase 4: 计算-通信重叠已配置
- [ ] Phase 5: 扩展效率 ≥ 80%（8卡）
- [ ] Phase 6: 配置已归档
- [ ] 通信占比 ≤ 30%
- [ ] 无 HCCL timeout / OOM 异常
