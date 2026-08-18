---
name: data-pipeline-optimizer
description: 数据加载与预处理流水线优化。流程：数据加载 Profiling → 识别 I/O 瓶颈 → 多进程 Worker 调优 → 预处理加速 → 数据缓存/预取 → 端到端验证。覆盖 NPU 场景下 CPU-GPU 数据传输、数据增强和加载策略。
keywords:
  - 数据加载
  - dataloader
  - 预处理
  - 数据流水线
  - I/O 优化
  - prefetch
  - cache
  - ascend
---

# 数据加载与预处理流水线优化 Skill

## 重要默认行为

- **I/O 不成为瓶颈**：数据加载延迟必须 ≤ 模型单步计算时间的 50%，避免 NPU 等待数据。
- **不影响数据质量**：优化预处理流程不得改变数据分布或引入伪影。
- **先诊断再优化**：必须通过 Profiling 确认 I/O 是瓶颈后，再开始优化。
- **可复现性**：优化后的数据加载流程必须保持随机种子可复现。
- **记录**：每次优化必须记录数据加载延迟、NPU 利用率、吞吐变化。

## 前置条件

| 检查项 | 说明 |
|--------|------|
| 环境 | PyTorch DataLoader / MindSpore GeneratorDataset |
| 数据路径 | 数据存储在高速磁盘（SSD/NVMe 优先） |
| 模型 | 模型可在 NPU 上以 DataLoader 加载数据运行 |
| 基线 | 至少有 100 步以上稳定运行的数据 |

## 工作流总览

```
Phase 1: 数据流水线 Profiling
    ├── 输入: 模型训练脚本（含 DataLoader）
    ├── 活动: 数据加载时间采集 → NPU 空闲分析 → I/O vs 计算时间对比
    └── 产出: 数据流水线瓶颈分析报告

Phase 2: DataLoader 参数调优
    ├── 输入: 瓶颈分析报告
    ├── 活动: num_workers / prefetch_factor / batch_size 调优
    └── 产出: DataLoader 最优参数配置

Phase 3: 预处理加速
    ├── 输入: DataLoader 配置
    ├── 活动: 预处理计算卸载 → 内存缓存 → 数据增强优化
    └── 产出: 预处理优化配置

Phase 4: 存储与 I/O 优化
    ├── 输入: 数据存储现状
    ├── 活动: 数据格式优化 → 数据排布优化 → 存储分级
    └── 产出: 存储优化方案

Phase 5: 端到端验证
    ├── 输入: 所有优化配置
    ├── 活动: 端到端训练 → Profiling → 对比基线
    └── 产出: 端到端数据流水线优化报告

Phase 6: 迭代决策
    ├── 达标 → 输出最终配置
    └── 不达标 → 返回 Phase 2
```

---

## Phase 1: 数据流水线 Profiling

### 1.1 数据加载 Profiling

```python
# profile_dataloader.py — 数据加载性能分析
import time
import torch
from torch.utils.data import DataLoader

def profile_dataloader(dataloader, num_steps=100):
    """分析 DataLoader 的加载性能"""
    iter_dataloader = iter(dataloader)
    load_times = []

    torch.npu.synchronize()
    for i in range(num_steps):
        t0 = time.perf_counter()
        batch = next(iter_dataloader)
        t1 = time.perf_counter()
        load_times.append((t1 - t0) * 1000)  # ms

    avg_load_time = sum(load_times) / len(load_times)
    max_load_time = max(load_times)
    p99_load_time = sorted(load_times)[int(len(load_times) * 0.99)]

    print(f"DataLoader Profiling ({num_steps} steps):")
    print(f"  Avg load time:  {avg_load_time:.1f} ms")
    print(f"  Max load time:  {max_load_time:.1f} ms")
    print(f"  P99 load time:  {p99_load_time:.1f} ms")
    print(f"  Batch size:     {batch_size}")
    print(f"  Throughput:     {batch_size / (avg_load_time / 1000):.1f} samples/s")

    return {
        "avg_ms": avg_load_time,
        "max_ms": max_load_time,
        "p99_ms": p99_load_time,
    }
```

### 1.2 NPU 空闲率分析

```bash
# 方案 1: 使用 npu-smi 监控 NPU 利用率
# 在另一个终端运行
npu-smi info watch -i 0 &  # 监控 NPU 利用率

# 同时运行训练
python train.py --max-steps=200

# 观察 NPU 利用率：
# - > 90%: NPU 充分使用，数据加载不是瓶颈
# - 70-90%: 轻微 I/O 瓶颈
# - 50-70%: 明显 I/O 瓶颈
# - < 50%: 严重 I/O 瓶颈，数据加载主导
```

**更精确的方法 — 使用 msprof 看 NPU 空闲时间**：

```bash
msprof --application="python train.py --max-steps=200" \
       --output=./TIMELINE_IO \
       --trace-level=1 \
       --profile-iterations=50

# 分析 timeline 中 NPU 空闲间隙（没有 AICore 事件的区间）
msprof --output=./TIMELINE_IO --level=0 -c timeline
```

### 1.3 I/O 瓶颈判定

```python
# io_bottleneck_diagnosis.py
def diagnose_io_bottleneck(
    step_time_ms,        # 单步总耗时
    compute_time_ms,     # 实际计算耗时（NPU 工作时间）
    load_time_ms,        # 数据加载耗时
):
    npu_idle = step_time_ms - compute_time_ms
    io_ratio = load_time_ms / step_time_ms

    print(f"Step time:    {step_time_ms:.1f} ms")
    print(f"Compute time: {compute_time_ms:.1f} ms")
    print(f"NPU idle:     {npu_idle:.1f} ms")
    print(f"Load time:    {load_time_ms:.1f} ms")

    if io_ratio > 0.5:
        print("🔴 严重 I/O 瓶颈：数据加载占主导")
    elif io_ratio > 0.3:
        print("🟡 中等 I/O 瓶颈：数据加载显著影响训练")
    elif io_ratio > 0.1:
        print("🟢 轻微 I/O 瓶颈：可优化")
    else:
        print("✅ I/O 正常：数据加载不是瓶颈")
```

### 1.4 ⚠️ 检查点：瓶颈确认

- [ ] 数据加载耗时已测量（avg / p99）
- [ ] NPU 利用率已确认（利用率 < 80% 则有优化空间）
- [ ] I/O 瓶颈等级已判定
- [ ] Profiling 报告已归档到 `docs/perf/baseline/io_profile.json`

---

## Phase 2: DataLoader 参数调优

### 2.1 核心参数调优

```python
from torch.utils.data import DataLoader

# DataLoader 可调参数
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,          # 数据加载进程数
    prefetch_factor=2,      # 每个 worker 预取批次数
    pin_memory=True,        # 锁页内存加速 CPU→NPU 传输
    persistent_workers=True,  # worker 跨 epoch 保持
    timeout=0,              # 超时（0=无限）
    drop_last=True,         # 丢弃最后一个不完整 batch
)
```

**参数调优建议**：

| 参数 | 建议值 | 调优方向 |
|------|--------|---------|
| `num_workers` | CPU 核数 / 2 | 过低 → NPU 等数据；过高 → CPU 上下文切换开销 |
| `prefetch_factor` | 2-4 | 增大可预取更多数据，但增加显存占用 |
| `pin_memory` | True | **始终开启**，减少 CPU→NPU 传输延迟 |
| `persistent_workers` | True | **始终开启**，避免每 epoch 重建 worker |
| `batch_size` | 模型可承受的最大值 | 越大越好（增加计算/IO 比） |

### 2.2 num_workers 扫描

```python
# find_optimal_workers.py — 搜索最优 num_workers
def scan_num_workers(dataset, batch_size, max_workers=None):
    if max_workers is None:
        import os
        max_workers = os.cpu_count() // 2

    results = []
    for nw in [0, 1, 2, 4, 8, 12, 16, max_workers]:
        # 只测试可用的值
        if nw > max_workers:
            continue

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
        )

        profile = profile_dataloader(loader, num_steps=50)
        results.append((nw, profile["avg_ms"]))
        print(f"num_workers={nw:2d}: avg_load={profile['avg_ms']:.1f}ms")

    # 找到最优值
    best_nw, best_time = min(results, key=lambda x: x[1])
    print(f"\n✅ 最优 num_workers={best_nw}, load_time={best_time:.1f}ms")
    return results
```

**num_workers 典型曲线**：

```
load_time (ms)
  ↑
  | ●────── num_workers=0 (主进程加载，最慢)
  |  ●───── num_workers=1 (稍有改善)
  |   ●──── num_workers=2
  |    ●─── num_workers=4 (拐点，继续增加收益递减)
  |     ●── num_workers=8 (接近最优)
  |      ●─ num_workers=12 (收益持平或略增)
  |       ● num_workers=16 (可能因上下文切换而变慢)
  └────────────────────────→ num_workers
```

### 2.3 ⚠️ 检查点：DataLoader 参数确认

- [ ] num_workers 已通过扫描确定最优值
- [ ] pin_memory=True 和 persistent_workers=True 已启用
- [ ] prefetch_factor 已调优
- [ ] 优化后数据加载延迟降低 ≥ 30%

---

## Phase 3: 预处理加速

### 3.1 预处理计算卸载

**原则：将耗时预处理从 DataLoader worker 中移出或加速**。

```python
# 优化前：预处理在主进程中串行执行
def transform_slow(sample):
    # 耗时操作：JPEG 解码 + resize + normalize
    image = decode_jpeg(sample["image"])   # I/O 密集型
    image = cv2.resize(image, (224, 224))  # CPU 密集型
    image = normalize(image)               # 计算密集型
    return image

# 优化后：将解码后的数据提前存为 .npy 或 .pth
# 方案 A：数据预处理，训练时直接加载 Tensor
def transform_fast(sample):
    # 直接从预处理的 tensor 加载
    image = torch.load(sample["preprocessed_path"])
    return image

# 方案 B：使用 DALI / 类 DALI 方案（如果可用）
# 或使用 torchvision 的 v2 transforms（更高效）
from torchvision.transforms import v2
transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 3.2 预处理耗时分析

```python
# profile_transforms.py — 分析每步预处理耗时
import time

def profile_transforms(dataset, transform_fn, num_samples=100):
    times = []
    for i in range(num_samples):
        sample = dataset[i]
        t0 = time.perf_counter()
        transformed = transform_fn(sample)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg = sum(times) / len(times)
    print(f"Transform profiling ({num_samples} samples):")
    print(f"  Average: {avg:.1f} ms")
    print(f"  Min:     {min(times):.1f} ms")
    print(f"  Max:     {max(times):.1f} ms")

    # 如果预处理 > 5ms，值得优化
    if avg > 5:
        print("⚠️ 预处理耗时较高，建议优化")
    return avg
```

### 3.3 数据缓存策略

```python
# 缓存已预处理的数据（适合数据集较小或可重复使用的情况）
import diskcache as dc

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform, cache_dir="./data_cache"):
        self.base_dataset = base_dataset
        self.transform = transform
        self.cache = dc.Cache(cache_dir)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 缓存命中则直接返回
        if idx in self.cache:
            return self.cache[idx]

        # 缓存未命中，预处理并缓存
        sample = self.base_dataset[idx]
        transformed = self.transform(sample)
        self.cache[idx] = transformed
        return transformed
```

**缓存策略选择**：

| 缓存类型 | 适用场景 | 加速效果 | 说明 |
|----------|---------|---------|------|
| 内存缓存 (RAM) | 数据集 ≤ 内存 | 100x+ | 最快，但受内存限制 |
| 磁盘缓存 (SSD) | 10GB-100GB 数据集 | 5-10x | 适合大数据集 |
| 预处理后持久化 | 固定数据集 | 100x+ | 训练前一次预处理，后续直接加载 |
| WebDataset / MDS | 大规模分布式 | 10-50x | 流式加载 + 打乱 |

### 3.4 ⚠️ 检查点：预处理优化确认

- [ ] 预处理耗时已 Profiling（< 5ms 为优）
- [ ] 数据缓存已实施（至少磁盘缓存）
- [ ] 预处理 CPU 计算已优化（向量化 / 预计算）

---

## Phase 4: 存储与 I/O 优化

### 4.1 数据格式优化

```python
# 数据格式对比
# 原始格式：大量小文件（每个样本一个 JPEG）
# ├── train/
# │   ├── sample_00001.jpg  (~100KB)
# │   ├── sample_00002.jpg  (~100KB)
# │   └── ...  × 100,000 files  (10GB, 100K 文件)

# 优化格式 1：打包为 WebDataset (tar)
# ├── train-00000.tar  (~500MB per shard)
# ├── train-00001.tar  (~500MB per shard)
# └── ...  × 20 shards  (10GB, 20 文件)

# 优化格式 2：Mosaic Sharded Dataset (MDS)
# ├── train/
# │   ├── shard-00000.mds
# │   ├── shard-00001.mds
# │   └── ...

# 优化格式 3：内存映射 (memmap)
# ├── data.bin  (二进制原始数据，可 mmap)
# ├── idx.pt    (索引文件)
```

**格式选择指南**：

| 格式 | 读取速度 | 随机访问 | 混洗支持 | 推荐场景 |
|------|---------|---------|---------|---------|
| 原始小文件 | ★ | ✅ | ✅ | 小数据集（<1K 文件） |
| WebDataset | ★★★★ | ❌ 适合顺序 | ✅ 打乱 shard | 大规模图像/视频 |
| MDS (Mosaic) | ★★★★ | ✅ | ✅ | 推荐通用格式 |
| LMDB | ★★★ | ✅ | ❌ | 键值对随机访问 |
| RecordIO | ★★★★ | ✅ | ❌ | MXNet 场景 |
| 内存映射 (memmap) | ★★★★★ | ✅ | ✅ | 固定大小数据 |

### 4.2 数据排布优化

```python
# 数据排布对性能的影响（NPU 场景）

# 优化前：行优先 (C x H x W, channel-last)
# 优化后：Channel-last (H x W x C) — PyTorch 默认格式已优化

# NPU 上更优的数据排布：
# 1. 尽量使用连续内存（.contiguous()）
# 2. 使用 NCHW 格式（Ascend 原生优化）
# 3. 对齐到 16/32 字节边界

def optimize_data_layout(tensor):
    """优化 NPU 上的数据排布"""
    # Ascend NPU 对 NCHW 格式有硬件加速
    if tensor.dim() == 4:  # NCHW
        # 确保内存连续且在 NPU 上
        return tensor.npu().contiguous()
    return tensor
```

### 4.3 存储分级

| 存储层级 | 延迟 | 带宽 | 容量 | 成本 | 适用数据 |
|----------|------|------|------|------|---------|
| CPU RAM | < 100ns | > 50 GB/s | ~1TB | 高 | 缓存当前 epoch 数据 |
| SSD (NVMe) | < 100μs | > 3 GB/s | ~10TB | 中 | 活跃数据集 |
| HDD | < 10ms | ~200 MB/s | ~100TB | 低 | 归档/冷数据 |
| 网络存储 (NFS) | 1-10ms | 1-10 Gb/s | 无限 | 中 | 共享数据集 |

**推荐配置**：
- 活跃数据集 → SSD（NVMe 优先）
- 当前 epoch 数据 → 预加载到内存（RAM Cache）
- 预处理后数据 → SSD 持久化

### 4.4 ⚠️ 检查点：存储优化确认

- [ ] 数据格式已优化（小文件打包 / 使用高效格式）
- [ ] 数据存储在 SSD 上（非 HDD / 远程网络）
- [ ] 数据排布适合 NPU 处理（NCHW 连续）

---

## Phase 5: 端到端验证

### 5.1 端到端对比

```bash
# 优化前
python train.py --max-steps=200 --dataloader-workers=0 --no-cache

# 优化后
python train.py --max-steps=200 \
    --dataloader-workers=8 \
    --prefetch-factor=4 \
    --pin-memory \
    --persistent-workers \
    --data-cache=memory
```

### 5.2 最终对比报告

```markdown
## 数据流水线优化报告

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 数据加载延迟 (avg) | 45.2 ms | 8.5 ms | **81.2%** |
| 数据加载延迟 (P99) | 128.6 ms | 15.2 ms | **88.2%** |
| NPU 利用率 | 52% | 91% | **75%** |
| 训练吞吐 (samples/s) | 185 | 320 | **73.0%** |
| 单步耗时 | 175 ms | 105 ms | **40.0%** |

**配置变更**:
- num_workers: 0 → 8
- prefetch_factor: 2 → 4
- pin_memory: False → True
- persistent_workers: False → True
- 数据格式: 原始 JPEG → MDS 打包
- 数据存储: HDD → NVMe SSD
```

### 5.3 ⚠️ 检查点：端到端确认

- [ ] 数据加载延迟降低 ≥ 50%
- [ ] NPU 利用率 ≥ 85%
- [ ] 整体吞吐提升 ≥ 20%
- [ ] 数据质量和分布与优化前一致

---

## Phase 6: 迭代决策

### 6.1 效果评判

| 等级 | 条件 | 决策 |
|------|------|------|
| 🟢 达标 | NPU 利用率 ≥ 85% + 加载延迟 < 计算时间 50% | 输出最终配置 |
| 🟡 可接受 | NPU 利用率 ≥ 70% + 加载延迟 < 计算时间 80% | 记录现状，条件允许时再优化 |
| 🔴 不达标 | NPU 利用率 < 70% | 返回 Phase 2 或升级存储硬件 |

### 6.2 最终交付物

| 交付物 | 路径 |
|--------|------|
| DataLoader 配置 | `docs/perf/final/dataloader_config.yaml` |
| 数据流水线报告 | `docs/perf/final/data_pipeline_report.md` |
| 预处理配置 | `docs/perf/final/transform_config.py` |

### 6.3 ⚠️ 检查点：迭代决策确认

- [ ] 数据流水线不再是训练瓶颈
- [ ] NPU 利用率达标
- [ ] 交付物已归档

---

## 异常处理

| 异常场景 | 可能原因 | 排查步骤 | 处理方式 |
|----------|----------|----------|----------|
| num_workers > 0 时训练更慢 | CPU 核心争抢 / 数据太小 | 降低 num_workers = 0/1/2 对比 | 减少 worker 数 |
| pin_memory=True 出错 | 数据对象不支持 pin | 检查 batch 中的数据类型 | 对非 tensor 数据不使用 pin_memory |
| 缓存后数据分布不一致 | 缓存了预处理但未注意随机增强 | 检查缓存逻辑：只缓存 base 特征 | 随机增强在缓存后在线做 |
| 内存不足 (CPU RAM) | prefetch_factor 太大 / 缓存大量数据 | 查看系统内存使用 | 降低 prefetch；减少缓存 |
| WebDataset 打乱不足 | 只打乱 shard 级别，内部未打乱 | 检查 shuffle 配置 | 设置 shard 内 shuffle buffer |
| 跨 epoch 数据重复 | persistent_workers 导致迭代器未重置 | 检查 DataLoader 行为 | 确保 dataloader 重新创建或 reset |

---

## 参考资源

### 关联 Skill

| Skill | 关系 |
|-------|------|
| [msprof-optimizer](../msprof-optimizer/SKILL.md) | **前置** – Profiling 识别数据加载是否为瓶颈 |
| [memory-optimizer](../memory-optimizer/SKILL.md) | **互补** – 预取数据占用显存，需整体平衡 |
| [distributed-optimizer](../distributed-optimizer/SKILL.md) | **互补** – 分布式场景下数据加载需多卡协同 |

### 数据流水线优化验证清单

- [ ] Phase 1: 数据流水线 Profiling 完成，瓶颈确认
- [ ] Phase 2: DataLoader 参数调优（num_workers / prefetch 等）
- [ ] Phase 3: 预处理优化实施（缓存 / 卸载）
- [ ] Phase 4: 存储优化完成（SSD / 高效格式）
- [ ] Phase 5: 端到端验证 NPU 利用率 ≥ 85%
- [ ] Phase 6: 交付物已归档
- [ ] 数据加载延迟 < 单步计算时间 50%
- [ ] 数据质量和分布不变
