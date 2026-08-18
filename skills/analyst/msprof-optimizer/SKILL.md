---
name: msprof-optimizer
description: 利用 msprof（函数级）和 msprof op（算子级）等 profiling 工具对昇腾 NPU 上的模型进行系统性性能分析、瓶颈定位与针对性调优。覆盖从全模型 Profiling、热点算子识别、算子级深度分析到优化迭代验证的完整闭环。
keywords:
  - msprof
  - msprof-op
  - profiling
  - performance-optimization
  - ascend
  - npu
  - bottleneck-analysis
  - operator-tuning
  - model-profiling
  - performance-tuning
---

# msprof 模型级 Profiling 与针对性调优 Skill

本 Skill 提供在昇腾 NPU 上利用 **msprof**（函数级）和 **msprof op**（算子级）工具进行模型级性能分析与针对性调优的标准化流程。覆盖从全模型 Profiling 基线采集、热点算子识别、算子级深度分析到优化迭代验证的完整闭环。

适用阶段：模型已在 NPU 上正常跑通（推理或训练），需要系统性分析性能瓶颈并进行针对性优化。

---

## 重要默认行为

1. **先确认可跑通**：本 Skill 假设模型已在 NPU 上正常运行。如果模型尚未迁移完成，请先完成迁移后再进入调优。

2. **分层递进**：严格遵循 **全模型 Profiling → 热点识别 → 算子级分析 → 修改 → 验证** 的迭代闭环。禁止跳过全模型 Profiling 直接进入算子级调优。

3. **一次只改一个点**：每次优化仅调整一个参数或一处代码，修改后重新采集验证，确保收益可追溯。

4. **基线先行**：每次优化前必须采集基线数据，优化后对比基线量化收益。基线数据需持久归档。

5. **异常即止，追问用户**：遇到环境问题、采集失败、数据异常等情况，先排障再继续，不盲目推进。

---

## 前置条件

| 检查项 | 要求 | 验证命令 |
|--------|------|---------|
| 昇腾 NPU 硬件 | Ascend 910/910B 系列（≥1 卡） | `npu-smi info` |
| CANN 工具包 | ≥ 7.0（推荐 8.0+） | `msprof --version` 或 `which msprof` |
| msprof 工具 | 已安装且可执行 | `msprof op --help` 应正常输出 |
| Python | 3.8 – 3.10 | `python3 --version` |
| 模型状态 | 能在 NPU 上正常跑通推理/训练 | 用户确认 |
| 脚本路径 | 记录模型启动脚本的完整路径 | 用户提供/确认 |

---

## 工作流总览

```
Phase 1: 全模型 Profiling 基线采集
    ├── 输入: 模型启动脚本/命令
    ├── 输出: msprof 函数级 Profiling 报告
    └── 产出: TIMELINE_${MODEL_NAME}/ 目录 + baseline_summary.csv

Phase 2: 热点分析与瓶颈定位
    ├── 输入: 基线 Profiling 数据
    ├── 输出: 热点算子 TOP-N 列表 + 瓶颈类型判定
    └── 判定: Vector-Bound / Cube-Bound / Memory-Bound / Host-Bound

Phase 3: 热点算子算子级采集（msprof op）
    ├── 输入: 热点算子清单
    ├── 输出: msprof op 算子级 CSV 数据 + PipeUtilization 分析
    └── 产出: OPPROF_*/ + 归档到 docs/perf/round_NNN/

Phase 4: 针对性优化
    ├── 输入: 瓶颈类型 + 算子级 CSV 分析结论
    ├── 输出: 修改后的算子/配置
    └── 策略: 参考瓶颈优化速查表

Phase 5: 优化效果验证
    ├── 输入: 修改后的算子/配置
    ├── 输出: 新一轮 Profiling 数据 + 对比报告
    └── 判定: 优化是否达标

Phase 6: 迭代决策
    ├── 达标 → 输出最终优化报告
    └── 不达标 → 返回 Phase 2 继续分析
```

---

## Phase 1: 全模型 Profiling 基线采集

### 1.1 确认 msprof 可用

```bash
# 检查 msprof 版本
msprof --version

# 确认 msprof 可执行
which msprof
```

### 1.2 执行全模型函数级 Profiling

**基本用法**：
```bash
# 采集整个模型运行过程的性能数据
msprof --application="python train.py" --output=./TIMELINE_BASELINE
```

**推荐用法（含预热 + 指定迭代）**：
```bash
msprof \
    --application="python train.py --epochs 1" \
    --output=./TIMELINE_${MODEL_NAME} \
    --trace-level=1 \
    --aic-metrics=Default \
    --profile-iterations=50 \
    --warmup-iterations=10
```

**关键参数说明**：

| 参数 | 说明 | 推荐值 | 使用场景 |
|------|------|--------|---------|
| `--application` | **必需**，指定要分析的应用程序命令 | — | 始终需要 |
| `--output` | 输出目录 | `./TIMELINE_${MODEL_NAME}` | 始终建议指定 |
| `--trace-level` | 采集级别：0=仅时间，1=含 AI Core 指标 | `1` | 首次建议 1，分析瓶颈时保持 1 |
| `--aic-metrics` | AI Core 性能指标采集 | `Default` | 常规场景默认即可 |
| `--profile-iterations` | 性能采集迭代次数 | `50` | 训练场景取 50+ 步 |
| `--warmup-iterations` | 预热迭代次数 | `5-10` | 始终建议，避免 DVFS 影响 |

### 1.3 采集输出文件

```
TIMELINE_${MODEL_NAME}/
├── summary.csv              # 性能摘要（整体耗时、算子统计）
├── op_summary.csv           # 算子级性能统计（算名、耗时、调用次数）
├── op_detail.csv            # 算子详细性能数据（各次调用详情）
├── timeline_trace.json      # 时间线数据（Chrome trace 可视化）
├── report.html              # 可视化报告
└── msprof.log               # 采集日志
```

### 1.4 ⚠️ 检查点：验证数据有效性

请用户确认以下事项：
- [ ] 采集完成，TIMELINE 目录已生成
- [ ] `msprof.log` 中无 ERROR 级别日志
- [ ] `op_summary.csv` 和 `summary.csv` 非空

> **遇到问题**：如果采集失败或数据异常 → 暂停，进入「异常处理」附表排查。

---

## Phase 2: 热点分析与瓶颈定位

### 2.1 读取 Profiling 数据

```python
import pandas as pd

# 读取算子汇总
df = pd.read_csv('./TIMELINE_${MODEL_NAME}/op_summary.csv')

# 按总耗时降序排列，取 TOP-N
top_ops = df.nlargest(10, 'total_time_us')
print("耗时最长的 10 个算子:")
print(top_ops[['op_name', 'total_time_us', 'call_count', 'avg_time_us']])
```

输出示例：
```
       op_name  total_time_us  call_count  avg_time_us
0  aten::matmul     1852034.2        1024       1808.6
1  aten::softmax    1230456.8         512       2403.2
2  aten::layer_norm  987654.3         256       3858.0
...
```

### 2.2 瓶颈类型宏观判定

从 `summary.csv` 和 `timeline_trace.json` 分析整体瓶颈：

| 信号 | 瓶颈类型 | 典型表现 |
|------|---------|---------|
| 单个算子耗时占比 > 30% | 算子热点 | 某个算子的 `total_time_us` 远高于其他 |
| Host 侧耗时占比高 | Host-Bound | timeline 中大量 Host 间隙 |
| 多个小算子连续出现 | 算子碎片 | 小算子数量多但每个耗时低 |

### 2.3 输出热点算子 TOP-N 清单

生成热点列表，格式如下：

| 排名 | 算子名 | 总耗时(us) | 调用次数 | 平均耗时(us) | 占比 |
|------|--------|-----------|---------|-------------|------|
| 1 | aten::matmul | 1852034 | 1024 | 1808.6 | 28.5% |
| 2 | aten::softmax | 1230457 | 512 | 2403.2 | 18.9% |
| 3 | aten::layer_norm | 987654 | 256 | 3858.0 | 15.2% |

**重点关注**：
- 总耗时占比 > 5% 的算子
- 单个调用平均耗时 > 500us 的算子
- 调用次数极多（>1000次）的算子

### 2.4 ⚠️ 检查点：确认热点目标

展示热点算子列表给用户，确认优化目标：

- [ ] 向用户展示 TOP-5 热点算子，解释每个算子的耗时占比
- [ ] 请用户确认优先优化的算子（默认从耗时第 1 的开始）
- [ ] 记录用户确认的优化目标到本轮笔记

---

## Phase 3: 热点算子算子级采集（msprof op）

### 3.1 准备算子级测试脚本

根据热点算子类型，编写或使用已有的算子级测试脚本。如果是 PyTorch 原生算子，使用 `torch_npu.profiler` 采集；如果是自研/自定义算子或 AscendC 算子，使用 `msprof op` 直接采集。

**方式 A：针对 PyTorch 原生算子**

```python
# benchmark_op.py - 热点算子独立性能测试
import torch
import torch_npu

def benchmark_matmul(M, N, K, dtype=torch.float16, device='npu'):
    # 创建测试数据
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    
    # 预热
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.npu.synchronize()
    
    # 正式采集
    for _ in range(50):
        c = torch.matmul(a, b)
    torch.npu.synchronize()

if __name__ == '__main__':
    benchmark_matmul(4096, 4096, 4096)
```

**方式 B：针对 AscendC 自定义算子**

```bash
# 使用 msprof op 直接采集自定义算子
msprof op \
    --warm-up=10 \
    --launch-count=5 \
    --output=./msprof_op_output \
    ./demo
```

**关键参数说明**：

| 参数 | 说明 | 推荐值 | 使用场景 |
|------|------|--------|---------|
| `--warm-up=N` | 预热 N 次后再采集 | `10` | **始终建议**，避免 DVFS 影响首次运行 |
| `--launch-count=N` | 运行 N 次取均值 | `5` | 需要统计稳定性时 |
| `--output=<dir>` | 指定输出目录 | `./msprof_op_output` | 避免结果散落 |
| `--kernel-name` | 指定 kernel 名 | 算子函数名 | 筛选特定 kernel |
| `--ai-core=on` | 采集 AI Core 指标 | `on` | 分析计算瓶颈 |
| `--aic-metrics=PipeUtilization` | 采集流水线利用率 | 按需 | 定位流水瓶颈 |

### 3.2 执行算子级采集

```bash
# 直接采集 PyTorch 算子
msprof op --warm-up=10 --launch-count=5 --output=./msprof_op_${OP_NAME} python benchmark_op.py

# 或使用 torch_npu.profiler 采集（更详细的 PyTorch 上下文）
python3 -c "
import torch
import torch_npu

prof = torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=1, warmup=5, active=3, repeat=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./profiler_log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)

a = torch.randn(4096, 4096, device='npu', dtype=torch.float16)
b = torch.randn(4096, 4096, device='npu', dtype=torch.float16)

prof.start()
for _ in range(10):
    c = torch.matmul(a, b)
    prof.step()
prof.stop()
"
```

### 3.3 算子级数据归档

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td ./msprof_op_${OP_NAME}/OPPROF_* 2>/dev/null | head -1)

if [ -z "$OPPROF_DIR" ]; then
    echo "⚠️  未找到 OPPROF 目录，请检查 msprof op 输出路径"
    exit 1
fi

# 归档到算子专属目录（自动创建递增轮次）
python3 ${SKILL_ROOT}/scripts/perf_summary.py $OPPROF_DIR ops/${OP_NAME}
```

### 3.4 读取分析算子级 CSV

```bash
# 查看算子基本信息（名称、核数、总耗时）
cat OPPROF_*/OpBasicInfo.csv

# 查看流水线利用率（判断瓶颈单元）
cat OPPROF_*/PipeUtilization.csv

# 查看内存带宽利用率
cat OPPROF_*/Memory.csv

# 查看资源冲突
cat OPPROF_*/ResourceConflictRatio.csv
```

### 3.5 ⚠️ 检查点：确认算子级数据完整性

- [ ] `OpBasicInfo.csv` 能读到算子名称、Block Dim、Task Duration
- [ ] `PipeUtilization.csv` 各流水单元数据完整
- [ ] `Memory.csv` 带宽数据正常
- [ ] 确认当前 NPU 频率正常（Current Freq ≈ Rated Freq）

---

## Phase 4: 针对性优化

### 4.1 瓶颈类型判定

读取 `PipeUtilization.csv` 中各流水单元的 `ratio`，对照下表判定瓶颈：

| 瓶颈类型 | 判定条件 | 典型 ratio 分布 |
|---------|---------|----------------|
| **Vector-Bound** | `aiv_vec_ratio` 最高 | VEC > 50%，MTE2 < 30% |
| **Cube-Bound** | `aic_cube_ratio` 最高 | CUBE > 50%，VEC/MTE2 < 20% |
| **MTE2-Bound** | `ai*_mte2_ratio` 最高 | MTE2 > 50%，VEC < 30% |
| **Scalar-Bound** | `ai*_scalar_ratio` > 30% | SCALAR > 30% 异常高 |
| **核间不均衡** | 各核 `ai*_time(us)` 差异 > 10% | 各核耗时明显不同 |
| **Bank-Conflict** | `vec_bank_cflt_ratio` > 5% | UB bank conflict 高 |
| **DoubleBuffer 未生效** | MTE2/VEC 重叠 < 5% | 流水线串行 |

### 4.2 优化策略速查表

| 瓶颈类型 | 优化方向 | 具体方法 |
|---------|---------|---------|
| **Vector-Bound** | 减少矢量计算量 | UB 融合、减少 Cast 指令、使用融合指令（如 `fused_add_relu`）、算符融合（如 FuseLayerNorm） |
| **Cube-Bound** | 提高 Cube 利用率 | 调整分块大小（TileShape）、使能 L0C 累加、增加 L1 数据复用、调整 Swizzle 策略 |
| **MTE2-Bound** | 优化数据搬运 | 增大搬运粒度（≥16KB）、确保 512B 对齐、使能 L2 Cache Mode、减少核间同地址访问 |
| **Scalar-Bound** | 减少标量开销 | 缩小 TilingData 大小、减少核数（降低核间同步开销） |
| **核间不均衡** | 优化任务切分 | 调整 Tiling 切分策略、核数设为 2 的幂次 |
| **Bank-Conflict** | 优化 UB 布局 | 调整 UB 地址偏移、添加 padding、调整数据块大小 |
| **算子碎片** | 算子融合 | 将多个小算子融合为单算子、使用 torch_npu.contrib 融合 API |
| **Host-Bound** | 减少 Host 开销 | 使能 `TASK_QUEUE_ENABLE=2`、`export COMBINED_ENABLE=1`、使用 `torch_npu.npu_format_code` |
| **头开销大** | 减少调度开销 | 减少 Block Dim 核数、缩小 TilingData、TPipe 外置 |

### 4.3 实施优化

**按照判定的瓶颈类型，从优化策略速查表中选择对应的优化方法。**

记录每次修改：

```
[优化笔记]
- 日期: YYYY-MM-DD
- 目标算子: aten::matmul
- 瓶颈类型: Cube-Bound
- 优化方法: 调整 MatMul TileShape 从 128x128 到 256x128
- 修改文件: matmul_tiling.h 第 45 行
- 修改前值: TILE_M=128, TILE_N=128
- 修改后值: TILE_M=256, TILE_N=128
```

### 4.4 ⚠️ 检查点：优化前确认

在执行修改之前，请用户确认：
- [ ] 瓶颈类型分析是否正确，是否与用户感知一致
- [ ] 优化方案是否合理，是否有其他优先方案
- [ ] 是否已保存优化笔记（便于回滚）

---

## Phase 5: 优化效果验证

### 5.1 重新采集算子级数据

修改代码或配置后，重复 Phase 3 的采集步骤：

```bash
# 记录优化前 baseline
cp -r ./msprof_op_${OP_NAME} ./msprof_op_${OP_NAME}_PRE

# 修改代码/配置后，重新采集
msprof op --warm-up=10 --launch-count=5 --output=./msprof_op_${OP_NAME}_POST python benchmark_op.py
```

### 5.2 全模型 Profiling 验证（最终确认）

避免"只见树木不见森林"，算子级优化后必须做一次全模型 Profiling 确认整体收益：

```bash
msprof \
    --application="python train.py" \
    --output=./TIMELINE_${MODEL_NAME}_POST \
    --trace-level=1 \
    --profile-iterations=50 \
    --warmup-iterations=10
```

### 5.3 对比分析指标

| 指标 | 优化前 | 优化后 | 变化率 | 说明 |
|------|-------|-------|-------|------|
| 热点算子耗时 (us) | 基线值 | 新值 | Δ% | 算子级提升 |
| 模型单步耗时 (us) | 基线值 | 新值 | Δ% | 整体收益 |
| 端到端吞吐 (samples/s) | 基线值 | 新值 | Δ% | 业务指标 |
| 流水线重叠率 | 基线值 | 新值 | — | DoubleBuffer 效果 |

### 5.4 结果判定

| 结果 | 操作 |
|------|------|
| 整体耗时下降 ≥ 5% | ✅ 验证通过，进入 Phase 6 |
| 整体耗时变化 < 5% | ⚠️ 收益有限，和用户确认是否接受 |
| 整体耗时不变或上升 | ❌ 回滚修改，重新分析瓶颈 |

### 5.5 ⚠️ 检查点：确认优化效果

- [ ] 算子级 profiling 数据显示性能提升
- [ ] 全模型 profiling 确认整体正向收益
- [ ] 收益已量化（Δ% 或绝对值）
- [ ] 数据已持久归档

---

## Phase 6: 迭代决策

- **当前算子优化效果显著**（≥ 5%）→ 询问用户是否继续优化下一个热点算子
- **当前算子优化效果有限**（< 5%）→ 询问用户是否接受当前结果，或换一个算子
- **当前算子无收益**（不变或变差）→ 回滚修改，重新分析瓶颈类型
- **所有热点已处理** → 输出最终优化报告

### 最终交付物

1. **优化报告**：包含优化前后对比表、瓶颈分析、修改记录
2. **归档数据**：所有 Profiling 数据按 `docs/perf/round_NNN/` 格式归档
3. **优化笔记**：记录所有尝试的配置和效果（含失败记录）

---

## 异常处理

| 异常场景 | 可能原因 | 排查步骤 | 处理方式 |
|---------|---------|---------|---------|
| `msprof: command not found` | CANN 未安装或环境变量未配置 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` | 重新 source 环境变量 |
| `msprof: cannot connect to driver` | NPU 驱动异常 | `npu-smi info` 检查 NPU 状态 | 重启驱动或硬件 |
| `OPPROF_*` 目录为空或缺失 CSV | 采集过程异常 | 检查 `msprof.log` 中 ERROR 信息 | 减少 `--launch-count` 或关闭 `--ai-core` |
| `OpBasicInfo.csv` 中 Current Freq << Rated Freq | NPU 未满频 | 检查芯片温度、功耗策略 | 确保 1 卡独占，关闭降频策略 |
| 算子级性能波动 > 10% | 系统负载不稳定 | 增加 `--warm-up` 值 | 设置 `--warm-up=30` |
| Profiling 数据过大导致磁盘满 | 采集轮次过多 | `du -sh *TIMELINE*` 检查 | 归档旧数据到远端存储 |
| 算子融合后精度不达标 | 融合实现有误 | 精度对比验证 | 回退融合，改用其他优化方式 |
| 用户脚本无法被 msprof 直接启动 | 脚本依赖复杂环境 | `msprof --application=...` 可能需要绝对路径 | 提供完整命令路径和参数 |

---

## 参考资源

### 文件清单

| 资源 | 类型 | 说明 | 使用时机 |
|------|------|------|---------|
| `references/msprof-function-level.md` | 参考文档 | msprof 函数级 Profiling 完整使用说明 | Phase 1 全模型 Profiling |
| `references/msprof-op-level.md` | 参考文档 | msprof op 算子级 Profiling 完整参数说明 | Phase 3 算子级采集 |
| `references/csv_fields_reference.md` | 参考文档 | 8 个 CSV 输出文件的完整字段定义和阈值 | Phase 3-4 数据分析 |
| `references/performance-data-analysis.md` | 参考文档 | 性能数据分析详解与 Python 分析脚本 | Phase 4 瓶颈定位 |
| `references/optimization_quickref.md` | 参考文档 | 各瓶颈类型的具体优化方法和案例 | Phase 4 优化实施 |
| `scripts/perf_summary.py` | 脚本 | CSV 归档 + 统计摘要生成 | Phase 3 数据归档 |
| `ai4s-perf-tuning` | 关联 Skill | 昇腾 NPU 通用性能调优（流水/绑核/tcmalloc/编译优化） | 补充优化手段 |
| `ops-profiling` | 关联 Skill | 算子级上板性能采集与分析 | Phase 3-4 算子级详细分析 |

### 关联 Skill 说明

| Skill | 关系 | 何时参考 |
|-------|------|---------|
| `ai4s-perf-tuning` (../ai4s-perf-tuning/SKILL.md) | **并行/互补**：本 Skill 专注 msprof Profiling 驱动的瓶颈定位+针对性优化；ai4s-perf-tuning 覆盖调度/OS/编译级通用优化 | 发现 Host-Bound 或希望做编译优化时参考 |
| `ops-profiling` (../../ascend/optimization/ops-profiling/SKILL.md) | **子集/细化**：提供更详细的算子级 CSV 指标解读和性能标准判定 | Phase 3-4 需要深度算子分析时参考 |
| `catlass-operator-performance-optim` (../catlass-operator-performance-optim/SKILL.md) | **算子类型特化**：针对 Catlass 算子的 Tiling 调优 | 热点算子为 Catlass 算子时参考 |

---

## 优化案例参考

### 案例 1：MatMul 算子 Cube-Bound 优化

**现象**：Profiling 显示 `aten::matmul` 占总耗时 35%，算子级 `ai_cube_ratio` = 65%

**分析**：Cube 利用率不足，Tiling 参数未适配硬件

**优化**：调整分块策略，使能 L1 数据复用

**结果**：MatMul 单算子耗时下降 22%，模型整体吞吐提升 15%

### 案例 2：Softmax + LayerNorm 算子碎片优化

**现象**：Profiling 显示多个小算子（softmax、layernorm、add）连续出现，总占比 25%

**分析**：算子碎片导致核启动开销占比高

**优化**：使用 `torch_npu.contrib` 融合 API 或手写融合算子

**结果**：融合后 3 个算子合并为 1 个，耗时下降 40%，模型整体吞吐提升 8%

### 案例 3：Host-Bound 优化

**现象**：timeline 显示大量 Host 间隙，NPU 利用率不足 50%

**分析**：CPU 侧调度成为瓶颈

**优化**：使能 `TASK_QUEUE_ENABLE=2` + `COMBINED_ENABLE=1`

**结果**：Host 间隙减少 60%，NPU 利用率提升至 80%，模型吞吐提升 20%

---

## 质量验证清单

- [ ] Phase 1: 全模型 Profiling 基线已采集并归档
- [ ] Phase 2: 热点算子 TOP-N 已列出，用户已确认优化目标
- [ ] Phase 3: 热点算子算子级数据已采集并归档
- [ ] Phase 3: `PipeUtilization.csv` 已读取，瓶颈类型已判定
- [ ] Phase 4: 优化方案已向用户展示并获确认
- [ ] Phase 4: 每次修改记录在优化笔记中
- [ ] Phase 5: 优化后算子级数据已重新采集
- [ ] Phase 5: 全模型 Profiling 确认整体收益
- [ ] Phase 5: 优化前后对比表已生成并展示
- [ ] Phase 6: 用户已确认是否继续迭代或接受结果
- [ ] 所有 Profiling 数据已按 `docs/perf/round_NNN/` 格式归档
- [ ] 失败尝试也已记录（便于后续避免重复踩坑）
