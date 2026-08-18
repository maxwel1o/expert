---
name: ascend-vllm-env-config
description: >
  vLLM-Ascend 部署环境变量完整配置与速查 Skill。
  按场景（单卡推理/多卡并行/高并发服务）分类整理所有关键环境变量，
  包含作用说明、推荐值、不设置时的默认行为，以及常见报错对应的环境变量修复方案。
  当用户提到 vLLM 环境变量配置、NPU 环境设置、OOM、日志权限、
  模型下载慢等问题时触发。
metadata:
  short-description: vLLM-Ascend 环境变量完整配置速查
  category: NPU-Model-Deploy
  tags: [ascend, npu, vllm, environment, configuration, deployment, troubleshooting]
---

# vLLM-Ascend 环境变量完整配置速查 Skill

本 Skill 提供 vLLM-Ascend 部署时涉及的所有关键环境变量的系统化整理。
以 `google/gemma-3-270m-it` 和 `SpatialLM-Llama-1B-LLM`
在 Atlas 800 A2 (NPU 910B4) 上的部署经验为参考。

## 使用方式

根据部署场景选择对应章节，复制环境变量块到启动脚本中。
每个变量表格包含：作用、推荐值、默认值、是否必须。

## 场景分类

| 场景 | 说明 | 章节 |
|------|------|------|
| 单卡推理 | 1 张 NPU，低并发，开发调试 | [单卡推理推荐配置](#单卡推理推荐配置) |
| 多卡并行 | Tensor Parallel / Data Parallel | [多卡并行推荐配置](#多卡并行推荐配置) |
| 高并发服务 | 生产环境，高 QPS | [高并发服务推荐配置](#高并发服务推荐配置) |

---

## 通用必设变量

以下变量在所有场景下均建议设置：

### 模型下载

| 变量 | 作用 | 推荐值 | 必须 |
|------|------|--------|------|
| `VLLM_USE_MODELSCOPE` | 使用 ModelScope 代替 HuggingFace 下载（国内推荐） | `true` | 否 |
| `HF_ENDPOINT` | HuggingFace 镜像 endpoint | `https://hf-mirror.com` | 否 |

### NPU 内存

| 变量 | 作用 | 推荐值 | 必须 |
|------|------|--------|------|
| `PYTORCH_NPU_ALLOC_CONF` | NPU 内存分配策略 | `expandable_segments:True` | 强烈建议 |

### 日志与调试

| 变量 | 作用 | 推荐值 | 必须 |
|------|------|--------|------|
| `ASCEND_LOG_PATH` | CANN 日志输出目录（解决默认目录无写权限问题） | `/tmp/ascend/log` 或自定义 | 建议 |

---

## 单卡推理推荐配置

适用于开发调试、模型验证、轻量级推理。

```bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LOG_PATH=/tmp/ascend/log
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
```

| 变量 | 作用 | 推荐值 | 说明 |
|------|------|--------|------|
| `OMP_PROC_BIND=false` | 禁用 OpenMP 线程绑定到特定 CPU 核心 | `false` | 避免与 vLLM 内部调度冲突 |
| `OMP_NUM_THREADS=1` | 每个进程 OpenMP 线程数 | `1` | 减少 CPU 资源争抢 |
| `TASK_QUEUE_ENABLE=1` | 启用 TaskQueue 优化 NPU 任务下发 | `1` | 提升推理吞吐 |

---

## 多卡并行推荐配置

适用于 TP (Tensor Parallel) 或 DP (Data Parallel) 部署。

```bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LOG_PATH=/tmp/ascend/log
export HCCL_BUFFSIZE=512
export HCCL_ALGO="level0:NA;level1:H-D_R"  # 可选，特定场景调优
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
```

| 变量 | 作用 | 推荐值 | 说明 |
|------|------|--------|------|
| `HCCL_BUFFSIZE` | HCCL 通信缓冲区大小（MB） | `512` | 多卡通信带宽调优 |
| `HCCL_ALGO` | HCCL 通信算法选择 | 视网络拓扑 | 通常默认即可，大集群可手动调优 |

---

## 高并发服务推荐配置

适用于生产环境部署，追求最大吞吐和稳定性。

```bash
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LOG_PATH=/tmp/ascend/log
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

| 变量 | 作用 | 推荐值 | 说明 |
|------|------|--------|------|
| `VLLM_WORKER_MULTIPROC_METHOD` | vLLM worker 进程启动方式 | `spawn` | 避免 fork 带来的 NPU 上下文问题 |

---

## 完整变量速查表

### NPU 运行时

| 变量 | 作用 | 推荐值 | 默认值 | 必须 |
|------|------|--------|--------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | 指定可见 NPU 设备 | `0,1,2,3` | 全部 | 否 |
| `ASCEND_GLOBAL_LOG_LEVEL` | CANN 全局日志级别 | `3` (ERROR) | `0` (DEBUG) | 否 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | CANN 日志输出到 stdout | `0` | `0` | 否 |
| `ASCEND_LOG_PATH` | CANN 日志目录 | `/tmp/ascend/log` | `~/ascend/log` | 建议 |

### PyTorch NPU

| 变量 | 作用 | 推荐值 | 默认值 | 必须 |
|------|------|--------|--------|------|
| `PYTORCH_NPU_ALLOC_CONF` | 内存分配策略 | `expandable_segments:True` | 无 | 强烈建议 |
| `TASK_QUEUE_ENABLE` | TaskQueue 优化 | `1` | `0` | 建议 |

### vLLM

| 变量 | 作用 | 推荐值 | 默认值 | 必须 |
|------|------|--------|--------|------|
| `VLLM_USE_MODELSCOPE` | 使用 ModelScope | `true` | `false` | 国内建议 |
| `VLLM_WORKER_MULTIPROC_METHOD` | Worker 启动方式 | `spawn` | `fork` | 建议 |
| `VLLM_LOGGING_LEVEL` | vLLM 日志级别 | `INFO` | `INFO` | 否 |

### OpenMP / CPU

| 变量 | 作用 | 推荐值 | 默认值 | 必须 |
|------|------|--------|--------|------|
| `OMP_PROC_BIND` | 线程绑定策略 | `false` | 系统决定 | 建议 |
| `OMP_NUM_THREADS` | OpenMP 线程数 | `1` | CPU 核心数 | 建议 |
| `OMP_WAIT_POLICY` | 空闲线程策略 | `PASSIVE` | `ACTIVE` | 可选 |

### HCCL (多卡通信)

| 变量 | 作用 | 推荐值 | 默认值 | 必须 |
|------|------|--------|--------|------|
| `HCCL_BUFFSIZE` | 通信缓冲区大小 | `512` | `120` | 多卡时建议 |
| `HCCL_ALGO` | 通信算法 | 视场景 | 自动 | 可选 |
| `HCCL_CONNECT_TIMEOUT` | 连接超时（秒） | `300` | `120` | 大集群建议 |

---

## 常见报错与环境变量修复

| 报错信息 | 原因 | 修复变量 |
|---------|------|---------|
| `can not create directory, directory: /home/xxx/ascend/log` | 默认日志目录无写权限 | `export ASCEND_LOG_PATH=/tmp/ascend/log` |
| `Out of memory` / `OOM` | NPU HBM 不足 | `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| 模型下载失败 / 连接超时 | HuggingFace 被墙或慢 | `export VLLM_USE_MODELSCOPE=true` |
| Worker 进程启动失败 | fork 方式导致 NPU 上下文错误 | `export VLLM_WORKER_MULTIPROC_METHOD=spawn` |
| 多卡通信超时 | HCCL 缓冲区不足 | `export HCCL_BUFFSIZE=512` |
| CPU 占用过高 | OpenMP 线程过多 | `export OMP_NUM_THREADS=1` |
| 日志刷屏 | CANN 默认 DEBUG 级别 | `export ASCEND_GLOBAL_LOG_LEVEL=3` |

---

## 一键检查脚本

使用本 Skill 提供的脚本快速检查环境：

```bash
python scripts/env_check.py
```

输出示例：

```
=== vLLM-Ascend Environment Check ===
[OK]   VLLM_USE_MODELSCOPE = true
[WARN] PYTORCH_NPU_ALLOC_CONF not set (recommend: expandable_segments:True)
[OK]   ASCEND_LOG_PATH = /tmp/ascend/log
[OK]   OMP_NUM_THREADS = 1
[INFO] HCCL_BUFFSIZE not set (only needed for multi-card)
[INFO] npu-smi: 1 NPU(s) available, Health: OK
```

---

## 环境变量设置模板

### 启动脚本模板

```bash
#!/bin/bash
# vLLM-Ascend environment setup

export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LOG_PATH=/tmp/ascend/log
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

# Optional: multi-card only
# export HCCL_BUFFSIZE=512

# Run vLLM
vllm serve /path/to/model \
  --tensor-parallel-size 1 \
  --served-model-name my-model
```

---

## 参考

- CANN 环境变量文档：<https://www.hiascend.com/document/detail/en/CANNCommunityEdition/80RC1alpha003/devguide/appdevg/aclpythondevg/aclpythondevg_0000.html>
- 本 Skill 脚本：`scripts/env_check.py`
