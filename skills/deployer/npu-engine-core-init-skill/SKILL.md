---
name: npu-engine-core-init-skill
description: >
  诊断并修复 vLLM-Ascend EngineCore 子进程 NPU 初始化失败问题。
  当用户遇到 "Engine core initialization failed"、"aclInit error 107001"、
  "Invalid device ID"、NPU 子进程无法启动等报错时自动触发。
  支持一键诊断环境、自动应用修复、验证修复结果。
triggers:
  - Engine core initialization failed
  - aclInit
  - 107001
  - Invalid device ID
  - NPU 子进程
  - SyncMPClient
  - EngineCore 初始化
  - failed core proc
  - vllm serve 启动失败 NPU
  - vllm bench 失败 NPU
---

# npu-engine-core-init-skill

## 触发条件

当用户输入中包含以下任一关键词时自动触发：
- `Engine core initialization failed`
- `aclInit error` / `107001`
- `Invalid device ID`
- `NPU 子进程` / `failed core proc`
- `vllm serve` / `vllm bench` 配合 `失败` / `报错` / `error`

## 核心工作流

```
1. 一键诊断 ──→ 2. 自动修复 ──→ 3. 验证结果
```

### Step 1: 一键诊断

**必须执行**：运行诊断脚本获取完整环境报告。

```bash
python3 ./scripts/diagnose.py
```

诊断脚本会检查：
- NPU 硬件状态 (`npu-smi info`)
- NPU Python API 可用性 (`torch_npu.npu.is_available()`)
- 设备文件权限 (`/dev/davinci*`)
- 当前 multiprocessing 启动方式
- 环境变量 (`ASCEND_RT_VISIBLE_DEVICES`, `VLLM_WORKER_MULTIPROC_METHOD`)
- vLLM 版本信息
- CANN 日志目录权限

### Step 2: 自动修复

根据诊断结果，自动选择并应用最合适的修复方案：

```bash
python3 ./scripts/fix.py
```

fix.py 会按优先级尝试：
1. **方案 A**：设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`
2. **方案 B**：取消不当的 `ASCEND_RT_VISIBLE_DEVICES`
3. **方案 C**：修复 CANN 日志目录权限
4. **方案 D**：生成绕过子进程的替代命令（bench 场景）

### Step 3: 验证结果

修复后，运行验证：

```bash
# 验证 NPU 基础可用性
python3 -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"

# 验证 vLLM 能否正常启动（干跑模式，不加载权重）
# 如果用户有模型路径，用真实路径替换
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /path/to/model \
  --load-format dummy \
  --max-model-len 1024 \
  --trust-remote-code \
  2>&1 | tail -20
```

## 手动修复方案速查

如果自动修复未完全解决问题，按以下优先级手动处理：

### 方案 A：强制 spawn（首选）

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /path/to/model ...
```

### 方案 B：单卡容器移除设备过滤

```bash
unset ASCEND_RT_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /path/to/model ...
```

### 方案 C：bench 场景绕过 SyncMPClient

`vllm bench latency/throughput` 不受 `spawn` 控制时，改用：

```bash
# 1. 先启动服务
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /path/to/model --host 0.0.0.0 --port 8000 ...

# 2. 用 bench serve 测在线性能
vllm bench serve --model <name> --host 127.0.0.1 --port 8000 ...

# 3. 用脚本测单请求延迟（替代 bench latency）
python3 ./scripts/measure_latency.py
```

## 输入参数

Skill 自动从用户上下文和环境中获取参数，无需手动输入。

可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | 用户上下文中的模型路径 | 验证时使用的模型路径 |
| `VLLM_PORT` | `8000` | 验证服务端口 |

## 输出格式

### 诊断报告 (JSON)

```json
{
  "diagnosis": {
    "npu_available": true,
    "npu_count": 1,
    "current_device": 0,
    "davinci_devices": ["/dev/davinci5"],
    "multiprocessing_method": "fork",
    "ascend_rt_visible_devices": "5",
    "vllm_version": "0.18.0+empty",
    "vllm_ascend_version": "0.18.0rc1",
    "torch_npu_version": "2.9.0.post1",
    "cann_log_dir_writable": false
  },
  "root_cause": "fork multiprocessing + ASCEND_RT_VISIBLE_DEVICES conflict",
  "recommended_fix": "A+B",
  "status": "diagnosed"
}
```

### 修复报告

```json
{
  "fixes_applied": [
    {"fix": "A", "action": "set VLLM_WORKER_MULTIPROC_METHOD=spawn", "status": "applied"},
    {"fix": "B", "action": "unset ASCEND_RT_VISIBLE_DEVICES", "status": "applied"},
    {"fix": "C", "action": "mkdir -p /home/user/ascend/log", "status": "applied"}
  ],
  "verification_command": "VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve ...",
  "status": "fixed"
}
```

## 错误处理

- **诊断失败**：记录环境信息，提示用户检查 NPU 驱动安装
- **修复失败**：回滚环境变量修改，输出手动修复指南
- **验证失败**：根据新报错重新诊断，尝试下一优先级方案

## 依赖

```bash
pip install torch torch_npu vllm vllm-ascend
```

系统依赖：
- `npu-smi` (CANN Toolkit)
- `/dev/davinci*` 设备文件

## 参考

- [vLLM-Ascend 官方文档](https://docs.vllm.ai/projects/ascend/)
- [CANN 错误码手册](https://www.hiascend.com/document/)
- PyTorch NPU multiprocessing: `torch.multiprocessing.set_start_method('spawn')`
