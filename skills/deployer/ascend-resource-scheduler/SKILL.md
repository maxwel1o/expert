---
name: ascend-resource-scheduler
description: >
  昇腾 NPU 资源冲突诊断与进程调度专家。当用户在已有 vLLM serve 服务的昇腾 NPU 上执行离线推理、
  benchmark 或精度测试时，自动检测同卡内存占用冲突，决策复用在线服务或无缝停服再测，
  实现 stop → run → restart 的全自动调度。触发场景包括：
  "NPU资源冲突"、"serve占用"、"停服务跑benchmark"、"device内存不足"、
  "Free memory on device is less than desired GPU memory utilization"、
  "同卡再跑测试"、"vLLM进程冲突"。
---

# ascend-resource-scheduler — 昇腾 NPU 资源冲突诊断与进程调度

## 核心工作流程

### 阶段 1：NPU 资源状态诊断

**任务 1.1：检测当前 NPU 设备状态**

```bash
# 查看 NPU 设备整体状态
npu-smi info

# 查看各卡内存占用详情
npu-smi info -t memory

# 查看各卡进程占用
npu-smi info -t processes
```

**任务 1.2：检测 vLLM serve 常驻进程**

```bash
# 查找 vLLM serve 进程
ps aux | grep -E "vllm.*serve|python.*vllm" | grep -v grep

# 查找 vllm serve 监听的端口
ss -tlnp | grep -E "vllm|python"

# 如果存在 vllm serve.pid 文件，读取 PID
cat vllm_serve.pid 2>/dev/null || echo "No pid file found"
```

**任务 1.3：评估内存冲突风险**

根据以下规则判断：

| 场景 | 判断标准 | 风险等级 |
|------|---------|---------|
| 无 vLLM 进程 | npu-smi 无 vllm/python 推理进程 | 🟢 安全 |
| vLLM serve 常驻 + 用户要跑离线测试 | 目标 NPU 已用内存 > 30% | 🔴 冲突 |
| vLLM serve 常驻 + 用户要做在线压测 | 服务端口可连通 | 🟢 可复用 |
| 存在多个 vLLM 进程 | 同一设备上 PID > 1 个 | 🔴 严重冲突 |

### 阶段 2：用户意图识别与决策

**任务 2.1：解析用户当前任务类型**

从用户输入中识别关键词：

| 用户意图关键词 | 任务类型 | 推荐策略 |
|---------------|---------|---------|
| "benchmark"、"性能测试"、"吞吐"、"延迟" | 性能基准测试 | 离线：先停服务再测 |
| "精度测试"、"验证精度"、"compare"、"离线推理" | 离线推理/精度验证 | 离线：先停服务再测 |
| "压测服务"、"在线测试"、"API测试"、"curl" | 在线服务压测 | 在线：复用现有服务 |
| "inference.py"、"跑推理"、"批量推理" | 批量离线推理 | 离线：先停服务再测 |

**任务 2.2：自动决策**

如果用户未明确说明意图，按以下默认策略：
- 检测到 `inference.py`、`benchmark`、`*_test.py` 等离线脚本执行 → **离线策略**
- 检测到 `ab`、`wrk`、`locust`、`vllm bench --url` 等在线压测工具 → **在线策略**
- 用户仅说"测试一下"、"验证一下" → 询问用户："您是要做离线推理测试，还是在线 API 压测？"

### 阶段 3：资源调度执行

#### 策略 A：离线测试 — 先停服务，测试后恢复

**步骤 3A.1：保存当前服务状态**

```bash
# 记录当前 vLLM serve 的完整启动命令
VLLM_PID=$(cat vllm_serve.pid 2>/dev/null || pgrep -f "vllm.*serve")
if [ -n "$VLLM_PID" ]; then
    # 获取启动命令行
    CMDLINE=$(ps -p $VLLM_PID -o args= 2>/dev/null)
    echo "$CMDLINE" > .vllm_serve.cmdline.bak
    echo "Service cmdline saved to .vllm_serve.cmdline.bak"
fi
```

**步骤 3A.2：优雅停止 vLLM serve**

```bash
VLLM_PID=$(cat vllm_serve.pid 2>/dev/null || pgrep -f "vllm.*serve")
if [ -n "$VLLM_PID" ]; then
    echo "Stopping vLLM serve (PID: $VLLM_PID)..."
    kill -TERM $VLLM_PID
    # 等待最多 30 秒
    for i in {1..30}; do
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM serve stopped successfully"
            break
        fi
        sleep 1
    done
    # 强制终止
    if kill -0 $VLLM_PID 2>/dev/null; then
        echo "Force killing vLLM serve..."
        kill -9 $VLLM_PID
    fi
    rm -f vllm_serve.pid
fi
```

**步骤 3A.3：确认 NPU 内存已释放**

```bash
# 等待内存释放
sleep 3
npu-smi info -t memory
# 确认目标卡已用内存 < 10%
```

**步骤 3A.4：执行用户指定的离线任务**

直接执行用户请求的 benchmark、inference.py 或精度测试脚本。

**步骤 3A.5：测试完成后自动恢复服务**

```bash
if [ -f .vllm_serve.cmdline.bak ]; then
    echo "Restarting vLLM serve..."
    nohup $(cat .vllm_serve.cmdline.bak) > vllm_serve.log 2>&1 &
    NEW_PID=$!
    echo $NEW_PID > vllm_serve.pid
    echo "vLLM serve restarted with PID: $NEW_PID"
    # 等待服务就绪
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "Service is ready"
            break
        fi
        sleep 1
    done
fi
```

#### 策略 B：在线压测 — 复用现有服务

**步骤 3B.1：探测现有服务端点**

```bash
# 从 vllm_serve.pid 或进程推断端口
PORT=$(ss -tlnp | grep -E "vllm|python" | head -1 | awk '{print $4}' | cut -d: -f2)
PORT=${PORT:-8000}
MODEL_NAME=$(curl -s http://localhost:$PORT/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "Detected service on port $PORT, model: $MODEL_NAME"
```

**步骤 3B.2：调整压测命令复用现有服务**

- 将 benchmark 命令中的 `--model` 改为探测到的 `MODEL_NAME`
- 将请求地址指向 `http://localhost:$PORT`
- 提醒用户："检测到已有 vLLM serve 在运行，将直接复用该服务进行在线压测，不会中断服务。"

### 阶段 4：调度报告生成

执行完成后，输出以下结构化报告：

```markdown
## NPU 资源调度报告

### 调度摘要
| 项目 | 内容 |
|------|------|
| 调度时间 | YYYY-MM-DD HH:mm:ss |
| 目标设备 | Ascend NPU (卡号 X) |
| 原服务状态 | 运行中 / 未运行 |
| 调度策略 | 离线暂停恢复 / 在线复用 |
| 任务结果 | 成功 / 失败 |

### 进程变更记录
| 操作 | PID | 命令 | 时间 |
|------|-----|------|------|
| 停止服务 | XXXX | vllm serve ... | HH:mm:ss |
| 启动服务 | XXXX | vllm serve ... | HH:mm:ss |

### NPU 内存变化
| 阶段 | 已用内存 | 可用内存 |
|------|---------|---------|
| 调度前 | XX MiB | XX MiB |
| 停服后 | XX MiB | XX MiB |
| 任务后 | XX MiB | XX MiB |

### 注意事项
- 若服务恢复失败，可手动执行：`nohup $(cat .vllm_serve.cmdline.bak) > vllm_serve.log 2>&1 &`
- 备份的启动命令保存在 `.vllm_serve.cmdline.bak`
```

## 异常处理规则

| 异常情况 | 处理方案 |
|---------|---------|
| vLLM serve 无法正常停止 | 先 `kill -TERM`，30 秒后 `kill -9`，记录警告 |
| 停止后内存未释放 | 检查是否有僵尸进程，提示用户手动 `npu-smi info -t processes` |
| 任务执行失败 | 仍尝试恢复服务，避免用户环境被破坏 |
| 服务恢复后端口被占用 | 尝试原端口 +1 递增，或提示用户手动处理 |
| 无 pid 文件且找不到进程 | 直接执行用户任务，视为无冲突 |
