# ascend-resource-scheduler

昇腾 NPU 资源冲突诊断与进程调度 Skill。

## 适用场景

在昇腾 NPU 上部署 vLLM-Ascend 模型后，`vllm serve` 常作为常驻服务运行。此时若再执行离线 benchmark、精度测试或批量推理，极易触发：

```
Free memory on device is less than desired GPU memory utilization
```

本 Skill 自动诊断此类冲突，并根据用户任务类型智能决策调度策略。

## 触发方式

在 Claude Code 中输入以下任意指令即可唤起：

- "NPU 资源冲突"
- "serve 占用"
- "停服务跑 benchmark"
- "device 内存不足"
- "同卡再跑测试"
- "vLLM 进程冲突"

## 核心能力

| 能力 | 说明 |
|------|------|
| 冲突检测 | 通过 `npu-smi info` + 进程扫描，自动发现同卡 vLLM 常驻服务 |
| 意图识别 | 根据用户关键词识别"离线测试"或"在线压测"意图 |
| 无缝调度 | 离线测试：自动保存启动命令 → 停服 → 执行测试 → 恢复服务 |
| 服务复用 | 在线压测：自动探测现有服务端点，复用而不中断 |
| 状态报告 | 生成包含进程变更、内存变化、注意事项的结构化报告 |

## 调度策略

```
用户请求测试
    │
    ▼
检测 NPU 上是否有 vLLM serve 进程
    │
    ├── 无进程 ──→ 直接执行用户任务
    │
    └── 有进程 ──→ 识别用户意图
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
        离线测试    在线压测      意图不明
            │           │           │
            ▼           ▼           ▼
        停服再测    复用服务      询问用户
        测完恢复    不中断
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `SKILL.md` | Claude Code Skill 定义文件，包含完整工作流与执行指令 |
| `README.md` | 本文件，提供使用说明和场景指南 |

## 依赖

- 昇腾 NPU 环境（CANN Toolkit 已安装）
- `npu-smi` 工具可用
- `curl`、`ss`、`ps` 等标准 Linux 工具

## 贡献

提交 PR 至：https://gitcode.com/Ascend/model-agent
