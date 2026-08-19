# 架构说明

## 四层交付结构

1. Hermes 本体层：`vendor/hermes-agent/` 保存 0.17.0 固定源码，提供对话、Profile、Kanban、Gateway、TUI 和工具运行时。
2. 团队角色层：默认 Agent 作为 Leader，四个独立 Profile 作为 Worker，边界由 `roles/*/SOUL.md` 固定。
3. 专业能力层：201 个角色级 Skill 按职责隔离安装，避免 Leader 读取全部专业说明。
4. 观察层：`team_progress` 利用 Hermes Kanban 事实源、生命周期 Hook 和 SQLite 汇总长任务进度与最终状态。

Hermes 依赖与用户运行状态不属于源码交付层；它们在目标机器安装或生成。API 配置始终属于用户本地运行环境。

## 责任模型

- 默认 Agent 是唯一 Leader，也是用户根任务的唯一入口。
- `deployer`、`tester`、`profiler`、`analyst` 是四个独立 Worker，不存在固定流水线。
- SOUL 定义角色边界、必需输入、缺失输入行为、交付内容和安全约束。
- Skill 按角色隔离安装；Worker 只在自己的 Skill 空间内选择能力。
- Kanban 记录任务、负责人、真实依赖和终态；Worker 不通过 delegation 绕过任务板自行创建隐形执行链。

## 任务路由

Leader 先理解任务，再基于“需要交付什么”选择角色：部署或服务运行交给 `deployer`；测试执行交给 `tester`；原始 Profiling 采集交给 `profiler`；证据解释和瓶颈判断交给 `analyst`。一个任务可以只使用一个 Worker，也可以使用任意组合。

只有这些情况建立依赖：后一步确实需要前一步产物；角色或设备资源互斥；用户明确要求顺序。否则应允许并行。

## 三层状态

1. 对话层：用户与 Leader 讨论目标和最终结果。
2. Kanban 层：Hermes 的任务事实源，负责派发和生命周期。
3. 观察层：`progress.db` 聚合细粒度事件、Hook 终态和 Kanban 对账，供第二终端查看。

观察层采用 job_id 隔离，减少多个长期任务的信息冲突。它不自动唤醒 Leader 去执行新动作；最终通知只允许汇总当前 Job，避免完成回调触发重复派发或额外修改。
