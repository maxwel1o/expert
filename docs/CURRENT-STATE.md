# 当前发布状态

本仓库记录的是已经运行验证过的五角色 Hermes NPU 专家团队链路 Demo，而不是后续规划中的九类专家系统。

## 角色

- 默认 Hermes Agent：Leader，也是所有用户根任务的唯一入口；不额外创建 `leader` Profile。
- `deployer`：模型部署、服务启停、部署环境和状态检查。
- `tester`：功能、精度、性能、并发和稳定性测试。
- `profiler`：Ascend NPU Profiling 采集、原始证据保存和完整性检查。
- `analyst`：测试及 Profiling 证据分析、性能瓶颈判断和优化建议。

四个 Worker 相互独立。没有真实输入输出依赖或独占资源冲突时，任意 Worker 组合都可以并行；不存在固定的部署→测试→采集→分析流水线。

## Skill

| 角色 | 团队发布 Skill 数 |
|---|---:|
| Leader | 1 |
| deployer | 127 |
| tester | 19 |
| profiler | 5 |
| analyst | 49 |
| 合计 | 201 |

这里只发布团队明确分配的 Skill。运行环境中可能还存在使用者自行安装或 Hermes 预置的其他 Leader Skill，它们不属于本团队发布范围。

## 进度观察

- `team_progress` 版本：`1.2.0`。
- SQLite 默认位置：`/opt/data/team-progress/state/progress.db`。
- Hermes Kanban 是任务事实源，`progress.db` 是细粒度观察与最终状态汇总层。
- lifecycle Hook 同步完成/阻塞终态；`wait-final` 对账 Kanban 数据库并生成一次最终摘要。
- 中间 heartbeat 通过第二终端查看，避免污染 Leader 主对话。

## 发布边界

角色、Skill 和进度组件来自实际部署状态的只读导出，并使用 [`manifests/skills.sha256`](../manifests/skills.sha256) 固定文件内容。仓库不包含 API 配置、`.env`、`config.yaml`、Token、数据库、服务器地址、日志、备份或历史任务数据。
