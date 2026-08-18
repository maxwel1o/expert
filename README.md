# Ascend NPU Hermes Expert Team（五角色链路 Demo）

这是一个可复现的 Hermes Agent 专家团队：用户始终与默认 Agent（Leader）对话，Leader 理解和拆解任务，再通过 Hermes Kanban 把任务派给四个彼此独立的 Worker。Worker 在各自的 Skill 空间中自主选择能力并回写结果，Leader 负责验收和最终汇总。

本仓库包含完整角色配置、201 个已分类 Skill、安装/验收/回滚脚本，以及可选的 SQLite 进度观察组件。仓库**不包含 API Key、Token、`.env`、服务器地址或运行数据库**；每位使用者必须先自行完成 Hermes 的模型/API 配置。

## 团队组成

| Profile | 定位 | Skill 数量 |
|---|---|---:|
| 默认 Agent（Leader） | 唯一任务入口；理解、拆解、派发、跟踪、验收、汇总 | 1 |
| `deployer` | 模型部署、服务启停、环境与部署状态检查 | 127 |
| `tester` | 功能、精度、性能、并发、稳定性测试 | 19 |
| `profiler` | Ascend NPU Profiling 采集、原始证据保存与完整性检查 | 5 |
| `analyst` | 测试/Profiling 数据分析、瓶颈定位与优化建议 | 49 |

四个 Worker 没有固定先后关系；只有真实输入输出依赖或资源冲突时才串行，否则可以并行。Leader 本身不替代 Worker 执行专业任务。

## 快速开始

```bash
git clone https://github.com/maxwel1o/expert.git
cd expert

# 1. 先按你安装的 Hermes 版本配置模型/API，然后检查配置
hermes config check

# 2. 安装团队与 201 个 Skill
chmod +x install.sh scripts/*.sh team_progress/assets/*
sudo ./install.sh

# 3. 如需第二终端的长期进度观察和最终状态对账
sudo ./install.sh --with-progress

# 4. 与默认 Agent（即 Leader）对话
hermes chat
# 或
hermes --tui
```

如果已经完成基础安装，只想补装进度观察层：

```bash
sudo ./scripts/install-team-progress.sh --dry-run
sudo ./scripts/install-team-progress.sh --apply
```

完整的新手说明见 [用户手册](docs/USER-GUIDE.md)，实际发布状态见 [CURRENT-STATE.md](docs/CURRENT-STATE.md)，架构见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)，故障处理见 [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

## 重要边界

- 这是五角色链路 Demo，不是后续九类 NPU 专家体系。
- Kanban 是任务事实源；`progress.db` 是观察与最终状态汇总层，不替代 Kanban。
- 进度信息默认在第二终端查看，避免持续 heartbeat 污染 Leader 的对话历史。
- 当前任务完成时，`wait-final` 可从 Hermes Kanban 数据库对账并输出一次 `TEAM_PROGRESS_FINAL`；能否自动显示回原 TUI，仍取决于 Hermes 启动后台命令时的会话通知生命周期。
- 第三方 Skill 的许可归各上游所有，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
