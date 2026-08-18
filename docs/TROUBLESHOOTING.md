# 故障排查

## `Unknown toolset 'kanban'`

不要执行 `hermes tools enable kanban`。在已验证的 Hermes v0.17.x 配置中，Kanban 是顶层 `toolsets` 项；`setup-team.sh` 会写入配置并执行 `hermes kanban init`。

## Worker 没有被 Leader 调用

检查 `hermes profile list`、`hermes kanban assignees`，并确认默认 Agent 的 `SOUL.md` 是 `roles/leader/SOUL.md`。所有根任务必须先进入默认 Agent；不要直接把独立 Worker 当作新的总入口。

## 任务完成但 TUI 没有出现最终消息

这通常是 Hermes 当前前台会话没有收到后台进程的完成通知，不代表 Worker 或 Kanban 没完成。先运行：

```bash
team-progress status JOB_ID
team-progress watch JOB_ID --once
team-progress wait-final JOB_ID --adapter hermes --source-db /opt/data/kanban.db
```

`wait-final` 解决的是终态对账和一次性最终摘要；它无法绕过 Hermes 本身的会话通知生命周期。第二终端观察是最稳定、也最不污染聊天历史的方式。

## `wait-final` 一直等待

检查 Job 是否登记了正确的 Kanban task ID、所有任务是否已进入终态，以及 `/opt/data/kanban.db` 是否是当前 Hermes 实例的数据源。再运行：

```bash
team-progress reconcile --adapter hermes --source-db /opt/data/kanban.db --job-id JOB_ID
team-progress status JOB_ID
```

## Skill 数量不符

运行 `python3 tools/validate_release.py` 检查仓库，再运行 `scripts/verify-team.sh` 检查安装目标。同名旧 Skill 会被移动到 `/opt/data/skill-backups/<stamp>/`。

## API 不可用

本仓库不提供 API 配置。回到默认 Hermes 配置流程，完成 provider、model、endpoint 和 Key 设置，然后运行 `hermes config check`。不要把凭证提交到 issue 或日志。
