# 日常操作

```bash
# 与 Leader 对话
hermes chat
hermes --tui

# 检查角色与任务板
hermes profile list
hermes kanban assignees
hermes kanban list --json

# 第二终端观察进度
team-progress status --all
team-progress watch JOB_ID
team-progress watch JOB_ID --once

# 切换默认观察目标（不会暂停其他 Job）
team-progress job focus JOB_ID

# 查看最终状态或等待最终汇总
team-progress status JOB_ID
team-progress wait-final JOB_ID --adapter hermes --source-db /opt/data/kanban.db
```

日常用户不需要手工调用 `team-progress start/update/complete`；这些命令写在 Worker 协议里。调试时可通过 `team-progress --help` 和各子命令的 `--help` 查看准确参数。
