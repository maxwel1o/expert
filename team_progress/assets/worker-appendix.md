<!-- BEGIN TEAM-PROGRESS-PROTOCOL v1 -->
## Worker 进度协议

领取 Kanban 卡片后，先读取其中的：

```text
Progress-Job
Progress-Task
Progress-Role
Progress-Resources
```

调用命令时可以显式传入三个标识，也可以设置：

```bash
export TEAM_PROGRESS_JOB_ID="JOB_ID"
export TEAM_PROGRESS_TASK_ID="KANBAN_TASK_ID"
export TEAM_PROGRESS_ROLE="当前角色"
```

开始业务工作前运行：

```bash
team-progress start \
  --phase prepare \
  --message "开始检查输入和环境" \
  --resource "任务声明的资源键"
```

阶段发生有意义的变化时运行：

```bash
team-progress update \
  --phase CURRENT_PHASE \
  --message "不含凭据的进度说明"
```

长时间外部命令优先交给包装器：

```bash
team-progress run \
  --phase CURRENT_PHASE \
  --resource "任务声明的资源键" \
  -- ACTUAL_COMMAND
```

`team-progress run` 自动负责开始、每 300 秒心跳、退出状态和租约释放。无法使用包装器时，至少每 300 秒运行：

```bash
team-progress heartbeat \
  --phase CURRENT_PHASE \
  --message "仍在运行，当前阶段未变化"
```

每次执行必须且只能以一种终态结束：

```bash
team-progress complete --phase done --message "完成摘要" --artifact /path/to/result
team-progress block --phase blocked --message "缺少的输入或权限"
team-progress fail --phase failed --message "失败位置和退出状态"
```

规则：

- 同角色锁或资源锁冲突时等待，不绕过锁、不抢占其他任务。
- 不知道进度百分比时不填写 `--percent`，不得伪造数字。
- 产物只记录路径和简短说明，不把大段日志写入进度消息。
- 不记录密码、Token、API Key、私钥、完整 `.env` 或带凭据 URL；使用 `credential:名称`。
- 完成、阻塞或失败后仍要按 Hermes Kanban 生命周期写回卡片结果。
- 主动 `start`、`update`、`heartbeat` 用于提供细粒度进度；Hermes lifecycle Hook 和 `wait-final` 自动对账负责终态一致性。两者互补，不能用自动对账替代阶段进度。
<!-- END TEAM-PROGRESS-PROTOCOL v1 -->
