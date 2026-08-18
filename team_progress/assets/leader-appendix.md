<!-- BEGIN TEAM-PROGRESS-PROTOCOL v1 -->
## 团队进度协议

每个用户根任务都必须经过 Leader，并且必须有独立的 `job_id`。在创建 Kanban 子任务前先运行：

```bash
team-progress job create --title "不含凭据的任务说明"
team-progress job focus JOB_ID
```

创建 Worker 卡片时先在正文写 `Progress-Task: pending`，再从 `hermes kanban create --json` 的输出读取真实 Kanban 卡片 ID。Hermes v0.17.0 没有普通卡片正文的 `update` 子命令，因此登记后必须通过受支持的 comment 把真实 ID 回填到卡片上下文：

```bash
team-progress task add \
  --job-id JOB_ID \
  --task-id KANBAN_TASK_ID \
  --role deployer

hermes kanban comment KANBAN_TASK_ID \
  "Progress-Job: JOB_ID
Progress-Task: KANBAN_TASK_ID
Progress-Role: deployer
Progress-Resources: none"
```

Kanban 卡片必须带下面的非秘密元数据：

```text
Progress-Job: JOB_ID
Progress-Task: KANBAN_TASK_ID
Progress-Role: deployer
Progress-Resources: npu:0,container:model-server-a
```

`Progress-Resources` 只写真实需要独占的 NPU、容器、端口或目录资源；没有资源冲突时写 `none`。角色之间只有存在真实输入输出关系时才建立 Kanban 依赖。

全部 Worker 卡片登记完成后，每个 `job_id` 只启动一个最终监听器。通过 Hermes `terminal` 工具调用：

```text
command="team-progress wait-final JOB_ID --adapter hermes --source-db /opt/data/kanban.db"
background=true
notify_on_complete=true
```

不要使用 `watch_patterns` 或 `cronjob`。启动成功后立即向用户返回 `job_id`，不要让当前模型回合持续等待。中间进度仍由用户在第二终端通过 `team-progress watch` 查看。

Hermes 的 `kanban_task_completed` 与 `kanban_task_blocked` lifecycle Hook 会即时把耐久 Kanban 终态同步到进度库；`wait-final` 每轮还会读取 `/opt/data/kanban.db` 自动对账。两条路径共用同一幂等接口，所以 Worker 漏掉 `team-progress complete`/`block` 时不会永久卡住，也不会生成重复终态。

当后台完成通知包含 `TEAM_PROGRESS_FINAL` 时，这是只读最终汇报回合。依次执行：

```bash
team-progress consume --consumer leader --job-id JOB_ID --important-only
team-progress status JOB_ID
```

核对 `final_summary`、各 Worker 终态、四类终态计数和产物引用后，只向用户汇报一次。该回合不得调用 `retry`，不得创建、修改或重新分配 Kanban 卡片，不得启动新的 Worker，不得执行部署、测试、采集、分析或修复命令，不得修改文件、配置或环境，也不得切换或中断其他 Job。

如果后台通知不含 `TEAM_PROGRESS_FINAL` 或命令非零退出，只报告“最终监听器失败”，不得把它描述为业务任务失败，也不得自动修复。

同一时刻只设置一个前台 Job。切换关注对象使用：

```bash
team-progress job focus JOB_ID
```

切换焦点只改变显示策略，不暂停或中断其他 Job。

每次准备向用户正式回复前，先读取当前 Job 尚未消费的重要事件：

```bash
team-progress consume \
  --consumer leader \
  --job-id JOB_ID \
  --important-only
```

单行速查：`team-progress consume --consumer leader --job-id JOB_ID --important-only`。

规则：

- 普通 `heartbeat` 只用于终端存活显示，不复制进正式聊天历史。
- `milestone`、`blocked`、`failed`、`completed` 必须在合适的正式回复中说明。
- 收到 `final_summary` 后，Leader 必须验收各 Worker 的摘要和产物引用，再向用户统一交付。
- Leader 始终是逻辑负责人，但不得让单个模型推理回合空转等待数小时。
- 不在 Kanban、命令参数、进度消息、SQLite 或文档中写入密码、Token、API Key、私钥和带凭据 URL；只引用 `credential:名称`。
<!-- END TEAM-PROGRESS-PROTOCOL v1 -->
