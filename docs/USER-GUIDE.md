# 五角色 NPU 专家团队用户手册

## 1. 这套系统解决什么问题

单个 Agent 同时承担部署、测试、Profiling 和性能分析时，容易出现职责混杂、上下文过长、Skill 误选和结果难验收。这里把“任务入口与最终责任”留给 Leader，把专业执行划分给四个 Worker：

```text
用户 → 默认 Agent / Leader → Hermes Kanban → 一个或多个 Worker
                                      └────→ Worker 结果 → Leader 验收与汇总
```

角色依据任务分派，Skill 依据角色职责组织。并行不是固定的“部署+测试”组合；任意两个或多个 Worker，只要没有数据依赖、同角色占用或 NPU/容器/端口等资源冲突，就可以并行。

## 2. 安装前准备

需要：

- Linux 或 Linux 容器；
- 能访问 Python/Node 依赖源；仓库已经包含 Hermes Agent 0.17.0 源码，无需预装 Hermes；
- `bash`、`python3`、SQLite（Python 标准库即可）和常用 coreutils；
- 安装进度 Hook 时需要 PyYAML；
- 有权写入 Hermes home；进度组件默认还写入 `/opt/data/team-progress` 和 `/usr/local/bin`，通常需要 root；
- NPU 任务本身需要正确安装驱动、固件、CANN 和对应框架，但这些不是建队脚本自动决定的“部署栈”。

先从固定源码安装 Hermes；已经安装 0.17.0 的用户可跳过：

```bash
./scripts/install-hermes.sh --dry-run
./install.sh --install-hermes
```

安装过程会在本机重建 Python/Node/Playwright 依赖，它们不会进入 Git。随后配置你自己的 API/模型；仓库不知道也不应知道你的供应商、Key 或 endpoint：

```bash
hermes setup
hermes config path
hermes config check
```

不要把 `.env`、Token 或带凭证 URL 写入仓库、Kanban 卡片或进度消息。

## 3. 获取与预检查

```bash
git clone https://github.com/maxwel1o/expert.git
cd expert
python3 tools/validate_release.py
chmod +x install.sh scripts/*.sh team_progress/assets/*
chmod +x vendor/hermes-agent/setup-hermes.sh
```

如果 Hermes 在容器中，在第二个终端先进入同一容器：

```bash
docker exec -it YOUR_HERMES_CONTAINER bash
cd /path/to/expert
```

## 4. 安装方式

新环境严格按照三个阶段执行：

```bash
# 阶段一：安装 Hermes Agent 0.17.0
./install.sh --install-hermes

# 阶段二：用户自行写入本机 API/模型配置
hermes setup
hermes config check

# 阶段三：安装五角色团队
./install.sh
```

已经有 Hermes 0.17.0 且完成 API 配置时，直接从阶段三开始。推荐先看团队变更范围：

```bash
./scripts/setup-team.sh --dry-run
sudo ./scripts/install-team-progress.sh --dry-run
```

基础团队（五角色 + 201 Skill）：

```bash
./install.sh
```

基础团队加进度观察：

```bash
./install.sh --with-progress
```

也可逐项执行：

```bash
./scripts/setup-team.sh --apply
./scripts/install-skills.sh
sudo ./scripts/install-team-progress.sh --apply   # 可选
./scripts/verify-team.sh
```

安装行为：

1. 默认 Agent 的 `SOUL.md` 替换为 Leader 规则；不会额外创建一个 `leader` Profile。
2. 创建或更新 `deployer`、`tester`、`profiler`、`analyst`。
3. 只有默认 Agent 保留 Kanban 编排 toolset；Worker 通过被派发的卡片工作。
4. Skill 安装前把同名旧目录移动到带时间戳的备份目录。
5. 可选进度层安装 lifecycle Hook、CLI 和 SQLite 状态库，并向五份 SOUL 追加进度协议。
6. 不修改仓库中的 API，不生成 API Key；若本机默认 Hermes 已配置 `.env`，建队脚本仅在同一机器内让新 Worker 继承该运行配置。

## 5. 验收

```bash
hermes config check
hermes profile list
hermes kanban assignees
sudo ./scripts/verify-team.sh
python3 tools/validate_release.py
```

预期看到默认 Agent，以及四个 Worker；仓库绑定的 Skill 数为 201。`verify-team.sh` 使用“至少”数量判断，因此不会误删使用者原先已有的其他 Skill。

## 6. 开始使用

进入默认 Agent：

```bash
hermes chat
# 或使用全屏界面
hermes --tui
```

直接用自然语言把根任务交给 Leader，例如：

```text
请评估当前 Ascend NPU 上的模型服务性能。先明确缺少的输入，再拆分任务；
没有真实依赖的 Worker 可以并行。所有执行任务都进入 Kanban，最后统一验收和汇总。
```

Leader 应先判断目标、输入、产出和完成标准，再选择角色。它知道四个 Worker 后才会正确匹配；角色定义来自 `roles/*/SOUL.md`，不是依靠模型临时猜测。Worker 依据自身 Skill 列表自主选择工具，Leader 不需要读取全部 201 个 Skill，也不应随意用 `--skill` 强绑具体 Skill。

## 7. 进度观察（可选）

进度层的原理是：

- Worker 主动把 `start`、里程碑、heartbeat 和终态写入 `progress.db`；
- Hermes 的 `kanban_task_completed` / `kanban_task_blocked` Hook 即时同步终态；
- `wait-final` 周期读取 `/opt/data/kanban.db` 对账，补偿 Worker 漏报；
- 全部已登记任务终止后只生成一次 `final_summary`；
- 第二终端只读显示，不把高频 heartbeat 写进聊天历史。

查看所有任务：

```bash
team-progress status --all
```

持续查看某个 Job：

```bash
team-progress watch JOB_ID
```

只看当前被 focus 的 Job：

```bash
team-progress watch
```

等待并对账最终结果：

```bash
team-progress wait-final JOB_ID \
  --adapter hermes \
  --source-db /opt/data/kanban.db
```

这里的 `progress.db` 确实记录进度事件、任务状态、资源锁和最终摘要；它不保存业务产物本体，也不替代 Hermes 的 Kanban 数据库。

不同 Job 使用不同 `job_id`，不会因为切换关注对象而中断。`team-progress job focus JOB_ID` 只改变默认显示目标。同一 Worker 或同一独占资源发生冲突时应等待；不同 Worker 且资源不冲突时可并行。

## 8. 目录与配置位置

仓库内：

```text
roles/leader/SOUL.md
roles/{deployer,tester,profiler,analyst}/SOUL.md
skills/<role>/<skill>/SKILL.md
manifests/skills.csv
team_progress/
vendor/hermes-agent/                       # 固定的 Hermes 0.17.0 源码
vendor/hermes-agent/VENDORED-SOURCE.md     # 版本、来源、许可证和导出哈希
```

安装后位置由 `hermes config path` 决定。常见容器布局为：

```text
/opt/data/SOUL.md                         # 默认 Agent / Leader
/opt/data/profiles/deployer/SOUL.md
/opt/data/profiles/tester/SOUL.md
/opt/data/profiles/profiler/SOUL.md
/opt/data/profiles/analyst/SOUL.md
/opt/data/profiles/<role>/skills/
/opt/data/team-progress/state/progress.db
```

## 9. 回滚

每次安装都会输出备份目录或 Skill stamp。先记录该值，再执行：

```bash
sudo ./scripts/rollback-team.sh setup /opt/data/team-change-backups/TIMESTAMP-team-setup
sudo ./scripts/rollback-team.sh skills TIMESTAMP
sudo ./scripts/rollback-team.sh progress /opt/data/team-change-backups/TIMESTAMP-team-progress
```

详细恢复边界见 [RECOVERY.md](RECOVERY.md)。
