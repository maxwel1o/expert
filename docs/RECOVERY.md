# 备份与恢复

`setup-team.sh` 在 Hermes home 的 `team-change-backups/` 中保存默认配置、Leader SOUL 和已有 Worker 配置。`install-skills.sh` 在 `skill-backups/<stamp>/` 保存被替换的同名 Skill，并在 `skill-install-records/` 保存本次安装清单。进度安装器在 `/opt/data/team-change-backups/` 保存 SQLite 一致性备份、SOUL、Hook 配置和旧安装。

```bash
sudo ./scripts/rollback-team.sh setup BACKUP_DIR
sudo ./scripts/rollback-team.sh skills STAMP
sudo ./scripts/rollback-team.sh progress BACKUP_DIR
```

回滚只接受 Hermes 备份根目录内的 setup/progress 路径，Skill stamp 只接受字母、数字、点、下划线和短横线。执行前先停止正在修改同一配置的进程；完成后运行 `hermes config check` 和 `scripts/verify-team.sh`。

## 仓库内 Hermes 安装恢复

`scripts/install-hermes.sh` 使用上游安装器在 `vendor/hermes-agent/venv/` 重建依赖，并通常把 `~/.local/bin/hermes` 链接到该环境。源码本身不会被安装器覆盖；API 配置属于用户本地状态，不由团队回滚脚本删除。

如果依赖安装损坏，优先在仓库根目录重新运行：

```bash
./scripts/install-hermes.sh --dry-run
cd vendor/hermes-agent
./setup-hermes.sh
```

如需完全重建依赖，先用 `readlink -f ~/.local/bin/hermes` 确认链接确实指向当前仓库的 `vendor/hermes-agent/venv/bin/hermes`，再移除该链接和当前仓库内的 `venv/`。不要删除 `~/.hermes`、Hermes home 或用户 `.env`，除非已经单独备份并明确希望清除 API 与历史状态。
