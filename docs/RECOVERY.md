# 备份与恢复

`setup-team.sh` 在 Hermes home 的 `team-change-backups/` 中保存默认配置、Leader SOUL 和已有 Worker 配置。`install-skills.sh` 在 `skill-backups/<stamp>/` 保存被替换的同名 Skill，并在 `skill-install-records/` 保存本次安装清单。进度安装器在 `/opt/data/team-change-backups/` 保存 SQLite 一致性备份、SOUL、Hook 配置和旧安装。

```bash
sudo ./scripts/rollback-team.sh setup BACKUP_DIR
sudo ./scripts/rollback-team.sh skills STAMP
sudo ./scripts/rollback-team.sh progress BACKUP_DIR
```

回滚只接受 Hermes 备份根目录内的 setup/progress 路径，Skill stamp 只接受字母、数字、点、下划线和短横线。执行前先停止正在修改同一配置的进程；完成后运行 `hermes config check` 和 `scripts/verify-team.sh`。
