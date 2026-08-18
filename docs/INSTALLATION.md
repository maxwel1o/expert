# 安装命令索引

```bash
# 发行包自检
python3 tools/validate_release.py

# 团队配置预览/执行
sudo ./scripts/setup-team.sh --dry-run
sudo ./scripts/setup-team.sh --apply

# 安装 201 Skill；可选参数为自定义安全 stamp
sudo ./scripts/install-skills.sh
sudo ./scripts/install-skills.sh my-install-001

# 可选进度层
sudo ./scripts/install-team-progress.sh --dry-run
sudo ./scripts/install-team-progress.sh --apply
sudo ./scripts/install-team-progress.sh --verify

# 总体验收
sudo ./scripts/verify-team.sh
```

脚本假定 Hermes 已正确配置。若 `hermes` 不在 PATH：

```bash
export HERMES_BIN=/opt/hermes/bin/hermes
sudo -E ./install.sh --with-progress
```

安装脚本不会重启 Gateway；验收后按你的 Hermes 部署方式决定是否重启。
