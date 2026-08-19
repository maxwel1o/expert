# 安装命令索引

```bash
# 发行包自检
python3 tools/validate_release.py

# 新机器：预览/安装仓库内固定的 Hermes Agent 0.17.0
./scripts/install-hermes.sh --dry-run
./scripts/install-hermes.sh --apply

# 安装器结束后，由用户自己配置模型/API
hermes setup
hermes config check

# 团队配置预览/执行
./scripts/setup-team.sh --dry-run
./scripts/setup-team.sh --apply

# 安装 201 Skill；可选参数为自定义安全 stamp
./scripts/install-skills.sh
./scripts/install-skills.sh my-install-001

# 可选进度层
sudo ./scripts/install-team-progress.sh --dry-run
sudo ./scripts/install-team-progress.sh --apply
sudo ./scripts/install-team-progress.sh --verify

# 总体验收
./scripts/verify-team.sh
```

Hermes 源码位于 `vendor/hermes-agent/`；`.venv`、`venv`、`node_modules` 和 Playwright 浏览器不会提交到 Git，安装时按 `uv.lock` 在本机重建。若已经安装 Hermes 0.17.0，可跳过源码安装。版本不一致时安装器会停止，避免团队配置写入错误版本。

团队安装必须使用拥有 Hermes 配置的同一用户。若 `hermes` 不在 PATH：

```bash
export HERMES_BIN=/absolute/path/to/hermes
./install.sh --with-progress
```

只有写入 `/usr/local/bin` 或系统级进度目录确实需要提权时才使用 `sudo -E`，并同时保留正确的 `HOME` 与 `HERMES_BIN`。安装脚本不会重启 Gateway；验收后按你的 Hermes 部署方式决定是否重启。
