# Hermes Agent 源码集成规范

## 目标

让本仓库从“需要预装 Hermes 的团队扩展包”升级为“包含 Hermes Agent 0.17.0 固定源码快照的可复现专家团队”。使用者只需克隆仓库、安装依赖、配置自己的模型/API，再安装五角色团队。

## 固定版本与来源

- 软件：Hermes Agent
- 版本：0.17.0（CLI 显示构建日期 2026.6.19）
- 上游：https://github.com/NousResearch/hermes-agent
- 许可证：MIT，版权归 Nous Research
- 快照来源：当前链路测试服务器容器 `/opt/hermes`

## 纳入范围

源码放在 `vendor/hermes-agent/`，保留运行 Hermes 所需的 Python、Web/TUI、Gateway、工具、插件、内置 Skill、锁文件、Docker 文件和官方安装脚本。

以下内容不得进入 Git：

- Python 环境：`.venv/`、`venv/`
- Node 依赖：`node_modules/`
- Playwright 浏览器：`.playwright/`
- 缓存与生成物：`.pytest_cache/`、`__pycache__/`、`*.pyc`、`*.egg-info/`
- 安装状态：`.install_method`
- 用户数据：`.env`、API Key、Token、Hermes 配置、数据库、日志、服务器地址

## 安装体验

1. `./scripts/install-hermes.sh --apply` 从仓库内的固定源码安装 Hermes。
2. 用户运行 `hermes setup`，自行填写模型/API。
3. `./install.sh [--with-progress]` 安装 Leader、四个 Worker、201 个团队 Skill，以及可选进度观察组件。
4. 已安装 Hermes 0.17.0 时保持现有安装，不重复覆盖；版本不匹配时明确报错并要求用户选择环境。

## 验收标准

- 固定源码包含 `pyproject.toml`、`uv.lock`、官方 `LICENSE` 和可执行安装脚本。
- `pyproject.toml` 中名称为 `hermes-agent`、版本为 `0.17.0`、许可证为 `MIT`。
- 禁止目录、API 配置和运行数据没有进入 Git。
- 原有五角色、201 个顶层 Skill、Skill 哈希和 54 项进度组件测试继续通过。
- 安装脚本通过 Shell 语法检查和 dry-run 验证。
- README 和用户手册明确区分“安装 Hermes”“配置 API”“安装专家团队”三个阶段。
