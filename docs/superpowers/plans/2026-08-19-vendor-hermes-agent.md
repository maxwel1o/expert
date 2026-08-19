# Vendor Hermes Agent 0.17.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将服务器当前运行的 Hermes Agent 0.17.0 固定源码快照纳入专家团队仓库，使新用户无需预装 Hermes。

**Architecture:** `vendor/hermes-agent/` 保存 MIT 许可的上游源码快照，不保存依赖、缓存和用户配置。`scripts/install-hermes.sh` 负责版本检查和调用固定源码的官方安装器；现有 `install.sh` 继续负责 API 配置完成后的团队层安装。

**Tech Stack:** Bash、Python 3.11、uv、Hermes Agent 0.17.0、Git、unittest

**Spec:** `docs/HERMES-INTEGRATION-SPEC.md`

## Global Constraints

- 固定 Hermes Agent 版本为 `0.17.0`。
- 固定上游为 `https://github.com/NousResearch/hermes-agent`，保留 MIT LICENSE。
- API Key、Token、`.env`、用户配置、数据库和日志不得进入仓库。
- `.venv`、`venv`、`node_modules`、`.playwright` 和缓存不得进入仓库。
- 原有 5 个角色、201 个角色级 Skill 和 team_progress 行为不得退化。

---

### Task 1: 固定源码契约测试

**Files:**
- Create: `tests/test_hermes_vendor.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `docs/HERMES-INTEGRATION-SPEC.md` 的版本和排除规则。
- Produces: `HermesVendorTests`，供后续源码、安装器和发布校验使用。

- [ ] **Step 1: 写入失败测试**

测试必须检查 `vendor/hermes-agent/pyproject.toml`、`LICENSE`、`uv.lock`、`setup-hermes.sh`，并断言版本 `0.17.0`、许可证 `MIT`、禁止目录不存在。

- [ ] **Step 2: 运行失败测试**

Run: `python -m unittest tests.test_hermes_vendor -v`

Expected: 因 `vendor/hermes-agent` 尚不存在而失败。

- [ ] **Step 3: 扩展忽略规则**

将 Hermes 运行依赖、缓存、`.env` 和安装状态路径加入 `.gitignore`。

### Task 2: 导入服务器权威 Hermes 源码

**Files:**
- Create: `vendor/hermes-agent/**`
- Create: `vendor/hermes-agent/VENDORED-SOURCE.md`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: 容器 `/opt/hermes` 和 Task 1 的契约测试。
- Produces: 可安装、具备来源和许可证记录的 Hermes 0.17.0 固定快照。

- [ ] **Step 1: 安全导出源码**

从容器导出 `/opt/hermes`，排除 `.venv`、`venv`、`node_modules`、`.playwright`、缓存、egg-info、`.install_method` 和用户配置。

- [ ] **Step 2: 写入来源说明**

记录版本、上游 URL、快照来源、排除项和 MIT 许可证位置。

- [ ] **Step 3: 运行源码契约测试**

Run: `python -m unittest tests.test_hermes_vendor -v`

Expected: 源码元数据和排除规则通过，安装脚本契约在 Task 3 前仍可失败。

### Task 3: 增加 Hermes 安装入口

**Files:**
- Create: `scripts/install-hermes.sh`
- Modify: `install.sh`
- Modify: `tests/test_hermes_vendor.py`

**Interfaces:**
- Consumes: `vendor/hermes-agent/setup-hermes.sh`。
- Produces: `scripts/install-hermes.sh [--dry-run|--apply]`；`HERMES_BIN` 可显式覆盖检测结果。

- [ ] **Step 1: 增加安装器失败测试**

测试 dry-run 输出固定版本、源码目录和官方安装器路径，并检查现有/不匹配版本分支。

- [ ] **Step 2: 实现最小安装器**

`--dry-run` 不写磁盘；`--apply` 在缺少 Hermes 时调用固定源码官方安装器，已存在 0.17.0 时保持不变，版本冲突时报错。

- [ ] **Step 3: 更新顶层入口**

`install.sh --install-hermes` 只安装 Hermes；团队安装仍要求用户先完成 `hermes setup`。

- [ ] **Step 4: 运行测试与 Shell 检查**

Run: `python -m unittest tests.test_hermes_vendor -v`

Run: `bash -n install.sh scripts/install-hermes.sh`

Expected: 全部通过。

### Task 4: 更新发布校验和文档

**Files:**
- Modify: `tools/validate_release.py`
- Modify: `README.md`
- Modify: `docs/USER-GUIDE.md`
- Modify: `docs/INSTALLATION.md`
- Modify: `docs/CURRENT-STATE.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `VALIDATION.md`

**Interfaces:**
- Consumes: Task 2 的源码快照和 Task 3 的安装命令。
- Produces: 新手可复现的三阶段安装说明和自动发布门禁。

- [ ] **Step 1: 扩展发布校验**

校验固定源码元数据、许可证、禁止目录和安装入口。

- [ ] **Step 2: 重写快速开始**

明确三阶段：安装 Hermes、用户配置 API、安装五角色团队；保留已有 Hermes 用户路径。

- [ ] **Step 3: 更新架构与当前状态**

说明 Hermes 本体、团队层、Skill 层和进度观察层的边界。

### Task 5: 全量验证与发布

**Files:**
- Modify: `VALIDATION.md`

**Interfaces:**
- Consumes: 所有前置任务交付物。
- Produces: 可审计提交、远端分支和合并后的 GitHub `main`。

- [ ] **Step 1: 运行全量验证**

Run: `python tools/validate_release.py`

Run: `python tools/verify_skill_hashes.py`

Run: `python -m unittest discover -s tests -t . -q`

Run: `bash -n install.sh scripts/*.sh team_progress/assets/*`

- [ ] **Step 2: 扫描敏感信息和禁止目录**

确认 `.env`、Token、API Key、用户配置、数据库、日志和依赖目录均未跟踪。

- [ ] **Step 3: 检查体积和差异**

记录 Git 对象大小、归档大小、文件数量和 `git diff --check` 结果。

- [ ] **Step 4: 提交和推送**

提交信息：`feat: bundle Hermes Agent 0.17.0 source`

- [ ] **Step 5: 核对 GitHub**

比较本地、服务器推送副本和 GitHub 远端提交号，确认默认分支为 `main`。
