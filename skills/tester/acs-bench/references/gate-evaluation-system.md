# Gate 评价体系 v2 — 多维门控

> 设计日期: 2026-06-04
> 替代旧方案: 单一"使用次数"门控
> 实现脚本: `agent-skill-commit/scripts/skill-usage-gate.py`

## 设计原则

- "用过" ≠ "用成功"，评价应基于成功而非原始次数
- 失败不抵消成功，但通过成功率门槛挡住"跑了很多但大部分失败"
- 失败分类：skill_defect（计入）vs environment（不计入）
- 多场景 skill 靠场景覆盖，单场景 skill 靠更多验证者
- coverage_dimensions 声明 → 精确检查；不声明 → 自动检测或走更严路径
- 不设时间窗口：历史记录始终有效

## Gate 1: dev → test

| 条件 | 阈值 |
|------|------|
| 成功次数 | ≥ 4 |
| 成功率 | ≥ 50%（skill_defect 类，总尝试 ≥ 5 时才计算） |

## Gate 2: test → prod

| 条件 | 多场景 skill | 单场景 skill |
|------|------------|------------|
| 成功次数 | ≥ 4 | ≥ 6 |
| 成功率 | ≥ 70% | ≥ 70% |
| 验证者（去重 author） | ≥ 2 人 | ≥ 3 人 |
| 场景覆盖 | ≥ 2 种 | N/A |
| 证据链 | 可验证 | 可验证 |

## 场景判定逻辑（三层）

1. SKILL.md 声明了 `coverage_dimensions` → 多场景
2. jsonl 记录中有 ≥ 2 种不同 `scenarios` → 多场景（自动检测）
3. 否则 → 单场景（走更严条件）

### 判定优先级

| 情况 | 判定 | prod 条件 |
|------|------|-----------|
| 声明了 coverage_dimensions | 精确多场景 | 4 成功 + 70% + 2 人 + 2 场景 |
| 没声明但 jsonl 有多种 scenarios | 自动多场景 | 4 成功 + 70% + 2 人 + 2 场景 |
| 没声明且 jsonl 场景单一 | 单场景 | 6 成功 + 70% + 3 人 |

## Gate 脚本实现

### Helper 函数

- `filter_valid_records(records, skill_name, stage)` — 排除 `failure_type=environment` 的记录
- `calc_success_rate(valid_records)` — 返回 (rate, total)
- `get_distinct_authors(success_records)` — 去重 author
- `get_distinct_scenarios(success_records)` — 去重 scenarios dict（JSON 序列化后比较）
- `get_coverage_dimensions(skill_name)` — 从 SKILL.md frontmatter 读取声明
- `is_multi_scenario(skill_name, success_records)` — 三层判定，返回 (bool, reason)

### Gate 1 输出示例

```
GATE 1: acs-bench — successes=4/4, success_rate=80% (5 attempts), rate_check=applied
GATE 1 PASS ✅
```

```
GATE 1: acs-bench — successes=1/4, success_rate=100% (1 attempts), rate_check=skipped (<5 attempts)
GATE 1 FAIL ❌ — need ≥4 successes (have 1)
```

### Gate 2 输出示例

```
GATE 2: acs-bench — mode=multi-scenario (declared coverage_dimensions)
  successes=4/4, rate=80% (need ≥70%), validators=2/2, scenario_coverage=2/2
GATE 2 PASS ✅
```

## jsonl 记录格式 v2

```json
{
  "skill": "acs-bench",
  "version": "3.0.0",
  "stage": "dev",
  "author": "zhangsan",
  "agent": "hermes",
  "action": "use",
  "result": "success",
  "failure_type": null,
  "scenarios": {"test_mode": "基准压测", "api": "MAAS"},
  "evidence": {"report_exists": true, "csv_valid": true, "fail_rate": "0%"},
  "timestamp": "2026-06-04T01:48:31+00:00"
}
```

### failure_type 分类

- `skill_defect`：workflow 步骤报错、输出格式错误、脚本 bug → 计入成功率分母
- `environment`：API 限流、网络超时、依赖缺失 → 不计入分母
- `user_cancel`：用户主动取消 → 不计入分母
- `null`：成功时

### scenarios 字段

记录本次使用的场景组合，用于场景覆盖判定。结构由 skill 自定义，Gate 脚本通过 JSON 序列化后比较去重。

### evidence 字段

可验证的成功证据，Gate 可抽查（文件是否存在、CSV 是否可解析）。

## SKILL.md frontmatter 声明

```yaml
coverage_dimensions:
  - test_mode: [基准压测, 摸高寻优, 纯吞吐爬坡]
  - api: [MAAS, vLLM, OpenAI]
```

单场景 skill 不声明此字段，自动走更严条件。

## 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 失败是否抵消成功 | 否 | 不合理，成功就是成功 |
| 时间窗口 | 不设 | 历史记录始终有效 |
| 成功率计算阈值 | Gate1 ≥5 次才计算 | 样本太少时成功率无意义 |
| 单场景 prod 条件 | 6 成功 + 3 人 | 用更多验证代替场景多样性 |
| 场景自动检测 | jsonl 去重 scenarios | 不依赖声明也能工作 |
