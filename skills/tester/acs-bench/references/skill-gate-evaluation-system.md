# Skill Gate Evaluation System (v2)

## Overview

The gate system controls skill lifecycle progression (dev → test → prod) based on **success count + success rate + validator diversity + scenario coverage**, not raw usage count.

## jsonl Record Format (v2)

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

### failure_type Classification

| Type | Meaning | Gate Impact |
|------|---------|-------------|
| `skill_defect` | Workflow step error, CSV validation fail, script bug | Counts as FAILED (enters denominator) |
| `environment` | API rate limit, network timeout, tokenizer download fail | Excluded from rate calculation entirely |
| `null` | Success (no failure) | Counts as SUCCESS |

## Gate 1: dev → test

**Purpose**: Prove the skill works stably.

```
✅ Success count ≥ 4 (skill_defect-classified only)
✅ Success rate ≥ 50% (only calculated when total valid attempts ≥ 5)
```

## Gate 2: test → prod

**Purpose**: Prove the skill is reliable and diversely validated.

```
✅ Success count ≥ 4 (≥6 for single-scenario)
✅ Success rate ≥ 70%
✅ Distinct validators ≥ 2 (≥3 for single-scenario)
✅ Evidence chain verifiable
✅ Scenario coverage ≥ 2 (if multi-scenario)
```

## Multi-scenario vs Single-scenario Detection

Three-tier logic:

1. **SKILL.md declares `coverage_dimensions`** → multi-scenario, check against declared dimensions
2. **No declaration but jsonl has ≥2 distinct `scenarios` dicts** → auto-detected multi-scenario
3. **Neither** → single-scenario (stricter: 6 successes + 3 validators)

### coverage_dimensions in SKILL.md frontmatter

```yaml
metadata:
  hermes:
    coverage_dimensions:
      test_mode: [基准压测, 摸高寻优, 纯吞吐爬坡]
      api: [MAAS, vLLM, OpenAI]
```

## Design Rationale

- **Failures don't cancel successes** — they lower the success rate
- **Environment failures excluded** — don't punish skills for infra issues
- **Validator diversity** — prevents self-certification (author + independent validators)
- **Scenario coverage** — prevents single-scenario gaming; auto-detection handles undeclared skills
- **Single-scenario skills** — compensated with higher success count + more validators instead of scenario coverage
