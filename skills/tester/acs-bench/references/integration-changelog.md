# ACS-Bench Skill Integration Changelog

## v3.0.0 — 2026-05-26

### Source
- Generic version: `$PROF_ROOT/work/code/agent-skills/acs-bench/` (5-step benchmark workflow, delivery pipeline, SLO判定, retry mechanism)
- Agent version: `~/.hermes/skills/mlops/acs-bench/` (7-step 摸高寻优, anti-rationalization, KV cache injection, CSV validation, ModelArts-specific paths)

### Integration Decisions
1. **通用优先，专用下沉**: SKILL.md写通用流程(MAAS/vLLM/OpenAI)，ModelArts特定配置放references/scenario-reference.md
2. **双模式共存**: 基准压测5步法 + 摸高寻优7步法并列
3. **交付闭环引入**: tar+send交付流程(旧版缺失)
4. **SLO判定引入**: TTFT/TPOT目标达标判定(旧版无此能力)
5. **重试机制引入**: 3次指数退避(120s→240s→480s)(旧版无重试)
6. **反合理化继承**: 旧版9条核心纪律+10处"无例外"标记+合理化借口表+红旗清单
7. **默认tokenizer改为Qwen3-32B**: 通用化，非ModelArts专用
8. **Label命名采用新版体系**: 90k贯穿全流程，仅CLI --input-length解析为90000

### TDD Verification
- RED: 无skill下agent遵守核心纪律(串行/注入/校验/报告)，拒绝用户压力捷径 ✅
- GREEN: 有skill下agent正确回答6项关键规范 ✅

### Backup
- Old version backed up to: `~/.hermes/skills/mlops/acs-bench.bak/`
