---
name: acs-bench
description: Use when benchmarking LLM API endpoints for throughput/latency, running stress tests on model services, finding optimal concurrency/QPS, or capacity planning for LLM deployments.
version: 3.0.0
author: Hermes Agent
license: MIT
coverage_dimensions:
  - test_mode: [基准压测, 摸高寻优, 纯吞吐爬坡]
  - api: [MAAS, vLLM, OpenAI]
metadata:
  hermes:
    tags: [benchmark, acs-bench, peak-finding, throughput, prof, llm-inference, stress-test]
    related_skills: [serving-llms-vllm, huggingface-hub]
---

# ACS-Bench LLM Serving Benchmark

## Overview

ACS-Bench is a general-purpose LLM inference serving performance benchmarking tool. It supports three operational modes: **baseline stress test** (benchmark mode), **peak-finding optimization** (摸高寻优), and **pure throughput climb** (纯吞吐爬坡). The tool covers the full workflow from provider configuration, dataset generation, stress testing, result analysis, to report delivery.

The benchmark measures key performance indicators including QPS, TTFT (Time To First Token), TPOT (Time Per Output Token), E2E latency, and token throughput under various concurrency levels and context lengths. It supports any OpenAI-compatible API endpoint (MAAS, vLLM, OpenAI, etc.) and provides SLO compliance checking against user-specified latency targets.

## When to Use

- User wants to benchmark an LLM API endpoint for throughput/latency
- User requests stress tests on a model serving endpoint
- User needs to find optimal concurrency or QPS limits
- User provides a benchmark spec with model, endpoint, input lengths, and performance targets
- User wants capacity planning for LLM deployments
- User asks to compare performance across context lengths or concurrency levels

**Don't use for:**
- Functional testing or correctness validation of model outputs
- Single-request latency measurement (use direct API calls instead)
- Training performance benchmarking (inference only)
- Load testing non-LLM HTTP endpoints (use wrk/k6 instead)

## Input Specification

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| 测试模型 | ✅ | - | 模型名称，如 `GLM-5`、`Qwen3-32B` |
| 测试方式 | ✅ | - | API调用方式：`MAAS API调用`、`vLLM本地`、`OpenAI` |
| 测试模式 | ✅ | - | 压测策略：`基准压测`（固定并发测延迟基线）、`摸高寻优`（并发爬升找退化拐点）、`纯吞吐爬坡`（多并发档位画吞吐曲线） |
| 输入长度 | ✅ | - | 上下文token长度列表，如 `90k, 150k, 200k` |
| 输出长度 | ❌ | `100` | 输出token长度 |
| 测试数据集 | ❌ | `随机生成` | 数据集来源（随机生成/真实数据集路径） |
| Tokenizer | ❌ | `Qwen3-32B` | 生成数据集用的tokenizer模型 |
| 测试数据量 | ❌ | `1000` | 每个长度的样本数 |
| 测试精度 | ❌ | - | 精度要求（如FP16/BF16） |
| 并发要求 | ❌ | `单并发` | 如 `单并发`、`1,2,4,8` |
| 测试时间 | ❌ | `立即执行` | 定时执行时间 |
| 测试工具 | ❌ | `acs-bench` | 压测工具 |
| TTFT要求 | ❌ | - | 首token延延迟目标，如 `10s` |
| TPOT要求 | ❌ | - | 每token输出时间目标，如 `33ms` |
| API key | ❌ | 使用openclaw配置 | API密钥 |

**Label命名规则：** 标签从用户输入贯穿全流程，不做二次转换：

```
用户输入: 90k, 150k, 200k
  → 数据集目录: dataset_90k/, dataset_150k/, dataset_200k/
  → 结果tag:   90k_nr1_cc1/, 150k_nr1_cc1/, ...
  → 报告长度列: 90k, 150k, 200k
  → CLI参数:   --input-length 90000  (仅此处解析为数字)
```

## Environment Setup（0-1构建）

> 首次使用时，agent必须先执行环境检查，不可跳过。
> **设计原则：所有路径通过环境变量引用（$PROF_ROOT等），agent从USER profile读取映射解析。本地skill与仓库skill使用同一份$VAR代码，区别仅在USER profile中的映射值。**

**Step 0: 环境检查** — 检查压测工具是否已安装，未安装则自动安装：

1. **环境检测**（优先执行，不可跳过）:
   ```bash
   # 检查 acs-bench 是否已安装
   if command -v acs-bench &>/dev/null || python3 -c "import acs_bench" 2>/dev/null; then
     echo "acs-bench 已安装: $(acs-bench --version 2>/dev/null || python3 -c 'import acs_bench; print(acs_bench.__version__)')"
   else
     echo "acs-bench 未安装，需要安装"
   fi
   ```
   - **已安装** → 要求用户提供以下信息：
     - `PROF_ROOT`: 压测工作根目录（如 `~/prof`）
     - `base_url`: API endpoint地址
     - `api_key`: API密钥
     - `model_name`: 模型名称
     - `tokenizer_path`: tokenizer本地路径（可选，缺省用HuggingFace自动下载）
   - **未安装** → 自动安装（见下方）

2. **自动安装流程**（用户无环境时）：
   ```bash
   # 创建工作目录
   PROF_ROOT=~/prof && mkdir -p $PROF_ROOT/{dataset,results,log,tokenizer,conf}

   # 安装acs-bench: 优先从华为云OBS下载whl，fallback到PyPI
   pip install https://hw-pbclouds.obs.cn-east-4.myhuaweicloud.com/pkg/acs_bench-1.4.1-py3-none-any.whl \
     || pip install acs-bench

   # 下载默认tokenizer（Qwen3-32B）
   huggingface-cli download Qwen/Qwen3-32B tokenizer.json tokenizer_config.json special_tokens_map.json \
     --local-dir $PROF_ROOT/tokenizer/Qwen3-32B

   # 设置fd限制
   ulimit -n 65535
   ```

3. **环境变量设置** — 安装完成后，将路径写入USER profile供后续使用：
   ```
   PROF_ROOT=~/prof
   ```

4. **验证安装**:
   ```bash
   acs-bench --version
   ls $PROF_ROOT/tokenizer/Qwen3-32B/tokenizer.json
   ulimit -n  # 应为65535
   ```

## Prerequisites

- `acs-bench` installed: 优先 `pip install https://hw-pbclouds.obs.cn-east-4.myhuaweicloud.com/pkg/acs_bench-1.4.1-py3-none-any.whl`，fallback `pip install acs-bench`
- Tokenizer files: default `Qwen3-32B` (local path or HuggingFace auto-download `Qwen/Qwen3-32B`)
- API endpoint credentials: `base_url` + `api_key` (from openclaw config, environment, or user input)
- Environment variables: `PROF_ROOT` (set during Environment Setup or from USER profile)

> **重要**: 本地skill也应使用`$PROF_ROOT`等环境变量，而非硬编码绝对路径。运行时从USER profile读取映射值。这样本地和仓库使用同一份skill。

## Workflow — 基准压测（5步法）

> Steps 1–5 must execute sequentially. No step may be skipped. Each step must confirm success before proceeding.

**Step 1: Configure** — Create `providers.yaml` based on 测试方式/测试模型. Provider schema: see `references/providers_schema.md`. Key templates: MAAS→`https://api.modelarts-maas.com/openai/v1`, vLLM→`http://localhost:8000/v1`, OpenAI→`https://api.openai.com/v1`.

**Step 2: Generate** — For each input length label, generate random token dataset:
```bash
acs-bench generate dataset --tokenizer <TOKENIZER> --dataset-type random \
  --output-path <WORKDIR>/dataset_<LABEL> --input-length <RESOLVED_LENGTH> --num-requests <N>
```
Use `nohup` for long runs. Reuse existing datasets but step itself cannot be skipped.

**Step 3: Stress** — Run benchmark matrix per concurrency spec:
```bash
acs-bench prof --provider <PROVIDER> --dataset-type custom --input-path <WORKDIR>/dataset_<LABEL>/ \
  --concurrency-backend threading-pool --backend openai-chat --warmup 1 --epochs 2 \
  --num-requests <NR> --concurrency <CC> --input-length <LEN> --output-length <OUT> \
  --benchmark-save-path <RESULTDIR>/<LABEL>_nr<NR>_cc<CC>/
```
Retry up to 3× with exponential backoff (120s→240s→480s) on failure.

**Step 4: Analyze** — Parse all `summary_*.csv`, generate report with SLO判定:
```bash
python3 scripts/parse_results.py <RESULTDIR> <WORKDIR>/压测报告.md [TTFT_SLO] [TPOT_SLO]
```

**Step 5: Deliver** — Compress → send file → report message (交付是汇报的前置条件):
```bash
python3 -c "import zipfile,os; zf=zipfile.ZipFile('results.zip','w',zipfile.ZIP_DEFLATED); [zf.write(os.path.join(r,f)) for r,d,fs in os.walk('results') for f in fs]; zf.write('压测报告.md'); zf.close()"
# Send archive + summary via configured channel
python3 scripts/deliver_results.py <report.md> <results.zip> [target_id] [channel]
```

> Detailed execution guide: `references/benchmark-execution.md`

## Workflow — 摸高寻优（7步法）

Three-phase iterative optimization to find peak throughput concurrency:

1. **Phase 1 — Baseline**: Run cc=1 baseline at each length, establish TTFT/TPOT/E2E baselines
2. **Phase 2 — Climb**: Increment concurrency using formula `cc_next = cc_prev × 2` (or `cc_prev + step`), run one round, analyze, adjust — **single variable only** per round
3. **Phase 3 — Confirm**: Re-run best concurrency 3× to confirm stability, generate final report

Key rules: single-variable progression (change only concurrency, not length/requests simultaneously); fresh data injection per round (真实数据集场景必需，随机数据集无需); target_e2e uses current tier's own target; analyze-then-adjust, never batch-preset multiple rounds.

> Detailed strategy: `references/peak-finding-strategy.md`

## Workflow — 纯吞吐爬坡

Climb mode iteratively increases concurrency to find maximum throughput before saturation. Differs from 摸高 in that it optimizes purely for throughput (tok/s) without SLO constraints. Key insight: **it/s decline at climb tail ≠ service saturation** — may indicate client-side bottleneck; verify with server-side metrics.

> See `references/peak-finding-strategy.md` §2.5

## Environment & Paths

> 所有路径通过环境变量解析，agent从USER profile读取映射。首次使用需先完成Environment Setup。

| Item | Variable | Default | Notes |
|------|----------|---------|-------|
| 工作根目录 | `$PROF_ROOT` | `~/prof` | 所有数据的父目录 |
| acs-bench | — | `pip install acs-bench` | 优先OBS whl, fallback PyPI |
| Default tokenizer | `$PROF_ROOT/tokenizer/Qwen3-32B/` | — | Fallback: HuggingFace `Qwen/Qwen3-32B` |
| Dataset dir | `$PROF_ROOT/dataset_<LABEL>/` | — | Named by input label |
| Results dir | `$PROF_ROOT/results/` | — | Subdirs: `<LABEL>_nr<NR>_cc<CC>/` |
| Report | `$PROF_ROOT/压测报告.md` | — | Markdown format |
| Archive | `$PROF_ROOT/results.zip` | — | Compressed deliverable (zip format, Feishu compatible) |
| Skill scripts | `<SKILL_DIR>/scripts/` | — | full_bench.sh, run_stress.sh, etc. |
| Provider config | `$PROF_ROOT/conf/<provider>.yaml` | — | User-specific, not in repo |
| ModelArts workdir | `$DATA_ROOT/acs-bench/` | `$DATA_ROOT/acs-bench` | (仅ModelArts) |
| ModelArts tokenizer | `$DATA_ROOT/tokenizer/` | `$DATA_ROOT/tokenizer` | (仅ModelArts) |

## Scripts Quick Reference

| Task | Command |
|------|---------|
| One-click full run | `bash scripts/full_bench.sh <workdir> [tokenizer] <provider.yaml> "90k 150k 200k"` |
| Generate datasets | `bash scripts/generate_datasets.sh <workdir> [tokenizer] 90k 150k 200k` |
| Run stress tests | `bash scripts/run_stress.sh <workdir> <provider.yaml> <results> "90k 150k" "1 2 4" "1 2 4"` |
| Parse results | `python3 scripts/parse_results.py <results_dir> <report.md> [ttft_slo] [tpot_slo]` |
| Deliver results | `python3 scripts/deliver_results.py <report.md> <results.zip> [target_id] [channel]` |
| Compress | `python3 -c "import zipfile,os; zf=zipfile.ZipFile('results.zip','w',zipfile.ZIP_DEFLATED); [zf.write(os.path.join(r,f)) for r,d,fs in os.walk('results') for f in fs]; zf.write('压测报告.md'); zf.close()"` |
| Check dataset progress | `grep -oP '\d+(?=/1000)' <log> \| tail -1` |
| Check bench progress | `grep -c "完成 ✅" <log>` |
| Find failures | `grep "失败" <log>` |

## Key Metrics

| Metric | Unit | Definition |
|--------|------|------------|
| QPS | req/s | Queries per second |
| TTFT | s | Time To First Token (prefill latency) |
| TPOT | s | Time Per Output Token (decode speed) |
| E2E | s | End-to-end latency = TTFT + TPOT × output_tokens |
| Throughput | tok/s | Total token throughput (input+output) |
| Fail_Rate | 0~1 | Request failure rate (rate limit / timeout) |

**SLO判定:** TTFT ≤ TTFT要求 → ✅; TPOT ≤ TPOT要求 → ✅. Report includes pass/fail per group and overall SLO达标率.

> Full metrics reference: `references/metrics-reference.md`
> Provider schema: `references/provider-schema.md`
> 基准压测详细规范: `references/benchmark-execution.md`
> 摸高寻优详细规范: `references/peak-finding-strategy.md`
> 全流程SOP: `references/workflow-sop.md`
> 长上下文90k压测实录: `references/long-context-90k-benchmark-session-20260604.md`
> 常见陷阱速查: `references/common-pitfalls.md`
> 场景与QPS参考(仅ModelArts): `references/scenario-reference.md`
> 数据管道: `references/benchmark-data-pipeline.md`
> CSV字段映射: `references/benchmark-summary-csv-field-mapping.md`
> DeepSeek官方API: `references/benchmark-deepseek-official-api-provider.md`
> 脚本层级: `references/workflow-script-hierarchy.md`
> 输出列规格: `references/workflow-output-columns-spec.md`
> 数据集顺序: `references/benchmark-dataset-order-analysis.md`
> 数据准备: `references/benchmark-data-preparation.md`
> vllm-bench参数映射: `references/vllm-bench-to-acs-bench-mapping.md`
> 整合变更日志: `references/integration-changelog.md`
> Gate评价体系(v2): `references/gate-evaluation-system.md`

## Execution Rules

1. **串行执行**: Steps run sequentially, no parallel execution within a workflow — 无例外
2. **确认前置**: Each step must confirm success before proceeding — 无例外
3. **数据注入**: Fresh round data per iteration (真实数据集场景必需，随机数据集无需) — 无例外
4. **CSV校验**: Validate CSV integrity before parsing — 无例外
5. **单变量递进**: Change only one variable (concurrency or rate, not both) per round in 摸高 — 无例外
6. **No parallel per provider**: Never run concurrent stress tests against same provider — 无例外
7. **fd limit**: Set `ulimit -n 65535` before stress tests to avoid fd exhaustion
8. **Retry**: 3× exponential backoff (120s→240s→480s) on failure
9. **交付前置**: File delivery must succeed before sending report message — 无例外
10. **摸高完成后必须执行报告输出与回传** — 无例外
11. **target_e2e必须用当前档位自身目标** — 无例外
12. **逐轮分析后调参，禁止批量预设多轮** — 无例外

## Gate Evaluation System

The skill lifecycle gate system (dev→test→prod) uses success rate, validator diversity, and scenario coverage — not raw usage count. See `references/gate-evaluation-system.md` for the full specification.

## References

- [Gate 评价体系 v2](references/gate-evaluation-system.md) — 多维门控设计文档（成功次数 × 成功率 × 验证者 × 场景覆盖）

## Long-Context Benchmarking (90k+ tokens)

> Long-context (90k+) benchmarks have distinct failure modes and tuning requirements compared to short-context tests. This section captures proven patterns.

### Key Findings (GLM-5.1 on MaaS, 2026-06-04)

| Finding | Detail |
|---------|--------|
| **MaaS rate limit (429) misreported as TimeoutError** | 90k requests with nr=5 → 70% "TimeoutError". Root cause: MaaS returns 429, acs-bench retries 3× then gives up and labels it TimeoutError. Increasing timeout does NOT fix it. Fix: reduce nr to 3, warmup=0, epochs=1 |
| **TPOT stable across concurrency** | TPOT ≈ 60ms/token at all cc levels (1–16). Decode speed is concurrency-independent |
| **TTFT decreases with concurrency** | cc=1→8: TTFT drops 8.4s→7.0s. Likely MaaS server-side batch prefill acceleration |
| **Throughput marginal diminishing** | cc 1→2: +93%, 2→4: +49%, 4→8: +12.7%, 8→16: +6.6%. Inflection at cc=8 |
| **Recommended cc for 90k** | cc=8, peak throughput ~25 tok/s, TTFT ~7s, 0% fail rate |

### Concurrency Climb Methodology (Pure Throughput)

For finding throughput inflection point without SLO constraints:

1. **Start at cc=1**: Establish baseline TTFT/TPOT/E2E/throughput
2. **Double concurrency**: cc=1→2→4→8→16, run one round each
3. **Calculate marginal gain**: If gain < 10% from previous cc, inflection reached
4. **Confirm**: Re-run inflection cc 3× for stability (QPS ±5%, E2E ±10%)
5. **Report**: Include throughput-vs-cc curve with bar chart

**Parameter tuning for long-context stability:**
- `--timeout 900` (not 300s — 90k requests need much longer)
- `--num-requests 3` (not 5+ — MaaS queues and drops excess)
- `--warmup 0 --epochs 1` (minimize total requests to avoid queueing)
- `ulimit -n 65535` (always, before any stress test)

> Detailed session log: `references/long-context-90k-benchmark-session-20260604.md`

## Common Pitfalls

1. **Long-context (90k+) MaaS rate limit misreported as TimeoutError**: nr=5 at 90k → 70% "TimeoutError". **Root cause is MaaS 429 rate limiting**, not client timeout. acs-bench catches the 429, retries 3× with backoff (`Rate limited. Waiting Xs (attempt N/3)...`), then gives up and labels it TimeoutError — misleading. Increasing timeout does NOT fix it. Fix: reduce nr to 3, warmup=0, epochs=1 to stay under MaaS rate limit window (~50s for 90k requests).
2. **Wrong AGENT_HOME when installing skill**: `AGENT_HOME` is `/opt/data` (NOT `/opt/data/home/.hermes`). Skills must be installed to `${AGENT_HOME}/skills/` = `/opt/data/skills/`. Installing to `/opt/data/home/.hermes/skills/` makes files invisible to `skill_view`/`skill_manage`/`skills_list`. The agent-skills initialization flow is: `skills0-dev/` → `cp -r` to `/opt/data/skills/<category>/`. Do NOT create alternative skill directories or symlinks without explicit user approval.
2. **Feishu Media cannot send .tar.gz**: Hermes's Feishu `send_message` with `MEDIA:` prefix supports images (jpg/png/webp), audio, and .zip files. It does NOT support .tar.gz — the API returns success but the file is not delivered. Always use .zip format for Step 5 (Deliver) when targeting Feishu channels.
3. **Feishu send_message requires explicit chat_id**: When using `send_message` to Feishu, the target must be `feishu:<chat_id>` (e.g. `feishu:oc_aa8152a155bd18917ab6aa751869cc34`), NOT just `feishu`. Without a configured home channel, bare `feishu` target will fail.
4. **Forgetting ulimit -n**: fd exhaustion at high concurrency → mysterious failures
5. **HuggingFace unreachable in container**: Containers often lack outbound HTTPS to huggingface.co. Always use a **local tokenizer path** with `--tokenizer <local_path> --trust-remote-code`. Fallback: use HF cache snapshot path like `$HF_HOME/hub/models--THUDM--glm-4-9b-chat/snapshots/<hash>/`. Never rely on `Qwen/Qwen3-32B` remote download inside containers without confirming network.
6. **acs-bench prof requires --input-path**: Unlike `generate dataset` (which has `--dataset-type random`), `prof` only supports `[custom|LongBench|CustomOpenAIChat|ShareGPT|...]` — there is no `random` mode. You **must** run `generate dataset` first, then pass the output JSON via `--input-path`. Cannot skip Step 2 (Generate).
7. **90k+ dataset generation is slow**: Generating 90k-token random datasets with GLM tokenizer takes ~7s per sample. 1000 samples ≈ 115 min. For quick validation or 摸高寻优, start with nr=5–10 to run the full workflow end-to-end, then scale up. Do NOT attempt nr=1000 at 90k without `nohup` and progress monitoring.
8. **Overriding user-specified defaults**: When user says "其他默认" or "use defaults", use the SKILL.md documented defaults (e.g. nr=1000). Do NOT silently reduce parameters for speed. If defaults are impractical for the context, ask the user — do not decide for them.
3. **Self-directed file/directory creation**: Do NOT create files, directories, or symlinks without explicit user approval. Ask first, act second.
2. **Parallel tests on same endpoint**: Skews results, triggers rate limits prematurely
3. **Reusing stale datasets across rounds**: Injects bias (真实数据集场景)
4. **Skipping CSV validation**: Corrupt CSV → silent wrong analysis
5. **Batch-presetting multiple rounds**: Must analyze-then-adjust, one round at a time
6. **Misreading it/s decline as saturation**: Could be client-side bottleneck, check server metrics
7. **Using wrong target_e2e**: Must use current tier's own target, not global target
8. **Skipping delivery step**: Report + archive delivery is mandatory, not optional
9. **Tokenizer mismatch**: Dataset token count ≠ actual prompt length if wrong tokenizer
10. **Label inconsistency**: 90k must stay 90k everywhere, only resolve to 90000 at CLI --input-length
11. **Hardcoded absolute paths**: Use $PROF_ROOT etc. everywhere, resolve from USER profile at runtime. Local and repo skill share the same $VAR code — no "local original + repo desensitized" dual maintenance

## Usage Hook

> 本skill每次成功执行后，必须记录使用日志。这是强制步骤，不可跳过。

**执行时机：** skill指导的任务成功完成后（非加载/浏览时）

**记录命令（成功）：**
```bash
echo '{"skill":"acs-bench","version":"3.0.0","stage":"<dev|test|prod>","author":"$(git config user.name)","agent":"hermes","session_id":"$HERMES_SESSION_ID","action":"use","result":"success","failure_type":null,"scenarios":{"test_mode":"<基准压测|摸高寻优|纯吞吐爬坡>","api":"<MAAS|vLLM|OpenAI>"},"evidence":{"report_exists":true,"csv_valid":true,"fail_rate":"<0%>"},"timestamp":"$(date -Iseconds)"}' >> ~/.hermes/logs/skill-usage.jsonl
```

**记录命令（失败）：**
```bash
echo '{"skill":"acs-bench","version":"3.0.0","stage":"<dev|test|prod>","author":"$(git config user.name)","agent":"hermes","session_id":"$HERMES_SESSION_ID","action":"use","result":"fail","failure_type":"<skill_defect|environment>","scenarios":{"test_mode":"<基准压测|摸高寻优|纯吞吐爬坡>","api":"<MAAS|vLLM|OpenAI>"},"evidence":null,"timestamp":"$(date -Iseconds)"}' >> ~/.hermes/logs/skill-usage.jsonl
```

**failure_type 分类：**
- `skill_defect`：workflow 步骤报错、CSV 校验失败、脚本 bug、输出格式错误
- `environment`：API 限流、网络超时、tokenizer 下载失败、服务不可达
- `null`：成功时不需要

**agent适配说明：**
- Hermes：直接执行上述命令
- OpenClaw/Claude Code/Codex：用各自shell工具执行等价命令
- 人工使用：手动执行或由agent代记录
- 路径 `~/.hermes/logs/` 为Hermes默认；其他agent需适配为 `<agent-home>/logs/`

**禁止事项：**
- 禁止跳过使用记录（agent-skill-commit Gate 3会检测）
- 禁止仅记录load不记录use（加载≠使用）
- 禁止依赖agent专属统计能力，必须通过本Hook写Log

---

## Verification Checklist (5-Step Benchmark)

- [ ] Provider YAML: endpoint reachable, model_name valid, tokenizer downloaded
- [ ] Dataset: `--prefix-mode shared` injection applied (真实数据集场景)
- [ ] Execution: no parallel stress tests on same endpoint
- [ ] CSV: `validate_csv_report.py` passed every round
- [ ] Delivery: zip+send succeeded before report message
- [ ] Report: 汇总对比表 + 5项分析结论 (TTFT/TPOT SLO判定)

## Verification Checklist (7-Step 摸高寻优)

- [ ] Round 0 baseline: c=1, r=5, single-variable only
- [ ] 每轮: validate_csv_report.py通过, Label含90k标识
- [ ] 每轮: target_e2e用当前档位非全局最宽松
- [ ] 每轮: 分析QPS/并发趋势后决定下轮参数(禁止预设)
- [ ] 甜点确认: 3轮验证QPS稳定(偏差<5%)
- [ ] 交付: zip+send → 报告输出与回传
- [ ] 无合理化借口: 对照红旗清单逐条排查

## Verification Checklist (General)

- [ ] `providers.yaml` created with correct base_url and api_key
- [ ] Tokenizer accessible (local path or HuggingFace download)
- [ ] All datasets generated: `dataset_<LABEL>/` exists for each input length
- [ ] Stress tests completed for all (label, nr, cc) combinations
- [ ] `summary_*.csv` files present in all result subdirectories
- [ ] CSV files valid (non-empty, correct headers, no corruption)
- [ ] Report generated: `压测报告.md` exists and is non-empty
- [ ] SLO判定 included (if TTFT/TPOT requirements specified)
- [ ] Archive created: `results.zip` exists and size > 0
- [ ] Delivery completed: file sent to target channel
- [ ] Report message sent with key findings summary
- [ ] ulimit -n set appropriately before stress tests
- [ ] No parallel tests ran against same provider endpoint

## Gates

- **check_single_variable**: Only one parameter changed between consecutive rounds
- **check_incremental_only**: Concurrency increases monotonically in 摸高/climb
- **check_fresh_data**: New round data injected (真实数据集场景) or confirmed reusable (随机数据集)
- **check_csv_validation**: CSV validated before any parsing
- **check_no_parallel**: No concurrent stress tests against same provider
- **check_tdd_cycle**: Skill modifications verified via TDD before deployment
- **check_report_output**: Report generated and delivered after 摸高 completion
- **check_target_e2e_per_tier**: target_e2e uses current tier's own target value

## Assertions

- **single_variable_changed**: Between any two rounds, exactly one parameter differs
- **one_round_at_a_time**: No batch-presetting of multiple rounds; analyze then adjust
- **round_data_fresh**: Each round uses freshly injected or confirmed-valid data
- **csv_validated**: All summary CSVs validated before parsing
- **no_parallel_processes**: No concurrent acs-bench prof against same provider
- **tdd_cycle_completed**: Any skill change passed TDD verification
- **report_generated**: Final report exists and is non-empty
- **target_e2e_correct_tier**: target_e2e value matches current tier, not a different tier

## Constraints

- Maximum 3 retries per stress test group with exponential backoff
- Steps execute strictly sequentially — no step may be skipped
- Delivery (5b) must succeed before report message (5c)
- Same provider endpoint: never parallel stress tests
- 摸高: single-variable progression only
- climb: it/s decline ≠ saturation (must verify server-side)
- Skill modifications require TDD verification cycle

## 反合理化（Rationalization Table + Red Flags）

### 合理化借口表

| 借口 | 事实 |
|------|------|
| "数据一样省掉注入" | 不注入=KV缓存命中=e2e虚低0.1~0.2s=结果作废必须重跑 |
| "缓存命中场景不需要注入" | 缓存命中场景用`--prefix-mode shared`注入，不是不注入 |
| "我直接看CSV数字就行" | 肉眼无法检测列偏移、编码错误、缺失行 |
| "上轮校验过了这轮应该没问题" | 每轮数据不同，必须每轮校验 |
| "并行跑更快结果差不多" | 同provider并行争抢连接池/服务端资源，数据不可信 |
| "同时调c和r加速收敛" | 双变量无法归因，过度并发致排队恶化，偏差15% |
| "QPS下降了就是拐点"（climb） | climb末尾it/s下降=请求数耗尽≠服务饱和 |
| "用户只看数字/下次补报告" | 甜点确认后必须立即执行报告输出与回传 |
| "用全局最宽松档更保守" | 低档e2e目标远小于高档，公式ceil(r×e2e)严重高估并发 |
| "预设多轮参数更高效" | 每轮结果影响下轮参数方向，QPS-并发非线性，预设无法适应 |
| "不用跑验证"/"赶时间"/"改动很小" | skill-authoring铁律：先修后测=反模式 |

### 红旗清单 — 出现以下任何一条立即停止纠正

- "这轮数据和上轮一样，省掉注入"
- "并行跑更快，结果差不多"
- "我直接看CSV数字就行，不用跑validate"
- "同时调c和r加速收敛"
- "QPS开始下降了，这就是拐点"（climb模式）
- "报告等下次再补"
- "用全局最宽松档更保守/差不多"
- "预设多轮参数更高效"
- "不用跑验证"/"赶时间"/"改动很小"（修skill相关）

**以上任何一条出现 → 结果不可信/修复无效，必须按正确流程重来。**
