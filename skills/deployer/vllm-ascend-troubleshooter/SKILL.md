---
name: vllm-ascend-troubleshooter
description: Troubleshoot vLLM-Ascend deployment, model loading, inference API, CANN/torch-npu, Ascend NPU visibility, ACLGraph/eager fallback, quantization, tensor parallel, multimodal, KV cache, memory, and performance issues. Use when users provide startup commands, logs, model paths, environment info, npu-smi output, stack traces, or failed validation results for Ascend NPU inference.
---

# vLLM-Ascend Troubleshooter

## TL;DR

- 先收集日志、启动命令、模型路径和环境信息；信息不足时使用 `scripts/collect_vllm_ascend_env.py` 采集
- 再分类故障：环境、启动、加载、推理、算子、量化、多模态、性能
- 故障分类时读取 `references/error-patterns.md` 匹配已知模式
- 选择恢复路径时读取 `references/fallback-matrix.md` 按序降级
- 设计验证方案时读取 `references/validation-checklist.md` 逐项过关
- 高风险或状态变更动作前必须暂停确认（PAUSE FOR USER CONFIRMATION）
- 最终按 `references/report-template.md` 输出中文排障报告
- 自测触发效果时参考 `references/test-prompts.md` 中的正负样本

## Scope

本 Skill 覆盖以下故障场景：

1. **服务启动失败**: vllm serve / vllm.entrypoints 启动报错、进程崩溃、端口占用
2. **模型加载失败**: 权重加载错误、safetensors 解析失败、OOM、dtype 不匹配
3. **推理请求失败**: API 返回 500/503、超时、OOM、输出乱码、空输出
4. **环境异常**: torch_npu 导入失败、CANN 版本不匹配、NPU 驱动异常、npu-smi 报错
5. **算子不兼容**: ACL 报错、算子不支持、fallback 失败、自定义算子注册问题
6. **精度异常**: 输出与 GPU 差异过大、NaN/Inf、loss 异常
7. **性能异常**: 吞吐低于预期、延迟过高、NPU 利用率低、host-bound
8. **高级特性问题**: ACLGraph 模式异常、eager mode 回退、tensor parallel 通信失败、KV cache 分配失败、量化模型 (AWQ/GPTQ/FP8) 加载异常
9. **多模态问题**: image / video 输入失败、processor 加载异常

## Directory And Resource Map

| Path | Purpose | Load When | Required |
|---|---|---|---|
| `SKILL.md` | 主 Skill 定义文件，包含工作流、检查点、边界条件 | 始终加载 | Yes |
| `references/error-patterns.md` | 已知错误模式库：14 类故障的症状、根因、检查、修复、fallback | Phase 2 故障分类时 | Yes |
| `references/fallback-matrix.md` | Fallback 恢复策略矩阵：10 类场景的有序降级路径 | Phase 4 Fallback 隔离时 | Yes |
| `references/validation-checklist.md` | 环境与功能验证清单：11 项检查的命令、标准、失败处理 | Phase 1 环境基线 + Phase 6 验证时 | Yes |
| `references/resource-manifest.md` | 资源清单：内部文件索引与外部官方资源链接 | 确认资源完整性时 | Yes |
| `references/report-template.md` | 中文排障报告模板：12 节结构化报告 | Phase 6 输出报告时 | Yes |
| `references/test-prompts.md` | 测试提示词：正负向触发样本与评测指标 | 自测 skill 触发效果时 | No |
| `scripts/collect_vllm_ascend_env.py` | 只读环境采集脚本：输出 JSON 格式的环境与版本信息 | 缺少环境信息时 | No |

## Required Inputs

### Minimum Inputs

至少需要以下信息中的一项才能开始排障：

- 完整的错误日志（stderr/stdout）
- 启动命令（完整的 vllm serve 命令及所有参数）
- 复现步骤描述

### Optional Inputs

以下信息可显著提升排障准确性：

- 模型名称/路径（本地路径或 HuggingFace model ID）
- 环境信息（NPU 型号、CANN 版本、torch_npu 版本、vLLM 版本）
- `npu-smi info` 输出
- 请求 payload（推理报错时）
- 多机多卡拓扑（分布式场景）
- 量化配置文件（量化模型场景）
- 模型 config.json 内容
- HCCL 日志（多卡通信问题时）

### Sensitive Inputs Never Requested

以下敏感信息不得要求用户提供，不得读取，不得出现在报告中：

- `.env` 文件内容
- token / API key / secret
- SSH 私钥
- `~/.ssh` 目录内容
- 云凭据（AWS Access Key、Azure SAS、GCP Service Account 等）
- 生产数据库密码
- 私有证书（TLS/SSL 私钥）
- 内部账号密码

## Expected Outputs

最终输出为中文排障报告，包含以下内容：

1. **中文结论摘要** — 一段话概括问题、根因和推荐修复方案
2. **用户提供信息摘要** — 用户已提供和缺失的关键信息列表
3. **环境与版本判断** — CANN / torch_npu / vLLM / vLLM-Ascend 版本兼容性结论
4. **故障分类** — 匹配的故障类别（环境/启动/加载/推理/算子/量化/多模态/性能）
5. **根因假设** — 明确的根因推导，不模棱两可
6. **证据链** — 从日志到根因的完整推理路径
7. **P0 / P1 / P2 风险等级** — 严重程度评估
8. **Fallback 路径** — 按优先级排列的降级恢复方案
9. **建议修复步骤** — 具体可执行的命令和操作
10. **验证命令** — 修复后验证问题已解决的命令和预期输出
11. **需要用户补充的信息** — 如果信息不足，列出需要补充的项
12. **用户确认记录** — 暂停确认点的用户决策记录

## Workflow

### Phase 0: Intake And Safety Screening

**Inputs**
- 用户原始消息

**Actions**
1. 检查用户是否提供了日志、启动命令、模型路径、环境信息中的至少一项
2. 扫描用户输入中是否包含敏感凭据（token、key、密码），如有则警告并要求脱敏
3. 判断问题是否属于 vLLM-Ascend 范畴，如不属于则引导用户到正确渠道
4. 记录用户已提供的信息清单和缺失信息清单

**Outputs**
- 信息完整性评估：sufficient / partial / insufficient
- 缺失信息列表
- 安全筛查结果

**Checkpoint**
- C0: 是否至少有一项关键输入（日志/命令/路径/环境）？如缺失，要求用户提供后再继续

### Phase 1: Environment Baseline

**Inputs**
- 用户提供的环境信息或采集脚本输出

**Actions**
1. 如用户未提供环境信息，建议运行 `scripts/collect_vllm_ascend_env.py` 收集
2. 读取 `references/validation-checklist.md`，执行 G0-G3 检查项
3. 核对 CANN × torch_npu × PyTorch × vLLM-Ascend 版本兼容性矩阵
4. 检查 NPU 可见性：`npu-smi info` 输出、`torch.npu.is_available()` 结果
5. 检查关键环境变量：ASCEND_HOME_PATH、LD_LIBRARY_PATH、PYTORCH_NPU_ALLOC_CONF

**Outputs**
- 环境基线报告（版本兼容性、NPU 状态、环境变量状态）
- 环境是否通过基线检查的结论

**Checkpoint**
- C1: 版本兼容性是否通过？NPU 是否可见？如不通过，标记为 P0 并进入 Phase 4

### Phase 2: Failure Classification

**Inputs**
- 用户提供的错误日志和复现步骤
- Phase 1 环境基线结论

**Actions**
1. 读取 `references/error-patterns.md`
2. 将错误日志与已知模式进行匹配：
   - 精确匹配：错误消息完全一致
   - 关键词匹配：提取错误中的关键算子名、异常类型、错误码
   - 场景匹配：根据故障现象（启动/加载/推理/性能）缩小范围
3. 确定故障分类：环境 / 启动 / 加载 / 推理 / 算子 / 量化 / 多模态 / 性能
4. 如匹配到已知模式，记录匹配的模式 ID

**Outputs**
- 故障分类结果
- 匹配的错误模式 ID（如 E-ENV-002、E-OP-002 等）
- 是否需要进入深度分析

**Checkpoint**
- C2: 是否匹配到已知错误模式？如匹配，直接跳到 Phase 5；如未匹配，继续 Phase 3

### Phase 3: Minimal Reproduction

**Inputs**
- 用户提供的启动命令和模型路径
- Phase 2 的故障分类

**Actions**
1. 构造最小复现命令：使用 `--enforce-eager`、单卡、最小参数集
2. 如适用，使用 dummy weights 验证模型架构是否可加载
3. 如适用，使用最小 prompt 验证 API 是否可用
4. 记录复现结果和新的错误信息

**Outputs**
- 最小复现命令
- 复现结果（成功/失败 + 错误日志）
- 根因缩小范围

**Checkpoint**
- C3: 最小复现是否成功定位问题？如已定位，跳到 Phase 5；如未定位，进入 Phase 4

### Phase 4: Fallback Isolation

**Inputs**
- Phase 3 的复现结果
- `references/fallback-matrix.md`

**Actions**
1. 读取 `references/fallback-matrix.md`，选择对应场景的降级路径
2. 按 ordered fallback ladder 逐级尝试：
   - 切换 `--enforce-eager`
   - 降低 `--max-model-len`
   - 降低 `--max-num-seqs`
   - 单卡复现
   - dummy load 验证
   - real weight load 验证
   - 最小 prompt API 验证
3. 记录每一级的结果，确定问题在哪一级消失

**Outputs**
- Fallback 逐级结果
- 问题消失的 fallback 级别
- 根因定位

**Checkpoint**
- C4: Fallback 隔离是否找到 workaround？如找到，进入 Phase 5 推荐修复；如未找到，升级为阻塞问题

### Phase 5: Fix Recommendation

**Inputs**
- Phase 2-4 的分析结果
- 匹配的错误模式和推荐方案

**Actions**
1. 按优先级排序解决方案：最可能解决 > 最容易实施 > 最安全
2. 每个方案包含：具体命令、预期效果、风险等级
3. 对于高风险方案（修改环境变量、升级依赖、改动版本），标记需要用户确认
4. 如涉及多卡问题，检查拓扑、rank、HCCL 日志
5. 如涉及性能问题，确认用户是否有基线指标

**Outputs**
- 按优先级排序的解决方案列表
- 每个方案的风险评估
- 需要用户确认的高风险操作列表

**Checkpoint**
- C5: 推荐方案是否涉及高风险操作（安装/升级依赖、修改环境变量、写入文件）？如涉及，必须 PAUSE FOR USER CONFIRMATION

### Phase 6: Validation And Report

**Inputs**
- 用户选择并执行的修复方案
- `references/validation-checklist.md`
- `references/report-template.md`

**Actions**
1. 读取 `references/validation-checklist.md`，设计验证方案
2. 按验证门禁逐项确认（G0-G11）
3. 读取 `references/report-template.md`，生成中文排障报告
4. 记录检查点和用户确认历史

**Outputs**
- 验证结果（每项 PASS/FAIL）
- 完整的中文排障报告
- 仍需用户补充的信息列表

**Checkpoint**
- C6: 所有验证门禁是否通过？如全部通过，报告完成；如有 FAIL 项，回到对应 Phase 重新排查

## Checkpoints And Pause-Confirm Mode

PAUSE FOR USER CONFIRMATION

在任何高风险或会改变系统状态的动作前，必须停止并要求用户确认。

### 需要暂停确认的动作

以下操作执行前必须暂停，等待用户明确确认：

- 安装或升级依赖（pip install / pip install --upgrade）
- 修改系统环境变量（export / .bashrc / .profile）
- 写入项目文件（修改 config.json、创建新文件）
- 删除文件（rm、清理缓存）
- 重启服务（systemctl restart、kill + restart）
- 杀进程（kill、pkill）
- 下载大模型（huggingface-hub download、modelscope download）
- 运行长时间 benchmark（vllm bench、压力测试）
- 远程 SSH 执行命令（ssh user@host "..."）
- 修改生产配置（线上服务配置、K8s ConfigMap）
- 改动 CANN / driver / torch-npu / vLLM / vLLM-Ascend 版本

### 不需要暂停确认的动作

以下只读操作可直接执行：

- 阅读用户提供的日志
- 分析启动命令
- 运行只读环境采集脚本（`scripts/collect_vllm_ascend_env.py`）
- 给出建议命令但不执行
- 生成排障报告
- 读取 `references/` 下的资源文件

### 检查点清单

| ID | Phase | Checkpoint Question | If Fail |
|---|---|---|---|
| C0 | Phase 0 | 是否至少有一项关键输入（日志/命令/路径/环境信息）？ | 要求用户提供后再继续 |
| C1 | Phase 1 | 版本兼容性是否通过？NPU 是否可见？ | 标记 P0，进入 Phase 4 |
| C2 | Phase 2 | 是否匹配到已知错误模式？ | 进入 Phase 3 深度分析 |
| C3 | Phase 3 | 最小复现是否成功定位问题？ | 进入 Phase 4 Fallback 隔离 |
| C4 | Phase 4 | Fallback 隔离是否找到 workaround？ | 升级为阻塞问题，输出 issue 材料 |
| C5 | Phase 5 | 推荐方案是否涉及高风险操作？ | PAUSE FOR USER CONFIRMATION |
| C6 | Phase 6 | 所有验证门禁是否通过？ | 回到对应 Phase 重新排查 |

## Failure Severity

| Level | Meaning | Examples | Required Action |
|---|---|---|---|
| P0 | 完全阻塞，服务无法运行 | NPU 不可见、CANN/torch_npu 无法 import、服务完全无法启动、CUDA-only 算子无 fallback | 立即处理，优先恢复基础环境 |
| P1 | 功能受损，核心路径异常 | real weight load 失败、graph mode 失败但 eager 可用、多卡失败、量化失败 | 尽快处理，提供 workaround |
| P2 | 体验问题，非阻塞 | 性能低于预期、日志噪音、参数不优、需要进一步基线数据 | 建议优化，不阻塞使用 |

## Boundary Conditions And Early Exit

| Condition | Why It Blocks | Required Output | Next User Action |
|---|---|---|---|
| 用户未提供日志、命令、模型路径、环境信息中的任何一项 | 无法开始排障 | 明确说明需要哪些信息 | 用户补充至少一项关键信息 |
| 用户要求读取敏感凭据（.env、token、SSH key、云凭据） | 安全边界 | 拒绝读取并说明原因 | 用户脱敏后重新提供 |
| NPU 不可见（npu-smi info 失败） | 基础环境不可用 | 输出 P0 结论 + 驱动/权限检查指引 | 用户检查 NPU 驱动和权限 |
| CANN / torch_npu 基础环境无法 import | 推理框架不可用 | 输出 P0 结论 + 版本兼容性矩阵 | 用户重新安装正确版本组合 |
| CUDA-only 算子无 torch / NPU fallback | 算子不兼容 | 输出不兼容算子列表 + 模型架构限制说明 | 用户换模型或等社区适配 |
| 用户要求直接修改生产环境但未确认 | 高风险操作 | PAUSE FOR USER CONFIRMATION | 用户明确确认风险后继续 |
| 用户只提供性能现象但没有基线指标 | 无法量化评估 | 要求提供基线指标（吞吐、延迟、NPU 利用率） | 用户运行 benchmark 获取基线 |
| 下载大模型或运行长 benchmark 但未确认 | 时间和资源消耗 | PAUSE FOR USER CONFIRMATION | 用户确认后继续 |
| 多机多卡问题缺少拓扑、rank、HCCL 日志 | 分布式问题无法定位 | 要求提供拓扑信息和 HCCL 日志 | 用户收集分布式信息 |

## Validation Gates

| Gate | Purpose | Command Or Evidence | Pass Criteria | If Failed |
|---|---|---|---|---|
| G0 环境信息完整性 | 确认收集到足够环境信息 | 用户提供或 `scripts/collect_vllm_ascend_env.py` 输出 | 包含 OS、Python、NPU、CANN、torch、torch_npu、vLLM 版本 | 要求用户补充或运行采集脚本 |
| G1 NPU 可见性 | 确认 NPU 设备可用 | `npu-smi info` | 设备状态 OK，数量正确 | 检查驱动、权限、设备文件 |
| G2 Python 包 import | 确认基础依赖可导入 | `python -c "import torch; import torch_npu; print(torch.npu.is_available())"` | `torch.npu.is_available()` 返回 True | 按 E-ENV-001 排查 |
| G3 vLLM-Ascend import | 确认 vLLM-Ascend 可用 | `python -c "import vllm; print(vllm.__version__)"` | 成功导入，版本正确 | 按 E-ENV-004 排查 |
| G4 dummy load | 验证模型架构可加载 | `vllm serve <MODEL_PATH> --load-format dummy --max-model-len 64 --device npu` | 服务启动无报错 | 检查模型架构支持、dtype |
| G5 real weight load | 验证真实权重可加载 | `vllm serve <MODEL_PATH> --max-model-len 64 --device npu` | 服务启动无报错 | 检查权重完整性、OOM |
| G6 /v1/models | 验证 API models 端点 | `curl http://localhost:<PORT>/v1/models` | 返回模型列表 JSON | 检查服务状态、端口 |
| G7 /v1/chat/completions | 验证推理 API | `curl http://localhost:<PORT>/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<MODEL>","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'` | 返回正常生成结果 | 按 E-INFER 系列排查 |
| G8 eager vs graph 对比 | 验证 ACLGraph 模式 | 分别用 `--enforce-eager` 和默认模式启动，对比推理结果 | 两种模式均正常，输出一致 | 按 E-OP-002 排查 |
| G9 tensor parallel sanity | 验证多卡通信 | `vllm serve <MODEL_PATH> --tensor-parallel-size <TP_SIZE> --device npu` | 服务启动无 HCCL 错误 | 按 E-DIST 系列排查 |
| G10 performance sanity | 验证基本性能 | `vllm bench ...` 或自定义 benchmark | 吞吐/延迟在合理范围 | 按 E-PERF 系列排查 |
| G11 multimodal sanity | 验证多模态输入 | 发送包含 image URL 的 chat 请求 | 返回正常描述结果 | 按多模态错误模式排查 |

## Output Format

排障报告使用中文输出，结构如下（完整模板见 `references/report-template.md`）：

```markdown
## vLLM-Ascend 排障报告

### 1. 结论摘要
[一段话概括问题、根因和推荐修复方案]

### 2. 用户提供的信息
[已提供和缺失的信息清单]

### 3. 环境与版本判断
[版本兼容性结论]

### 4. 故障现象
[用户描述和错误日志]

### 5. 根因假设
[明确的根因推导]

### 6. 证据链
[从日志到根因的推理路径]

### 7. 风险等级
[P0 / P1 / P2]

### 8. Fallback 路径
[降级恢复方案]

### 9. 建议修复步骤
[具体命令和操作]

### 10. 验证方案
[验证命令和预期输出]

### 11. 检查点与用户确认记录
[各检查点的决策记录]

### 12. 需要用户补充的信息
[如有缺失信息，列出]
```

## Resource Loading Rules

| When | Load | Purpose |
|---|---|---|
| 分类故障时 | `references/error-patterns.md` | 匹配已知错误模式，获取症状、根因、修复方案 |
| 选择 fallback 时 | `references/fallback-matrix.md` | 按场景选择有序降级路径 |
| 设计验证方案时 | `references/validation-checklist.md` | 获取验证命令、通过标准、失败处理 |
| 输出报告时 | `references/report-template.md` | 按模板生成结构化中文报告 |
| 自测 skill 触发效果时 | `references/test-prompts.md` | 使用正负样本验证触发准确性 |
| 确认资源完整性时 | `references/resource-manifest.md` | 检查所有内部文件和外部链接 |
| 缺少环境信息时 | `scripts/collect_vllm_ascend_env.py` | 只读采集环境和版本信息，输出 JSON |

## Safety Rules

1. **不猜测根因**: 信息不足时明确要求用户补充，不基于假设给出可能错误的解决方案
2. **不读取敏感信息**: 不读取 .env、token、SSH key、云凭据、生产数据库密码
3. **高风险操作暂停确认**: 安装/升级依赖、修改环境变量、写入文件、重启服务、杀进程、下载大模型、长 benchmark、SSH 执行、改版本前必须 PAUSE FOR USER CONFIRMATION
4. **版本敏感**: vLLM-Ascend 不同版本的参数名、行为可能不同，务必确认版本
5. **安全边界**: 不建议用户修改 NPU 驱动或 CANN 底层组件，除非问题明确指向那里
6. **量化区分**: 区分 vLLM 原生量化 (AWQ/GPTQ) 和昇腾特有量化 (Ascend W8A8/FP8)，两者排查路径不同
7. **优先级排序**: 解决方案按"最可能解决 > 最容易实施 > 最安全"排序
8. **只读默认**: 默认行为是分析和建议，不执行修改命令

### 常见环境变量参考

```bash
# 必须设置
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest

# 推荐设置
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

# 性能调优
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export COMBINED_ENABLE=1

# 调试用
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=1
export HCCL_DEBUG=1
```

### 常见启动命令模板

```bash
# 标准单卡推理
vllm serve /path/to/model \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --max-model-len 4096 \
    --device npu

# 多卡 tensor parallel
vllm serve /path/to/model \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-model-len 8192 \
    --device npu \
    --distributed-executor-backend mp

# 量化模型
vllm serve /path/to/model \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --quantization ascend \
    --device npu

# 强制 eager mode（排查 ACLGraph 问题时）
vllm serve /path/to/model \
    --enforce-eager \
    --device npu
```

### 版本兼容性矩阵（快速参考）

| vLLM-Ascend | vLLM | PyTorch | torch_npu | CANN | 推荐昇腾驱动 |
|-------------|------|---------|-----------|------|-------------|
| 0.9.x | 0.9.x | 2.5.1 | 2.5.1 | 8.1.RC1 | 24.1.RC3+ |
| 0.8.x | 0.8.x | 2.5.1 | 2.5.1 | 8.1.RC1 | 24.1.RC3+ |
| 0.7.x | 0.7.x | 2.4.0 | 2.4.0 | 8.0.T6+ | 24.1.RC2+ |
