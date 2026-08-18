---
name: model-deploy
description: "Use when deploying LLM models on Ascend NPU with vLLM-Ascend. Triggers on: 部署模型, 启动推理服务, 拉起模型, vllm serve, vllm-ascend, model deployment, 推理部署. Not for CPU/GPU deployment or non-vLLM serving."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [vllm, ascend, npu, deployment, model-deploy, 推理部署]
    related_skills: [vllm-bench, serving-llms-vllm]
---

# vLLM-Ascend 模型部署

在昇腾 NPU 环境下部署大模型推理服务。

## 一键部署 (推荐)

```bash
cd $DEPLOY_SKILL_ROOT/scripts

# 查看可用模型
./quick_deploy.sh --list

# 一键部署
./quick_deploy.sh qwen3-32b
./quick_deploy.sh deepseek-v4-flash
./quick_deploy.sh glm-5
```

**Agent 一键部署流程:**

当用户请求部署模型时：
1. 确认模型名称（同一模型有多个版本）
2. 执行 `./quick_deploy.sh <模型别名>`
3. 等待部署完成，返回服务地址

## 重要提示

**模型名称需与用户确认具体版本！**
- 同一模型有多个版本（w8a8、w4a8 量化版等）
- ModelScope ID 可能与本地命名不同
- 查询地址: https://www.modelscope.cn/models

## 配置来源

脚本整合了以下来源的最佳实践：
- `$VLLM_TOOLS_SCRIPT_ROOT/`
- `$VLLM_TOOLS_ROOT/`
- https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/tutorials/models/

## 脚本位置

```
$DEPLOY_SKILL_ROOT/scripts/
├── quick_deploy.sh      # 一键部署 (推荐)
├── deploy_model.sh      # 完整部署 (自定义参数)
├── model_registry.sh    # 模型配置注册表
└── common.sh            # 公共环境配置
```

## 整合的配置

### 环境变量
- `VLLM_USE_V1=1`, `VLLM_VERSION=0.18.0`
- `TASK_QUEUE_ENABLE` (根据模式配置)
- `HCCL_*` 网络和通信配置
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`

### 模型特定优化
- **DeepSeek V4**: `VLLM_ASCEND_APPLY_DSV4_PATCH`, 专用 tokenizer
- **Qwen3**: `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE`, FlashComm
- **Qwen3.5 MoE**: `qwen3_5_mtp` MTP 配置
- **GLM-5**: `deepseek_mtp` MTP, glm47 tool-call-parser

### MTP 配置
- DeepSeek 系列: `deepseek_mtp`, num_speculative_tokens=1
- Qwen3.5 MoE: `qwen3_5_mtp`, num_speculative_tokens=3
- GLM-5: `deepseek_mtp`, num_speculative_tokens=2

## 镜像与模型来源

- **镜像**: https://quay.io/repository/ascend/vllm-ascend?tab=tags
- **模型**: https://www.modelscope.cn/models

## PD 部署模式

详见 `references/pd-deployment.md`

### mix-pd (混合部署)
- Prefill 和 Decode 在同一进程
- `TASK_QUEUE_ENABLE=0/1`
- 配置简单，适合单机

### full-pd (分离部署)
- Prefill 和 Decode 分离，通过 MooncakeConnector 传输 KV Cache
- Prefill: `TASK_QUEUE_ENABLE` 可配置 (常见值: 1, 2, 6, 10 等)
- Decode: `TASK_QUEUE_ENABLE=0`
- 适合大规模生产

## 预定义模型

详见 `references/models.md`

快速参考:
- **Qwen**: qwen3-32b, qwen3-30b-a3b, qwen3-235b-a22b, qwen2.5-72b/32b/14b/7b, qwen3.5-397b/122b, qwq-32b
- **DeepSeek**: deepseek-v3, deepseek-r1, deepseek-v4-flash
- **GLM**: glm-4-9b, glm-4-9b-chat, glm-5, glm-5.1
- **LLaMA**: llama3-70b, llama3-8b
- **Kimi**: kimi-k2, kimi-k2.5
- **MiniMax**: minimax-m2.5

## 文件结构

```
$DEPLOY_SKILL_ROOT/
├── SKILL.md                    # 本文档
├── references/
│   ├── models.md               # 预定义模型详情
│   └── pd-deployment.md        # PD 部署模式说明
└── scripts/
    ├── quick_deploy.sh         # 一键部署脚本
    ├── deploy_model.sh         # 完整部署脚本
    ├── model_registry.sh       # 模型配置注册表
    └── common.sh               # 公共环境配置
```

## Common Pitfalls

1. **模型版本未确认** — 同一模型有多个量化版本(w8a8/w4a8)，部署前必须与用户确认具体版本
2. **ModelScope ID 不匹配** — 本地别名与 ModelScope 模型 ID 可能不同，查询 https://www.modelscope.cn/models 确认
3. **NPU 编号冲突** — 多模型部署时 NPU 编号可能重叠，需在 deploy_model.sh 中指定不同 NPU 范围
4. **环境变量未设置** — `$DEPLOY_SKILL_ROOT`, `$VLLM_TOOLS_ROOT`, `$VLLM_TOOLS_SCRIPT_ROOT` 必须在 USER profile 中定义
5. **PD 模式选择错误** — 单机用 mix-pd，大规模生产用 full-pd，混淆会导致性能问题

## Usage Hook

> 本skill每次成功执行后，必须记录使用日志。这是强制步骤，不可跳过。

**执行时机：** skill指导的任务成功完成后（非加载/浏览时）

**记录命令：**
```bash
echo '{"skill":"model-deploy","version":"1.0.0","stage":"dev","author":"$(git config user.name)","agent":"hermes","session_id":"$HERMES_SESSION_ID","timestamp":"$(date -Iseconds)","action":"use"}' >> ~/.hermes/logs/skill-usage.jsonl
```

**agent适配说明：**
- Hermes：直接执行上述命令
- 其他agent：用各自shell工具执行等价命令
- 路径 `~/.hermes/logs/` 为Hermes默认；其他agent需适配为 `<agent-home>/logs/`
