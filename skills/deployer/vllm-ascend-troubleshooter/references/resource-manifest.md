# Resource Manifest

## Internal Files

| File | Type | Purpose | Used By Phase | Notes |
|---|---|---|---|---|
| `SKILL.md` | Skill 定义 | 主工作流、检查点、边界条件、安全规则 | 始终加载 | 核心文件 |
| `references/error-patterns.md` | 参考文档 | 14 类已知错误模式的症状、根因、检查、修复、fallback | Phase 2 故障分类 | 优先匹配 |
| `references/fallback-matrix.md` | 参考文档 | 10 类场景的有序降级路径和 fallback ladder | Phase 4 Fallback 隔离 | 按序降级 |
| `references/validation-checklist.md` | 参考文档 | 11 项验证检查的命令、通过标准、失败处理 | Phase 1 环境基线 + Phase 6 验证 | 逐项过关 |
| `references/resource-manifest.md` | 索引文件 | 内部文件和外部资源链接清单 | 确认资源完整性时 | 本文件 |
| `references/report-template.md` | 模板文件 | 12 节中文排障报告模板 | Phase 6 输出报告 | 结构化输出 |
| `references/test-prompts.md` | 测试文件 | 正负向触发样本和评测指标 | 自测 skill 触发效果时 | 可选 |
| `scripts/collect_vllm_ascend_env.py` | 脚本 | 只读环境采集，输出 JSON | 缺少环境信息时 | 只读、不上传 |

## External Resources

| Resource | URL | Purpose | Verify Before Use |
|---|---|---|---|
| vLLM documentation | https://docs.vllm.ai/ | vLLM 官方文档，参数说明、模型支持列表 | Yes |
| vLLM-Ascend repository | https://github.com/vllm-project/vllm-ascend | vLLM-Ascend 源码、issue、版本发布 | Yes |
| Ascend / CANN documentation | https://www.hiascend.com/document | CANN 安装指南、算子支持、版本说明 | Yes |
| torch-npu project | https://github.com/ascend/torch_pytorch | torch_npu 源码和安装说明 | Yes |
| OpenAI-compatible server API | https://platform.openai.com/docs/api-reference | API 请求格式参考 | Yes |
| Hugging Face model config | https://huggingface.co/docs/transformers/model_doc/config | 模型 config.json 字段说明 | Yes |
| Ascend NPU error codes | https://www.hiascend.com/document/detail/zh/canncommercial | ACL 错误码查询 | Yes |
| vLLM-Ascend compatibility matrix | https://github.com/vllm-project/vllm-ascend/blob/main/README.md | 版本兼容性矩阵 | Yes |

注：外部资源链接可能随上游更新而变化。使用前建议验证链接有效性。标记 Verify Before Use: Yes 的资源需要在引用前确认可达性。
