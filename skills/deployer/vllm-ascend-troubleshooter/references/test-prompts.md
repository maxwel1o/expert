# Test Prompts

用于验证 vllm-ascend-troubleshooter skill 的触发准确性和覆盖范围。

---

## Positive Trigger Prompts

以下提示词应触发本 Skill：

1. "vLLM-Ascend 启动报错：RuntimeError: No available npu device found，怎么排查？"
2. "npu-smi info 输出显示 NPU 状态为 Fault，vLLM 服务启动不了"
3. "ACLGraph capture failed，eager mode 可以跑，graph mode 失败，怎么解决？"
4. "量化模型加载报错：ValueError: Quantization method 'ascend' is not supported"
5. "vllm serve 启动后 /v1/chat/completions 返回 500 Internal Server Error"
6. "tensor parallel 4 卡启动，HCCL timeout，rank 2 超时"
7. "torch_npu 导入失败：ModuleNotFoundError: No module named 'torch_npu'，CANN 已安装"
8. "多模态模型输入图片时报错：PIL cannot identify image file"
9. "vLLM-Ascend 推理吞吐比 GPU 低很多，NPU 利用率只有 30%"
10. "KV cache 分配失败：RuntimeError: Failed to allocate KV cache，max-model-len 设了 32768"
11. "safetensors 加载报错：SafetensorError: Error while deserializing header"
12. "CANN 版本 8.0.T5，torch_npu 2.5.1，启动 vLLM 报各种 ACL 错误码"

---

## Negative Trigger Prompts

以下提示词不应触发本 Skill（属于其他领域）：

1. "Python 语法错误：IndentationError: unexpected indent" — 纯 Python 语法问题
2. "React 组件样式怎么居中？CSS flexbox 用法" — 前端 UI 设计
3. "MySQL 查询优化：SELECT * FROM users WHERE id > 1000 很慢" — 数据库优化
4. "Docker 容器网络配置：bridge vs host mode 区别" — 容器网络
5. "Git merge conflict 解决方法" — 版本控制
6. "Kubernetes Pod OOMKilled，如何调整 resource limits" — K8s 运维（非 vLLM 相关）
7. "PyTorch GPU 训练报错 CUDA out of memory" — GPU/CUDA 问题（非 NPU）

---

## Evaluation Metrics

| Metric | Target | Evidence |
|---|---|---|
| frontmatter trigger coverage | >= 80% positive prompts 匹配 | 所有正向提示词中至少 10/12 能触发 Skill |
| phase completeness | Phase 0-6 全部存在 | SKILL.md 中包含 Phase 0 到 Phase 6 共 7 个 Phase |
| checkpoint count | >= 7 个 Checkpoint | SKILL.md 中 C0-C6 共 7 个 Checkpoint |
| boundary table coverage | >= 9 条边界条件 | Boundary Conditions 表格至少 9 行 |
| fallback coverage | >= 10 个场景 | fallback-matrix.md 至少覆盖 10 个场景 |
| resource references | >= 8 个外部资源 | resource-manifest.md External Resources 至少 8 行 |
| validation gates | >= 12 个门禁 | SKILL.md Validation Gates 表格至少 12 行 (G0-G11) |
| Chinese report output | 报告模板包含 12 节 | report-template.md 包含 1-12 节 |
| error patterns coverage | >= 14 类故障 | error-patterns.md 总览表格至少 14 行 |
| test prompts positive | >= 8 条正向触发 | test-prompts.md Positive Trigger 至少 8 条 |
| test prompts negative | >= 5 条负向不触发 | test-prompts.md Negative Trigger 至少 5 条 |
| script safety | 不读取敏感信息 | collect_vllm_ascend_env.py 不读取 .env/SSH/凭据 |
| no false claims | 无伪造 benchmark/PASS/skills-eval 声明 | 全文不包含伪造评测结果 |
