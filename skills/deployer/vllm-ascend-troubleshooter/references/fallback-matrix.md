# Fallback 恢复策略矩阵

当主要路径失败时，按以下矩阵选择降级恢复方案。

---

## 场景降级矩阵

| Scenario | Primary Path | Fallback 1 | Fallback 2 | Fallback 3 | Stop Condition |
|---|---|---|---|---|---|
| graph mode 失败 | 默认 ACLGraph 模式启动 | `--enforce-eager` 强制 eager | 升级 CANN 到 8.1.RC1+ | 保持 eager mode | eager 也失败则非 graph 问题 |
| 多模态失败 | 完整多模态推理 | 检查 Pillow/timm 版本 | 使用纯文本 prompt 验证 | 降级到纯文本模型 | 纯文本也不可用 |
| 量化失败 | `--quantization ascend` | 移除量化参数，使用原始精度 | 尝试 `--quantization awq` / `gptq` | 使用非量化模型 | 原始精度也 OOM |
| 多卡失败 | `--tensor-parallel-size N` | 降低 TP size 到 N/2 | 单卡复现 (`--tensor-parallel-size 1`) | 减小 `--max-model-len` 后单卡 | 单卡也失败 |
| 内存不足 (OOM) | 默认参数启动 | 设置 `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` | 减小 `--max-model-len` | 减小 `--max-num-seqs` | 最小参数仍 OOM |
| API 失败 | `/v1/chat/completions` | 检查服务端日志 | 简化请求 payload（减小 max_tokens） | 使用 `/v1/completions` 替代 | 最简请求仍 500 |
| tokenizer 失败 | AutoTokenizer 加载 | 检查 tokenizer 文件完整性 | 指定 `--tokenizer` 路径 | 使用默认 tokenizer | tokenizer 模型不支持 |
| CUDA-only 算子失败 | 默认模式运行 | `--enforce-eager` 跳过编译优化 | 检查 torch_npu 是否有替代算子 | 等待社区适配或换模型 | 核心算子无替代 |
| NPU 不可见 | `torch.npu.is_available()` | 检查 npu-smi info | 检查 /dev/davinci* 设备文件 | 检查用户组权限 (HwHiAiUser) | 设备硬件故障 |
| 性能异常 | 默认参数 + 性能优化 env | 开启 TASK_QUEUE_ENABLE=1 + CPU_AFFINITY_CONF=1 | `--enforce-eager` 排除 ACLGraph 开销 | 减小 batch size 降低压力 | 需要 profiler 深度分析 |

---

## Ordered Fallback Ladder

当遇到未知问题或已知模式的方案无效时，按以下顺序逐级降级：

### Level 1: 复现并固定命令

- 记录完整的启动命令、环境变量、模型路径
- 确保问题可稳定复现
- 固定随机种子排除随机性

```bash
# 固定环境变量
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

### Level 2: 切换 --enforce-eager

- 排除 ACLGraph 相关问题
- 如果 eager mode 正常，问题定位到 graph mode

```bash
vllm serve <MODEL_PATH> --enforce-eager --device npu --port <PORT>
```

### Level 3: 降低 --max-model-len

- 减少 KV cache 内存占用
- 排除长序列导致的 OOM

```bash
vllm serve <MODEL_PATH> --max-model-len 2048 --enforce-eager --device npu --port <PORT>
```

### Level 4: 降低 --max-num-seqs

- 减少并发 batch 内存压力
- 排除 batch 过大导致的 OOM

```bash
vllm serve <MODEL_PATH> --max-model-len 2048 --max-num-seqs 4 --enforce-eager --device npu --port <PORT>
```

### Level 5: 单卡复现

- 排除 tensor parallel / HCCL 通信问题
- 排除多卡拓扑问题

```bash
vllm serve <MODEL_PATH> --tensor-parallel-size 1 --max-model-len 2048 --enforce-eager --device npu --port <PORT>
```

### Level 6: dummy load 验证

- 排除权重文件问题
- 验证模型架构是否可加载

```bash
vllm serve <MODEL_PATH> --load-format dummy --max-model-len 64 --enforce-eager --device npu --port <PORT>
```

### Level 7: real weight load 验证

- 确认真实权重可加载
- 验证权重完整性

```bash
vllm serve <MODEL_PATH> --max-model-len 64 --enforce-eager --device npu --port <PORT>
```

### Level 8: 最小 prompt API 验证

- 验证推理 API 可用
- 使用最简单的请求排除复杂因素

```bash
curl http://localhost:<PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"<MODEL>","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0}'
```

### Level 9: 升级为阻塞问题

如果 Level 1-8 均无法解决问题或找到 workaround：

1. 汇总所有尝试过的 fallback 结果
2. 收集完整错误日志和环境信息
3. 输出结构化 issue 材料：
   - 环境信息摘要
   - 复现命令
   - 错误日志
   - 已尝试的 fallback 和结果
4. 建议用户向 vLLM-Ascend 社区提交 issue

---

## 使用指南

1. 根据故障分类选择对应的场景行
2. 从 Primary Path 开始尝试
3. Primary Path 失败后按 Fallback 1 → 2 → 3 顺序降级
4. 到达 Stop Condition 时停止该路径的降级
5. 如果场景矩阵无法解决，使用 Ordered Fallback Ladder 从 Level 1 开始逐级尝试
6. 每级降级的结果必须记录，用于最终报告
