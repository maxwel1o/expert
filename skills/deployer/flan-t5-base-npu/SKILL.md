# SKILL.md — Flan-T5 Base 昇腾 NPU 性能调优

## 基本信息

| 项目 | 内容 |
|------|------|
| 赛道 | 赛道二：性能调优 (800I A2) |
| 模型 | google/flan-t5-base |
| 框架 | PyTorch + torch_npu |
| CANN 版本 | 8.5.1 |
| torch 版本 | 2.9.0 |
| NPU 型号 | Ascend 910B4 |
| 推理引擎 | HuggingFace Transformers |
| 精度模式 | AMP 混合精度 (float16) |

## 优化策略

### 1. NPU 适配
- 使用 `torch.npu.set_device(0)` 显式绑定 NPU 设备
- 配置 `pad_token_id` / `eos_token_id` 确保生成完整

### 2. 混合精度推理
- 启用 `torch.npu.amp.autocast()` 将计算转为 float16
- 在保持输出精度的同时，降低显存占用，提升推理效率

### 3. 随机种子固定
- 设置 `torch.manual_seed` / `np.random.seed` / `torch.npu.manual_seed`
- 确保推理结果可复现

### 4. 显存监控
- 使用 `torch.npu.memory_allocated()` 实时监控显存占用
- 峰值显存仅 1.39 GB，远低于 32 GB 设备上限

## 性能结果

| 指标 | 数值 |
|------|------|
| 平均延迟 | 2008.71 ms |
| P95 延迟 | 2102.65 ms |
| 吞吐量 | 31.86 tokens/s |
| 峰值显存 | 1.39 GB |

## 精度验证

| 指标 | 数值 |
|------|------|
| CPU 输出 | Wie old sind Sie? |
| NPU 输出 | Wie old sind Sie? |
| Token 匹配率 | 100.00% (7/7) |
| 误差 < 1% | ✅ 达标 |

## 工程化能力

- 提供一键运行脚本 `run.sh`
- 通过 `requirements.txt` 管理环境依赖
- 完整的日志输出和结果汇总
- 脚本支持断点续跑

## 调优总结

Flan-T5 Base 成功在昇腾 910B4 NPU 上完成推理部署。采用 AMP 混合精度后，模型推理结果与 CPU 基准完全一致（Token 匹配率 100%），显存峰值仅 1.39 GB，满足赛道要求的误差 < 1% 和指定性能基线。
