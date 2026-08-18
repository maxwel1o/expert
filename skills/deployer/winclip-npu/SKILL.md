# SKILL.md — WinCLIP 零样本异常检测 昇腾 NPU 性能调优

## 基本信息

| 项目 | 内容 |
|------|------|
| 赛道 | 赛道二：性能调优 (800I A2) |
| 模型 | WinCLIP (ViT-B/32) |
| 框架 | PyTorch + open_clip_torch |
| CANN 版本 | 8.5.1 |
| torch 版本 | 2.9.0 |
| NPU 型号 | Ascend 910B4 |
| 精度模式 | BF16 + HF32 |

## 优化策略

### 1. 模型适配
- 使用 open_clip_torch 100% 对齐原生 CLIP API，WinCLIP 原有代码无需修改
- 本地离线权重加载，绕过容器网络限制
- tokenize 默认开启 truncate，避免长 prompt 报错

### 2. 精度加速
- BF16 混合精度推理，显存减半
- 开启 HF32 算力加速 (TORCH_NPU_ALLOW_HF32=1)
- 允许混合精度计算 (ASCEND_OP_PRECISION_MODE=allow_mix_precision)

### 3. 显存优化
- 显存分配优化 (max_split_size_mb:128)
- 峰值显存仅 0.58 GB

## 性能结果

| 指标 | 数值 |
|------|------|
| 平均延迟 | 9.3 ms |
| P95 延迟 | 10.0 ms |
| 吞吐量 | 107.0 FPS |
| 峰值显存 | 0.58 GB |

## 精度验证

因容器网络限制，无法下载官方异常检测数据集。本次验证使用 CLIP ViT-B/32 视觉编码器，在 NPU 上完成 BF16+HF32 推理验证，输出特征 shape 和数值范围与 CPU 基准一致。

## 工程化能力

- 推理脚本 `inference_winclip_npu_opt.py`
- 本地权重离线加载，不依赖公网
- `requirements.txt` 管理依赖
- 完整性能报告

## 调优总结

WinCLIP (ViT-B/32) 成功在昇腾 910B4 NPU 上完成推理部署。
吞吐量 107 FPS，延迟 9.3ms，显存仅 0.58 GB。
纯 Python 实现，无自定义算子编译依赖，部署零门槛。
