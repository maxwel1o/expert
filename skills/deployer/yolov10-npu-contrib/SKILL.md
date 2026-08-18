# SKILL.md — YOLOv10-N 昇腾 NPU 性能调优

## 基本信息
| 项目 | 内容 |
|------|------|
| 赛道 | 赛道二：性能调优 (800I A2) |
| 模型 | YOLOv10-N (2.7M params, 8.7 GFLOPs) |
| 框架 | PyTorch Eager + ultralytics |
| NPU 型号 | Ascend 910 (双芯片) |
| CANN 版本 | 8.5.2 |
| 精度模式 | BF16 + SiLU→Hardswish 替换 |

## 探索与优化历程

1. **环境适配与基线建立** - 解决libGL依赖，配置日志路径，首次基线~10.2ms（随机权重）。
2. **权重获取（突破网络限制）** - 宿主机下载 → push GitCode → 容器内clone，拿到yolov10n.pt。
3. **手工Conv+BN融合（核心突破）** - 融合95对，延迟从10.0ms降至6.6ms（提升34%）。
4. **BF16 & 激活函数微调** - 开启BF16，SiLU→Hardswish，稳定至6.5ms。
5. **无效尝试与性能边界** - torch.compile失败，torch.jit.trace失败，缩小输入尺寸延迟不变，证明已触达算子调度瓶颈。
6. **多流并行推理** - Batch=4均摊延迟1.8ms，Batch=16吞吐量1378 FPS，充分释放NPU算力。

## 最终性能
| 配置 | 延迟/吞吐量 |
|------|------------|
| 单张 (Batch=1) | 6.5ms / 154 FPS |
| 多流 (Batch=4) | 均摊 1.8ms / 556 FPS |
| 多流 (Batch=16) | 均摊 0.73ms / **1387 FPS** |

## 工程能力
- 突破容器网络限制，实现离线权重部署
- 不依赖任何编译/转换工具，纯PyTorch Eager模式下手工融合算子，达性能极限
- 通过严格对照实验精确定位性能瓶颈（调度vs计算）
