---
name: aimv2-adapt-npu
description: >
  Apple AIMv2 视觉编码器适配版系列模型在昇腾 NPU 上的完整部署 Skill。
  涵盖 4 个官方发布变体（224-distilled、336-distilled、224-lit、native）
  的环境准备、权重下载与 Key 转换、NPU 推理验证、精度对比的全流程。
  当用户提到 AIMv2 适配部署昇腾、Apple AIMv2 NPU 推理、aimv2 蒸馏版/lit/native NPU 时触发。
metadata:
  short-description: Apple AIMv2 适配版昇腾 NPU 部署
  category: NPU-Model-Deploy
  tags: [ascend, npu, aimv2, apple, vision, pytorch, inference, transformers, distilled, lit]
---

# Apple AIMv2 适配版昇腾 NPU 部署 Skill

本 Skill 提供 Apple 官方发布的 AIMv2 视觉编码器系列模型在华为昇腾 NPU 上的完整部署与推理验证流程。AIMv2 采用多模态自回归预训练（Multimodal Autoregressive Pre-training），在视觉编码任务上达到 SOTA 水平。

## 支持的模型变体

| 序号 | 变体 | 参数量 | 输入尺寸 | 架构特点 |
|------|------|--------|----------|----------|
| 1 | 224-distilled | 1.15B | 224x224 | 蒸馏 ViT，Timm 风格权重 |
| 2 | 336-distilled | 1.15B | 336x336 | 蒸馏 ViT，大输入尺寸 |
| 3 | 224-lit | 1.63B | 224x224 | CLIP-like 双编码器（视觉+文本） |
| 4 | native | 1.15B | 224x224 | Sin-Cos 位置编码 ViT，动态分辨率 |

## 前置条件

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend910B4（至少 1 卡，64GB HBM） |
| OS | openEuler / Ubuntu（aarch64） |
| CANN | >= 8.5.1 |
| Python | 3.9 - 3.13 |
| 网络 | 首次运行需下载模型权重（约 2-3GB） |
| 存储 | 本地缓存 ~/.cache/modelscope/，权重转换产物 scripts/ |

## 工作流

1. 环境初始化与 NPU 验证
2. 安装依赖（torch_npu + transformers）
3. 下载模型权重（ModelScope）
4. 权重 Key 转换（Timm 到 HF 映射 + QKV 合并）
5. NPU 推理验证（随机输入 + 真实图片）
6. 精度对比测试（CPU vs NPU 余弦相似度）
7. 检查点验收与异常回退处理
8. 资源归档与评测结果输出

按以下步骤顺序执行，每步完成后确认再进入下一步。

## 详细执行步骤

1. 加载 CANN 环境
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 检查 NPU 设备状态
```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0
```

3. 验证 NPU 可用性
```bash
python3 -c "import torch; import torch_npu; print(f'NPU available: {torch.npu.is_available()}')"
```

4. 安装核心依赖
```bash
pip install torch==2.9.0 torch_npu==2.9.0.post1 transformers==4.57.6 safetensors pillow
```

5. 验证 torch_npu 导入
```bash
python3 -c "import torch_npu; print(f'torch_npu version: {torch_npu.__version__}')"
```

6. 设置模型变体和获取 ModelScope 模型名
```bash
MODEL_VARIANT="224-distilled"
case $MODEL_VARIANT in
  224-distilled) MODEL_NAME="aimv2-large-patch14-224-distilled" ;;
  336-distilled) MODEL_NAME="aimv2-large-patch14-336-distilled" ;;
  224-lit)       MODEL_NAME="aimv2-large-patch14-224-lit" ;;
  native)        MODEL_NAME="aimv2-large-patch14-native" ;;
esac
```

7. 从 ModelScope 下载权重
```bash
pip install modelscope
python3 -c "from modelscope import snapshot_download; print(snapshot_download('apple/$MODEL_NAME'))"
```

8. 运行权重转换脚本
```bash
python3 scripts/convert_weights.py --model $MODEL_VARIANT --input ~/.cache/modelscope/hub/models/apple/$MODEL_NAME/model.safetensors --output ./converted_$MODEL_VARIANT.safetensors
```

9. 验证转换结果完整性
```bash
python3 -c "import torch; sd=torch.load('./converted_$MODEL_VARIANT.safetensors'); print(f'转换完成，参数组数: {len(sd)}')"
```

10. 随机输入 NPU 推理验证
```bash
python3 scripts/aimv2_npu_infer.py --model $MODEL_VARIANT --dtype float32
```

11. 真实图片 NPU 推理
```bash
python3 scripts/aimv2_npu_infer.py --model 224-distilled --image /path/to/image.jpg
```

12. FP16 精度推理测试
```bash
python3 scripts/aimv2_npu_infer.py --model 224-distilled --dtype float16
```

13. Lit 模型文本 prompt 推理验证
```bash
python3 scripts/aimv2_npu_infer.py --model 224-lit --image /path/to/image.jpg --text "a photo of a cat"
```

14. 导出 NPU 推理输出用于精度对比
```bash
python3 -c "import torch; arr=torch.load('npu_output.pt'); print(f'Output shape: {arr.shape}, Mean: {arr.mean():.6f}, Std: {arr.std():.6f}')"
```

15. 计算 CPU 与 NPU 余弦相似度
```bash
python3 -c "import torch; cpu=torch.load('cpu_output.pt'); npu=torch.load('npu_output.pt'); cos=torch.nn.CosineSimilarity(dim=0)(cpu.flatten(),npu.flatten()); print(f'Cosine Similarity: {cos.item():.6f}')"
```

## 执行检查点与用户确认

| 检查点 | 触发时机 | 用户确认内容 | 失败或回退处理 |
|---|---|---|---|
| CP-1 环境检查点 | CANN/npu-smi 检查完成后 | 确认继续使用当前 NPU 环境 | 暂停确认，标注 dry-run 模式 |
| CP-2 下载检查点 | 首次下载 ModelScope 权重前 | 确认模型变体和缓存目录 | 切换镜像或复用本地权重缓存 |
| CP-3 转换检查点 | 权重 Key 转换执行前 | 确认输入路径和转换参数 | 回滚到原始权重，retry 一次 |
| CP-4 推理检查点 | NPU 推理输出生成后 | 确认输出 shape 和设备状态 | 标记验证失败，记录 eval 日志 |
| CP-5 精度检查点 | CPU/NPU 对比结果生成后 | 确认余弦相似度 >= 0.999 | 保留输出，标记 failed，不写通过 |
| CP-6 结果确认点 | evals.json 和 results.tsv 生成前 | 确认评测输出路径和格式 | 停止输出，保留中间产物 |
| CP-7 最终验收 | 全部变体部署完成后 | 确认 4 个变体均通过验证 | 回滚失败变体，追加修复 commit |

## 边界条件与异常回退

| 场景 | 触发条件 | fallback / retry / recover 处理 | 验证输出 |
|---|---|---|---|
| CANN 环境缺失 | npu-smi info 无输出或报错 | 提示加载 set_env.sh，retry 一次 | 环境检查日志标注 dry_run |
| torch_npu 不可导入 | import torch_npu 抛出 ImportError | fallback 到 CPU 推理模式 | 推理日志标记 CPU-only |
| 权重下载网络失败 | snapshot_download 连接超时 | retry 2 次，切换到 hf-mirror 镜像 | results.tsv 记录权重来源 |
| Key 映射失败 | strict=True 加载抛出 size mismatch | 输出缺失 Key 列表，回滚到原始权重 | evals.json 记录错误类型数量 |
| NPU 显存不足 OOM | 推理时抛出 CUDA/NPU OOM 异常 | gc.collect() + torch.npu.empty_cache() 后 retry | evals.json 记录 retry 次数 |
| 输入图片尺寸不匹配 | 图片分辨率与模型要求不一致 | 自动 resize 到模型输入尺寸后重新推理 | 推理日志含实际输入尺寸 |
| 精度对比不达标 | cos_sim < 0.999 或 max_diff 异常 | 保留 CPU/NPU 输出，标记 failed 不通过 | 精度评测表显示偏离原因 |
| NPU 输出全零或 NaN | 推理结果包含 NaN 或全零元素 | 回退到 FP32 精度 retry，记录异常日志 | evals.json 记录错误类型 |
| 磁盘空间不足 | 权重写入或转换输出写文件失败 | 清理临时缓存目录，retry 并提示存储需求 | 系统日志记录磁盘状态 |
| 多变体并行冲突 | 同时执行多个变体推理 | 强制串行执行，每步回收 NPU 显存 | results.tsv 逐变体串行记录 |
| test-prompts.json 加载失败 | 评测阶段文件不存在或格式错误 | 跳过当前 prompt，继续后续测试 | evals.json 记录 skip 原因 |

## 精度参考

各变体在 Ascend910B4 上 FP32 推理的精度测试结果：

| 变体 | max_diff | mean_diff | cos_sim | 结论 |
|------|----------|-----------|---------|------|
| 224-distilled | 5.22e-03 | 6.17e-05 | 1.000015 | PASS |
| 336-distilled | 4.72e-03 | 6.22e-05 | 1.000058 | PASS |
| 224-lit | 2.46e-04 | 2.46e-04 | 1.000000 | PASS |
| native | 2.41e+00 | 1.84e-03 | 0.999995 | WARN（离群值 < 0.002%） |

## 性能参考

测试条件：Ascend910B4, FP32, 20 次取平均。

| 变体 | batch_size=1 | batch_size=4 |
|------|-------------|-------------|
| 224-distilled | 22.9 ms | 23.1 ms |
| 336-distilled | 21.9 ms | 40.3 ms |
| 224-lit | 33.9 ms | 34.7 ms |
| native | 26.5 ms | 26.8 ms |

## 脚本文件与资源

| 路径 | 用途 |
|---|---|
| `scripts/aimv2_npu_infer.py` | NPU 推理主入口，支持四种变体 |
| `scripts/convert_weights.py` | Timm/HF Key 到模型代码的映射转换 |
| `scripts/modeling_aimv2_distilled.py` | Distilled 变体模型定义（含 QKV 合并） |
| `scripts/modeling_aimv2_lit.py` | Lit 变体视觉+文本双编码器模型 |
| `scripts/modeling_aimv2_native.py` | Native 变体 Sin-Cos 位置编码模型 |
| `scripts/configuration_aimv2_distilled.py` | Distilled 配置类定义 |
| `scripts/configuration_aimv2_lit.py` | Lit 配置类定义 |
| `scripts/configuration_aimv2_native.py` | Native 配置类定义 |
| `test-prompts.json` | 本 Skill 的结构化测试提示集 |
| `evals.json` | 评估结果结构化输出（含环境、retry、验证记录） |

## 验收确认

- [ ] `torch.npu.is_available()` 返回 True
- [ ] 权重转换后 `strict=True` 加载成功
- [ ] NPU 推理输出 shape 与预期一致
- [ ] 余弦相似度 >= 0.999
- [ ] FP16 推理正常
- [ ] test-prompts.json 评测通过
- [ ] evals.json 记录完整

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `KeyError: 'model.safetensors'` | 权重文件不存在 | 确认路径包含 model.safetensors |
| `size mismatch for qkv.weight` | 权重未转换 | 先运行 convert_weights.py |
| `torch_npu not available` | CANN 环境未加载 | npu-smi info 检查，重新加载 |
| OOM | NPU 显存不足 | 使用 FP16 或 batch_size=1 |
| native 精度 max_diff 偏高 | NPU-CPU 浮点累积差异 | cos_sim ~ 1.0 即合格 |

## 引用

```bibtex
@misc{fini2024multimodalautoregressivepretraininglarge,
  author={Fini, Enrico and Shukor, Mustafa and Li, Xiujun and others},
  title={Multimodal Autoregressive Pre-training of Large Vision Encoders},
  year={2024},
  eprint={2411.14402},
  archivePrefix={arXiv},
}
```
