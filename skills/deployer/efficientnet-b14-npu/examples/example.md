# EfficientNet NPU Deployment Example

## 示例 1：使用已发布的模型仓库

### 克隆模型仓库

```bash
git clone https://gitcode.com/m0_74196153/tf_efficientnetv2_s.in1k-npu.git
cd tf_efficientnetv2_s.in1k-npu
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 执行 CPU 推理

```bash
python3 inference.py tf_efficientnetv2_s.in1k --device cpu
```

预期输出：
```
[INFO] Running inference for: tf_efficientnetv2_s.in1k
[INFO] Device: cpu
[RESULTS] Top-5 predictions (CPU):
  1. class 285: 0.3766
  2. class 281: 0.1607
  3. class 282: 0.1258
```

### 执行 NPU 推理

```bash
python3 inference.py tf_efficientnetv2_s.in1k --device npu
```

预期输出：
```
[INFO] Running inference for: tf_efficientnetv2_s.in1k
[INFO] Device: npu
[RESULTS] Top-5 predictions (NPU):
  1. class 285: 0.3766
  2. class 281: 0.1606
  3. class 282: 0.1258
```

### CPU/NPU 精度对比

```bash
python3 compare_cpu_npu.py tf_efficientnetv2_s.in1k
```

预期输出：
```
[COMPARE] Model: tf_efficientnetv2_s.in1k
  Mean Absolute Error: 0.00026292
  Cosine Similarity: 1.00000000
  Top-1 Agreement: True
  Top-5 Agreements: 5/5
  CPU: 0.0780s | NPU: 0.0070s | Speedup: 11.14x
[VERDICT] PASS: NPU误差<1%
```

## 示例 2：批量处理多个模型

```bash
# 使用 batch_runner 处理一批模型
python3 batch_runner.py --start 0 --end 5
```

## 示例 3：生成终端截图

```bash
python3 scripts/terminal_screenshot.py --text \
  "$ python3 inference.py tf_efficientnetv2_s.in1k --device cpu" \
  "[INFO] Running inference for: tf_efficientnetv2_s.in1k" \
  "[INFO] Device: cpu" \
  "[RESULTS] Top-5 predictions (CPU):" \
  "  1. class 285: 0.3766" \
  "  2. class 281: 0.1607" \
  "  3. class 282: 0.1258"
```

## 结果说明

- **MAE**: 平均绝对误差，衡量 CPU 和 NPU 输出差异
- **余弦相似度**: 接近 1.0 表示输出高度一致
- **Top-1 一致率**: NPU 和 CPU 的 Top-1 预测类别是否相同
- **Speedup**: NPU 推理速度相对于 CPU 的加速比
