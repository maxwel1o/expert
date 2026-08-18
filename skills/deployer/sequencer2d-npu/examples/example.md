# Sequencer2D NPU Deployment Skill - 使用示例

## 1. 运行全部 3 个模型（串行执行）

```bash
python3 scripts/batch_runner.py
```

该命令会依次执行所有 Sequencer2D 模型（s/m/l three variants）的：
- 从 ModelScope 下载权重
- CPU 推理并保存 logits
- NPU 推理并保存 logits
- CPU/NPU 精度对比
- 生成终端截图
- 推送至 GitCode 模型仓库

## 2. 运行单个模型

```bash
python3 scripts/batch_runner.py --model sequencer2d_s.in1k
```

## 3. 跳过推理，只生成截图和文档

```bash
python3 scripts/batch_runner.py --skip-inference
```

## 4. 跳过 GitCode 推送

```bash
python3 scripts/batch_runner.py --skip-push
```

## 5. 手动运行单个模型的完整流程

### 5.1 环境准备

```bash
pip install -r requirements.txt
```

### 5.2 CPU 推理

```bash
python3 scripts/inference.py --model sequencer2d_s.in1k \
  --weights /path/to/model.safetensors \
  --device cpu --dump logits_cpu.npy
```

### 5.3 NPU 推理

```bash
python3 scripts/inference.py --model sequencer2d_s.in1k \
  --weights /path/to/model.safetensors \
  --device npu --dump logits_npu.npy
```

### 5.4 CPU/NPU 精度对比

```bash
python3 scripts/compare_cpu_npu.py --cpu logits_cpu.npy \
  --npu logits_npu.npy --model sequencer2d_s
```

## 6. 精度指标说明

- **误差率** = max(|CPU_logits - NPU_logits|) / (max(CPU_logits) - min(CPU_logits))
- 要求误差率 < 1%
- 额外验证 Top-1/Top-5 类别一致性、余弦相似度
