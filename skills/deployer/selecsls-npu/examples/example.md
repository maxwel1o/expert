# SelecSLS NPU 部署示例

## 示例 1：单模型 NPU 推理

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 运行 NPU 推理
python3 scripts/inference.py --model selecsls60b.in1k --image test_image.jpg --device npu
```

预期输出：
```
Loading model: selecsls60b.in1k on npu...
Results saved to npu_results.json
Mean inference time: 0.0077s
Top-5 indices: [735, 533, 539, 741, 911]
Top-5 probs: [0.147678, 0.104253, 0.082703, 0.033201, 0.025226]
```

## 示例 2：CPU/NPU 精度对比

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/compare_cpu_npu.py --model selecsls60b.in1k --image test_image.jpg
```

预期输出：
```
=== Comparison Results ===
Max absolute diff: 0.00281978
Cosine similarity: 0.99999988
Top-1 match: True (CPU: 735, NPU: 735)
Top-5 overlap: 5/5
Max diff %: 0.050174%
Precision: PASS (error < 1%)
```

## 示例 3：串行执行全部模型

```bash
bash scripts/run_all.sh
```

该脚本将依次处理 `selecsls60b.in1k`、`selecsls60.in1k`、`selecsls42b.in1k` 三个模型，每个模型完成推理和精度测试后自动释放 NPU 显存。

## 示例 4：生成终端截图

```bash
python3 /path/to/terminal_screenshot.py \
  --input terminal_output.txt \
  --output terminal_screenshot.png
```

## 精度对比结果汇总

| 模型 | Top-1 一致 | Top-5 重叠 | 余弦相似度 | 最大误差 % |
|:---|---:|:---:|:---:|:---:|
| selecsls60b.in1k | 是 | 5/5 | 0.99999988 | 0.05% |
| selecsls60.in1k | 是 | 5/5 | 0.99999988 | 0.04% |
| selecsls42b.in1k | 是 | 5/5 | 0.99999964 | 0.07% |
