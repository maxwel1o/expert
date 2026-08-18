# Swin Transformer NPU Deployment 示例

## 示例 1：下载所有模型权重

```bash
python3 scripts/download_all.py
```

## 示例 2：单个模型 CPU 推理

```bash
python3 scripts/inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device cpu
```

## 示例 3：单个模型 NPU 推理

```bash
python3 scripts/inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device npu
```

## 示例 4：CPU/NPU 精度对比

```bash
python3 scripts/compare_cpu_npu.py --model swin_tiny_patch4_window7_224.ms_in1k
```

## 示例 5：批量执行全部 19 个模型

```bash
python3 scripts/batch_runner.py
```

## 示例 6：使用 Skill 参数指定单个模型

```json
{
  "model_name": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
  "skip_inference": false,
  "skip_push": false
}
```

## 示例 7：跳过推理，只生成文档

```json
{
  "model_name": "all",
  "skip_inference": true,
  "skip_push": false
}
```

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `inference.py` | NPU/CPU 通用推理脚本 |
| `compare_cpu_npu.py` | CPU/NPU 精度对比脚本 |
| `download_all.py` | 批量权重下载脚本 |
| `batch_runner.py` | 串行批量执行脚本 |
| `screenshots/*.png` | 终端输出截图 |
| `README.md` | 每个模型的中文文档 |
