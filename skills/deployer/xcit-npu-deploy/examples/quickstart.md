# XCiT NPU Deployment Quick Start

## Single model inference

```bash
# NPU inference
python3 scripts/run_inference.py --model xcit_tiny_12_p16_384 --device npu:0

# CPU inference
python3 scripts/run_inference.py --model xcit_tiny_12_p16_384 --device cpu

# With custom batch size and iterations
python3 scripts/run_inference.py --model xcit_large_24_p8_224 --device npu:0 --batch-size 4 --warmup 5 --iters 50
```

## Precision comparison

```bash
# Compare CPU vs NPU output for a single model
python3 scripts/run_compare.py --model xcit_tiny_12_p16_384
```

## Run all models serially

```bash
# Run all 5 XCiT models (precision comparison + NPU benchmark)
bash scripts/run_all.sh

# Run all models, skip benchmark (faster)
bash scripts/run_all.sh --skip-benchmark

# Run a single model
bash scripts/run_all.sh --model xcit_tiny_12_p16_384
```

## Verify NPU environment

```bash
python3 -c "
import torch
print('NPU available:', torch.npu.is_available())
print('NPU count:', torch.npu.device_count())
print('NPU name:', torch.npu.get_device_name(0))
"
```
