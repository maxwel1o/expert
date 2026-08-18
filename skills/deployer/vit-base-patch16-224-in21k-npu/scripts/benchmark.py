#!/usr/bin/env python3
"""ViT-Base-Patch16-224-IN21K 昇腾 NPU 综合测评脚本"""
import os
import sys
import time
import pickle
import json
import numpy as np
import torch

# NPU 自动迁移
import torch_npu
from torch_npu.contrib import transfer_to_npu

from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

MODEL_PATH = '/opt/atomgit/vit-base-patch16-224-npu/model_weights/google/vit-base-patch16-224-in21k'
RESULT_DIR = '/opt/atomgit/vit-base-patch16-224-npu/benchmark_results'
os.makedirs(RESULT_DIR, exist_ok=True)

def get_model_and_processor(device_str):
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device(device_str)
    model = model.to(device)
    return model, processor, device

def make_inputs(processor, device, batch_size=1):
    # 重新固定种子，确保 CPU/NPU 在相同 batch_size 下得到完全相同的输入
    np.random.seed(42 + batch_size)
    images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(batch_size)]
    inputs = processor(images=images, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

def measure_latency(model, inputs, warmup=5, runs=20):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
        if inputs['pixel_values'].device.type == 'npu':
            torch.npu.synchronize()
        else:
            torch.cpu.synchronize() if hasattr(torch.cpu, 'synchronize') else None

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(**inputs)
            if inputs['pixel_values'].device.type == 'npu':
                torch.npu.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    return times

def benchmark_cpu():
    print("[Benchmark] CPU baseline...")
    # 重置种子，确保模型随机初始化权重与 NPU 一致
    torch.manual_seed(42)
    np.random.seed(42)
    model, processor, device = get_model_and_processor('cpu')

    # 单图延迟
    inputs = make_inputs(processor, device, batch_size=1)
    times = measure_latency(model, inputs, warmup=3, runs=10)
    cpu_single = {
        'device': 'cpu',
        'batch_size': 1,
        'mean_ms': float(np.mean(times)),
        'median_ms': float(np.median(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p99_ms': float(np.percentile(times, 99)),
    }

    # batch=4
    inputs4 = make_inputs(processor, device, batch_size=4)
    times4 = measure_latency(model, inputs4, warmup=2, runs=5)
    cpu_b4 = {
        'device': 'cpu',
        'batch_size': 4,
        'mean_ms': float(np.mean(times4)),
        'median_ms': float(np.median(times4)),
    }

    # 保存输出用于精度对比
    with torch.no_grad():
        out = model(**inputs)
    cpu_logits = out.logits.cpu().numpy()

    print(f"  CPU single mean: {cpu_single['mean_ms']:.2f} ms")
    print(f"  CPU batch=4 mean: {cpu_b4['mean_ms']:.2f} ms")
    return cpu_single, cpu_b4, cpu_logits

def benchmark_npu():
    print("[Benchmark] NPU...")
    # 重置种子，确保模型随机初始化权重与 CPU 一致
    torch.manual_seed(42)
    np.random.seed(42)
    model, processor, device = get_model_and_processor('npu:0')

    # 单图延迟
    inputs = make_inputs(processor, device, batch_size=1)
    times = measure_latency(model, inputs, warmup=10, runs=50)
    npu_single = {
        'device': 'npu',
        'batch_size': 1,
        'mean_ms': float(np.mean(times)),
        'median_ms': float(np.median(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p99_ms': float(np.percentile(times, 99)),
        'std_ms': float(np.std(times)),
    }

    # batch=4
    inputs4 = make_inputs(processor, device, batch_size=4)
    times4 = measure_latency(model, inputs4, warmup=5, runs=20)
    npu_b4 = {
        'device': 'npu',
        'batch_size': 4,
        'mean_ms': float(np.mean(times4)),
        'median_ms': float(np.median(times4)),
        'min_ms': float(np.min(times4)),
        'max_ms': float(np.max(times4)),
        'std_ms': float(np.std(times4)),
    }

    # batch=8
    inputs8 = make_inputs(processor, device, batch_size=8)
    times8 = measure_latency(model, inputs8, warmup=3, runs=10)
    npu_b8 = {
        'device': 'npu',
        'batch_size': 8,
        'mean_ms': float(np.mean(times8)),
        'median_ms': float(np.median(times8)),
        'min_ms': float(np.min(times8)),
        'max_ms': float(np.max(times8)),
        'std_ms': float(np.std(times8)),
    }

    # 保存输出用于精度对比
    with torch.no_grad():
        out = model(**inputs)
    npu_logits = out.logits.cpu().numpy()

    print(f"  NPU single mean: {npu_single['mean_ms']:.2f} ms")
    print(f"  NPU batch=4 mean: {npu_b4['mean_ms']:.2f} ms")
    print(f"  NPU batch=8 mean: {npu_b8['mean_ms']:.2f} ms")
    return npu_single, npu_b4, npu_b8, npu_logits

def compute_accuracy(cpu_logits, npu_logits):
    abs_diff = np.abs(cpu_logits - npu_logits)
    rel_diff = abs_diff / (np.abs(cpu_logits) + 1e-8)
    return {
        'max_abs_diff': float(np.max(abs_diff)),
        'mean_abs_diff': float(np.mean(abs_diff)),
        'max_rel_diff': float(np.max(rel_diff)),
        'mean_rel_diff': float(np.mean(rel_diff)),
        'top1_match_rate': float(np.sum(np.argmax(cpu_logits, -1) == np.argmax(npu_logits, -1)) / len(cpu_logits) * 100),
    }

def main():
    print("="*60)
    print("ViT-Base-Patch16-224-IN21K NPU Benchmark")
    print("="*60)

    cpu_single, cpu_b4, cpu_logits = benchmark_cpu()
    npu_single, npu_b4, npu_b8, npu_logits = benchmark_npu()
    acc = compute_accuracy(cpu_logits, npu_logits)

    report = {
        'model': 'google/vit-base-patch16-224-in21k',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cpu': {
            'single': cpu_single,
            'batch4': cpu_b4,
        },
        'npu': {
            'single': npu_single,
            'batch4': npu_b4,
            'batch8': npu_b8,
        },
        'accuracy': acc,
    }

    result_path = os.path.join(RESULT_DIR, 'benchmark_report.json')
    with open(result_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print("Accuracy Comparison (CPU vs NPU)")
    print("="*60)
    for k, v in acc.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nReport saved to: {result_path}")

if __name__ == '__main__':
    main()
