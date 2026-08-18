#!/usr/bin/env python3
"""
AIMv2 通用昇腾 NPU 推理脚本
支持所有 12 个变体（large/huge/1B/3B × 224/336/448）

用法: MODEL_SIZE=3B IMG_SIZE=224 python3 inference.py
"""
import os, sys, time, json
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoConfig

# ============ 配置 ============
MODEL_SIZE = os.environ.get('MODEL_SIZE', '3B')
IMG_SIZE = int(os.environ.get('IMG_SIZE', '224'))
MODEL_NAME = f'aimv2-{MODEL_SIZE}-patch14-{IMG_SIZE}'
MODEL_PATH = os.path.expanduser(f'~/.cache/modelscope/hub/models/apple/{MODEL_NAME}')
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'npu:0'
dtype = torch.float16


def load_model():
    """加载转换后的 AIMv2 模型到 NPU"""
    print(f"Loading {MODEL_NAME} ({MODEL_SIZE}, {IMG_SIZE}×{IMG_SIZE})")
    print(f"Model path: {MODEL_PATH}")

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Config: image_size={config.image_size}, layers={config.num_hidden_layers}, "
          f"hidden={config.hidden_size}")

    model = AutoModel.from_config(config, trust_remote_code=True)

    # 尝试加载转换后的权重
    converted_path = os.path.join(MODEL_PATH, 'converted_model.pth')
    if not os.path.exists(converted_path):
        print(f"! converted_model.pth not found at {converted_path}")
        print("! Run aimv2_weight_convert.py first")
        sys.exit(1)

    print(f"Loading weights from: {converted_path}")
    state_dict = torch.load(converted_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    print(f"✓ Weights loaded (strict=True, {len(state_dict)} groups)")

    model = model.to(device).to(dtype)
    model.eval()
    return model, config


def preprocess_image(size=224):
    """预处理图像为模型输入"""
    img = Image.fromarray(
        (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    )
    print("  (using random test image)")

    img = img.resize((size, size), Image.BICUBIC)
    img_array = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


@torch.no_grad()
def run_inference(model, img_tensor, warmup=3, runs=10):
    """推理 + 性能测试"""
    img_tensor = img_tensor.to(device).to(dtype)

    # Warmup
    for _ in range(warmup):
        _ = model(img_tensor)
    torch.npu.synchronize()

    # Benchmark
    latencies = []
    for _ in range(runs):
        torch.npu.synchronize()
        t0 = time.time()
        out = model(img_tensor)
        torch.npu.synchronize()
        latencies.append(time.time() - t0)

    latencies = np.array(latencies)
    stats = {
        'mean_ms': float(latencies.mean() * 1000),
        'median_ms': float(np.median(latencies) * 1000),
        'p99_ms': float(np.percentile(latencies, 99) * 1000),
        'min_ms': float(latencies.min() * 1000),
        'max_ms': float(latencies.max() * 1000),
        'throughput_img_per_s': float(1.0 / latencies.mean()),
    }

    print(f"  Mean latency: {stats['mean_ms']:.1f} ms")
    print(f"  Throughput: {stats['throughput_img_per_s']:.2f} img/s")

    return out, stats


def analyze_output(output):
    """分析模型输出特征"""
    if hasattr(output, 'last_hidden_state'):
        features = output.last_hidden_state
    elif isinstance(output, torch.Tensor):
        features = output
    else:
        return {}

    pooled = features.mean(dim=1)
    results = {
        'feature_shape': list(features.shape),
        'feature_norm': float(features.norm().cpu().item()),
        'feature_mean': float(features.mean().cpu().item()),
        'feature_std': float(features.std().cpu().item()),
        'pooled_shape': list(pooled.shape),
        'pooled_norm': float(pooled.norm().cpu().item()),
    }
    return results


def main():
    print("=" * 60)
    print(f"AIMv2 {MODEL_NAME} 昇腾 NPU 推理")
    print("=" * 60)
    print(f"Device: {device} ({torch.npu.get_device_name(0)})")
    print(f"Dtype:  {dtype}")

    t_start = time.time()

    model, config = load_model()
    img_tensor = preprocess_image(size=IMG_SIZE)
    output, perf_stats = run_inference(model, img_tensor)
    out_analysis = analyze_output(output)

    t_total = time.time() - t_start

    results = {
        "model": MODEL_NAME,
        "model_size": MODEL_SIZE,
        "image_size": IMG_SIZE,
        "device": torch.npu.get_device_name(0),
        "dtype": str(dtype),
        "total_time_s": round(t_total, 2),
        "performance": perf_stats,
        "output": out_analysis,
    }

    report_path = os.path.join(OUTPUT_DIR, f'inference_report_{MODEL_NAME}.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved: {report_path}")
    print(f"Total time: {t_total:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
