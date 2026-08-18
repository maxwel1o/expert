#!/usr/bin/env python3
"""
GIT-NPU 性能基准测试脚本
测量 NPU 和 CPU 上的推理延迟和吞吐量
支持 GIT-Base / GIT-Large 及 COCO / TextCaps 微调版本
"""
import argparse
import logging
import time
from PIL import Image

import torch
import torch_npu
from transformers import AutoProcessor, GitForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def benchmark_model(processor, model, device, num_runs=10, warmup=3):
    """运行性能基准测试"""
    image = Image.new("RGB", (224, 224), color="white")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Warmup
    logger.info(f"Warming up ({warmup} runs) ...")
    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=20)

    # Benchmark
    logger.info(f"Benchmarking ({num_runs} runs) ...")
    times = []
    for i in range(num_runs):
        start = time.time()
        _ = model.generate(**inputs, max_new_tokens=20)
        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"  Run {i+1}/{num_runs}: {elapsed:.4f}s")

    avg = sum(times) / len(times)
    logger.info(f"Average inference time: {avg:.4f}s")
    logger.info(f"Min: {min(times):.4f}s, Max: {max(times):.4f}s")
    return times


def main():
    parser = argparse.ArgumentParser(description="GIT-NPU Performance Benchmark")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    device = args.device if torch.npu.is_available() and "npu" in args.device else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = GitForCausalLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    times = benchmark_model(processor, model, device, num_runs=args.runs)

    print()
    print("## 性能测试结果")
    print()
    print(f"| 指标 | 数值 |")
    print(f"| --- | --- |")
    print(f"| 设备 | {device} |")
    print(f"| 运行次数 | {args.runs} |")
    print(f"| 平均耗时 | {sum(times)/len(times):.4f}s |")
    print(f"| 最小耗时 | {min(times):.4f}s |")
    print(f"| 最大耗时 | {max(times):.4f}s |")


if __name__ == "__main__":
    main()
