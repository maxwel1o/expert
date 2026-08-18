"""
OpenJourney 昇腾 NPU 性能基准测试
"""
import torch
import torch_npu
import time
import numpy as np
from diffusers import StableDiffusionPipeline
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/openjourney", help="模型路径")
    parser.add_argument("--output", default="./benchmark_results.json", help="输出文件")
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("npu")
    pipe.enable_attention_slicing()

    prompt = "mdjrnm-syle portrait photo of a girl, highly detailed, digital painting"
    results = []

    for steps in [10, 20, 30]:
        print(f"\nBenchmark: {steps} steps")

        # Cold run (w/ compilation)
        gen = torch.Generator(device="npu").manual_seed(42)
        start = time.time()
        _ = pipe(prompt, num_inference_steps=steps, generator=gen).images[0]
        cold_time = time.time() - start

        # Warm runs
        warm_times = []
        for _ in range(3):
            gen = torch.Generator(device="npu").manual_seed(42)
            start = time.time()
            _ = pipe(prompt, num_inference_steps=steps, generator=gen).images[0]
            warm_times.append(time.time() - start)

        avg = np.mean(warm_times)
        std = np.std(warm_times)
        print(f"  Cold: {cold_time:.2f}s | Warm: {avg:.2f}s +- {std:.3f}s | {steps/avg:.2f} step/s")

        results.append({
            "steps": steps,
            "cold_time_s": round(cold_time, 2),
            "warm_avg_s": round(avg, 2),
            "warm_std_s": round(std, 3),
            "steps_per_sec": round(steps / avg, 2),
        })

    mem = torch.npu.mem_get_info(0)
    report = {
        "model": "prompthero/openjourney",
        "device": "Ascend 910B4",
        "resolution": "512x512",
        "dtype": "float16",
        "benchmark": results,
        "memory_gb": {
            "free": round(mem[0] / 1024**3, 2),
            "total": round(mem[1] / 1024**3, 2),
            "used": round((mem[1] - mem[0]) / 1024**3, 2),
        },
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
