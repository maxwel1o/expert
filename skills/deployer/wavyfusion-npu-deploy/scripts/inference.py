#!/usr/bin/env python3
"""
WavyFusion — Stable Diffusion 1.5 Fine-tune on Ascend NPU
==========================================================
Model: wavymulder/wavyfusion
Hardware: Ascend 910B4 NPU
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch_npu
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def get_device():
    if torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def load_pipeline(model_path, device, dtype=torch.float16):
    print(f"[INFO] Loading pipeline (dtype={dtype}, device={device})...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"[INFO] Pipeline loaded successfully.", flush=True)
    return pipe


@torch.no_grad()
def generate(
    pipe,
    prompt,
    seed=42,
    steps=25,
    guidance_scale=7.5,
    height=512,
    width=512,
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    ).images[0]
    return image


def benchmark(pipe, prompt, steps=25, num_runs=5):
    print(f"[BENCH] Running {num_runs}x benchmark (steps={steps})...", flush=True)
    times = []
    for i in range(num_runs):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        generate(pipe, prompt, seed=42 + i, steps=steps)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s", flush=True)

    times = times[1:]  # skip first (cold start)
    mean = np.mean(times)
    std = np.std(times)
    print(f"[BENCH] Mean: {mean:.3f}s ± {std:.3f}s")
    print(f"[BENCH] Throughput: {steps / mean:.1f} steps/s")
    return {"mean": mean, "std": std, "throughput": steps / mean}


def main():
    parser = argparse.ArgumentParser(description="WavyFusion NPU Inference")
    parser.add_argument("--model-path", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--prompt", type=str,
                        default="a beautiful landscape in wavy art style, surreal waves of color")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--benchmark", action="store_true", default=False)
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Device: {device}", flush=True)

    pipe = load_pipeline(args.model_path, device)

    print(f"[INFO] Generating: \"{args.prompt}\"", flush=True)
    t0 = time.time()
    image = generate(pipe, args.prompt, args.seed, args.steps, args.guidance)
    elapsed = time.time() - t0
    print(f"[INFO] Done in {elapsed:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image.save(args.output)
    print(f"[INFO] Saved: {args.output}", flush=True)

    if args.benchmark and device.type == "npu":
        benchmark(pipe, args.prompt, args.steps)


if __name__ == "__main__":
    main()
