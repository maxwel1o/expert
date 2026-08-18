"""
prompthero/openjourney 昇腾 NPU 推理脚本
基于 diffusers + torch_npu

Usage:
    python inference.py \
        --model ./models/openjourney \
        --prompt "mdjrnm-syle portrait" \
        --steps 25 \
        --seed 42 \
        --output ./output
"""
import torch
import torch_npu
import time
import argparse
import os
from diffusers import StableDiffusionPipeline


def load_model(model_path: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("npu")
    pipe.enable_attention_slicing()
    return pipe


def generate(pipe, prompt: str, steps: int = 25, guidance: float = 7.5, seed: int = None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device="npu").manual_seed(seed)
    return pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]


def main():
    parser = argparse.ArgumentParser(description="openjourney on Ascend NPU")
    parser.add_argument("--model", default="./models/openjourney", help="模型路径")
    parser.add_argument("--prompt", nargs="+", required=True, help="生成提示词")
    parser.add_argument("--steps", type=int, default=25, help="推理步数")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--output", default="./output", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading model from {args.model} ...")
    pipe = load_model(args.model)
    print("Model loaded on Ascend NPU")

    for i, prompt in enumerate(args.prompt):
        print(f"[{i+1}/{len(args.prompt)}] Generating: '{prompt}'")
        start = time.time()
        image = generate(pipe, prompt, args.steps, args.guidance, args.seed)
        elapsed = time.time() - start
        fpath = os.path.join(args.output, f"openjourney_{i}.png")
        image.save(fpath)
        print(f"  Done in {elapsed:.1f}s -> {fpath}")

    print(f"All images saved to {args.output}/")


if __name__ == "__main__":
    main()
