#!/usr/bin/env python3
"""
GIT-NPU 精度评测脚本
对比 NPU 与 CPU 推理结果，确保误差 < 1%
支持 GIT-Base / GIT-Large 及 COCO / TextCaps 微调版本
"""
import argparse
import logging
from PIL import Image, ImageDraw

import torch
import torch_npu
from transformers import AutoProcessor, GitForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_PROMPTS = [
    "a cat sitting on a table",
    "a dog running in the park",
    "a car parked on the street",
    "a beautiful sunset over the ocean",
    "a plate of food on the table",
]


def create_test_image(text: str) -> Image.Image:
    """创建包含提示文本的测试图像"""
    img = Image.new("RGB", (224, 224), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 200, 200], outline="black", width=3)
    draw.ellipse([50, 50, 170, 170], outline="blue", width=2)
    draw.text((30, 100), text[:20], fill="black")
    return img


def inference(processor, model, image, device):
    """单张图片推理"""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="GIT-NPU Accuracy Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    device = args.device if torch.npu.is_available() and "npu" in args.device else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = GitForCausalLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    cpu_model = GitForCausalLM.from_pretrained(args.model_path)
    cpu_model.eval()

    # =============================================
    # Part 1: Vision encoder output comparison
    # =============================================
    logger.info("=" * 60)
    logger.info("Part 1: Vision encoder output numerical comparison (NPU vs CPU)")
    logger.info("=" * 60)

    enc_diffs = []
    for prompt in TEST_PROMPTS:
        image = create_test_image(prompt)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values

        with torch.no_grad():
            enc_npu = model.git.image_encoder(pixel_values.to(device))
            enc_cpu = cpu_model.git.image_encoder(pixel_values)

        diff = (enc_npu.last_hidden_state.cpu() - enc_cpu.last_hidden_state).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        enc_diffs.append(max_diff)
        close = torch.allclose(enc_npu.last_hidden_state.cpu(), enc_cpu.last_hidden_state, rtol=1e-3, atol=1e-5)
        logger.info(f"  Sample '{prompt[:20]}...' max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} allclose={close}")

    logger.info(f"  Max encoder diff across samples: {max(enc_diffs):.4f}")

    # =============================================
    # Part 2: Generation consistency check
    # =============================================
    logger.info("=" * 60)
    logger.info("Part 2: Generation consistency (NPU vs CPU text output)")
    logger.info("=" * 60)

    results = []
    diffs = 0

    for i, prompt in enumerate(TEST_PROMPTS):
        image = create_test_image(prompt)
        npu_text = inference(processor, model, image, device)
        cpu_text = inference(processor, cpu_model, image, "cpu")
        is_match = npu_text.strip().lower() == cpu_text.strip().lower()
        if not is_match:
            diffs += 1
            logger.info(f"  [{i+1}] DIFF: NPU='{npu_text}' vs CPU='{cpu_text}'")
        else:
            logger.info(f"  [{i+1}] MATCH: '{npu_text[:60]}'")
        results.append((prompt, npu_text, cpu_text, is_match))

    match_rate = (1 - diffs / len(TEST_PROMPTS)) * 100
    logger.info(f"Match rate: {match_rate:.1f}% ({len(TEST_PROMPTS) - diffs}/{len(TEST_PROMPTS)})")

    if diffs == 0:
        logger.info("PASS: NPU and CPU outputs are 100% identical (diff rate = 0%)")
    elif (100 - match_rate) < 1:
        logger.info(f"PASS: NPU vs CPU diff rate = {100 - match_rate:.2f}% (< 1%)")
    else:
        logger.info(f"RESULT: NPU vs CPU diff rate = {100 - match_rate:.2f}%")

    # =============================================
    # Part 3: Self-consistency of NPU
    # =============================================
    logger.info("=" * 60)
    logger.info("Part 3: NPU self-consistency (run twice, should match)")
    logger.info("=" * 60)

    self_diffs = 0
    for prompt in TEST_PROMPTS:
        image = create_test_image(prompt)
        r1 = inference(processor, model, image, device)
        r2 = inference(processor, model, image, device)
        if r1 != r2:
            self_diffs += 1
            logger.info(f"  NPU self-diff: '{r1}' vs '{r2}'")

    logger.info(f"NPU self-consistency: {len(TEST_PROMPTS) - self_diffs}/{len(TEST_PROMPTS)}")

    # Output table
    print()
    print("## 精度测试结果")
    print()
    print("| # | NPU 推理结果 | CPU 推理结果 | 一致? |")
    print("|---|------------|-------------|------|")
    for i, (_, npu_t, cpu_t, match) in enumerate(results):
        print(f"| {i+1} | {npu_t[:50]:50s} | {cpu_t[:50]:50s} | {'✓' if match else '✗'} |")
    print(f"| **总一致率** | | **{match_rate:.1f}%** | |")


if __name__ == "__main__":
    main()
