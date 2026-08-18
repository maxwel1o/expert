#!/usr/bin/env python3
"""
TrOCR-Large-Printed NPU 精度评测脚本
核心目标：对比 NPU 与 CPU 推理结果，确保误差 < 1%
"""
import argparse
import logging
from PIL import Image, ImageDraw, ImageFont

import torch
import torch_npu
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEST_SAMPLES = [
    "Hello World\nOCR Test\nPrinted 123",
    "The quick brown fox\njumps over the lazy dog",
    "Transformer models\nAttention Is All You Need",
    "Machine Learning\nDeep Learning\nNeural Networks",
    "Python Programming\nData Science\nAI Applications",
]


def create_test_image(text: str) -> Image.Image:
    img = Image.new("RGB", (384, 384), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except (OSError, IOError):
        font = ImageFont.load_default()
    y = 20
    for line in text.split("\n"):
        draw.text((20, y), line, fill="black", font=font)
        y += 50
    return img


def inference(processor, model, image, device):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="TrOCR NPU Accuracy Evaluation")
    parser.add_argument("--model_path", type=str, default="./trocr-large-printed")
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    device = args.device if torch.npu.is_available() and "npu" in args.device else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from {args.model_path} ...")
    processor = TrOCRProcessor.from_pretrained(args.model_path)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    cpu_model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    cpu_model.eval()

    # =============================================
    # Part 1: Encoder output comparison
    # =============================================
    logger.info("=" * 60)
    logger.info("Part 1: Encoder output numerical comparison (NPU vs CPU)")
    logger.info("=" * 60)

    enc_diffs = []
    for sample in TEST_SAMPLES:
        image = create_test_image(sample)
        px = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            enc_npu = model.get_encoder()(px.to(device))
            enc_cpu = cpu_model.get_encoder()(px)

        diff = (enc_npu.last_hidden_state.cpu() - enc_cpu.last_hidden_state).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        enc_diffs.append(max_diff)
        close = torch.allclose(enc_npu.last_hidden_state.cpu(), enc_cpu.last_hidden_state, rtol=1e-3, atol=1e-5)
        logger.info(f"  Sample '{sample[:20]}...' max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} allclose={close}")

    logger.info(f"  Max encoder diff across samples: {max(enc_diffs):.4f}")

    # =============================================
    # Part 2: Generation consistency check
    # =============================================
    logger.info("=" * 60)
    logger.info("Part 2: Generation consistency (NPU vs CPU text output)")
    logger.info("=" * 60)

    results = []
    diffs = 0

    for i, sample_text in enumerate(TEST_SAMPLES):
        image = create_test_image(sample_text)
        npu_text = inference(processor, model, image, device)
        cpu_text = inference(processor, cpu_model, image, "cpu")
        is_match = npu_text.strip().lower() == cpu_text.strip().lower()
        if not is_match:
            diffs += 1
            logger.info(f"  [{i+1}] DIFF: NPU='{npu_text}' vs CPU='{cpu_text}'")
        else:
            logger.info(f"  [{i+1}] MATCH: '{npu_text[:40]}'")
        results.append((sample_text, npu_text, cpu_text, is_match))

    match_rate = (1 - diffs / len(TEST_SAMPLES)) * 100
    logger.info(f"Match rate: {match_rate:.1f}% ({len(TEST_SAMPLES) - diffs}/{len(TEST_SAMPLES)})")

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
    for sample in TEST_SAMPLES:
        image = create_test_image(sample)
        r1 = inference(processor, model, image, device)
        r2 = inference(processor, model, image, device)
        if r1 != r2:
            self_diffs += 1
            logger.info(f"  NPU self-diff: '{r1}' vs '{r2}'")

    logger.info(f"NPU self-consistency: {len(TEST_SAMPLES) - self_diffs}/{len(TEST_SAMPLES)}")

    # Output table
    print()
    print("## 精度测试结果")
    print()
    print("| # | NPU 推理结果 | CPU 推理结果 | 一致? |")
    print("|---|------------|-------------|------|")
    for i, (_, npu_t, cpu_t, match) in enumerate(results):
        print(f"| {i+1} | {npu_t[:40]:40s} | {cpu_t[:40]:40s} | {'✓' if match else '✗'} |")
    print(f"| **总一致率** | | **{match_rate:.1f}%** | |")


if __name__ == "__main__":
    main()
