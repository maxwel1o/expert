#!/usr/bin/env python3
"""
TrOCR-Large-Printed NPU 推理脚本
基于 transformers + torch_npu 在 Ascend NPU 上运行
"""

import argparse
import logging
import time
from PIL import Image

import torch
import torch_npu

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "npu:0"):
    """加载 TrOCR 模型和处理器"""
    logger.info(f"Loading model from {model_path} on {device} ...")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
    return processor, model


def inference_single_image(processor, model, image_path: str, device: str = "npu:0") -> str:
    """对单张图片进行 OCR 推理"""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="TrOCR-Large-Printed NPU Inference")
    parser.add_argument("--model_path", type=str, default="./trocr-large-printed",
                        help="Model directory path")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--device", type=str, default="npu:0",
                        help="Device to run inference on (npu:0 or cpu)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark with warmup")
    args = parser.parse_args()

    device = args.device if torch.npu.is_available() and "npu" in args.device else "cpu"
    if device != args.device:
        logger.warning(f"NPU not available, falling back to CPU")

    processor, model = load_model(args.model_path, device)

    if args.benchmark:
        # Warmup
        logger.info("Warming up ...")
        image = Image.new("RGB", (384, 384), color="white")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(pixel_values)

        # Benchmark
        logger.info(f"Running inference on {args.image_path} ...")
        start = time.time()
        text = inference_single_image(processor, model, args.image_path, device)
        elapsed = time.time() - start
        logger.info(f"Result: {text}")
        logger.info(f"Inference time: {elapsed:.4f}s")
    else:
        text = inference_single_image(processor, model, args.image_path, device)
        print(text)


if __name__ == "__main__":
    main()
