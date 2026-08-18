"""
DETR NPU 推理脚本（通用版，支持 DETR-ResNet-50 和 DETR-ResNet-101）
"""
import argparse
import os
import warnings
from PIL import Image, ImageDraw, ImageFont

import torch
import torch_npu  # noqa: F401
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection


def parse_args():
    parser = argparse.ArgumentParser(description="DETR NPU Inference")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--image", type=str,
                        default="http://images.cocodataset.org/val2017/000000039769.jpg")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="npu", choices=["cpu", "npu"])
    parser.add_argument("--save", type=str, default="result.jpg")
    return parser.parse_args()


def load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        return Image.open(requests.get(source, stream=True, timeout=30).raw)
    return Image.open(source)


@torch.no_grad()
def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    device = "npu" if (args.device == "npu" and torch.npu.is_available()) else "cpu"
    if device == "npu":
        print(f"[INFO] NPU: {torch.npu.get_device_name(0)}")
    else:
        print("[INFO] Using CPU")

    processor = DetrImageProcessor.from_pretrained(args.model_path)
    model = DetrForObjectDetection.from_pretrained(args.model_path)
    model.eval().to(device)

    image = load_image(args.image)
    orig = image.copy()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    target_sizes = torch.tensor([orig.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=args.threshold, target_sizes=target_sizes
    )[0]

    print(f"\n[RESULTS] Detected {len(results['labels'])} objects:")
    for s, l, b in zip(results["scores"], results["labels"], results["boxes"]):
        print(f"  {model.config.id2label[l.item()]:15s} score={s.item():.3f}  "
              f"box=[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]")

    if args.save:
        draw = ImageDraw.Draw(orig)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.tolist()
            text = f"{model.config.id2label[label.item()]}: {score.item():.3f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(bbox, fill="red")
            draw.text((x1, y1), text, fill="white", font=font)
        orig.save(args.save)
        print(f"[INFO] Result saved: {args.save}")


if __name__ == "__main__":
    main()
