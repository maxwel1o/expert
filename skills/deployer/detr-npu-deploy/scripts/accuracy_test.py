"""
DETR NPU 精度对比测试脚本（通用版）
对比 CPU (float32) 和 NPU (float32) 的输出差异
"""
import argparse
import os
import warnings
import json
import time

import torch
import torch_npu  # noqa: F401
import requests
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


def parse_args():
    parser = argparse.ArgumentParser(description="DETR NPU Accuracy Test")
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--image", type=str,
                        default="http://images.cocodataset.org/val2017/000000039769.jpg")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="accuracy_report.json")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    warnings.filterwarnings("ignore")

    if not torch.npu.is_available():
        print("[ERROR] NPU not available")
        return

    model_path = args.model_path
    model_name = os.path.basename(model_path.rstrip("/"))

    print("=" * 60)
    print(f"  {model_name} Accuracy Test: NPU vs CPU")
    print("=" * 60)

    processor = DetrImageProcessor.from_pretrained(model_path)

    # CPU model
    model_cpu = DetrForObjectDetection.from_pretrained(model_path)
    model_cpu.eval()
    cpu_params = sum(p.numel() for p in model_cpu.parameters()) / 1e6

    # NPU model
    model_npu = DetrForObjectDetection.from_pretrained(model_path)
    model_npu.eval().to("npu")
    npu_dev = torch.npu.get_device_name(0)

    # Load image
    image_source = args.image
    if image_source.startswith(("http://", "https://")):
        image = Image.open(requests.get(image_source, stream=True, timeout=30).raw)
    else:
        image = Image.open(image_source)
    orig_size = image.size
    print(f"[INFO] Image: {image_source} ({orig_size})")

    # CPU inference
    inputs = processor(images=image, return_tensors="pt")
    t0 = time.perf_counter()
    outputs_cpu = model_cpu(**inputs)
    t_cpu = time.perf_counter() - t0

    # NPU inference
    inputs_npu = {k: v.to("npu") for k, v in inputs.items()}
    torch.npu.synchronize()
    t0 = time.perf_counter()
    outputs_npu = model_npu(**inputs_npu)
    torch.npu.synchronize()
    t_npu = time.perf_counter() - t0

    # Calculate accuracy metrics
    logits_cpu, logits_npu = outputs_cpu.logits, outputs_npu.logits.cpu()
    boxes_cpu, boxes_npu = outputs_cpu.pred_boxes, outputs_npu.pred_boxes.cpu()

    logits_abs_diff = (logits_cpu - logits_npu).abs()
    logits_rel_diff = logits_abs_diff / (logits_cpu.abs() + 1e-8)
    boxes_abs_diff = (boxes_cpu - boxes_npu).abs()

    results = processor.post_process_object_detection(
        outputs_cpu, threshold=args.threshold,
        target_sizes=torch.tensor([orig_size[::-1]])
    )[0]
    results_npu = processor.post_process_object_detection(
        outputs_npu, threshold=args.threshold,
        target_sizes=torch.tensor([orig_size[::-1]])
    )[0]

    metrics = {
        "model": model_name,
        "device_npu": npu_dev,
        "params_m": round(cpu_params, 1),
        "test_image": image_source,
        "image_size": list(orig_size),
        "cpu_time_s": round(t_cpu, 4),
        "npu_time_s": round(t_npu, 4),
        "logits": {
            "max_abs_diff": round(logits_abs_diff.max().item(), 8),
            "mean_abs_diff": round(logits_abs_diff.mean().item(), 8),
            "mean_rel_diff_pct": round(logits_rel_diff.mean().item() * 100, 4),
        },
        "boxes": {
            "max_abs_diff": round(boxes_abs_diff.max().item(), 8),
            "mean_abs_diff": round(boxes_abs_diff.mean().item(), 8),
        },
        "passed": logits_rel_diff.mean().item() < 0.01,
    }

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Accuracy Report")
    print(f"{'=' * 60}")
    print(f"  Logits mean rel error: {metrics['logits']['mean_rel_diff_pct']:.4f}%")
    print(f"  Requirement: < 1%")
    print(f"  Result: {'✅ PASS' if metrics['passed'] else '❌ FAIL'}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Report saved: {args.output}")


if __name__ == "__main__":
    main()
