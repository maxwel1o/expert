#!/usr/bin/env python3
"""Swin Transformer inference on Ascend NPU and CPU.

Reusable for all swin_* variants from timm. Usage:
  python3 inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device cpu
  python3 inference.py --model swin_tiny_patch4_window7_224.ms_in1k --device npu
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model
from timm.data import create_transform, resolve_data_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_class_labels(model_name: str) -> list[str] | None:
    """Try to load ImageNet class labels from timm or local cache."""
    try:
        from timm.data import IMAGENET_1k_CLASSES as cls
        return cls
    except ImportError:
        pass
    try:
        import json
        p = Path.home() / ".cache" / "timm" / "imagenet_classes.json"
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None


def load_image(path: str | Path, input_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    transform = create_transform(
        input_size=input_size,
        is_training=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return transform(img).unsqueeze(0)  # [1,3,H,W]


def get_test_image() -> torch.Tensor:
    """Return a dummy test tensor if no image file is available."""
    import numpy as np
    rand_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    from torchvision.transforms import functional as TF
    img = TF.to_tensor(rand_img).unsqueeze(0)
    return img


def main():
    parser = argparse.ArgumentParser(description="Swin Transformer inference")
    parser.add_argument("--model", required=True, help="timm model name")
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--image", default=None, help="input image path")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override num_classes. Auto-detected from model config if not set.")
    args = parser.parse_args()

    # Auto-detect num_classes from model config
    if args.num_classes is None:
        model_safe = args.model.replace('/', '--')
        cfg_paths = list(Path(f"/opt/atomgit/.cache/huggingface/hub/models--timm--{model_safe}").glob("**/config.json"))
        if cfg_paths:
            try:
                cfg = json.loads(cfg_paths[0].read_text())
                args.num_classes = cfg.get("num_classes", 1000)
            except Exception:
                args.num_classes = 1000
        else:
            args.num_classes = 1000

    device = torch.device(args.device)

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")
    print(f"num_classes: {args.num_classes}")
    t0 = time.time()

    model = create_model(args.model, pretrained=True, num_classes=args.num_classes)
    model.eval()

    model = model.to(device)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    data_config = resolve_data_config(model=model, use_test_size=True)
    input_size = data_config.get("input_size", (3, 224, 224))[-1]
    print(f"Input size from config: {data_config.get('input_size', '(3,224,224)')}")

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {args.image}, using random input")
            input_tensor = get_test_image()
        else:
            input_tensor = load_image(image_path, input_size=input_size)
    else:
        input_tensor = get_test_image()

    if input_tensor.shape[-1] != input_size:
        input_tensor = F.interpolate(input_tensor, size=(input_size, input_size),
                                     mode="bilinear", align_corners=False)

    input_tensor = input_tensor.to(device)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    # Timed inference
    torch.cuda.synchronize() if args.device == "npu" else None
    t0 = time.time()
    n_runs = 10
    all_outputs = []
    with torch.no_grad():
        for i in range(n_runs):
            out = model(input_tensor)
            all_outputs.append(out)
    if args.device == "npu":
        torch.npu.synchronize()
    elapsed = time.time() - t0
    print(f"Inference: {n_runs} runs in {elapsed:.3f}s, avg {elapsed/n_runs*1000:.2f}ms/run")

    output = all_outputs[-1]
    probs = F.softmax(output, dim=1)
    top5_val, top5_idx = torch.topk(probs, 5, dim=1)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"\nTop-5 predictions:")
    classes = load_class_labels(args.model)
    for i in range(5):
        idx = top5_idx[0, i].item()
        label = classes[idx] if classes and idx < len(classes) else f"class_{idx}"
        print(f"  {i + 1}. {label}: {top5_val[0, i].item():.6f}")

    # Save output for comparison
    out_path = Path(f"{args.model.replace('/', '_')}_{args.device}.pt")
    torch.save(output.cpu(), out_path)
    print(f"\nOutput saved to: {out_path}")

    # Resource cleanup
    del model, input_tensor, output, all_outputs
    gc.collect()
    if args.device == "npu":
        torch.npu.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
