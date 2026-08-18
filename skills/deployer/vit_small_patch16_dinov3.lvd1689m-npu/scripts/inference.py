"""vit_small_patch16_dinov3.lvd1689m  --  NPU 推理脚本

Usage:
    # 单张图片推理
    python3 inference.py --image /path/to/image.jpg

    # 随机输入推理（测试）
    python3 inference.py --random
"""
import os
import argparse
import urllib.request
import torch
import torch_npu
import numpy as np
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "npu:0"
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
CKPT_PATH = os.path.join(os.path.dirname(__file__),
                         "vit_small_patch16_dinov3_lvd1689m.pth")
# DINOv3 preprocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_model(device=DEVICE, use_local_ckpt=True):
    import timm
    model = timm.create_model(MODEL_NAME, pretrained=not use_local_ckpt)
    if use_local_ckpt and os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"  Loaded local checkpoint: {CKPT_PATH}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def preprocess_image(img_path, input_size=256):
    """Load and preprocess an image for DINOv3."""
    img = Image.open(img_path).convert("RGB")

    # Resize to 256
    img = img.resize((input_size, input_size), Image.BICUBIC)

    # To tensor
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW

    # Normalize
    for c in range(3):
        img_tensor[c] = (img_tensor[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]

    return img_tensor.unsqueeze(0)  # 1CHW


def download_sample_image():
    """Download a sample image for quick testing."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    local_path = "/tmp/dog.jpg"
    if not os.path.exists(local_path):
        print(f"  Downloading sample image from {url}")
        urllib.request.urlretrieve(url, local_path)
    return local_path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="DINOv3 ViT-Small NPU Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--random", action="store_true", help="Use random input")
    parser.add_argument("--npu-only", action="store_true", help="Skip CPU comparison")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load NPU model
    print("\n[1/4] Loading model on NPU...")
    model = load_model(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load / create input
    print("\n[2/4] Preparing input...")
    if args.random:
        x = torch.randn(1, 3, 256, 256)
        print("  Using random input")
    elif args.image:
        x = preprocess_image(args.image)
        print(f"  Loaded image: {args.image}")
    else:
        sample = download_sample_image()
        x = preprocess_image(sample)
        print(f"  Using sample image: {sample}")

    x_npu = x.to(DEVICE)

    # NPU inference
    print("\n[3/4] Running NPU inference...")
    torch.npu.synchronize(DEVICE)
    import time
    t0 = time.perf_counter()
    out = model(x_npu)
    torch.npu.synchronize(DEVICE)
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"  Latency: {t_ms:.2f} ms")
    print(f"  Output shape: {out.shape}")
    print(f"  Output (first 10 dims): {out[0, :10].tolist()}")

    # Print feature statistics
    out_cpu = out.float().cpu()
    print(f"\n  Feature stats:")
    print(f"    mean: {out_cpu.mean().item():.6f}")
    print(f"    std:  {out_cpu.std().item():.6f}")
    print(f"    min:  {out_cpu.min().item():.6f}")
    print(f"    max:  {out_cpu.max().item():.6f}")

    # Compare with CPU if requested
    if not args.npu_only:
        print("\n[4/4] Comparing with CPU baseline...")
        model_cpu = load_model("cpu")
        t0 = time.perf_counter()
        out_cpu = model_cpu(x)
        t_cpu = (time.perf_counter() - t0) * 1000
        out_np = out.float().cpu()

        cos = torch.nn.functional.cosine_similarity(out_np, out_cpu, dim=1).item()
        mae = (out_np - out_cpu).abs().mean().item()
        log = f"  Cosine similarity: {cos:.6f}  |  MAE: {mae:.6f}  |  CPU {t_cpu:.0f}ms -> NPU {t_ms:.0f}ms ({t_cpu/t_ms:.0f}x)"
        print(log)

    print("\nDone.")


if __name__ == "__main__":
    main()
