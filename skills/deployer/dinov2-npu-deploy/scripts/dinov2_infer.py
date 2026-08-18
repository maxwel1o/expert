#!/usr/bin/env python3
"""
vit_base_patch14_dinov2.lvd142m — Ascend NPU 推理脚本

Usage:
    # 单张图片推理
    python3 inference.py --image /path/to/image.jpg

    # 批量推理
    python3 inference.py --image-dir /path/to/images/

    # 指定精度
    python3 inference.py --image img.jpg --dtype fp16
"""

import argparse
import os
import sys
from typing import List, Optional, Union

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from transformers import AutoModel, AutoImageProcessor


MODEL_PATH = os.environ.get(
    "DINOV2_MODEL_PATH",
    "/opt/atomgit/.cache/modelscope/facebook/dinov2-base",
)


class Dinov2Inference:
    """DINOv2 ViT-B/14 推理封装 (NPU/CPU)."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: Optional[str] = None,
        dtype: str = "fp32",
    ):
        if device is None:
            device = "npu:0" if torch.npu.is_available() else "cpu"

        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32

        print(f"[Dinov2Inference] Loading model from {model_path}")
        print(f"[Dinov2Inference] Device: {self.device}, dtype: {dtype}")

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device, dtype=self.dtype)

        self.processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def preprocess(self, image: Union[np.ndarray, "PIL.Image.Image"]) -> torch.Tensor:
        """预处理单张图片，返回 NPU tensor [1, 3, 224, 224]."""
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def embed(self, image: Union[np.ndarray, "PIL.Image.Image"]) -> np.ndarray:
        """单张图片 -> CLS embedding [768]."""
        pixel_values = self.preprocess(image)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0].cpu().numpy().flatten()

    @torch.no_grad()
    def embed_batch(self, images: List[Union[np.ndarray, "PIL.Image.Image"]]) -> np.ndarray:
        """多张图片批量推理 -> [N, 768]."""
        pixel_values = torch.cat([self.preprocess(img) for img in images], dim=0)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0].cpu().numpy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del self.model
        if "npu" in str(self.device):
            torch.npu.empty_cache()


def load_image(path: str):
    """加载图片为 numpy array (HWC uint8)."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")

    img = Image.open(path).convert("RGB")
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description="DINOv2 NPU Inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--image-dir", type=str, help="Directory of input images")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.print_help()
        sys.exit(1)

    with Dinov2Inference(model_path=args.model_path, dtype=args.dtype) as infer:
        if args.image:
            img = load_image(args.image)
            emb = infer.embed(img)
            print(f"Embedding shape: {emb.shape}")
            print(f"Embedding[:10]:  {emb[:10].tolist()}")
            print(f"Stats: mean={emb.mean():.6f}, std={emb.std():.6f}")

        if args.image_dir:
            import glob
            paths = sorted(glob.glob(os.path.join(args.image_dir, "*")))
            paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if not paths:
                print(f"No images found in {args.image_dir}")
                sys.exit(1)
            print(f"Found {len(paths)} images")
            images = [load_image(p) for p in paths]
            embs = infer.embed_batch(images)
            print(f"Batch embeddings shape: {embs.shape}")


if __name__ == "__main__":
    main()
