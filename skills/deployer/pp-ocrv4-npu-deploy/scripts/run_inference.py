#!/usr/bin/env python3
"""
Generic ONNX inference script for PP-OCRv4 models on CPU and Ascend NPU.
Usage: python3 run_inference.py --model_path <path> --image_path <path> --device <cpu|npu> --model_type <detection|recognition>
"""

import argparse, os, sys, time
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.onnx_utils import create_session, run_inference, clean_npu_cache


def preprocess_det(image, det_limit_side_len=960):
    h, w = image.shape[:2]
    ratio = 1.0
    if max(h, w) > det_limit_side_len:
        ratio = det_limit_side_len / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
    elif min(h, w) < 32:
        ratio = 32.0 / min(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
    else:
        new_h, new_w = h, w
    pad_h, pad_w = (32 - new_h % 32) % 32, (32 - new_w % 32) % 32
    img = cv2.resize(image, (new_w, new_h))
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, axis=0)


def preprocess_rec(crop_img, rec_img_h=48):
    h, w = crop_img.shape[:2]
    new_w = max(int(w * rec_img_h / h), 1)
    img = cv2.resize(crop_img, (new_w, rec_img_h))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"])
    parser.add_argument("--model_type", type=str, default="detection", choices=["detection", "recognition"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {args.image_path}")

    if args.model_type == "detection":
        img_tensor = preprocess_det(image)
    else:
        img_tensor = preprocess_rec(image)

    session = create_session(args.model_path, args.device)
    outputs, timing = run_inference(session, {session.get_inputs()[0].name: img_tensor.astype(np.float32)})

    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Time (mean): {timing['mean']:.4f}s")

    if args.output:
        np.save(args.output, outputs[0])
        print(f"  Saved to: {args.output}")

    clean_npu_cache()


if __name__ == "__main__":
    main()
