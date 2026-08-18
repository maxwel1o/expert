#!/usr/bin/env python3
"""YOLOv10n NPU 综合性能基准测试脚本

运行 9 组配置对比, 输出完整性能表格。

用法:
  python yolov10_npu_benchmark.py --model yolov10n.pt --image test.jpg --runs 50

依赖:
  pip install ultralytics pillow numpy torch torch_npu
"""

import argparse
import time
import sys
import os
import threading
from queue import Queue
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch_npu

from ultralytics.utils import ops


CONFIGS = [
    # (name, device, half, compile, npu_opt, stream, batch_size)
    ("01-CPU FP32",           "cpu",  False, False, False, False, 1),
    ("02-NPU FP32",           "npu", False, False, False, False, 1),
    ("03-NPU FP16",           "npu", True,  False, False, False, 1),
    ("04-NPU FP16+compile",   "npu", True,  True,  False, False, 1),
    ("05-NPU FP16+compile+opt", "npu", True, True, True, False, 1),
    ("06-NPU FP16+compile+opt-autotune", "npu", True, True, True, False, 1),
    ("07-NPU FP16+compile+batch4", "npu", True, True, True, False, 4),
    ("08-NPU FP16+compile+stream", "npu", True, True, True, True, 1),
    ("09-NPU FP16+compile+batch+stream", "npu", True, True, True, True, 4),
]


def preprocess(img_path, imgsz=640, device="cpu", half=False, npu_opt=False):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil)[:, :, ::-1]
    orig_shape = img_np.shape[:2]
    h, w = img_np.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * r), int(w * r)

    if npu_opt and str(device).startswith("npu"):
        img_resized = cv2_resize(img_np, new_w, new_h)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        pad_h = (imgsz - new_h) / 2
        pad_w = (imgsz - new_w) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_tensor = F.pad(img_tensor, (left, right, top, bottom), mode="constant", value=0)
        img_tensor = img_tensor / 255.0
    else:
        img_resized = cv2_resize(img_np, new_w, new_h)
        pad_h = (imgsz - new_h) / 2
        pad_w = (imgsz - new_w) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_padded = cv2_border(img_resized, top, bottom, left, right)
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0)
        if half:
            img_tensor = img_tensor.half().to(device) / 255.0
        else:
            img_tensor = img_tensor.float().to(device) / 255.0

    return img_tensor, orig_shape, (r, (pad_w, pad_h))


def postprocess(preds, orig_shape, pad_info, nms=True):
    r, (pad_w, pad_h) = pad_info
    if isinstance(preds, dict):
        preds = preds["one2one"]
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if nms and str(preds.device).startswith("npu"):
        preds = postprocess_npu_opt(preds, orig_shape, pad_info)
        return preds

    preds_cpu = preds.float().cpu()
    preds_t = preds_cpu.transpose(-1, -2)
    bboxes, scores, labels = ops.v10postprocess(preds_t, 300, 80)
    bboxes = ops.xywh2xyxy(bboxes)
    preds_final = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
    mask = preds_final[..., 4] > 0.25
    filtered = preds_final[0][mask[0]]
    detections = []
    if len(filtered) > 0:
        filtered[:, [0, 2]] = (filtered[:, [0, 2]] - pad_w) / r
        filtered[:, [1, 3]] = (filtered[:, [1, 3]] - pad_h) / r
        filtered[:, [0, 2]] = filtered[:, [0, 2]].clamp(0, orig_shape[1])
        filtered[:, [1, 3]] = filtered[:, [1, 3]].clamp(0, orig_shape[0])
        for det in filtered:
            detections.append({
                "cls": int(det[5]),
                "conf": float(det[4]),
                "bbox": [float(x) for x in det[:4]],
            })
    return detections


def postprocess_npu_opt(preds, orig_shape, pad_info):
    r, (pad_w, pad_h) = pad_info
    preds_t = preds.transpose(-1, -2)
    bboxes, scores, labels = ops.v10postprocess(preds_t, 300, 80)
    bboxes = ops.xywh2xyxy(bboxes)
    preds_final = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
    mask = preds_final[..., 4].float() > 0.25
    filtered = preds_final[0][mask[0]]
    detections = []
    if len(filtered) > 0:
        filtered[:, [0, 2]] = (filtered[:, [0, 2]] - pad_w) / r
        filtered[:, [1, 3]] = (filtered[:, [1, 3]] - pad_h) / r
        filtered[:, [0, 2]] = filtered[:, [0, 2]].clamp(0, orig_shape[1])
        filtered[:, [1, 3]] = filtered[:, [1, 3]].clamp(0, orig_shape[0])
        filtered_cpu = filtered.float().cpu()
        for det in filtered_cpu:
            detections.append({
                "cls": int(det[5]),
                "conf": float(det[4]),
                "bbox": [float(x) for x in det[:4]],
            })
    return detections


def cv2_resize(img, new_w, new_h):
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(img)
    img_r = img_pil.resize((new_w, new_h), PILImage.LANCZOS)
    return np.array(img_r)


def cv2_border(img, top, bottom, left, right):
    h, w = img.shape[:2]
    new_h = h + top + bottom
    new_w = w + left + right
    bordered = np.full((new_h, new_w, 3), 114, dtype=img.dtype)
    bordered[top:top + h, left:left + w] = img
    return bordered


def load_model(model_path, device="cpu", half=False, compile_model=False):
    from ultralytics import YOLO
    t0 = time.time()
    model = YOLO(model_path)
    model.to(device)
    if half:
        model.model.half()
    if compile_model and str(device).startswith("npu"):
        model.model = torch.compile(model.model, backend="npu")
    return model


def benchmark_config(model_path, test_image, config, warmup=10, runs=100, imgsz=640):
    name, device, half, compile_m, npu_opt, stream, batch_size = config

    print(f"\n{'='*60}")
    print(f"  配置: {name}")
    print(f"{'='*60}")

    model = load_model(model_path, device, half, compile_m)

    # warmup
    for _ in range(warmup):
        tensor, shape, pad = preprocess(test_image, imgsz, device, half, npu_opt)
        with torch.no_grad():
            preds = model.model(tensor)
        _ = postprocess(preds, shape, pad, nms=npu_opt)
    print(f"  预热: {warmup}次")

    if stream:
        return _benchmark_stream(model, test_image, warmup, runs, imgsz, half, npu_opt, name)
    return _benchmark_single(model, test_image, warmup, runs, imgsz, half, npu_opt, batch_size, name)


def _benchmark_single(model, test_image, warmup, runs, imgsz, half, npu_opt, batch_size, name):
    pre_times, inf_times, post_times, total_times = [], [], [], []
    device = model.device if hasattr(model, "device") else model.predictor.device

    for _ in range(runs):
        if batch_size > 1:
            tensors, shapes, pads = [], [], []
            t0 = time.time()
            for _ in range(batch_size):
                t, s, p = preprocess(test_image, imgsz, device, half, npu_opt)
                tensors.append(t)
                shapes.append(s)
                pads.append(p)
            t1 = time.time()
            batch_tensor = torch.cat(tensors, dim=0)
            with torch.no_grad():
                if str(device).startswith("npu"):
                    torch.npu.synchronize()
                preds = model.model(batch_tensor)
                if str(device).startswith("npu"):
                    torch.npu.synchronize()
            t2 = time.time()
            for i in range(batch_size):
                single_pred = preds[i:i+1] if isinstance(preds, torch.Tensor) else preds
                _ = postprocess(single_pred, shapes[i], pads[i], nms=npu_opt)
            t3 = time.time()
            pre_times.append((t1 - t0) * 1000 / batch_size)
            inf_times.append((t2 - t1) * 1000 / batch_size)
            post_times.append((t3 - t2) * 1000 / batch_size)
            total_times.append((t3 - t0) * 1000 / batch_size)
        else:
            t0 = time.time()
            tensor, shape, pad = preprocess(test_image, imgsz, device, half, npu_opt)
            t1 = time.time()
            with torch.no_grad():
                if str(device).startswith("npu"):
                    torch.npu.synchronize()
                preds = model.model(tensor)
                if str(device).startswith("npu"):
                    torch.npu.synchronize()
            t2 = time.time()
            _ = postprocess(preds, shape, pad, nms=npu_opt)
            t3 = time.time()
            pre_times.append((t1 - t0) * 1000)
            inf_times.append((t2 - t1) * 1000)
            post_times.append((t3 - t2) * 1000)
            total_times.append((t3 - t0) * 1000)

    avg_pre = np.mean(pre_times)
    avg_inf = np.mean(inf_times)
    avg_post = np.mean(post_times)
    avg_total = np.mean(total_times)
    fps = 1000 / avg_total

    print(f"  预处理: {avg_pre:.2f}ms | 推理: {avg_inf:.2f}ms | 后处理: {avg_post:.2f}ms")
    print(f"  端到端: {avg_total:.2f}ms | {fps:.1f} FPS")

    return {
        "config": name, "device": device, "half": half, "compile": True,
        "pre_ms": avg_pre, "inf_ms": avg_inf, "post_ms": avg_post,
        "total_ms": avg_total, "fps": fps,
    }


def _benchmark_stream(model, test_image, warmup, runs, imgsz, half, npu_opt, name):
    n_frames = max(warmup + runs, 30)
    device = model.device if hasattr(model, "device") else model.predictor.device
    pre_queue = Queue(maxsize=2)

    # warmup
    for _ in range(warmup):
        t, s, p = preprocess(test_image, imgsz, device, half, npu_opt)
        with torch.no_grad():
            _ = model.model(t)

    # benchmark
    def _prefetch(n):
        for _ in range(n):
            t, s, p = preprocess(test_image, imgsz, device, half, npu_opt)
            pre_queue.put((t, s, p))
        pre_queue.put(None)

    t = threading.Thread(target=_prefetch, args=(n_frames,), daemon=True)
    t.start()

    t_start = time.time()
    for idx in range(n_frames):
        item = pre_queue.get()
        if item is None:
            break
        tensor, shape, pad = item
        with torch.no_grad():
            if str(device).startswith("npu"):
                torch.npu.synchronize()
            preds = model.model(tensor)
            if str(device).startswith("npu"):
                torch.npu.synchronize()
        _ = postprocess(preds, shape, pad, nms=npu_opt)

    elapsed = time.time() - t_start
    fps = n_frames / elapsed
    avg_total = elapsed / n_frames * 1000

    print(f"  端到端: {avg_total:.2f}ms | {fps:.1f} FPS (流式)")

    return {
        "config": name, "device": device, "half": half, "compile": True,
        "pre_ms": 0, "inf_ms": 0, "post_ms": 0,
        "total_ms": avg_total, "fps": fps,
    }


def run_benchmarks(args, configs, warmup, runs, imgsz, test_image):
    results = []
    for config in configs:
        result = benchmark_config(args.model, test_image, config, warmup, runs, imgsz)
        results.append(result)
    return results


def print_summary(results):
    print(f"\n\n{'='*80}")
    print(f"  YOLOv10n 性能基准测试结果汇总")
    print(f"{'='*80}")
    print(f"  {'配置':<38} {'预处理(ms)':>10} {'推理(ms)':>10} {'后处理(ms)':>10} {'E2E(ms)':>8} {'FPS':>8}")
    print(f"  {'-'*84}")
    for r in results:
        name = r["config"]
        pre = f"{r['pre_ms']:.2f}" if r["pre_ms"] > 0 else "-"
        inf = f"{r['inf_ms']:.2f}" if r["inf_ms"] > 0 else "-"
        post = f"{r['post_ms']:.2f}" if r["post_ms"] > 0 else "-"
        total = f"{r['total_ms']:.2f}"
        fps = f"{r['fps']:.1f}"
        print(f"  {name:<38} {pre:>10} {inf:>10} {post:>10} {total:>8} {fps:>8}")
    print(f"  {'-'*84}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv10n NPU 综合性能基准测试")
    parser.add_argument("--model", default="yolov10n.pt", help="模型路径")
    parser.add_argument("--image", default=None, help="测试图像路径")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--runs", type=int, default=50, help="测试次数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--output", default="benchmark_result.json", help="结果输出路径")
    return parser.parse_args()


def main():
    args = parse_args()

    # 找一张测试图
    test_image = args.image
    if test_image is None:
        for candidate in ["test.jpg", "bus.jpg", "test_images/bus.jpg", "../test.jpg"]:
            if os.path.exists(candidate):
                test_image = candidate
                break
        if test_image is None:
            print("请指定测试图像路径: --image test.jpg")
            sys.exit(1)

    print(f"测试图像: {test_image}")
    print(f"预热: {args.warmup}, 测试: {args.runs}")

    results = run_benchmarks(args, CONFIGS, args.warmup, args.runs, args.imgsz, test_image)

    print_summary(results)

    # 保存结果
    import json
    output_path = args.output
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items()}
        for k in ["pre_ms", "inf_ms", "post_ms", "total_ms", "fps"]:
            sr[k] = round(float(sr[k]), 2) if isinstance(sr[k], (int, float)) else sr[k]
        serializable.append(sr)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {output_path}")


if __name__ == "__main__":
    main()
