#!/usr/bin/env python3
"""YOLOv10n NPU 推理脚本 (自包含版, 包含所有优化)

使用方式:
  python yolov10_npu_infer.py --model yolov10n.pt --image test.jpg --device npu --half --compile

优化模式:
  --half          FP16 半精度
  --compile       torch.compile + NPU 后端
  --npu-opt      NPU 后处理 + NPU 图像缩放
  --stream       流式流水线 (多图时)
"""

import argparse
import time
import os
import threading
from queue import Queue
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
import torch_npu

try:
    from ultralytics import YOLO
except ImportError:
    print("请先安装 ultralytics: pip install ultralytics")
    exit(1)


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class YOLOv10nNPUInference:
    """YOLOv10n NPU 推理器 (含单图/批量/流式三种模式)"""

    def __init__(self, model_path, device="npu:0", imgsz=640,
                 half=True, compile_model=True, npu_opt=True):
        self.device = torch.device(device)
        self.imgsz = imgsz
        self.half = half
        self.compile_model = compile_model
        self.npu_opt = npu_opt and (self.device.type == "npu")

        print(f"加载模型: {model_path}")
        t0 = time.time()
        self.model = YOLO(model_path)
        print(f"  模型加载: {time.time() - t0:.2f}s")

        # 迁移到 NPU
        self.model.to(self.device)

        # FP16
        if half:
            print(f"  精度: FP16")
            self.model.model.half()

        # torch.compile
        if compile_model and self.device.type == "npu":
            print(f"  torch.compile (backend=npu)...")
            tc0 = time.time()
            self.model.model = torch.compile(self.model.model, backend="npu")
            print(f"  编译完成: {time.time() - tc0:.2f}s")

        self.warmed_up = False

    def warmup(self, n=3):
        """预热模型"""
        dummy = torch.randn(1, 3, self.imgsz, self.imgsz,
                           dtype=torch.float16 if self.half else torch.float32)
        dummy = dummy.to(self.device)
        for i in range(n):
            with torch.no_grad():
                _ = self.model.model(dummy)
            if self.device.type == "npu":
                torch.npu.synchronize()
        self.warmed_up = True
        print(f"  预热完成 ({n}次)")

    def preprocess(self, img_source):
        """预处理: 图像加载 -> resize -> pad -> normalize"""
        if isinstance(img_source, str):
            img = Image.open(img_source)
        elif isinstance(img_source, Image.Image):
            img = img_source
        elif isinstance(img_source, np.ndarray):
            img = Image.fromarray(img_source[..., ::-1])
        else:
            raise TypeError(f"不支持的图像类型: {type(img_source)}")

        orig_shape = (img.height, img.width)
        img_np = np.array(img)[:, :, ::-1]  # RGB -> BGR
        h, w = img_np.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(h * r), int(w * r)
        pad_h = (self.imgsz - new_h) / 2
        pad_w = (self.imgsz - new_w) / 2

        if self.npu_opt:
            # NPU 图像缩放
            img_resized = cv2_resize_fallback(img_np, new_w, new_h)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            dtype = torch.float16 if self.half else torch.float32
            img_tensor = img_tensor.to(dtype)
            top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
            left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
            img_tensor = F.pad(img_tensor, (left, right, top, bottom), mode="constant", value=0)
            img_tensor = img_tensor / 255.0
        else:
            # CPU 缩放
            img_resized = cv2_resize_fallback(img_np, new_w, new_h)
            top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
            left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
            img_padded = cv2_copy_make_border(img_resized, top, bottom, left, right)
            img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0)
            dtype = torch.float16 if self.half else torch.float32
            img_tensor = img_tensor.to(dtype).to(self.device) / 255.0

        return img_tensor, orig_shape, (r, (pad_w, pad_h))

    def postprocess(self, preds, orig_shape, pad_info):
        """YOLOv10 后处理 (支持 NPU 优化)"""
        r, (pad_w, pad_h) = pad_info
        if isinstance(preds, dict):
            preds = preds["one2one"]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if self.npu_opt and str(preds.device).startswith("npu"):
            from ultralytics.utils import ops
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
        else:
            preds_cpu = preds.float().cpu()
            from ultralytics.utils import ops
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

    def predict(self, img_source, verbose=True):
        """单图推理"""
        t0 = time.time()
        tensor, shape, pad = self.preprocess(img_source)
        t1 = time.time()
        with torch.no_grad():
            if self.device.type == "npu":
                torch.npu.synchronize()
            preds = self.model.model(tensor)
            if self.device.type == "npu":
                torch.npu.synchronize()
        t2 = time.time()
        detections = self.postprocess(preds, shape, pad)
        t3 = time.time()

        if verbose:
            print(f"\n检测结果 ({len(detections)} 个目标):")
            for d in detections:
                cls_name = COCO_NAMES[d["cls"]] if d["cls"] < len(COCO_NAMES) else f"cls{d['cls']}"
                x1, y1, x2, y2 = d["bbox"]
                print(f"  {cls_name}: conf={d['conf']:.4f}, bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            print(f"延迟: pre={t1-t0:.1f}ms inf={t2-t1:.1f}ms post={t3-t2:.1f}ms total={t3-t0:.1f}ms")

        return detections

    def predict_stream(self, img_sources, verbose=False):
        """流式流水线: 后台线程预处理, 主线程推理"""
        n = len(img_sources)
        pre_queue = Queue(maxsize=2)

        def _prefetch():
            for idx in range(n):
                tensor, shape, pad = self.preprocess(img_sources[idx])
                pre_queue.put((tensor, shape, pad))
            pre_queue.put(None)

        t = threading.Thread(target=_prefetch, daemon=True)
        t.start()

        results = []
        t_start = time.time()
        for idx in range(n):
            item = pre_queue.get()
            if item is None:
                break
            tensor, shape, pad = item
            with torch.no_grad():
                if self.device.type == "npu":
                    torch.npu.synchronize()
                preds = self.model.model(tensor)
                if self.device.type == "npu":
                    torch.npu.synchronize()
            detections = self.postprocess(preds, shape, pad)
            results.append(detections)
            if verbose:
                print(f"[{idx+1}/{n}] {len(detections)} 个目标")

        elapsed = time.time() - t_start
        fps = n / elapsed
        if verbose:
            print(f"\n流式推理: {n}张, {elapsed:.2f}s, {fps:.1f} FPS")
        return results

    def benchmark(self, img_source, warmup=10, runs=100, stream=False, verbose=True):
        """性能基准测试"""
        if not self.warmed_up:
            self.warmup()

        if stream:
            return self._benchmark_stream(img_source, warmup, runs, verbose)
        return self._benchmark_single(img_source, warmup, runs, verbose)

    def _benchmark_single(self, img_source, warmup, runs, verbose):
        # 预热
        for _ in range(warmup):
            self.predict(img_source, verbose=False)

        # 计时
        pre_times, inf_times, post_times, total_times = [], [], [], []
        for _ in range(runs):
            t0 = time.time()
            tensor, shape, pad = self.preprocess(img_source)
            t1 = time.time()
            with torch.no_grad():
                if self.device.type == "npu":
                    torch.npu.synchronize()
                preds = self.model.model(tensor)
                if self.device.type == "npu":
                    torch.npu.synchronize()
            t2 = time.time()
            _ = self.postprocess(preds, shape, pad)
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

        if verbose:
            print(f"\n基准测试 (warmup={warmup}, runs={runs})")
            print(f"  预处理: {avg_pre:.2f}ms")
            print(f"  推理:   {avg_inf:.2f}ms")
            print(f"  后处理: {avg_post:.2f}ms")
            print(f"  端到端: {avg_total:.2f}ms ({fps:.1f} FPS)")

        return {"pre_ms": avg_pre, "inf_ms": avg_inf, "post_ms": avg_post,
                "total_ms": avg_total, "fps": fps}

    def _benchmark_stream(self, img_source, warmup, runs, verbose):
        sources = [img_source] * max(warmup + runs, 1)
        t_start = time.time()
        results = self.predict_stream(sources, verbose=False)
        elapsed = time.time() - t_start
        fps = len(sources) / elapsed
        if verbose:
            print(f"\n流式基准测试: {len(sources)}张, {elapsed:.2f}s, {fps:.1f} FPS")
        return {"total_ms": elapsed / len(sources) * 1000, "fps": fps}

    def visualize(self, img_source, detections, output_path="result.jpg"):
        """画检测框"""
        if isinstance(img_source, str):
            img = Image.open(img_source).convert("RGB")
        else:
            img = img_source
        draw = ImageDraw.Draw(img)

        colors = [
            "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
            "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
            "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF",
            "#FF95C8", "#FF37C7"
        ]

        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cls_name = COCO_NAMES[d["cls"]] if d["cls"] < len(COCO_NAMES) else str(d["cls"])
            color = colors[d["cls"] % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{cls_name} {d['conf']:.2f}"
            bbox = draw.textbbox((x1, y1), label)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), label, fill="white")

        img.save(output_path)
        print(f"结果保存至: {output_path}")


def cv2_resize_fallback(img_np, new_w, new_h):
    """cv2.resize fallback (使用 PIL)"""
    from PIL import Image
    img_pil = Image.fromarray(img_np)
    img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img_resized)


def cv2_copy_make_border(img, top, bottom, left, right):
    """cv2.copyMakeBorder fallback"""
    h, w = img.shape[:2]
    new_h = h + top + bottom
    new_w = w + left + right
    if img.ndim == 3:
        bordered = np.full((new_h, new_w, 3), 114, dtype=img.dtype)
    else:
        bordered = np.full((new_h, new_w), 114, dtype=img.dtype)
    bordered[top:top + h, left:left + w] = img
    return bordered


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv10n NPU 推理")
    parser.add_argument("--model", default="yolov10n.pt", help="模型路径")
    parser.add_argument("--image", required=True, help="输入图像路径（支持多张用逗号分隔）")
    parser.add_argument("--device", default="npu:0", choices=["npu:0", "cpu"], help="推理设备")
    parser.add_argument("--half", action="store_true", help="FP16 半精度")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--npu-opt", action="store_true", help="NPU 后处理+缩放优化")
    parser.add_argument("--stream", action="store_true", help="流式流水线模式")
    parser.add_argument("--benchmark", action="store_true", help="基准测试模式")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--runs", type=int, default=100, help="测试次数")
    parser.add_argument("--visualize", action="store_true", help="可视化并保存")
    parser.add_argument("--output", default="result.jpg", help="输出图像路径")
    return parser.parse_args()


def main():
    args = parse_args()

    infer = YOLOv10nNPUInference(
        model_path=args.model,
        device=args.device,
        half=args.half,
        compile_model=args.compile,
        npu_opt=args.npu_opt,
    )
    infer.warmup()

    images = [s.strip() for s in args.image.split(",")]

    if args.benchmark:
        infer.benchmark(images[0], warmup=args.warmup, runs=args.runs, stream=args.stream)
        return

    if args.stream and len(images) > 1:
        results_list = infer.predict_stream(images, verbose=True)
    else:
        for img_path in images:
            detections = infer.predict(img_path, verbose=True)
            if args.visualize:
                infer.visualize(img_path, detections, args.output)


if __name__ == "__main__":
    main()
