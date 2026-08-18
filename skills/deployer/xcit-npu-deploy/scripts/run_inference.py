#!/usr/bin/env python3
"""
run_inference.py — XCiT model NPU inference script (unified).

Usage:
  python3 run_inference.py --model xcit_tiny_12_p16_384 --device npu:0
  python3 run_inference.py --model xcit_large_24_p8_224 --device cpu
"""
import argparse, gc, os, sys, time, torch, torch.nn.functional as F, safetensors.torch

try:
    import torch_npu
    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    HAS_NPU = False

# Model name mapping: short name -> (timm_model_name, input_size, modelscope_path)
MODEL_MAP = {
    "xcit_tiny_12_p16_384": ("xcit_tiny_12_p16_384.fb_dist_in1k", 384, "timm/xcit_tiny_12_p16_384.fb_dist_in1k"),
    "xcit_large_24_p8_224": ("xcit_large_24_p8_224.fb_in1k", 224, "timm/xcit_large_24_p8_224.fb_in1k"),
    "xcit_medium_24_p16_384": ("xcit_medium_24_p16_384.fb_dist_in1k", 384, "timm/xcit_medium_24_p16_384.fb_dist_in1k"),
    "xcit_tiny_12_p8_384": ("xcit_tiny_12_p8_384.fb_dist_in1k", 384, "timm/xcit_tiny_12_p8_384.fb_dist_in1k"),
    "xcit_small_12_p8_224": ("xcit_small_12_p8_224.fb_in1k", 224, "timm/xcit_small_12_p8_224.fb_in1k"),
}


def load_model(timm_name: str, modelscope_path: str, device: torch.device):
    import timm
    from modelscope import snapshot_download
    model_dir = snapshot_download(modelscope_path)
    model = timm.create_model(timm_name, pretrained=False)
    p = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(p):
        model.load_state_dict(safetensors.torch.load_file(p))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"),
                                          map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def run_inference(model, x):
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    return logits, probs, torch.topk(probs, k=5, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("npu:0" if HAS_NPU else "cpu")
    else:
        device = torch.device(args.device)

    timm_name, input_size, ms_path = MODEL_MAP[args.model]
    print(f"[INFO] Model: {timm_name}")
    print(f"[INFO] Device: {device}")

    model = load_model(timm_name, ms_path, device)
    x = torch.randn(args.batch_size, 3, input_size, input_size).to(device)

    if device.type == "npu":
        torch.npu.synchronize()

    for _ in range(args.warmup):
        run_inference(model, x)

    if device.type == "npu":
        torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        logits, probs, top5 = run_inference(model, x)
    if device.type == "npu":
        torch.npu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / args.iters * 1000
    print(f"\n[RESULT] Avg inference time: {avg_ms:.2f} ms")
    print(f"[RESULT] Throughput: {1000/avg_ms*args.batch_size:.2f} samples/sec")

    for i in range(min(args.batch_size, 5)):
        items = [f"{j+1}: class {top5.indices[i][j].item()} ({probs[i][top5.indices[i][j]].item():.4f})"
                 for j in range(5)]
        print(f"  Sample {i}: {', '.join(items)}")

    torch.save({"logits": logits.cpu(), "probs": probs.cpu()}, "inference_output.pt")
    print("[INFO] Output saved to inference_output.pt")

    gc.collect()
    if HAS_NPU:
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
