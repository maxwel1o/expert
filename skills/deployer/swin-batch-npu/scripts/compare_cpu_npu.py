#!/usr/bin/env python3
"""Compare CPU vs NPU inference outputs for Swin Transformer models.

Usage:
  python3 compare_cpu_npu.py --model swin_tiny_patch4_window7_224.ms_in1k
"""
import argparse
import gc
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from timm import create_model
from timm.data import resolve_data_config


def get_random_input(batch_size: int = 1, num_channels: int = 3,
                     height: int = 224, width: int = 224) -> torch.Tensor:
    return torch.randn(batch_size, num_channels, height, width)


def main():
    parser = argparse.ArgumentParser(description="CPU vs NPU comparison for Swin")
    parser.add_argument("--model", required=True, help="timm model name")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override num_classes. Auto-detected from model config if not set.")
    parser.add_argument("--atol", type=float, default=1e-3, help="absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance")
    args = parser.parse_args()

    if not torch.npu.is_available():
        print("ERROR: NPU is not available")
        return 1

    # Auto-detect num_classes from model config
    if args.num_classes is None:
        model_safe = args.model.replace('/', '--')
        cfg_paths = list(Path(f"/opt/atomgit/.cache/huggingface/hub/models--timm--{model_safe}").glob("**/config.json"))
        if cfg_paths:
            try:
                cfg = json.loads(cfg_paths[0].read_text())
                args.num_classes = cfg.get("num_classes", 1000)
                print(f"Model: {args.model} (auto-detected num_classes={args.num_classes})")
            except Exception:
                args.num_classes = 1000
                print(f"Model: {args.model} (default num_classes={args.num_classes})")
        else:
            args.num_classes = 1000
            print(f"Model: {args.model} (default num_classes={args.num_classes})")
    else:
        print(f"Model: {args.model} (num_classes={args.num_classes})")

    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print("=" * 60)

    # --- CPU inference ---
    print("\n[1/4] Loading model on CPU...")
    t0 = time.time()
    model_cpu = create_model(args.model, pretrained=True, num_classes=args.num_classes)
    model_cpu.eval()
    print(f"  CPU model loaded in {time.time() - t0:.2f}s")

    data_config = resolve_data_config(model=model_cpu, use_test_size=True)
    input_size = data_config.get("input_size", (3, 224, 224))
    print(f"  Input size: {input_size}")
    h, w = input_size[1], input_size[2]

    input_tensor = get_random_input(1, 3, h, w)
    print(f"  Input shape: {input_tensor.shape}")

    print("\n[2/4] CPU inference...")
    with torch.no_grad():
        for _ in range(3):
            _ = model_cpu(input_tensor)
    t0 = time.time()
    with torch.no_grad():
        cpu_out = model_cpu(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    cpu_time = time.time() - t0
    print(f"  CPU inference: {cpu_time * 1000:.2f}ms")

    cpu_probs = F.softmax(cpu_out, dim=1)
    cpu_top5_val, cpu_top5_idx = torch.topk(cpu_probs, 5, dim=1)
    print("  CPU Top-5 indices:", cpu_top5_idx[0].tolist())
    print("  CPU Top-5 probs:", [round(v, 6) for v in cpu_top5_val[0].tolist()])

    del model_cpu
    gc.collect()

    # --- NPU inference ---
    print("\n[3/4] Loading model on NPU...")
    t0 = time.time()
    model_npu = create_model(args.model, pretrained=True, num_classes=args.num_classes)
    model_npu.eval()
    model_npu = model_npu.npu()
    print(f"  NPU model loaded in {time.time() - t0:.2f}s")

    input_npu = input_tensor.npu()

    print("\n[4/4] NPU inference...")
    with torch.no_grad():
        for _ in range(3):
            _ = model_npu(input_npu)
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        npu_out = model_npu(input_npu)
    torch.npu.synchronize()
    npu_time = time.time() - t0
    print(f"  NPU inference: {npu_time * 1000:.2f}ms")

    npu_probs = F.softmax(npu_out.cpu(), dim=1)
    npu_top5_val, npu_top5_idx = torch.topk(npu_probs, 5, dim=1)
    print("  NPU Top-5 indices:", npu_top5_idx[0].tolist())
    print("  NPU Top-5 probs:", [round(v, 6) for v in npu_top5_val[0].tolist()])

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    cpu_out_np = cpu_out.cpu().float().numpy()
    npu_out_np = npu_out.cpu().float().numpy()

    # Per-element differences
    abs_diff = abs(cpu_out_np - npu_out_np)
    max_abs_err = float(abs_diff.max())
    mean_abs_err = float(abs_diff.mean())
    std_abs_err = float(abs_diff.std())

    # Relative error (avoid div-by-zero)
    denom = abs(cpu_out_np) + 1e-8
    rel_diff = abs_diff / denom
    max_rel_err = float(rel_diff.max())
    mean_rel_err = float(rel_diff.mean())

    # Cosine similarity between flattened logits
    c = cpu_out_np.flatten()
    n = npu_out_np.flatten()
    cos_sim = float((c @ n) / ((c @ c) ** 0.5 * (n @ n) ** 0.5 + 1e-12))

    # Top-5 agreement rate
    cpu_top5 = cpu_top5_idx[0].tolist()
    npu_top5 = npu_top5_idx[0].tolist()
    top5_overlap = len(set(cpu_top5) & set(npu_top5))
    top5_agree = all(c == n for c, n in zip(cpu_top5, npu_top5))

    # Probability differences for top-1
    top1_prob_diff = abs(float(cpu_top5_val[0, 0]) - float(npu_top5_val[0, 0]))

    print(f"\n  Max absolute error:    {max_abs_err:.8f}")
    print(f"  Mean absolute error:   {mean_abs_err:.8f}")
    print(f"  Std absolute error:    {std_abs_err:.8f}")
    print(f"  Max relative error:    {max_rel_err:.8f}")
    print(f"  Mean relative error:   {mean_rel_err:.8f}")
    print(f"  Cosine similarity:     {cos_sim:.8f}")
    print(f"\n  Top-1 prob difference: {top1_prob_diff:.8f}")
    print(f"  Top-5 exact match:     {top5_agree}")
    print(f"  Top-5 overlap:         {top5_overlap}/5")
    print(f"\n  CPU inference time:    {cpu_time * 1000:.2f}ms")
    print(f"  NPU inference time:    {npu_time * 1000:.2f}ms")
    print(f"  Speedup:              {cpu_time / npu_time:.2f}x")

    # Accuracy conclusion
    error_pct = max_abs_err * 100
    print(f"\n  Overall error (max_abs * 100%): {error_pct:.4f}%")
    if max_abs_err < 0.01:
        print("  ✓ PASS: NPU vs CPU error < 1% (tolerance: max_abs < 0.01)")
    else:
        print(f"  ✗ FAIL: NPU vs CPU error = {error_pct:.4f}% > 1%")

    # Save outputs
    torch.save({
        "model": args.model,
        "cpu_output": cpu_out.cpu(),
        "npu_output": npu_out.cpu(),
        "cpu_time_ms": cpu_time * 1000,
        "npu_time_ms": npu_time * 1000,
        "max_abs_error": max_abs_err,
        "mean_abs_error": mean_abs_err,
        "cosine_similarity": cos_sim,
    }, f"{args.model.replace('/', '_')}_comparison.pt")
    print(f"\n  Comparison data saved to: {args.model.replace('/', '_')}_comparison.pt")

    # Cleanup
    del model_npu, input_npu, cpu_out, npu_out
    gc.collect()
    torch.npu.empty_cache()

    return 0 if max_abs_err < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
