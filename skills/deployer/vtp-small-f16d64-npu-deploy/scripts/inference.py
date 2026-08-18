#!/usr/bin/env python3
"""
VTP-Small-f16d64 NPU Inference and Precision Verification Script.

Usage:
    # Run NPU inference with CPU baseline comparison
    python run_npu_inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64

    # Quick smoke test
    python run_npu_inference.py --model_path ./model_weights/MiniMax/VTP-Small-f16d64 --quick
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure VTP repo is on PYTHONPATH
project_root = Path(__file__).parent / "vtp-repo"
sys.path.insert(0, str(project_root))

from vtp.models.vtp_hf import VTPConfig, VTPModel


def get_device(prefer_npu=True):
    """Auto-detect available device: npu > cuda > cpu."""
    if prefer_npu:
        try:
            import torch_npu
            if torch.npu.is_available():
                return torch.device("npu:0")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def adapt_model_for_npu(model):
    """Adapt VTP model for NPU inference by fixing dtype-sensitive layers.

    The pixel_decoder's RoPE position embedding defaults to bfloat16, which
    can cause numerical divergence between NPU and CPU. We force it to float32
    for deterministic, high-precision computation.
    """
    if hasattr(model, "pixel_decoder") and model.pixel_decoder is not None:
        rope = model.pixel_decoder.rope_embed
        if rope.dtype == torch.bfloat16:
            rope.dtype = torch.float32
            rope.periods = rope.periods.to(torch.float32)
            rope._init_weights()
            print("  [NPU Adapt] pixel_decoder RoPE dtype: bf16 -> fp32")
    return model


def generate_random_image(batch_size=1, image_size=256, dtype=torch.float32):
    """Generate random image tensor for testing."""
    return torch.randn(batch_size, 3, image_size, image_size, dtype=dtype)


def generate_random_text(batch_size=1, context_length=77, vocab_size=49408):
    """Generate random text token IDs for testing."""
    return torch.randint(0, vocab_size, (batch_size, context_length))


def tensor_allclose(a, b, rtol=1e-3, atol=1e-3):
    """Compare two tensors with tolerance."""
    if a is None and b is None:
        return True, 0.0, 0.0
    if a is None or b is None:
        return False, float('inf'), float('inf')
    a_np = a.detach().cpu().numpy().flatten()
    b_np = b.detach().cpu().numpy().flatten()
    diff = np.abs(a_np - b_np)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    relative_diff = max_diff / (np.abs(b_np).max() + 1e-8)
    passed = np.allclose(a_np, b_np, rtol=rtol, atol=atol)
    return passed, max_diff, mean_diff, relative_diff


def run_inference(model, images, texts, device, precision="fp32", warmup=1, runs=3):
    """Run inference on target device and measure latency."""
    model = model.to(device).eval()
    images = images.to(device)
    texts = texts.to(device)

    if precision in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif precision in ("fp16", "float16"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Determine autocast device_type
    if device.type == "npu":
        autocast_device = "npu"
    elif device.type == "cuda":
        autocast_device = "cuda"
    else:
        autocast_device = "cpu"

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if autocast_device != "cpu" and dtype != torch.float32:
                with torch.amp.autocast(device_type=autocast_device, dtype=dtype):
                    _ = model.get_clip_image_feature(images)
                    if model.config.train_reconstruction:
                        _ = model.get_reconstruction_latents(images)
            else:
                _ = model.get_clip_image_feature(images)
                if model.config.train_reconstruction:
                    _ = model.get_reconstruction_latents(images)

    # Synchronize before timing
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    results = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            if autocast_device != "cpu" and dtype != torch.float32:
                with torch.amp.autocast(device_type=autocast_device, dtype=dtype):
                    img_feat = model.get_clip_image_feature(images)
                    rec_latents = None
                    rec_image = None
                    if model.config.train_reconstruction:
                        rec_latents = model.get_reconstruction_latents(images)
                        rec_image = model.get_latents_decoded_images(rec_latents)
            else:
                img_feat = model.get_clip_image_feature(images)
                rec_latents = None
                rec_image = None
                if model.config.train_reconstruction:
                    rec_latents = model.get_reconstruction_latents(images)
                    rec_image = model.get_latents_decoded_images(rec_latents)

            if device.type == "npu":
                torch.npu.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)
            results.append({
                "img_feat": img_feat.cpu().clone(),
                "rec_latents": rec_latents.cpu().clone() if rec_latents is not None else None,
                "rec_image": rec_image.cpu().clone() if rec_image is not None else None,
            })

    avg_latency = np.mean(latencies)
    return results[0], avg_latency


def calculate_psnr_tensor(a, b, max_val=1.0):
    """Calculate PSNR between two tensors."""
    mse = torch.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def verify_precision(cpu_results, npu_results, tolerance=0.01):
    """Verify NPU output precision against CPU baseline.

    Criteria:
      - All tensors: normalized max error (max_diff / data_range) < 1%
      - Image tensors: PSNR > 35 dB as additional guard
    """
    print("\n" + "=" * 60)
    print("Precision Verification (NPU vs CPU)")
    print("=" * 60)

    all_passed = True
    for key in cpu_results:
        cpu_val = cpu_results[key]
        npu_val = npu_results[key]
        if cpu_val is None or npu_val is None:
            continue

        max_diff, mean_diff, rel_diff = tensor_allclose(
            npu_val, cpu_val, rtol=tolerance, atol=tolerance
        )[1:]

        cpu_flat = cpu_val.detach().cpu().flatten()
        npu_flat = npu_val.detach().cpu().flatten()

        # Normalized max error: max_diff / (max_abs_value + eps)
        max_abs_val = float(torch.abs(cpu_flat).max())
        norm_max_err = max_diff / (max_abs_val + 1e-8)
        norm_mean_err = mean_diff / (max_abs_val + 1e-8)

        # For image outputs, also compute PSNR
        psnr = None
        if key == "rec_image":
            psnr = calculate_psnr_tensor(cpu_flat, npu_flat, max_val=1.0)

        # Pass criteria: normalized max error < 1%
        passed = norm_max_err < tolerance
        if key == "rec_image" and psnr is not None:
            passed = passed and psnr > 35.0

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        psnr_str = f" | PSNR={psnr:.2f}dB" if psnr else ""
        print(f"  {key:20s}: {status} | norm_max_err={norm_max_err:.6e} | norm_mean_err={norm_mean_err:.6e}{psnr_str}")

    print("=" * 60)
    if all_passed:
        print(f"All precision checks PASSED (normalized max error < {tolerance * 100:.0f}%)")
    else:
        print("Some precision checks FAILED")
    print("=" * 60)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="VTP NPU Inference and Verification")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VTP HuggingFace model directory")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Inference precision")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test without CPU baseline")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (npu, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(prefer_npu=True)

    print("=" * 60)
    print("VTP-Small-f16d64 NPU Inference")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = VTPModel.from_pretrained(args.model_path)
    config = model.config
    print(f"  Image size: {config.image_size}")
    print(f"  Vision embed dim: {config.vision_embed_dim}")
    print(f"  Vision depth: {config.vision_depth}")
    print(f"  Train clip: {config.train_clip}")
    print(f"  Train reconstruction: {config.train_reconstruction}")

    # Apply NPU adaptation if running on NPU
    if device.type == "npu":
        model = adapt_model_for_npu(model)

    # Generate random inputs
    images = generate_random_image(args.batch_size, config.image_size)
    texts = generate_random_text(args.batch_size, config.text_context_length)

    # Run on target device
    print(f"\nRunning inference on {device}...")
    npu_results, npu_latency = run_inference(
        model, images, texts, device, precision=args.precision
    )
    print(f"  Avg latency: {npu_latency:.2f} ms")
    print(f"  Image feature shape: {npu_results['img_feat'].shape}")
    if npu_results['rec_latents'] is not None:
        print(f"  Reconstruction latents shape: {npu_results['rec_latents'].shape}")
    if npu_results['rec_image'] is not None:
        print(f"  Reconstructed image shape: {npu_results['rec_image'].shape}")

    # CPU baseline comparison
    if not args.quick:
        print("\nRunning CPU baseline for precision comparison...")
        cpu_device = torch.device("cpu")
        # Re-create fresh model on CPU to avoid state contamination
        cpu_model = VTPModel.from_pretrained(args.model_path)
        cpu_model = adapt_model_for_npu(cpu_model)
        cpu_results, cpu_latency = run_inference(
            cpu_model, images, texts, cpu_device, precision="fp32", warmup=0, runs=1
        )
        print(f"  CPU latency: {cpu_latency:.2f} ms")

        # Verify precision
        passed = verify_precision(cpu_results, npu_results, tolerance=0.01)
        if not passed:
            print("\nWARNING: NPU precision does not meet <1% requirement.")
            sys.exit(1)
    else:
        print("\nQuick mode: skipping CPU baseline comparison.")

    print("\nInference completed successfully.")


if __name__ == "__main__":
    main()
