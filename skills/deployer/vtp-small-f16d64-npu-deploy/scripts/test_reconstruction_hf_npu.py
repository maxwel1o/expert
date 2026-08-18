#!/usr/bin/env python3
"""
ImageNet reconstruction evaluation script for VTP HuggingFace model (NPU-adapted).

Usage:
    # Single NPU
    python test_reconstruction_hf_npu.py --model_path /path/to/vtp_hf_model --data_path /path/to/imagenet/val

    # Multi-NPU with DDP
    torchrun --nproc_per_node=8 test_reconstruction_hf_npu.py \
        --model_path /path/to/vtp_hf_model \
        --data_path /path/to/imagenet/val \
        --use_ddp
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import Normalize
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# NPU compatibility layer
sys.path.insert(0, str(Path(__file__).parent))
from npu_compat import get_device, adapt_model_for_npu, synchronize_device

project_root = Path(__file__).parent / "vtp-repo"
sys.path.insert(0, str(project_root))

from vtp.models.vtp_hf import VTPConfig, VTPModel
from vtp.utils.image_utils import center_crop_arr
from vtp.utils.lpips import LPIPS as LPIPS_VTP

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def calculate_psnr(original: torch.Tensor, processed: torch.Tensor) -> float:
    mse = torch.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(255.0) / torch.sqrt(mse)).item()


class LPIPS:
    def __init__(self, device='npu:0'):
        self.device = device
        self.model = LPIPS_VTP().to(device).eval()

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(img1, img2)


def get_ssim_metric(device):
    try:
        from torchmetrics import StructuralSimilarityIndexMeasure
        return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except ImportError:
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            print("Warning: torchmetrics not installed, SSIM will not be calculated")
        return None


def test_reconstruction(
    model_path: str,
    data_path: str,
    output_path: str = "reconstruction_output",
    device: str = None,
    batch_size: int = 32,
    precision: str = "bf16",
    save_images: bool = True,
    use_ddp: bool = False,
    num_workers: int = 4,
    max_samples: int = None,
):
    # Initialize DDP if needed
    if use_ddp:
        if not dist.is_initialized():
            backend = "hccl" if get_device().type == "npu" else "nccl"
            dist.init_process_group(backend=backend)
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_obj = get_device()
        if device_obj.type == "npu":
            torch.npu.set_device(local_rank)
        elif device_obj.type == "cuda":
            torch.cuda.set_device(local_rank)
        device = torch.device(f'{device_obj.type}:{local_rank}')
        is_main = (local_rank == 0)
    else:
        local_rank, world_size = 0, 1
        if device is None:
            device = str(get_device())
        device = torch.device(device)
        is_main = True

    if is_main:
        print("=" * 60)
        print("ImageNet Reconstruction Evaluation (VTP NPU)")
        print("=" * 60)
        print(f"Model path: {model_path}")
        print(f"Data path: {data_path}")
        print(f"Device: {device}" + (f", DDP: {world_size} NPUs" if use_ddp else ""))
        print(f"Precision: {precision}")
        print(f"Batch size: {batch_size}")
        print()

    # Load model
    if is_main:
        print("Loading model...")
    model = VTPModel.from_pretrained(model_path)
    model = adapt_model_for_npu(model)
    model = model.to(device).eval()

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    config = model.module.config if use_ddp else model.config
    image_size = config.image_size

    if is_main:
        print(f"Image size: {image_size}")

    # Setup transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    transform_rev = Normalize(
        [-m / s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
        [1 / s for s in IMAGENET_DEFAULT_STD]
    )

    # Load dataset
    if is_main:
        print(f"Loading dataset: {data_path}")
    dataset = ImageFolder(root=data_path, transform=transform)
    total_samples = len(dataset)

    if max_samples is not None and max_samples < total_samples:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(max_samples))
        total_samples = max_samples
        if is_main:
            print(f"Limiting to {max_samples} samples for quick test")

    if is_main:
        print(f"Number of samples: {total_samples}")

    # Setup output directories
    save_dir = os.path.join(output_path, 'reconstructed')
    ref_dir = os.path.join(output_path, 'reference')
    skip_generation = False

    if save_images and is_main:
        existing_recon = sum(1 for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))) if os.path.exists(save_dir) else 0
        existing_ref = sum(1 for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))) if os.path.exists(ref_dir) else 0
        if existing_recon >= total_samples and existing_ref >= total_samples:
            print(f"Found {existing_recon} reconstructed and {existing_ref} reference images. Skipping generation...")
            skip_generation = True

    if use_ddp:
        skip_tensor = torch.tensor([1 if skip_generation else 0], device=device)
        dist.broadcast(skip_tensor, src=0)
        skip_generation = skip_tensor.item() == 1

    if skip_generation:
        fid = None
        if is_main and os.path.exists(save_dir) and os.path.exists(ref_dir):
            try:
                print("Computing rFID...")
                from pytorch_fid import fid_score
                fid = fid_score.calculate_fid_given_paths([ref_dir, save_dir], batch_size=50, device=device, dims=2048, num_workers=4)
            except Exception as e:
                print(f"Error during FID calculation: {e}")

        if is_main:
            print("\n" + "=" * 60)
            print("Results (from cached images):")
            if fid is not None:
                print(f"  rFID: {fid:.3f}")
            print(f"  Samples: {total_samples}")
            print("=" * 60)

        if use_ddp:
            dist.destroy_process_group()
        return {'fid': fid, 'psnr': None, 'lpips': None, 'ssim': None, 'num_samples': total_samples}

    # Create dataloader
    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Create output directories
    if save_images:
        if is_main:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(ref_dir, exist_ok=True)
        if use_ddp:
            dist.barrier()

    # Initialize metrics
    lpips_values, psnr_values, ssim_values = [], [], []
    lpips_metric = LPIPS(device=device)
    ssim_metric = get_ssim_metric(device)

    # Get precision dtype
    if precision in ('bf16', 'bfloat16'):
        dtype = torch.bfloat16
    elif precision in ('fp16', 'float16'):
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Determine autocast device_type
    autocast_device = 'npu' if device.type == 'npu' else ('cuda' if device.type == 'cuda' else 'cpu')

    if is_main:
        print(f"\nProcessing {total_samples} samples...")

    progress_bar = tqdm(dataloader, desc=f"Rank {local_rank}") if is_main else dataloader

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            actual_model = model.module if use_ddp else model

            if autocast_device != 'cpu' and dtype != torch.float32:
                with torch.amp.autocast(device_type=autocast_device, dtype=dtype):
                    latents = actual_model.get_reconstruction_latents(images)
                with torch.amp.autocast(device_type=autocast_device, dtype=torch.float32):
                    recon = actual_model.get_latents_decoded_images(latents)
                    recon_denorm = transform_rev(recon)
                    recon_denorm = torch.clamp(recon_denorm, 0, 1)
            else:
                latents = actual_model.get_reconstruction_latents(images)
                recon = actual_model.get_latents_decoded_images(latents)
                recon_denorm = transform_rev(recon)
                recon_denorm = torch.clamp(recon_denorm, 0, 1)

            orig_denorm = transform_rev(images)
            orig_denorm = torch.clamp(orig_denorm, 0, 1)

            if lpips_metric.model is not None:
                orig_lpips = orig_denorm * 2.0 - 1.0
                recon_lpips = recon_denorm * 2.0 - 1.0
                lpips_val = lpips_metric(orig_lpips, recon_lpips)
                if lpips_val is not None:
                    if isinstance(lpips_val, torch.Tensor) and lpips_val.numel() > 1:
                        lpips_values.append(lpips_val.mean().item())
                    else:
                        lpips_values.append(float(lpips_val))

            if ssim_metric is not None:
                ssim_values.append(ssim_metric(orig_denorm, recon_denorm).item())

            for i in range(images.size(0)):
                psnr = calculate_psnr(orig_denorm[i] * 255.0, recon_denorm[i] * 255.0)
                psnr_values.append(psnr)

            if save_images:
                orig_np = (orig_denorm.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                recon_np = (recon_denorm.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

                for i in range(images.size(0)):
                    global_idx = batch_idx * batch_size * world_size + local_rank * batch_size + i
                    if global_idx >= total_samples:
                        break
                    Image.fromarray(orig_np[i]).save(os.path.join(ref_dir, f"ref_{global_idx:06d}.png"))
                    Image.fromarray(recon_np[i]).save(os.path.join(save_dir, f"rec_{global_idx:06d}.png"))

    if use_ddp:
        dist.barrier()

    if use_ddp:
        def reduce_mean(values):
            if not values:
                return None
            t = torch.tensor(np.mean(values), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return (t / world_size).item()

        avg_lpips = reduce_mean(lpips_values)
        avg_psnr = reduce_mean(psnr_values)
        avg_ssim = reduce_mean(ssim_values)
    else:
        avg_lpips = np.mean(lpips_values) if lpips_values else None
        avg_psnr = np.mean(psnr_values) if psnr_values else None
        avg_ssim = np.mean(ssim_values) if ssim_values else None

    fid = None
    if save_images and is_main and os.path.exists(save_dir) and os.path.exists(ref_dir):
        try:
            print("\nComputing rFID...")
            from pytorch_fid import fid_score
            fid = fid_score.calculate_fid_given_paths([ref_dir, save_dir], batch_size=50, device=device, dims=2048, num_workers=4)
        except Exception as e:
            print(f"Error during FID calculation: {e}")

    if is_main:
        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        if fid is not None:
            print(f"  rFID: {fid:.3f}")
        if avg_psnr is not None:
            print(f"  PSNR: {avg_psnr:.2f} dB")
        if avg_lpips is not None:
            print(f"  LPIPS: {avg_lpips:.4f}")
        if avg_ssim is not None:
            print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  Samples: {len(psnr_values) * world_size}")
        print("=" * 60)

    if use_ddp:
        dist.destroy_process_group()

    return {
        'fid': fid,
        'psnr': avg_psnr,
        'lpips': avg_lpips,
        'ssim': avg_ssim,
        'num_samples': len(psnr_values) * world_size
    }


def main():
    parser = argparse.ArgumentParser(description="ImageNet reconstruction evaluation for VTP NPU")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VTP HuggingFace model directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ImageNet validation dataset")
    parser.add_argument("--output_path", type=str, default="reconstruction_output",
                        help="Output directory for reconstructed images")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per NPU")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., npu:0, cpu)")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Precision for inference")
    parser.add_argument("--no_save_images", action="store_true",
                        help="Do not save reconstructed images (skip FID calculation)")
    parser.add_argument("--use_ddp", action="store_true",
                        help="Use Distributed Data Parallel for multi-NPU")
    parser.add_argument("--local_rank", type=int, default=None,
                        help="Local rank for DDP (usually set automatically)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for quick testing)")
    args = parser.parse_args()

    if args.use_ddp:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    test_reconstruction(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
        save_images=not args.no_save_images,
        use_ddp=args.use_ddp,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
