"""
NPU Compatibility Layer for VTP models.

Provides device auto-detection, dtype-sensitive layer adaptation,
and unified autocast helpers for Ascend NPU inference.
"""

import os
from typing import Optional

import torch


def get_device(prefer_npu: bool = True) -> torch.device:
    """Auto-detect available accelerator: npu > cuda > cpu."""
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


def is_npu_available() -> bool:
    """Check if Ascend NPU is available."""
    try:
        import torch_npu
        return torch.npu.is_available()
    except Exception:
        return False


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across devices."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def adapt_model_for_npu(model: torch.nn.Module) -> torch.nn.Module:
    """Adapt VTP model for NPU inference.

    Fixes:
      - pixel_decoder RoPE defaults to bfloat16 which causes NPU/CPU
        numerical divergence. We force float32 for deterministic results.
    """
    if hasattr(model, "pixel_decoder") and model.pixel_decoder is not None:
        rope = model.pixel_decoder.rope_embed
        if rope.dtype == torch.bfloat16:
            rope.dtype = torch.float32
            rope.periods = rope.periods.to(torch.float32)
            rope._init_weights()
    # Also handle legacy VTP class (non-HF)
    if hasattr(model, "pixel_decoder") and model.pixel_decoder is not None:
        rope = model.pixel_decoder.rope_embed
        if hasattr(rope, "dtype") and rope.dtype == torch.bfloat16:
            rope.dtype = torch.float32
            if hasattr(rope, "periods"):
                rope.periods = rope.periods.to(torch.float32)
            if hasattr(rope, "_init_weights"):
                rope._init_weights()
    return model


def get_autocast_kwargs(device: torch.device, precision: str = "fp32"):
    """Return (device_type, dtype) for torch.amp.autocast.

    Args:
        device: target torch.device
        precision: one of fp32, fp16, bf16

    Returns:
        (device_type_str, torch_dtype or None)
    """
    if device.type == "npu":
        device_type = "npu"
    elif device.type == "cuda":
        device_type = "cuda"
    else:
        device_type = "cpu"

    if precision in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif precision in ("fp16", "float16"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    return device_type, dtype


def maybe_autocast(device: torch.device, precision: str = "fp32"):
    """Context manager that returns autocast context or a no-op context.

    Usage:
        with maybe_autocast(device, "bf16"):
            output = model(input)
    """
    device_type, dtype = get_autocast_kwargs(device, precision)
    if device_type == "cpu" or dtype == torch.float32:
        from contextlib import nullcontext
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def synchronize_device(device: torch.device):
    """Synchronize the target device before/after timing."""
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
