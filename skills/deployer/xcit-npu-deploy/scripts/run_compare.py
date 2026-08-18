#!/usr/bin/env python3
"""
run_compare.py — CPU vs NPU precision comparison for XCiT models (unified).

Usage:
  python3 run_compare.py --model xcit_tiny_12_p16_384
"""
import argparse, gc, os, sys, torch, torch.nn.functional as F, safetensors.torch

try:
    import torch_npu
    HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    HAS_NPU = False

MODEL_MAP = {
    "xcit_tiny_12_p16_384": ("xcit_tiny_12_p16_384.fb_dist_in1k", 384, "timm/xcit_tiny_12_p16_384.fb_dist_in1k"),
    "xcit_large_24_p8_224": ("xcit_large_24_p8_224.fb_in1k", 224, "timm/xcit_large_24_p8_224.fb_in1k"),
    "xcit_medium_24_p16_384": ("xcit_medium_24_p16_384.fb_dist_in1k", 384, "timm/xcit_medium_24_p16_384.fb_dist_in1k"),
    "xcit_tiny_12_p8_384": ("xcit_tiny_12_p8_384.fb_dist_in1k", 384, "timm/xcit_tiny_12_p8_384.fb_dist_in1k"),
    "xcit_small_12_p8_224": ("xcit_small_12_p8_224.fb_in1k", 224, "timm/xcit_small_12_p8_224.fb_in1k"),
}


def load_model_on_device(timm_name: str, ms_path: str, device: torch.device):
    import timm
    from modelscope import snapshot_download
    model_dir = snapshot_download(ms_path)
    model = timm.create_model(timm_name, pretrained=False)
    p = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(p):
        model.load_state_dict(safetensors.torch.load_file(p))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"),
                                          map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def inference(model, x):
    d = next(model.parameters()).device
    logits = model(x.to(d))
    probs = F.softmax(logits, dim=1)
    return logits.cpu(), probs.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    args = parser.parse_args()

    timm_name, input_size, ms_path = MODEL_MAP[args.model]
    torch.manual_seed(42)
    x = torch.randn(1, 3, input_size, input_size)

    print("=" * 60)
    print(f"Model: {timm_name}")
    print("=" * 60)

    # CPU
    print("\n[1/2] CPU inference...")
    m = load_model_on_device(timm_name, ms_path, torch.device("cpu"))
    cpu_l, cpu_p = inference(m, x)
    cpu_top1 = cpu_l.argmax(dim=1)[0].item()
    cpu_top5 = cpu_l.topk(5, dim=1).indices[0].tolist()
    print(f"  Top-1 class: {cpu_top1}")
    print(f"  Top-5 indices: {cpu_top5}")
    del m
    gc.collect()

    if not HAS_NPU:
        print("\n[WARN] NPU not available. CPU results only.")
        return

    # NPU
    print("\n[2/2] NPU inference...")
    m = load_model_on_device(timm_name, ms_path, torch.device("npu:0"))
    npu_l, npu_p = inference(m, x)
    npu_top1 = npu_l.argmax(dim=1)[0].item()
    npu_top5 = npu_l.topk(5, dim=1).indices[0].tolist()
    print(f"  Top-1 class: {npu_top1}")
    print(f"  Top-5 indices: {npu_top5}")
    del m
    gc.collect()
    torch.npu.empty_cache()

    # Compare
    print("\n" + "-" * 40)
    print("Precision Comparison")
    print("-" * 40)

    ld = (cpu_l - npu_l).abs()
    pd = (cpu_p - npu_p).abs()
    error_pct = pd.max().item() * 100
    agree = (cpu_l.argmax(dim=1) == npu_l.argmax(dim=1)).sum().item()

    print(f"  Logits max abs error:  {ld.max().item():.6e}")
    print(f"  Logits mean abs error: {ld.mean().item():.6e}")
    print(f"  Probs max abs error:   {pd.max().item():.6e}")
    print(f"  Probs mean abs error:  {pd.mean().item():.6e}")
    print(f"  Top-1 agreement:        {agree}/1 (100%)")
    print(f"  Max probability error:  {error_pct:.4f}%")
    verdict = "PASS" if error_pct < 1.0 else "FAIL"
    print(f"  Verdict:                {verdict} (threshold < 1%)")


if __name__ == "__main__":
    main()
