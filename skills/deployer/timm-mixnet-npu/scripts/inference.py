#!/usr/bin/env python3
"""timm MixNet NPU inference script - supports tf_mixnet_s/m/l.in1k"""

import os
import argparse
import time
import json
import torch
import torch_npu
import numpy as np
from safetensors.torch import load_file
from timm import create_model

MODEL_NAMES = ["tf_mixnet_s.in1k", "tf_mixnet_m.in1k", "tf_mixnet_l.in1k"]


def load_model(model_name: str, checkpoint_dir: str, device: torch.device):
    safetensors_path = f"{checkpoint_dir}/model.safetensors"
    bin_path = f"{checkpoint_dir}/pytorch_model.bin"

    print(f"[INFO] Creating model: {model_name}")
    model = create_model(model_name, pretrained=False)
    model.eval()

    if os.path.exists(safetensors_path):
        print(f"[INFO] Loading from safetensors: {safetensors_path}")
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        print(f"[INFO] Loading from bin: {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    # Filter state dict if needed (remove unexpected keys)
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())
    extra_keys = sd_keys - model_keys
    if extra_keys:
        print(f"[INFO] Skipping {len(extra_keys)} unexpected keys in checkpoint")
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print(f"[INFO] Model loaded on {device}")
    return model


def main():
    parser = argparse.ArgumentParser(description="timm MixNet NPU Inference")
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_NAMES,
                        help="Model name (e.g. tf_mixnet_s.in1k)")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--device", type=str, default="npu",
                        choices=["cpu", "npu"],
                        help="Inference device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    import os

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" else f"npu:0")

    model = load_model(args.model, args.checkpoint_dir, device)

    # Dummy input (matching model's expected input size 3x224x224)
    print("[INFO] Using dummy RGB image tensor (3x224x224)")
    input_tensor = torch.randn(1, 3, 224, 224)

    # Warm-up
    with torch.no_grad():
        _ = model(input_tensor.to(device))

    # Timed inference
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        start = time.time()
        output = model(input_tensor)
        torch.npu.synchronize() if args.device == "npu" else None
        elapsed = time.time() - start

    output_cpu = output.cpu()
    probs = torch.nn.functional.softmax(output_cpu[0], dim=0)
    top5 = probs.topk(5)

    print(f"\n=== Results on {args.device} ===")
    print(f"Inference time: {elapsed:.4f}s")
    print(f"Output shape: {output_cpu.shape}")
    print(f"Top-5 predictions:")
    for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i + 1}. class {idx.item():4d}: {score.item():.6f}")

    result = {
        "model": args.model,
        "device": args.device,
        "time_seconds": round(elapsed, 4),
        "output_shape": list(output_cpu.shape),
        "top5_indices": [idx.item() for idx in top5.indices],
        "top5_scores": [round(s.item(), 6) for s in top5.values],
    }
    with open(f"output_{args.device}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Results saved to output_{args.device}.json")

    torch.save(output_cpu, f"logits_{args.device}.pt")
    print(f"[INFO] Full logits saved to logits_{args.device}.pt")


if __name__ == "__main__":
    main()
