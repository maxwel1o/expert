#!/usr/bin/env python3
"""Compare CPU vs NPU inference results for a LeViT model.

Usage:
  python3 compare.py --model levit-128
"""

import argparse
import json
import os

import numpy as np


MODEL_NAMES = {
    "128": "levit-128", "128s": "levit-128S", "192": "levit-192",
    "256": "levit-256", "384": "levit-384",
}


def resolve_model(model_arg: str) -> str:
    if model_arg in MODEL_NAMES:
        return MODEL_NAMES[model_arg]
    return model_arg


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs NPU results")
    parser.add_argument("--model", default="levit-128", help="Model name")
    args = parser.parse_args()

    model_name = resolve_model(args.model)

    with open(f"/tmp/{model_name}_cpu_results.json") as f:
        cpu = json.load(f)
    with open(f"/tmp/{model_name}_npu_results.json") as f:
        npu = json.load(f)

    cpu_l = np.array(cpu["logits"])
    npu_l = np.array(npu["logits"])
    cpu_p = np.array(cpu["probabilities"])
    npu_p = np.array(npu["probabilities"])

    # Logits
    l_diff = np.abs(cpu_l - npu_l)
    l_mae, l_maxae = np.mean(l_diff), np.max(l_diff)
    cos_sim = np.dot(cpu_l[0], npu_l[0]) / (
        np.linalg.norm(cpu_l[0]) * np.linalg.norm(npu_l[0]) + 1e-12
    )

    # Probabilities
    p_diff = np.abs(cpu_p - npu_p)
    p_mae, p_maxae = np.mean(p_diff), np.max(p_diff)

    cpu_top1 = int(np.argmax(cpu_l, axis=-1)[0])
    npu_top1 = int(np.argmax(npu_l, axis=-1)[0])

    print(f"Model: {model_name}")
    print(f"Logits MAE:    {l_mae:.8f}")
    print(f"Logits MaxAE:  {l_maxae:.8f}")
    print(f"Probs MAE:     {p_mae:.8f}")
    print(f"Probs MaxAE:   {p_maxae:.8f} ({p_maxae*100:.6f}%)")
    print(f"Cosine Sim:    {cos_sim:.8f}")
    print(f"CPU Top-1:     class {cpu_top1}")
    print(f"NPU Top-1:     class {npu_top1}")
    print(f"Top-1 Match:   {'YES' if cpu_top1 == npu_top1 else 'NO'}")
    print(f"Speedup:       {cpu['time_ms']/npu['time_ms']:.1f}x")
    print(f"\nResult: {'PASS' if p_maxae < 0.01 else 'FAIL'} (error {p_maxae*100:.4f}% {'<' if p_maxae < 0.01 else '>='} 1%)")


if __name__ == "__main__":
    main()
