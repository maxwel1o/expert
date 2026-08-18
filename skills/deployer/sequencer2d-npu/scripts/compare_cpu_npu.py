#!/usr/bin/env python3
"""Compare CPU and NPU inference results for Sequencer2D models."""
import argparse
import numpy as np
from numpy.linalg import norm


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", default="logits_cpu.npy", help="CPU logits file")
    parser.add_argument("--npu", default="logits_npu.npy", help="NPU logits file")
    parser.add_argument("--model", default="sequencer2d", help="model name")
    args = parser.parse_args()

    cpu_logits = np.load(args.cpu)
    npu_logits = np.load(args.npu)

    print(f"=== CPU vs NPU Comparison for {args.model} ===\n")

    cpu_top1 = np.argmax(cpu_logits[0])
    npu_top1 = np.argmax(npu_logits[0])
    cpu_top5 = np.argsort(cpu_logits[0])[-5:][::-1]
    npu_top5 = np.argsort(npu_logits[0])[-5:][::-1]

    print(f"Top-1 class (CPU): {cpu_top1}")
    print(f"Top-1 class (NPU): {npu_top1}")
    print(f"Top-1 match: {cpu_top1 == npu_top1}")
    print(f"Top-5 match: {np.array_equal(cpu_top5, npu_top5)}")

    cpu_probs = softmax(cpu_logits)
    npu_probs = softmax(npu_logits)

    abs_diff = np.abs(cpu_logits - npu_logits)
    prob_abs_diff = np.abs(cpu_probs - npu_probs)
    l2_diff = norm(cpu_logits[0] - npu_logits[0])
    cosine_sim = np.dot(cpu_logits[0], npu_logits[0]) / (norm(cpu_logits[0]) * norm(npu_logits[0]))
    cpu_range = cpu_logits.max() - cpu_logits.min()
    error_rate = abs_diff.max() / cpu_range if cpu_range > 0 else 0

    print(f"\n--- Logits Comparison ---")
    print(f"Max absolute error:       {abs_diff.max():.8e}")
    print(f"Mean absolute error:      {abs_diff.mean():.8e}")
    print(f"Max probability diff:     {prob_abs_diff.max():.8e}")
    print(f"Mean probability diff:    {prob_abs_diff.mean():.8e}")
    print(f"L2 distance:              {l2_diff:.8e}")
    print(f"Cosine similarity:        {cosine_sim:.8f}")
    print(f"Error rate:               {error_rate:.6%}")
    print(f"Threshold (< 1%):         {'PASS' if error_rate < 0.01 else 'FAIL'}")

    print(f"\n=== Conclusion ===")
    if error_rate < 0.01:
        print(f"NPU and CPU inference results match within 1% tolerance.")
    else:
        print(f"NPU and CPU inference difference exceeds 1% tolerance!")

    return error_rate


if __name__ == "__main__":
    main()
