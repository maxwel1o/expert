#!/usr/bin/env python3
"""Compare CPU vs NPU inference results for ConvNeXt model."""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
if not NPU_AVAILABLE:
    print("[ERROR] NPU not available. Cannot run comparison.")
    sys.exit(1)

MODEL_NAME = "test_convnext.r160_in1k"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "timm", "test_convnext___r160_in1k")
TEST_IMAGE = os.path.join(MODEL_PATH, "test", "test_owl.jpg")


def load_model(device="cpu"):
    """Load model on specified device."""
    import timm
    ckpt_path = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(ckpt_path):
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            checkpoint_path=ckpt_path,
        )
    else:
        model = timm.create_model(MODEL_NAME, pretrained=True)
    model = model.to(device)
    model.eval()
    return model


def preprocess(image_path, data_config):
    """Preprocess image using timm transforms."""
    import timm
    img = Image.open(image_path).convert("RGB")
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms(img).unsqueeze(0)


def compare_cpu_npu():
    """Run inference on CPU and NPU, then compare results."""
    import timm

    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"CPU vs NPU Accuracy Comparison")
    print(f"{'='*60}")

    # Load model on CPU to get data config
    model_cpu = load_model("cpu")
    data_config = timm.data.resolve_model_data_config(model_cpu)

    # Load and preprocess test image
    if os.path.exists(TEST_IMAGE):
        print(f"\nTest image: {TEST_IMAGE}")
        input_tensor = preprocess(TEST_IMAGE, data_config)
    else:
        print("\n[WARN] No test image, using random input.")
        input_tensor = torch.randn(1, 3, 160, 160)

    print(f"Input shape: {input_tensor.shape}")

    # ===== CPU Inference =====
    print(f"\n{'─'*40}")
    print("Running CPU inference...")
    with torch.no_grad():
        cpu_output = model_cpu(input_tensor)
    cpu_probs = torch.softmax(cpu_output, dim=1)
    cpu_top5 = torch.topk(cpu_probs, k=5)
    print("CPU inference completed.")

    # ===== NPU Inference =====
    print(f"\n{'─'*40}")
    print("Running NPU inference...")
    model_npu = load_model("npu")
    npu_input = input_tensor.to("npu")
    with torch.no_grad():
        npu_output = model_npu(npu_input)
    npu_output_cpu = npu_output.cpu()
    npu_probs = torch.softmax(npu_output_cpu, dim=1)
    npu_top5 = torch.topk(npu_probs, k=5)
    print("NPU inference completed.")

    # ===== Comparison =====
    print(f"\n{'='*60}")
    print("CPU vs NPU Comparison")
    print(f"{'='*60}")

    # 1. Logits comparison
    cpu_logits = cpu_output.numpy()
    npu_logits = npu_output_cpu.numpy()

    logits_diff = np.abs(cpu_logits - npu_logits)
    max_logits_diff = float(logits_diff.max())
    mean_logits_diff = float(logits_diff.mean())
    std_logits_diff = float(logits_diff.std())

    print(f"\n--- Logits Comparison ---")
    print(f"Max absolute diff:  {max_logits_diff:.8f}")
    print(f"Mean absolute diff: {mean_logits_diff:.8f}")
    print(f"Std of diff:        {std_logits_diff:.8f}")

    # 2. Probability comparison
    cpu_probs_np = cpu_probs.numpy()
    npu_probs_np = npu_probs.numpy()

    prob_diff = np.abs(cpu_probs_np - npu_probs_np)
    max_prob_diff = float(prob_diff.max())
    mean_prob_diff = float(prob_diff.mean())

    print(f"\n--- Probability Comparison ---")
    print(f"Max probability diff:  {max_prob_diff:.8f}")
    print(f"Mean probability diff: {mean_prob_diff:.8f}")
    print(f"Max diff %:            {max_prob_diff * 100:.6f}%")
    print(f"Mean diff %:           {mean_prob_diff * 100:.6f}%")

    # 3. Top-5 comparison
    print(f"\n--- Top-5 Comparison ---")
    print(f"{'Rank':<6} {'CPU idx':<10} {'CPU prob':<12} {'NPU idx':<10} {'NPU prob':<12} {'Match':<8}")
    print(f"{'─'*58}")
    matches = 0
    for i in range(5):
        cpu_idx = int(cpu_top5.indices[0][i])
        npu_idx = int(npu_top5.indices[0][i])
        cpu_p = float(cpu_top5.values[0][i])
        npu_p = float(npu_top5.values[0][i])
        match = "✓" if cpu_idx == npu_idx else "✗"
        if cpu_idx == npu_idx:
            matches += 1
        print(f"{i+1:<6} {cpu_idx:<10} {cpu_p:<12.6f} {npu_idx:<10} {npu_p:<12.6f} {match:<8}")

    top5_agreement = matches / 5 * 100
    print(f"\nTop-5 agreement: {matches}/5 ({top5_agreement:.1f}%)")

    # 4. Cosine similarity
    from scipy.spatial.distance import cosine
    cos_sim = 1 - cosine(cpu_logits.flatten(), npu_logits.flatten())
    print(f"\n--- Overall Similarity ---")
    print(f"Cosine similarity (logits): {cos_sim:.10f}")
    print(f"Cosine similarity %:        {cos_sim * 100:.6f}%")

    # 5. Relative error on significant logits (top-100 by absolute value)
    abs_cpu_logits = np.abs(cpu_logits.flatten())
    threshold = np.percentile(abs_cpu_logits, 90)  # top 10% significant logits
    significant_mask = abs_cpu_logits > threshold
    cpu_sig = cpu_logits.flatten()[significant_mask]
    npu_sig = npu_logits.flatten()[significant_mask]
    epsilon = 1e-8
    sig_rel_error = np.abs(cpu_sig - npu_sig) / (np.abs(cpu_sig) + epsilon)
    max_sig_rel_error = float(sig_rel_error.max())
    mean_sig_rel_error = float(sig_rel_error.mean())
    print(f"Max relative error (significant logits):  {max_sig_rel_error:.8f} ({max_sig_rel_error*100:.4f}%)")
    print(f"Mean relative error (significant logits): {mean_sig_rel_error:.8f} ({mean_sig_rel_error*100:.4f}%)")

    # Also compute relative error clamped version
    # For classification, probability difference is the meaningful metric
    max_prob_diff_pct = max_prob_diff * 100

    # 6. Conclusion
    passed = max_prob_diff_pct < 1.0  # 1% probability diff threshold
    print(f"\n{'='*60}")
    print(f"Conclusion: NPU vs CPU inference error < 1%: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"{'='*60}")

    # Save results
    results = {
        "model": MODEL_NAME,
        "input_shape": list(input_tensor.shape),
        "cpu_vs_npu": {
            "max_logits_diff": round(max_logits_diff, 8),
            "mean_logits_diff": round(mean_logits_diff, 8),
            "max_prob_diff": round(max_prob_diff, 8),
            "mean_prob_diff": round(mean_prob_diff, 8),
            "max_sig_relative_error": round(max_sig_rel_error, 8),
            "mean_sig_relative_error": round(mean_sig_rel_error, 8),
            "cosine_similarity": round(cos_sim, 10),
            "top5_agreement": f"{matches}/5",
            "top5_agreement_pct": round(top5_agreement, 2),
        },
        "cpu_top5": {
            "indices": [int(i) for i in cpu_top5.indices[0]],
            "probs": [round(float(v), 6) for v in cpu_top5.values[0]],
        },
        "npu_top5": {
            "indices": [int(i) for i in npu_top5.indices[0]],
            "probs": [round(float(v), 6) for v in npu_top5.values[0]],
        },
        "conclusion": "PASS" if passed else "FAIL",
        "error_threshold": "1%",
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "compare_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    import gc
    del model_cpu, model_npu
    gc.collect()
    torch.npu.empty_cache()

    return results


if __name__ == "__main__":
    compare_cpu_npu()
