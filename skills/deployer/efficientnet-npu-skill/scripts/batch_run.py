#!/usr/bin/env python3
"""Batch runner for EfficientNet NPU deployment skill - run all models sequentially."""

import gc
import subprocess
import sys
import torch

MODELS = [
    "test_efficientnet.r160_in1k",
    "test_efficientnet_evos.r160_in1k",
    "test_efficientnet_gn.r160_in1k",
    "test_efficientnet_ln.r160_in1k",
    "tf_efficientnet_b0.aa_in1k",
    "tf_efficientnet_b0.ap_in1k",
    "tf_efficientnet_b0.in1k",
    "tf_efficientnet_b0.ns_jft_in1k",
    "tf_efficientnet_b1.aa_in1k",
    "tf_efficientnet_b1.ap_in1k",
    "tf_efficientnet_b1.in1k",
    "tf_efficientnet_b1.ns_jft_in1k",
    "tf_efficientnet_b2.aa_in1k",
    "tf_efficientnet_b2.ap_in1k",
    "tf_efficientnet_b2.in1k",
    "tf_efficientnet_b2.ns_jft_in1k",
    "tf_efficientnet_b3.aa_in1k",
    "tf_efficientnet_b3.ap_in1k",
    "tf_efficientnet_b3.in1k",
    "tf_efficientnet_b3.ns_jft_in1k",
    "tf_efficientnet_b4.aa_in1k",
    "tf_efficientnet_b4.ap_in1k",
    "tf_efficientnet_b4.in1k",
    "tf_efficientnet_b4.ns_jft_in1k",
    "tf_efficientnet_b5.aa_in1k",
    "tf_efficientnet_b5.ap_in1k",
    "tf_efficientnet_b5.in1k",
    "tf_efficientnet_b5.ns_jft_in1k",
    "tf_efficientnet_b5.ra_in1k",
    "tf_efficientnet_b6.aa_in1k",
    "tf_efficientnet_b6.ap_in1k",
    "tf_efficientnet_b6.ns_jft_in1k",
    "tf_efficientnet_b7.aa_in1k",
    "tf_efficientnet_b7.ap_in1k",
    "tf_efficientnet_b7.ns_jft_in1k",
    "tf_efficientnet_b7.ra_in1k",
    "tf_efficientnet_b8.ap_in1k",
    "tf_efficientnet_b8.ra_in1k",
    "tf_efficientnet_cc_b0_4e.in1k",
    "tf_efficientnet_cc_b0_8e.in1k",
    "tf_efficientnet_cc_b1_8e.in1k",
    "tf_efficientnet_el.in1k",
    "tf_efficientnet_em.in1k",
    "tf_efficientnet_es.in1k",
    "tf_efficientnet_l2.ns_jft_in1k",
    "tf_efficientnet_l2.ns_jft_in1k_475",
    "tf_efficientnet_lite0.in1k",
    "tf_efficientnet_lite1.in1k",
    "tf_efficientnet_lite2.in1k",
    "tf_efficientnet_lite3.in1k",
    "tf_efficientnet_lite4.in1k",
    "tf_efficientnetv2_b0.in1k",
    "tf_efficientnetv2_b1.in1k",
    "tf_efficientnetv2_b2.in1k",
    "tf_efficientnetv2_b3.in1k",
    "tf_efficientnetv2_b3.in21k",
    "tf_efficientnetv2_b3.in21k_ft_in1k",
    "tf_efficientnetv2_l.in1k",
    "tf_efficientnetv2_l.in21k",
    "tf_efficientnetv2_l.in21k_ft_in1k",
    "tf_efficientnetv2_m.in1k",
    "tf_efficientnetv2_m.in21k",
    "tf_efficientnetv2_m.in21k_ft_in1k",
    "tf_efficientnetv2_s.in1k",
    "tf_efficientnetv2_s.in21k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "tf_efficientnetv2_xl.in21k",
    "tf_efficientnetv2_xl.in21k_ft_in1k",
]


def run_model(model_name, device="npu"):
    """Run inference and comparison for a single model."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name} on {device}")
    print(f"{'='*60}\n")

    # Step 1: NPU inference
    print(f"[1/3] Running {device} inference...")
    result = subprocess.run(
        [sys.executable, "inference.py", model_name, "--device", device],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] Inference failed for {model_name}")
        return False

    # Step 2: CPU inference (for comparison)
    if device == "npu":
        print(f"\n[2/3] Running CPU inference for comparison...")
        result = subprocess.run(
            [sys.executable, "inference.py", model_name, "--device", "cpu"],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[ERROR] CPU inference failed for {model_name}")
            return False

        # Step 3: Accuracy comparison
        print(f"\n[3/3] Running CPU vs NPU accuracy comparison...")
        result = subprocess.run(
            [sys.executable, "compare_cpu_npu.py", model_name],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[ERROR] Comparison failed for {model_name}")
            return False

    # Memory cleanup
    print(f"\n[Cleanup] Releasing resources...")
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()

    print(f"\n>> Completed: {model_name}")
    return True


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "npu"
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else len(MODELS)

    selected = MODELS[start_idx:end_idx]
    print(f"Batch processing models [{start_idx}:{end_idx}], total={len(selected)}")

    for i, model_name in enumerate(selected):
        print(f"\n--- Model {start_idx + i + 1}/{len(MODELS)}: {model_name} ---")
        success = run_model(model_name, device)
        if not success:
            print(f"[FAILED] {model_name} - continuing with next model")

    successes = sum(1 for m in selected[:end_idx - start_idx] if True)  # approximate
    print(f"\n{'='*60}")
    print(f"Batch processing complete! Processed {len(selected)} models.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
