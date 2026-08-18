"""
Compare NPU inference output against CPU baseline for wav2vec2 CTC model.
Uses multiple metrics suitable for speech recognition logits evaluation.
"""
import sys
import numpy as np

def compare(ref_path, npu_path):
    ref = np.load(ref_path)
    npu = np.load(npu_path)

    print(f"Reference shape: {ref.shape}, dtype: {ref.dtype}")
    print(f"NPU shape      : {npu.shape}, dtype: {npu.dtype}")

    if ref.shape != npu.shape:
        print(f"[ERROR] Shape mismatch: {ref.shape} vs {npu.shape}")
        sys.exit(1)

    abs_diff = np.abs(ref - npu)
    max_abs_err = np.max(abs_diff)
    mean_abs_err = np.mean(abs_diff)
    mse = np.mean((ref - npu) ** 2)
    rmse = np.sqrt(mse)

    # Relative to output dynamic range (standard for logits comparison)
    out_range = np.max(ref) - np.min(ref)
    rel_range = max_abs_err / out_range

    # Relative to mean absolute value
    mean_abs_ref = np.mean(np.abs(ref))
    rel_mean = max_abs_err / mean_abs_ref

    # Cosine similarity
    cos_sim = np.dot(ref.flatten(), npu.flatten()) / (
        np.linalg.norm(ref.flatten()) * np.linalg.norm(npu.flatten())
    )

    # Argmax prediction consistency (most important for CTC decoding)
    ref_pred = np.argmax(ref, axis=-1)
    npu_pred = np.argmax(npu, axis=-1)
    pred_match_rate = np.mean(ref_pred == npu_pred)

    print("\n========== Accuracy Comparison ==========")
    print(f"Max Absolute Error           : {max_abs_err:.6e}")
    print(f"Mean Absolute Error          : {mean_abs_err:.6e}")
    print(f"RMSE                         : {rmse:.6e}")
    print(f"Cosine Similarity            : {cos_sim:.8f}")
    print(f"Argmax Match Rate            : {pred_match_rate*100:.2f}%")
    print(f"Rel Error (to full range)    : {rel_range*100:.4f}%")
    print(f"Rel Error (to mean abs)      : {rel_mean*100:.4f}%")

    # Pass criterion for wav2vec2 CTC logits:
    # 1. Prediction consistency must be 100% (decoded output identical)
    # 2. Relative error to output range < 1%
    passed = True
    if pred_match_rate < 1.0:
        print(f"\n[FAIL] Argmax match rate {pred_match_rate*100:.2f}% < 100%")
        passed = False
    else:
        print(f"\n[PASS] Argmax match rate = 100% (decoded output identical)")

    if rel_range >= 0.01:
        print(f"[FAIL] Relative error to range {rel_range*100:.4f}% >= 1%")
        passed = False
    else:
        print(f"[PASS] Relative error to range {rel_range*100:.4f}% < 1%")

    if cos_sim < 0.9999:
        print(f"[FAIL] Cosine similarity {cos_sim:.6f} < 0.9999")
        passed = False
    else:
        print(f"[PASS] Cosine similarity {cos_sim:.6f} >= 0.9999")

    if passed:
        print("\n========== OVERALL: PASS ==========")
        return 0
    else:
        print("\n========== OVERALL: FAIL ==========")
        return 1

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ref_path = sys.argv[1]
        npu_path = sys.argv[2]
    else:
        ref_path = "/opt/atomgit/wav2vec2-npu-adapt/ref_logits_cpu.npy"
        npu_path = "/opt/atomgit/wav2vec2-npu-adapt/logits_npu.npy"

    sys.exit(compare(ref_path, npu_path))
