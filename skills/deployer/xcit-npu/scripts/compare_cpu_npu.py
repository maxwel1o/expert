#!/usr/bin/env python3
"""
xcit_tiny_24_p8_384.fb_dist_in1k - CPU vs NPU Precision Comparison
Compare inference results between CPU and NPU for accuracy verification.
"""
import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image

MODEL_DIR = os.path.expanduser("~/.cache/modelscope/hub/models/timm/xcit_tiny_24_p8_384___fb_dist_in1k")
MODEL_FILE = os.path.join(MODEL_DIR, "pytorch_model.bin")
os.environ["TORCH_HOME"] = os.path.dirname(MODEL_DIR)


def load_image(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def load_imagenet_labels():
    try:
        import requests
        url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
        response = requests.get(url, timeout=10)
        class_idx = json.loads(response.text)
        labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
        return labels
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def get_model_output(model_name, device, input_tensor, model_file):
    """Get model output on specified device."""
    import timm
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval()
    model.to(device)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    del model
    return output.cpu()


def main():
    model_name = "xcit_tiny_24_p8_384.fb_dist_in1k"
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_input.jpg"

    print("=" * 60)
    print("CPU vs NPU Precision Comparison")
    print(f"Model: {model_name}")
    print(f"Image: {image_path}")
    print("=" * 60)

    # Get transforms by creating model once
    import timm
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    del model

    from timm.data import create_transform
    transform = create_transform(**data_config, is_training=False)
    input_tensor = load_image(image_path, transform)
    print(f"\nInput shape: {input_tensor.shape}")

    # Get CPU output
    print("\n--- CPU Inference ---")
    t0 = time.time()
    cpu_output = get_model_output(model_name, "cpu", input_tensor.clone(), MODEL_FILE)
    cpu_time = time.time() - t0
    print(f"CPU inference time: {cpu_time*1000:.2f}ms")

    # Get NPU output
    print("\n--- NPU Inference ---")
    t0 = time.time()
    npu_output = get_model_output(model_name, "npu:0", input_tensor.clone(), MODEL_FILE)
    npu_time = time.time() - t0
    print(f"NPU inference time: {npu_time*1000:.2f}ms")

    # ===== Precision Analysis =====
    print("\n" + "=" * 60)
    print("Precision Analysis")
    print("=" * 60)

    cpu_logits = cpu_output.numpy().flatten()
    npu_logits = npu_output.numpy().flatten()

    cpu_probs = torch.softmax(cpu_output, dim=1).numpy().flatten()
    npu_probs = torch.softmax(npu_output, dim=1).numpy().flatten()

    # Error metrics
    abs_diff = np.abs(cpu_logits - npu_logits)
    max_abs_error = float(abs_diff.max())
    mean_abs_error = float(abs_diff.mean())
    mse = float(np.mean((cpu_logits - npu_logits) ** 2))
    rmse = float(np.sqrt(mse))

    # Cosine similarity
    cos_sim = float(np.dot(cpu_logits, npu_logits) /
                   (np.linalg.norm(cpu_logits) * np.linalg.norm(npu_logits)))

    # Probability differences
    prob_abs_diff = np.abs(cpu_probs - npu_probs)
    max_prob_diff = float(prob_abs_diff.max())
    mean_prob_diff = float(prob_abs_diff.mean())

    # Top-1 / Top-5 comparison
    cpu_top1 = int(cpu_output.argmax(dim=1)[0])
    npu_top1 = int(npu_output.argmax(dim=1)[0])

    cpu_top5 = set(cpu_output.topk(5, dim=1).indices[0].tolist())
    npu_top5 = set(npu_output.topk(5, dim=1).indices[0].tolist())

    top1_match = cpu_top1 == npu_top1
    top5_intersection = cpu_top5 & npu_top5
    top5_overlap = len(top5_intersection) / 5.0

    labels = load_imagenet_labels()

    print(f"\n--- Error Metrics ---")
    print(f"Max Absolute Error (logits): {max_abs_error:.6e}")
    print(f"Mean Absolute Error (logits): {mean_abs_error:.6e}")
    print(f"RMSE: {rmse:.6e}")
    print(f"Cosine Similarity: {cos_sim:.10f}")

    print(f"\n--- Probability Metrics ---")
    print(f"Max Probability Difference: {max_prob_diff:.6e}")
    print(f"Mean Probability Difference: {mean_prob_diff:.6e}")

    print(f"\n--- Top-1 Prediction ---")
    cpu_label = labels[cpu_top1] if cpu_top1 < len(labels) else f"class_{cpu_top1}"
    npu_label = labels[npu_top1] if npu_top1 < len(labels) else f"class_{npu_top1}"
    print(f"CPU: class {cpu_top1} ({cpu_label})")
    print(f"NPU: class {npu_top1} ({npu_label})")
    print(f"Match: {'YES' if top1_match else 'NO'}")

    print(f"\n--- Top-5 Overlap ---")
    print(f"Overlapping classes: {len(top5_intersection)}/5 = {top5_overlap*100:.0f}%")
    for i, idx in enumerate(cpu_output.topk(5, dim=1).indices[0].tolist()):
        match = "Y" if idx in npu_top5 else "N"
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  CPU top-{i+1}: class {idx} ({label}) {match}")
    for i, idx in enumerate(npu_output.topk(5, dim=1).indices[0].tolist()):
        match = "Y" if idx in cpu_top5 else "N"
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  NPU top-{i+1}: class {idx} ({label}) {match}")

    # Relative error check
    epsilon = 1e-8
    relative_error = np.abs(cpu_probs - npu_probs) / (np.abs(cpu_probs) + epsilon)
    max_relative_error = float(relative_error.max()) * 100
    mean_relative_error = float(relative_error.mean()) * 100
    print(f"\n--- Relative Error ---")
    print(f"Max Relative Error (probabilities): {max_relative_error:.4f}%")
    print(f"Mean Relative Error (probabilities): {mean_relative_error:.4f}%")

    # Conclusion
    print(f"\n" + "=" * 60)
    passed = max_prob_diff < 0.01
    print(f"Precision Check: {'PASSED' if passed else 'FAILED'}")
    print(f"NPU vs CPU max probability difference: {max_prob_diff*100:.4f}%")
    print(f"Requirement: < 1%")
    print(f"Cosine similarity: {cos_sim:.10f}")
    print("=" * 60)

    # Save comparison results
    result = {
        "model": model_name,
        "input_shape": list(input_tensor.shape),
        "cpu_inference_time_ms": round(cpu_time * 1000, 2),
        "npu_inference_time_ms": round(npu_time * 1000, 2),
        "speedup": round(cpu_time / npu_time, 2) if npu_time > 0 else float('inf'),
        "metrics": {
            "max_abs_error_logits": max_abs_error,
            "mean_abs_error_logits": mean_abs_error,
            "rmse": rmse,
            "cosine_similarity": cos_sim,
            "max_prob_diff": max_prob_diff,
            "mean_prob_diff": mean_prob_diff,
            "max_relative_error_pct": round(max_relative_error, 6),
            "mean_relative_error_pct": round(mean_relative_error, 6),
        },
        "top1": {
            "cpu": {"index": cpu_top1, "label": cpu_label},
            "npu": {"index": npu_top1, "label": npu_label},
            "match": top1_match,
        },
        "top5": {
            "cpu_indices": cpu_output.topk(5, dim=1).indices[0].tolist(),
            "npu_indices": npu_output.topk(5, dim=1).indices[0].tolist(),
            "overlap": len(top5_intersection),
            "overlap_pct": round(top5_overlap * 100, 1),
        },
        "conclusion": {
            "max_prob_diff_pct": round(max_prob_diff * 100, 4),
            "passed": bool(max_prob_diff < 0.01),
            "requirement": "NPU vs CPU error < 1%",
        }
    }

    with open("precision_results.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nResults saved to precision_results.json")

    # Clean up NPU memory
    import gc
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
