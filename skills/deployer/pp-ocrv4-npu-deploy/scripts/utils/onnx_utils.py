"""
Common utilities for PP-OCRv4 ONNX model inference on CPU and Ascend NPU.
"""

import time
import numpy as np
import onnxruntime as ort


def create_session(model_path, device="cpu"):
    """Create ONNX Runtime session for CPU or NPU (CANN)."""
    if device in ("npu", "cann", "ascend"):
        providers = ["CANNExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
    else:
        providers = ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
    return session


def run_inference(session, input_dict, warmup=1, num_runs=5):
    """Run ONNX inference with warmup and return outputs + timing."""
    for _ in range(warmup):
        session.run(None, input_dict)

    times = []
    outputs = None
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, input_dict)
        end = time.perf_counter()
        times.append(end - start)

    return outputs, {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "num_runs": num_runs,
    }


def compare_outputs(np_result, cpu_result, tolerance=1e-5):
    """Compare NPU and CPU inference outputs.

    Returns a dict with various comparison metrics.
    """
    metrics = {}

    if isinstance(np_result, list):
        np_result = np_result[0]
    if isinstance(cpu_result, list):
        cpu_result = cpu_result[0]

    np_arr = np.array(np_result, dtype=np.float64)
    cpu_arr = np.array(cpu_result, dtype=np.float64)

    diff = np.abs(np_arr - cpu_arr)
    metrics["max_abs_error"] = float(np.max(diff))
    metrics["mean_abs_error"] = float(np.mean(diff))
    metrics["median_abs_error"] = float(np.median(diff))

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(cpu_arr), 1e-8)
    rel_err = diff / denom
    metrics["max_rel_error"] = float(np.max(rel_err))
    metrics["mean_rel_error"] = float(np.mean(rel_err))

    # Cosine similarity
    flat_np = np_arr.flatten().astype(np.float64)
    flat_cpu = cpu_arr.flatten().astype(np.float64)
    dot = np.dot(flat_np, flat_cpu)
    norm_np = np.linalg.norm(flat_np)
    norm_cpu = np.linalg.norm(flat_cpu)
    if norm_np > 0 and norm_cpu > 0:
        metrics["cosine_sim"] = float(dot / (norm_np * norm_cpu))
    else:
        metrics["cosine_sim"] = 1.0

    # Percentage of elements within tolerance
    within_tol = diff < tolerance
    metrics["within_tol_pct"] = float(np.mean(within_tol) * 100)

    # L1 and L2 norms
    metrics["l1_error"] = float(np.sum(diff))
    metrics["l2_error"] = float(np.sqrt(np.sum(diff ** 2)))

    # Signal-to-noise ratio
    signal_power = np.mean(cpu_arr ** 2)
    noise_power = np.mean(diff ** 2)
    if noise_power > 0:
        metrics["snr_db"] = float(10 * np.log10(signal_power / noise_power))
    else:
        metrics["snr_db"] = float("inf")

    return metrics


def clean_npu_cache():
    """Release NPU memory cache."""
    import gc
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass
