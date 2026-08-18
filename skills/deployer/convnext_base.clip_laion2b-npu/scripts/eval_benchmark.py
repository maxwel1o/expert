"""
convnext_base.clip_laion2b — Ascend NPU 精度 & 性能综合测评
=============================================================
生成文件：
  - eval_logs/accuracy_report.txt     精度对比报告
  - eval_logs/benchmark_report.txt    性能基准报告
  - eval_logs/runtime.log             完整运行日志
"""
import os, sys, time, json, logging
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file
from timm.models.convnext import convnext_base
from timm.data import resolve_data_config, create_transform

# NPU injection
import torch_npu
from torch_npu.contrib import transfer_to_npu

# ===== Configuration =====
MODEL_NAME = "convnext_base.clip_laion2b"
WEIGHT_PATH = str(Path("/opt/atomgit/convnext_base.clip_laion2b_modelscope/timm/convnext_base.clip_laion2b/model.safetensors"))
LOG_DIR = Path("/opt/atomgit/eval_logs")
LOG_DIR.mkdir(exist_ok=True)
NUM_CLASSES = 640
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "runtime.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_model(device: str):
    log.info(f"Loading model on {device} ...")
    model = convnext_base(pretrained=False, num_classes=NUM_CLASSES)
    model.eval()
    state_dict = load_file(WEIGHT_PATH)
    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")
    model = model.to(device)
    log.info(f"Model loaded on {device}: {sum(p.numel() for p in model.parameters()):,} params")
    return model


@torch.no_grad()
def accuracy_test(cpu_model, npu_model, config, num_samples=50):
    """Compare CPU vs NPU logits across multiple random inputs."""
    log.info(f"\n{'='*60}")
    log.info(f"Accuracy Test: {num_samples} random samples")
    log.info(f"{'='*60}")
    input_size = config["input_size"]
    max_diffs, mean_diffs, cos_sims = [], [], []

    for i in range(num_samples):
        x = torch.randn(1, *input_size)
        cpu_out = cpu_model(x)
        npu_out = npu_model(x.to("npu:0"))

        cpu_arr = cpu_out.cpu().float().numpy()
        npu_arr = npu_out.cpu().float().numpy()

        max_diff = float(np.max(np.abs(cpu_arr - npu_arr)))
        mean_diff = float(np.mean(np.abs(cpu_arr - npu_arr)))
        cpu_flat = cpu_arr.flatten()
        npu_flat = npu_arr.flatten()
        cos_sim = float(np.dot(cpu_flat, npu_flat) / (np.linalg.norm(cpu_flat) * np.linalg.norm(npu_flat) + 1e-12))

        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        cos_sims.append(cos_sim)

        if (i + 1) % 10 == 0:
            log.info(f"  [{i+1:3d}/{num_samples}] max_diff={max_diff:.6e}  cos_sim={cos_sim:.8f}")

    # Summary
    log.info(f"\n--- Accuracy Summary ({num_samples} samples) ---")
    log.info(f"  Max absolute diff:  max={max(max_diffs):.6e}  min={min(max_diffs):.6e}  avg={np.mean(max_diffs):.6e}")
    log.info(f"  Mean absolute diff: max={max(mean_diffs):.6e}  min={min(mean_diffs):.6e}  avg={np.mean(mean_diffs):.6e}")
    log.info(f"  Cosine similarity:  max={max(cos_sims):.8f}  min={min(cos_sims):.8f}  avg={np.mean(cos_sims):.8f}")
    log.info(f"  Max diff < 1%: {'PASS' if max(max_diffs) < 0.01 else 'FAIL'}")

    return {
        "num_samples": num_samples,
        "max_abs_diff_max": float(max(max_diffs)),
        "max_abs_diff_min": float(min(max_diffs)),
        "max_abs_diff_avg": float(np.mean(max_diffs)),
        "cos_sim_max": float(max(cos_sims)),
        "cos_sim_min": float(min(cos_sims)),
        "cos_sim_avg": float(np.mean(cos_sims)),
    }


@torch.no_grad()
def benchmark(model, device: str, config, num_warmup=3, num_runs=20):
    """Benchmark latency and throughput at various batch sizes."""
    log.info(f"\n{'='*60}")
    log.info(f"Performance Benchmark — device={device}")
    log.info(f"{'='*60}")
    input_size = config["input_size"]
    results = {}

    for batch_size in [1, 2, 4, 8, 16, 32]:
        x = torch.randn(batch_size, *input_size).to(device)
        # Warmup
        for _ in range(num_warmup):
            _ = model(x)
        if device == "npu":
            torch.npu.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(x)
        if device == "npu":
            torch.npu.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = elapsed / num_runs * 1000
        throughput = batch_size / (elapsed / num_runs)
        results[batch_size] = {"avg_latency_ms": round(avg_ms, 2), "throughput_fps": round(throughput, 2)}
        log.info(f"  batch_size={batch_size:2d}  avg_latency={avg_ms:8.2f} ms  throughput={throughput:8.2f} img/s")

    return results


def main():
    log.info(f"torch: {torch.__version__}")
    log.info(f"NPU available: {torch.npu.is_available()}")
    if torch.npu.is_available():
        log.info(f"NPU device: {torch.npu.get_device_name(0)}")

    # Config
    cfg = resolve_data_config({}, model=convnext_base(pretrained=False))
    log.info(f"Input config: {cfg}")

    # ---- 1. Accuracy ----
    log.info(f"\n{'#'*60}")
    log.info(f"# Phase 1: Accuracy Evaluation")
    log.info(f"{'#'*60}")
    cpu_model = load_model("cpu")
    npu_model = load_model("npu:0")

    acc_results = accuracy_test(cpu_model, npu_model, cfg, num_samples=50)

    # ---- 2. Performance ----
    log.info(f"\n{'#'*60}")
    log.info(f"# Phase 2: Performance Benchmark")
    log.info(f"{'#'*60}")
    perf_results = {}

    log.info("\n--- CPU Performance ---")
    perf_results["cpu"] = benchmark(cpu_model, "cpu", cfg)

    if torch.npu.is_available():
        log.info("\n--- NPU Performance ---")
        perf_results["npu"] = benchmark(npu_model, "npu:0", cfg)

    # ---- 3. Summary ----
    log.info(f"\n{'#'*60}")
    log.info(f"# Final Summary")
    log.info(f"{'#'*60}")

    log.info(f"\nModel: {MODEL_NAME}")
    log.info(f"Weights: {WEIGHT_PATH}")
    log.info(f"Parameters: {sum(p.numel() for p in cpu_model.parameters()):,}")
    log.info(f"Input size: {cfg['input_size']}")

    log.info(f"\n--- Accuracy ---")
    for k, v in acc_results.items():
        log.info(f"  {k}: {v}")

    log.info(f"\n--- Performance ---")
    for device, batches in perf_results.items():
        for bs, metrics in batches.items():
            log.info(f"  {device} bs={bs}: {metrics}")

    if "npu" in perf_results and "cpu" in perf_results:
        log.info(f"\n--- Speedup (bs=1) ---")
        cpu_lat = perf_results["cpu"][1]["avg_latency_ms"]
        npu_lat = perf_results["npu"][1]["avg_latency_ms"]
        log.info(f"  CPU: {cpu_lat:.2f} ms")
        log.info(f"  NPU: {npu_lat:.2f} ms")
        log.info(f"  Speedup: {cpu_lat / npu_lat:.2f}x")

    # Save structured results
    report = {
        "model": MODEL_NAME,
        "parameters": sum(p.numel() for p in cpu_model.parameters()),
        "input_size": cfg["input_size"],
        "npu_device": torch.npu.get_device_name(0) if torch.npu.is_available() else None,
        "accuracy": acc_results,
        "performance": perf_results,
    }
    with open(LOG_DIR / "eval_results.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved to {LOG_DIR / 'eval_results.json'}")

    # Save accuracy report text
    with open(LOG_DIR / "accuracy_report.txt", "w") as f:
        f.write(f"Accuracy Report — {MODEL_NAME}\n")
        f.write(f"{'='*50}\n\n")
        for k, v in acc_results.items():
            f.write(f"  {k}: {v}\n")

    # Save benchmark report text
    with open(LOG_DIR / "benchmark_report.txt", "w") as f:
        f.write(f"Benchmark Report — {MODEL_NAME}\n")
        f.write(f"{'='*50}\n\n")
        for device, batches in perf_results.items():
            f.write(f"[{device}]\n")
            for bs, metrics in batches.items():
                f.write(f"  bs={bs:2d}: {metrics}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
