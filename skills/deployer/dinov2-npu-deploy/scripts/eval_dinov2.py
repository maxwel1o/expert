#!/usr/bin/env python3
"""
vit_base_patch14_dinov2.lvd142m — Ascend NPU 精度 & 性能评测

评测项:
  1) 精度: CPU (fp32) vs NPU (fp32/fp16) 输出 embedding 的余弦相似度
  2) 性能: NPU 单卡延迟 (bs=1) 与吞吐 (bs=1/4/8)
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dinov2_npu_adapt import Dinov2NPUModel, precision_report

import torch

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_DIR = "/opt/atomgit/.cache/modelscope/facebook/dinov2-base"
OUTPUT_DIR = "/opt/atomgit/npu_ref/eval_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_IMAGES_PRECISION = 10   # 精度测试图片数
NUM_IMAGES_PERF = 50        # 性能测试 warmup + bench
WARMUP = 5
BATCH_SIZES = [1, 4, 8]

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def gen_images(n: int, size: int = 518):
    """生成 n 张随机测试图 (numpy HWC uint8)."""
    rng = np.random.RandomState(42)
    return [rng.randint(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def perf_bench(model: Dinov2NPUModel, images, batch_size: int = 1) -> dict:
    """对给定 batch_size 做吞吐 & 延迟测试."""
    device = model.device
    dtype = model.dtype

    # 预处理: 拼接 batch
    pixel_values_list = [model.preprocess(img) for img in images]
    batch = torch.cat(pixel_values_list[:batch_size], dim=0)

    # warmup
    for _ in range(WARMUP):
        _ = model.forward(batch)
    torch.npu.synchronize()

    # benchmark
    n_iters = max(1, NUM_IMAGES_PERF // batch_size)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model.forward(batch)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    total_samples = n_iters * batch_size
    avg_latency = elapsed / n_iters * 1000  # ms
    throughput = total_samples / elapsed    # samples/s

    return {
        "batch_size": batch_size,
        "iters": n_iters,
        "total_samples": total_samples,
        "total_time_s": round(elapsed, 4),
        "avg_latency_ms": round(avg_latency, 4),
        "throughput_samples_per_sec": round(throughput, 2),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    log_lines = []
    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"# DINOv2 NPU Evaluation — {datetime.now().isoformat()}")
    log(f"Model: {MODEL_DIR}")
    log(f"NPU: {torch.npu.get_device_name(0)}")

    images = gen_images(max(NUM_IMAGES_PRECISION, max(BATCH_SIZES)))
    log(f"Test images: {len(images)}")

    # ==================================================================
    # 1. 精度评测
    # ==================================================================
    log("\n" + "=" * 60)
    log("1. PRECISION EVALUATION")
    log("=" * 60)

    # CPU baseline
    log("\n--- Loading CPU baseline ...")
    cpu_model = Dinov2NPUModel(model_path=MODEL_DIR, device="cpu", dtype="fp32")
    cpu_embs = torch.cat(
        [cpu_model.infer(images[i])["embedding"] for i in range(NUM_IMAGES_PRECISION)], dim=0
    )
    del cpu_model
    log(f"CPU embeddings: {cpu_embs.shape}")

    precision_results = {}
    for dtype in ["fp32", "fp16"]:
        log(f"\n--- NPU dtype={dtype} ...")
        npu_model = Dinov2NPUModel(model_path=MODEL_DIR, device="npu:0", dtype=dtype)
        npu_embs = torch.cat(
            [npu_model.infer(images[i])["embedding"] for i in range(NUM_IMAGES_PRECISION)], dim=0
        )

        # batch comparison
        cos_sim = torch.nn.functional.cosine_similarity(cpu_embs.float(), npu_embs.float(), dim=1)
        abs_diff = (npu_embs.float() - cpu_embs.float()).abs()

        results = {
            "cosine_similarity_mean": round(cos_sim.mean().item(), 8),
            "cosine_similarity_min": round(cos_sim.min().item(), 8),
            "mean_abs_diff": round(abs_diff.mean().item(), 8),
            "max_abs_diff": round(abs_diff.max().item(), 8),
        }

        log(f"  Cosine similarity (mean): {results['cosine_similarity_mean']:.8f}")
        log(f"  Cosine similarity (min):  {results['cosine_similarity_min']:.8f}")
        log(f"  Mean absolute diff:       {results['mean_abs_diff']:.8e}")
        log(f"  Max absolute diff:        {results['max_abs_diff']:.8e}")

        precision_results[dtype] = results
        del npu_model
        torch.npu.empty_cache()

    # ==================================================================
    # 2. 性能评测
    # ==================================================================
    log("\n" + "=" * 60)
    log("2. PERFORMANCE EVALUATION")
    log("=" * 60)

    perf_results = {}
    for dtype in ["fp32", "fp16"]:
        log(f"\n--- NPU dtype={dtype} ...")
        model = Dinov2NPUModel(model_path=MODEL_DIR, device="npu:0", dtype=dtype)
        res_list = []
        for bs in BATCH_SIZES:
            res = perf_bench(model, images, batch_size=bs)
            res_list.append(res)
            log(f"  batch_size={bs}:  latency={res['avg_latency_ms']} ms  "
                f"throughput={res['throughput_samples_per_sec']} samples/s")
        perf_results[dtype] = res_list
        del model
        torch.npu.empty_cache()

    # ==================================================================
    # 3. 汇总报告
    # ==================================================================
    log("\n" + "=" * 60)
    log("3. SUMMARY")
    log("=" * 60)

    summary = {
        "model": "vit_base_patch14_dinov2.lvd142m",
        "npu": torch.npu.get_device_name(0),
        "timestamp": datetime.now().isoformat(),
        "precision": precision_results,
        "performance": perf_results,
    }

    log(json.dumps(summary, indent=2))

    # Save log
    log_path = os.path.join(OUTPUT_DIR, "eval_report.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nReport saved to {log_path}")

    # Also save raw log text
    txt_path = os.path.join(OUTPUT_DIR, "eval_output.log")
    with open(txt_path, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Log saved to {txt_path}")

    # Pass/fail verdict
    all_pass = True
    for dtype, res in precision_results.items():
        threshold = 0.02 if dtype == "fp16" else 0.015
        ok = res["cosine_similarity_mean"] > 0.999 and res["mean_abs_diff"] < threshold
        all_pass = all_pass and ok
        log(f"  Precision {dtype}: {'PASS' if ok else 'FAIL'} (cos={res['cosine_similarity_mean']:.6f}, "
            f"mean_abs={res['mean_abs_diff']:.6f})")

    log(f"\n>>> Overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
