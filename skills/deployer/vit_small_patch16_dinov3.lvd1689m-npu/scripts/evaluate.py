"""vit_small_patch16_dinov3.lvd1689m  --  精度/性能综合评测脚本

Output:
  - evaluation_report.json   (评测报告)
  - evaluation.log            (运行日志)
  - Console output            (用于截图)
"""
import os, sys, time, json, subprocess
import torch
import torch_npu
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "npu:0"
CKPT_PATH = "/opt/atomgit/vit_dinov3_adapt/vit_small_patch16_dinov3_lvd1689m.pth"
NUM_WARMUP = 10
NUM_BENCH = 100
REPORT_PATH = "/opt/atomgit/vit_dinov3_adapt/evaluation_report.json"
LOG_PATH = "/opt/atomgit/vit_dinov3_adapt/evaluation.log"

# ---------- helpers ----------
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def load_model(device, use_local_ckpt=True):
    import timm
    model = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=not use_local_ckpt)
    if use_local_ckpt:
        state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

@torch.no_grad()
def infer(model, x):
    return model(x)

def get_env_info():
    info = {}
    try:
        import timm
        info["timm_version"] = timm.__version__
    except: pass
    try:
        info["torch_version"] = torch.__version__
    except: pass
    try:
        info["torch_npu_version"] = torch_npu.__version__
    except: pass
    try:
        r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
        info["npu_smi"] = r.stdout
    except: pass
    return info

# ---------- accuracy test ----------
def test_accuracy():
    log("=" * 60)
    log("Accuracy Evaluation")
    log("=" * 60)

    # CPU baseline
    log("Loading CPU model...")
    model_cpu = load_model("cpu")

    # NPU model
    log("Loading NPU model...")
    model_npu = load_model(DEVICE)

    results = []
    seeds = [42, 123, 256, 512, 1024]

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        x = torch.randn(1, 3, 256, 256)

        out_cpu = infer(model_cpu, x)
        x_npu = x.to(DEVICE)
        out_npu = infer(model_npu, x_npu).float().cpu()

        a = out_npu.numpy().flatten()
        b = out_cpu.numpy().flatten()
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        mae = float(np.abs(a - b).mean())
        max_ae = float(np.abs(a - b).max())
        mse = float(((a - b) ** 2).mean())
        nmae = mae / (float(np.abs(b).mean()) + 1e-12)

        r = {"seed": seed, "cosine_similarity": cos_sim, "mae": mae,
             "max_abs_error": max_ae, "mse": mse, "normalized_mae": nmae,
             "passed": cos_sim > 0.999 and nmae < 0.01}
        results.append(r)
        log(f"  seed={seed:4d}  cos={cos_sim:.6f}  mae={mae:.6f}  nmae={nmae:.4%}  {'PASS' if r['passed'] else 'FAIL'}")

    return results

# ---------- performance benchmark ----------
def test_performance():
    log("\n" + "=" * 60)
    log("Performance Benchmark")
    log("=" * 60)

    model_npu = load_model(DEVICE)
    x = torch.randn(1, 3, 256, 256).to(DEVICE)

    # warmup
    log(f"Warmup ({NUM_WARMUP} runs)...")
    for _ in range(NUM_WARMUP):
        _ = infer(model_npu, x)
    torch.npu.synchronize(DEVICE)

    # benchmark with batch_size=1
    log(f"Benchmark ({NUM_BENCH} runs, batch_size=1)...")
    torch.npu.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(NUM_BENCH):
        _ = infer(model_npu, x)
    torch.npu.synchronize(DEVICE)
    t_total = time.perf_counter() - t0
    lat_avg = t_total / NUM_BENCH * 1000  # ms

    log(f"  Avg latency: {lat_avg:.2f} ms")

    # throughput
    throughput = 1000.0 / lat_avg  # samples/s
    log(f"  Throughput:  {throughput:.2f} samples/s")

    # batch_size sweep
    log("\nBatch-size sweep:")
    bs_results = {}
    for bs in [1, 2, 4, 8]:
        x_bs = torch.randn(bs, 3, 256, 256).to(DEVICE)
        # warmup
        for _ in range(5):
            _ = infer(model_npu, x_bs)
        torch.npu.synchronize(DEVICE)

        t0 = time.perf_counter()
        for _ in range(50):
            _ = infer(model_npu, x_bs)
        torch.npu.synchronize(DEVICE)
        t_bs = (time.perf_counter() - t0) / 50 * 1000
        thr = bs / (t_bs / 1000)
        bs_results[str(bs)] = {"latency_ms": round(t_bs, 2), "throughput_samples_per_s": round(thr, 2)}
        log(f"  bs={bs:2d}  latency={t_bs:.2f} ms  throughput={thr:.2f} samples/s")

    return {
        "batch_size_1": {
            "avg_latency_ms": round(lat_avg, 2),
            "throughput_samples_per_s": round(throughput, 2),
        },
        "batch_size_sweep": bs_results,
    }


# ---------- main ----------
def main():
    open(LOG_PATH, "w").close()

    log("=" * 60)
    log("vit_small_patch16_dinov3.lvd1689m  Comprehensive Evaluation")
    log("=" * 60)

    env = get_env_info()
    log(f"\nEnvironment:")
    for k, v in env.items():
        if k != "npu_smi":
            log(f"  {k}: {v}")

    # accuracy
    acc_results = test_accuracy()
    acc_passed = all(r["passed"] for r in acc_results)
    acc_avg_cos = float(np.mean([r["cosine_similarity"] for r in acc_results]))
    acc_avg_nmae = float(np.mean([r["normalized_mae"] for r in acc_results]))
    log(f"\nAccuracy summary: avg_cos={acc_avg_cos:.6f}  avg_nmae={acc_avg_nmae:.4%}  overall={'PASS' if acc_passed else 'FAIL'}")

    # performance
    perf_results = test_performance()

    # build report
    report = {
        "model": "vit_small_patch16_dinov3.lvd1689m",
        "framework": "timm + torch_npu",
        "device": "Ascend910B4",
        "environment": env,
        "accuracy": {
            "seeds": acc_results,
            "summary": {
                "avg_cosine_similarity": acc_avg_cos,
                "avg_normalized_mae": acc_avg_nmae,
                "passed": acc_passed,
            },
        },
        "performance": perf_results,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\nReport saved to {REPORT_PATH}")

    print("\n\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))

    return 0 if acc_passed else 1


if __name__ == "__main__":
    sys.exit(main())
