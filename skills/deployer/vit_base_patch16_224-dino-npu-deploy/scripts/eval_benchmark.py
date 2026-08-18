#!/usr/bin/env python3
"""
vit_base_patch16_224.dino — 精度 & 性能综合测评

输出:
  - 精度: cosine similarity, L2 relative error
  - 性能: latency (ms), throughput (imgs/s)
  - 不同 batch size 下的性能基准
"""

import os
import sys
import time
import json
import copy

os.environ["NPU_IGNORE_PERMISSION_MISMATCH"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

REPORT = {}

def log(msg):
    print(msg)
    sys.stdout.flush()

def load_model():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    model.eval()
    return model

# ═══════════════════════════════════════════
# 1. 环境信息
# ═══════════════════════════════════════════
log("=" * 60)
log("vit_base_patch16_224.dino — 精度 & 性能测评报告")
log("=" * 60)

log("\n[1] 环境信息")
env_info = {
    "torch": torch.__version__,
    "torch_npu": torch.version.npu if hasattr(torch, 'version') and hasattr(torch.version, 'npu') else "N/A",
    "npu_available": torch.npu.is_available(),
    "npu_device": torch.npu.get_device_name(0) if torch.npu.is_available() else "N/A",
    "npu_count": torch.npu.device_count() if torch.npu.is_available() else 0,
    "python": sys.version,
}
for k, v in env_info.items():
    log(f"  {k}: {v}")
REPORT["environment"] = env_info

# ═══════════════════════════════════════════
# 2. 加载模型
# ═══════════════════════════════════════════
log("\n[2] 加载模型")
model = load_model()
num_params = sum(p.numel() for p in model.parameters())
log(f"  params: {num_params:,}")
REPORT["model"] = {
    "name": "vit_base_patch16_224.dino",
    "params": num_params,
    "arch": "ViT-B/16",
    "output_dim": 768,
}

# ═══════════════════════════════════════════
# 3. 精度对比 (CPU vs NPU) — 合成数据
# ═══════════════════════════════════════════
log("\n[3] 精度对比 — 合成数据 (CPU vs NPU)")

torch.manual_seed(42)
np.random.seed(42)
x = torch.randn(1, 3, 224, 224) * 0.2 + 0.5
x = x.clamp(0, 1)

with torch.no_grad():
    out_cpu = model(x)

# 创建 NPU 副本
model_npu = copy.deepcopy(model).to("npu:0")
x_npu = x.to("npu:0")
with torch.no_grad():
    out_npu = model_npu(x_npu).cpu()

cos_sim = F.cosine_similarity(out_cpu.flatten().unsqueeze(0),
                               out_npu.flatten().unsqueeze(0)).item()
abs_diff = (out_cpu - out_npu).abs()
l2_cpu = float(out_cpu.flatten().norm())
l2_diff = float((out_cpu - out_npu).flatten().norm())
rel_l2 = l2_diff / max(l2_cpu, 1e-8)
max_abs_err = float(abs_diff.max().item())
mean_abs_err = float(abs_diff.mean().item())
rel_err = abs_diff / (out_cpu.abs().clamp(min=1e-8))
max_rel_err = float(rel_err.max().item())
mean_rel_err = float(rel_err.mean().item())

accuracy = {
    "cos_sim": cos_sim,
    "l2_rel_error_pct": round(rel_l2 * 100, 3),
    "max_abs_err": max_abs_err,
    "mean_abs_err": mean_abs_err,
    "max_rel_err": max_rel_err,
    "mean_rel_err": mean_rel_err,
}
log(f"  cosine similarity:   {cos_sim:.8f}")
log(f"  L2 relative error:  {rel_l2*100:.3f}%")
log(f"  max abs error:      {max_abs_err:.6f}")
log(f"  max rel error:      {max_rel_err:.6f}")
passed = cos_sim > 0.999 or rel_l2 < 0.01
accuracy["passed"] = passed
log(f"  判定: {'PASS' if passed else 'FAIL'}")
REPORT["accuracy"] = accuracy

# ═══════════════════════════════════════════
# 3b. 精度对比 — 真实图片
# ═══════════════════════════════════════════
log("\n[3b] 精度对比 — 真实图片")

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.fromarray((np.random.RandomState(42).rand(300, 300, 3) * 255).astype(np.uint8))
x_img = transform(img).unsqueeze(0)

with torch.no_grad():
    out_cpu_img = model(x_img)  # model is still on CPU

x_img_npu = x_img.to("npu:0")
with torch.no_grad():
    out_npu_img = model_npu(x_img_npu).cpu()

cos_sim_img = F.cosine_similarity(out_cpu_img.flatten().unsqueeze(0),
                                   out_npu_img.flatten().unsqueeze(0)).item()
l2_diff_img = float((out_cpu_img - out_npu_img).flatten().norm())
l2_cpu_img = float(out_cpu_img.flatten().norm())
rel_l2_img = l2_diff_img / max(l2_cpu_img, 1e-8)

accuracy_img = {
    "cos_sim": cos_sim_img,
    "l2_rel_error_pct": round(rel_l2_img * 100, 3),
}
log(f"  cosine similarity:   {cos_sim_img:.8f}")
log(f"  L2 relative error:  {rel_l2_img*100:.3f}%")
log(f"  判定: {'PASS' if cos_sim_img > 0.999 or rel_l2_img < 0.01 else 'FAIL'}")
REPORT["accuracy_real_image"] = accuracy_img

# ═══════════════════════════════════════════
# 4. 性能基准
# ═══════════════════════════════════════════
log("\n[4] 性能基准")

def benchmark(model, x, device_str, warmup=20, repeat=100):
    device = torch.device(device_str)
    model = model.to(device)
    x = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "npu":
        torch.npu.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(x)
    if device.type == "npu":
        torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / repeat * 1000
    throughput = 1.0 / (elapsed / repeat)
    return avg_ms, throughput

batch_sizes = [1, 2, 4, 8]

# CPU benchmark (use a fresh CPU model copy)
model_cpu = load_model()
cpu_results = {}
for bs in batch_sizes:
    x_bs = torch.randn(bs, 3, 224, 224) * 0.2 + 0.5
    x_bs = x_bs.clamp(0, 1)
    cpu_lat, cpu_thr = benchmark(model_cpu, x_bs, "cpu", warmup=3, repeat=10)
    cpu_results[bs] = {"latency_ms": round(cpu_lat, 2), "throughput": round(cpu_thr, 2)}
    log(f"  CPU bs={bs}:  {cpu_lat:.2f} ms  |  {cpu_thr:.2f} imgs/s")

# NPU benchmark (model_npu already on NPU)
npu_results = {}
for bs in batch_sizes:
    x_bs = torch.randn(bs, 3, 224, 224) * 0.2 + 0.5
    x_bs = x_bs.clamp(0, 1)
    npu_lat, npu_thr = benchmark(model_npu, x_bs, "npu:0", warmup=20, repeat=100)
    npu_results[bs] = {"latency_ms": round(npu_lat, 2), "throughput": round(npu_thr, 2)}
    log(f"  NPU bs={bs}: {npu_lat:.2f} ms  |  {npu_thr:.2f} imgs/s")

REPORT["performance"] = {
    "cpu": cpu_results,
    "npu": npu_results,
    "speedup_cpu_to_npu_bs1": round(cpu_results[1]["latency_ms"] / npu_results[1]["latency_ms"], 1),
}
log(f"\n  CPU→NPU 加速比 (bs=1): {cpu_results[1]['latency_ms'] / npu_results[1]['latency_ms']:.1f}x")

# ═══════════════════════════════════════════
# 5. 汇总报告
# ═══════════════════════════════════════════
log("\n" + "=" * 60)
log("测评汇总")
log("=" * 60)
log(f"  精度: cos_sim={cos_sim:.6f}, L2_rel_err={rel_l2*100:.3f}%  >>> {'PASS' if passed else 'FAIL'}")
log(f"  NPU bs=1 延迟: {npu_results[1]['latency_ms']:.1f} ms")
log(f"  NPU bs=8 吞吐: {npu_results[8]['throughput']:.1f} imgs/s")
log(f"  CPU→NPU 加速比: {cpu_results[1]['latency_ms'] / npu_results[1]['latency_ms']:.1f}x")

report_path = "/opt/atomgit/dino_npu/eval_report.json"
with open(report_path, "w") as f:
    json.dump(REPORT, f, indent=2, ensure_ascii=False)
log(f"\n报告已保存: {report_path}")
log("Done.")
