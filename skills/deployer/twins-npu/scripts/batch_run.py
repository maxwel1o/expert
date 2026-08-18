#!/usr/bin/env python3
"""Batch run all 6 Twins models sequentially - CPU/NPU comparison"""
import subprocess, sys, os, gc, shutil, time

MODELS = [
    "twins_svt_small.in1k",
    "twins_svt_large.in1k",
    "twins_svt_base.in1k",
    "twins_pcpvt_small.in1k",
    "twins_pcpvt_large.in1k",
    "twins_pcpvt_base.in1k",
]

BASE_DIR = "/opt/atomgit/batch8"
ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"

os.chdir(BASE_DIR)

for i, model in enumerate(MODELS):
    tag = model.replace(".", "_")
    model_dir = os.path.join(BASE_DIR, model)
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(model_dir, f"{tag}_compare.log")
    screenshot_file = os.path.join(model_dir, f"{tag}_screenshot.txt")

    print(f"\n{'='*70}")
    print(f"  [{i+1}/{len(MODELS)}] {model}")
    print(f"{'='*70}")

    # Run comparison
    print(f"\nRunning CPU/NPU comparison...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "compare_cpu_npu.py", "--model-name", model],
        capture_output=True, text=True, timeout=900, env=ENV
    )

    with open(log_file, "w") as f:
        f.write(result.stdout)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    # Parse results
    passed = "结论: 通过" in result.stdout
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    # Log key metrics
    for line in result.stdout.split("\n"):
        for key in ["Max Prob Difference", "Cosine Similarity", "Top-1 Match",
                     "Top-5 Overlap", "CPU Time", "NPU Time", "Speedup"]:
            if key in line:
                print(f"  {line.strip()}")

    # Generate screenshot
    subprocess.run([
        sys.executable, "gen_screenshot.py",
        log_file, screenshot_file, f"Model: {model} CPU vs NPU"
    ])

    # Copy scripts
    for script in ["inference.py", "compare_cpu_npu.py", "ms_loader.py", "requirements.txt"]:
        src = os.path.join(BASE_DIR, script)
        dst = os.path.join(model_dir, script)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Copy comparison data
    compare_dir = f"{tag}_compare"
    if os.path.exists(compare_dir):
        shutil.copytree(compare_dir, os.path.join(model_dir, "compare_data"),
                       dirs_exist_ok=True)

    # Cleanup
    gc.collect()
    try:
        import torch
        torch.npu.empty_cache()
    except Exception:
        pass

print(f"\n{'='*70}")
print(f"  All models completed!")
print(f"{'='*70}")
