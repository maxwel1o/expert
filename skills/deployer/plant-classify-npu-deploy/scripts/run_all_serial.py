#!/usr/bin/env python3
"""
串行执行多个植物分类模型的 NPU 推理和 CPU/NPU 精度对比
逐个模型串行执行，防止 NPU 显存爆炸
"""
import os
import sys
import gc
import subprocess
import time

MODELS = [
    {
        "name": "plant-classify-densenet121",
        "model_id": "flowscolors/plant-classify-densenet121",
        "arch": "densenet121",
    },
    {
        "name": "plant-classify-resnet18",
        "model_id": "flowscolors/plant-classify-resnet18",
        "arch": "resnet18",
    },
    {
        "name": "plant-classify-resnet50",
        "model_id": "flowscolors/plant-classify-resnet50",
        "arch": "resnet50",
    },
]

WORK_DIR = "/opt/atomgit/models"
CACHE_DIR = os.path.expanduser("~/.cache/modelscope/hub/models")


def ensure_model_downloaded(model_id, model_name):
    """确保模型已下载"""
    model_dir = os.path.join(CACHE_DIR, model_id)
    if not os.path.exists(os.path.join(model_dir, "pytorch_model.pt")):
        print(f"[INFO] Downloading model {model_id}...")
        from modelscope import snapshot_download
        snapshot_download(model_id)
    else:
        print(f"[INFO] Model {model_name} already cached at {model_dir}")


def run_command(cmd, cwd=None):
    """运行命令并打印输出"""
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    return result.returncode


def release_resources():
    """释放 NPU 显存和 CPU 内存"""
    print("\n>>> Releasing resources...")
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
            print("[INFO] NPU cache cleared")
    except Exception as e:
        print(f"[INFO] torch not available: {e}")
    print("[INFO] Resources released\n")


def main():
    results = []

    for i, model in enumerate(MODELS):
        name = model["name"]
        model_id = model["model_id"]
        arch = model["arch"]

        print("\n" + "=" * 70)
        print(f"  Processing model {i + 1}/{len(MODELS)}: {name}")
        print("=" * 70)

        model_work_dir = os.path.join(WORK_DIR, name)
        os.makedirs(model_work_dir, exist_ok=True)

        try:
            # Step 1: Ensure model is downloaded
            ensure_model_downloaded(model_id, name)

            # Step 2: Generate test image
            gen_script = os.path.join(model_work_dir, "generate_test_image.py")
            if os.path.exists(gen_script):
                run_command([sys.executable, gen_script], cwd=model_work_dir)

            # Step 3: Run CPU inference
            inf_script = os.path.join(model_work_dir, "inference.py")
            if os.path.exists(inf_script):
                run_command([
                    sys.executable, inf_script,
                    "test_inputs/test_plant.jpg", "cpu"
                ], cwd=model_work_dir)

            # Step 4: Release CPU memory
            release_resources()

            # Step 5: Run NPU inference
            if os.path.exists(inf_script):
                run_command([
                    sys.executable, inf_script,
                    "test_inputs/test_plant.jpg", "npu:0"
                ], cwd=model_work_dir)

            # Step 6: Run CPU vs NPU comparison
            compare_script = os.path.join(model_work_dir, "compare_cpu_npu.py")
            if os.path.exists(compare_script):
                ret = run_command([sys.executable, compare_script], cwd=model_work_dir)
                if ret == 0:
                    results.append((name, "PASS"))
                else:
                    results.append((name, f"FAIL (compare script returned {ret})"))
            else:
                results.append((name, "FAIL (compare_cpu_npu.py not found)"))

        except Exception as e:
            print(f"[ERROR] Model {name} failed: {e}")
            results.append((name, f"FAIL: {str(e)}"))
            import traceback
            traceback.print_exc()

        # Step 7: Release all resources before next model
        release_resources()

        # Small delay between models
        time.sleep(2)

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for name, status in results:
        print(f"  {name:40s} {status}")
    print("=" * 70)

    all_pass = all(s == "PASS" for _, s in results)
    if all_pass:
        print("\n[PASS] All models passed!")
    else:
        print("\n[WARN] Some models failed. Check logs above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
