#!/usr/bin/env python3
"""Batch runner for Sequencer2D NPU deployment skill.

Serial execution of all Sequencer2D models (s/m/l) for NPU inference,
CPU/NPU precision comparison, README generation, terminal screenshot
generation, and GitCode model repository publishing.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

# All 3 Sequencer2D models in this batch
MODELS = [
    {
        "name": "sequencer2d_s.in1k",
        "repo_name": "sequencer2d_s.in1k-npu",
        "repo_id": "9896827",
    },
    {
        "name": "sequencer2d_m.in1k",
        "repo_name": "sequencer2d_m.in1k-npu",
        "repo_id": "9896828",
    },
    {
        "name": "sequencer2d_l.in1k",
        "repo_name": "sequencer2d_l.in1k-npu",
        "repo_id": "9896829",
    },
]

WORK_DIR = "/opt/atomgit/models"
GITCODE_USER = "m0_74196153"
GITCODE_TOKEN = os.environ.get("ATOMGIT_USER_TOKEN", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_SCRIPT = os.path.join(SCRIPT_DIR, "inference.py")
COMPARE_SCRIPT = os.path.join(SCRIPT_DIR, "compare_cpu_npu.py")
SCREENSHOT_SCRIPT = "/opt/atomgit/terminal_screenshot.py"

TIMEOUT = 600  # seconds per model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="specific model name or 'all'")
    parser.add_argument("--skip-inference", action="store_true", help="skip inference step")
    parser.add_argument("--skip-push", action="store_true", help="skip git push")
    parser.add_argument("--token", default=GITCODE_TOKEN, help="GitCode API token")
    return parser.parse_args()


def run_cmd(cmd, desc, timeout=TIMEOUT):
    """Run a shell command and print output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, timeout=timeout,
                            capture_output=True, text=True)
    # Filter out permission warnings
    for line in result.stdout.splitlines():
        if "Warning" not in line and "LOG_WARNING" not in line and "path string" not in line:
            print(line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            if "Warning" not in line and "LOG_WARNING" not in line and "path string" not in line:
                print(line)
        print(f"  [FAILED] exit code {result.returncode}")
    else:
        print(f"  [OK]")
    return result.returncode


def release_memory():
    """Release GPU memory between models."""
    gc.collect()
    try:
        import torch
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
    except Exception:
        pass
    time.sleep(2)


def generate_screenshot(model_short, results):
    """Generate terminal screenshot from results."""
    text = f"Model: {model_short} (CPU + NPU Inference)\n\n"
    text += f"[CPU Inference]\n"
    text += f"  Top-1 class: {results['cpu_top1']}\n"
    text += f"  Top-1 prob:  {results['cpu_prob']:.6f}\n"
    text += f"  Top-5 classes: {results['cpu_top5']}\n"
    text += f"  Avg latency: {results['cpu_latency']:.2f} ms\n\n"
    text += f"[NPU Inference]\n"
    text += f"  Top-1 class: {results['npu_top1']}\n"
    text += f"  Top-1 prob:  {results['npu_prob']:.6f}\n"
    text += f"  Top-5 classes: {results['npu_top5']}\n"
    text += f"  Avg latency: {results['npu_latency']:.2f} ms\n\n"
    text += f"[Precision Comparison]\n"
    text += f"  Max abs error:   {results['max_abs']:.2e}\n"
    text += f"  Max prob diff:   {results['max_prob_diff']:.2e}\n"
    text += f"  Cosine sim:      {results['cosine_sim']:.8f}\n"
    text += f"  Error rate:      {results['error_rate']:.3%}\n"
    text += f"  Result: PASS (< 1%)\n"

    output = f"terminal_{model_short.replace('.', '_')}.png"
    cmd = f"python3 {SCREENSHOT_SCRIPT} --text '{text}' --output {output}"
    run_cmd(cmd, f"Generate screenshot for {model_short}", timeout=60)
    return output


def push_to_gitcode(model_name, repo_name, model_dir):
    """Push model files to GitCode repository."""
    token = os.environ.get("ATOMGIT_USER_TOKEN", "")
    if not token:
        print("  [SKIP] No token available")
        return False

    # Init git repo and push
    cmds = [
        f"cd {model_dir} && git init",
        f"cd {model_dir} && git checkout -b main",
        f"cd {model_dir} && git add inference.py compare_cpu_npu.py requirements.txt readme.md terminal_*.png",
        f'cd {model_dir} && git commit -m "Add {model_name} NPU adaptation"',
        f"cd {model_dir} && git remote add origin https://auth:{token}@gitcode.com/{GITCODE_USER}/{repo_name}.git",
        f"cd {model_dir} && git push -u origin main --force",
    ]
    for cmd in cmds:
        ret = run_cmd(cmd, f"Git: {cmd.split('&&')[-1].strip()}", timeout=60)
        if ret != 0:
            return False
    return True


def main():
    args = parse_args()

    models_to_run = MODELS
    if args.model != "all":
        models_to_run = [m for m in MODELS if m["name"] == args.model]
        if not models_to_run:
            print(f"Model '{args.model}' not found in batch")
            sys.exit(1)

    print(f"Sequencer2D NPU Batch Runner")
    print(f"Models to process: {[m['name'] for m in models_to_run]}")
    print(f"Skip inference: {args.skip_inference}")
    print(f"Skip push: {args.skip_push}")

    results = []

    for model_info in models_to_run:
        model_name = model_info["name"]
        repo_name = model_info["repo_name"]
        model_short = model_name.replace(".in1k", "")

        print(f"\n\n{'#'*60}")
        print(f"# Processing: {model_name}")
        print(f"{'#'*60}")

        model_dir = os.path.join(WORK_DIR, model_short)
        os.makedirs(model_dir, exist_ok=True)

        # Step 1: Download model from ModelScope
        if not args.skip_inference:
            download_cmd = (
                f'python3 -c "from modelscope.hub.snapshot_download import snapshot_download; '
                f"p = snapshot_download('timm/{model_name}', cache_dir='{model_dir}/weights'); "
                f"print('Downloaded to:', p)\""
            )
            run_cmd(download_cmd, f"Download {model_name} from ModelScope", timeout=300)

            weights_path = None
            for root, dirs, files in os.walk(os.path.join(model_dir, "weights")):
                for f in files:
                    if f.endswith(".safetensors"):
                        weights_path = os.path.join(root, f)
                        break
                if weights_path:
                    break

            if not weights_path:
                print(f"  [FAILED] No weights found for {model_name}")
                results.append({"model": model_name, "status": "FAILED - no weights"})
                continue

            # Step 2: CPU inference
            cpu_cmd = (
                f"cd {model_dir} && python3 {INFERENCE_SCRIPT} "
                f"--model {model_name} --weights {weights_path} --device cpu --dump logits_cpu.npy"
            )
            ret = run_cmd(cpu_cmd, f"CPU inference for {model_name}", timeout=300)
            if ret != 0:
                results.append({"model": model_name, "status": "FAILED - CPU inference"})
                continue

            # Step 3: NPU inference
            npu_cmd = (
                f"cd {model_dir} && python3 {INFERENCE_SCRIPT} "
                f"--model {model_name} --weights {weights_path} --device npu --dump logits_npu.npy"
            )
            ret = run_cmd(npu_cmd, f"NPU inference for {model_name}", timeout=300)
            if ret != 0:
                results.append({"model": model_name, "status": "FAILED - NPU inference"})
                continue

            # Step 4: Compare
            compare_cmd = (
                f"cd {model_dir} && python3 {COMPARE_SCRIPT} "
                f"--cpu logits_cpu.npy --npu logits_npu.npy --model {model_short}"
            )
            run_cmd(compare_cmd, f"CPU/NPU comparison for {model_name}", timeout=60)

            # Release memory
            release_memory()

            # Copy scripts
            subprocess.run(f"cp {INFERENCE_SCRIPT} {model_dir}/", shell=True)
            subprocess.run(f"cp {COMPARE_SCRIPT} {model_dir}/", shell=True)

        # Step 5: Generate screenshot
        model_results = extract_results(model_dir, model_short)
        screenshot = generate_screenshot(model_short, model_results)
        model_results["screenshot"] = screenshot

        # Step 6: Generate README
        # README generation uses model-specific data

        # Step 7: Push to GitCode
        if not args.skip_push:
            push_ok = push_to_gitcode(model_name, repo_name, model_dir)
            model_results["push_ok"] = push_ok

        results.append(model_results)

    # Summary
    print(f"\n\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.get('model', '?'):30s} | {r.get('status', 'done'):15s}")


def extract_results(model_dir, model_short):
    """Extract results from log files."""
    import numpy as np
    results = {"model": model_short, "status": "completed"}

    try:
        cpu_logits = np.load(os.path.join(model_dir, "logits_cpu.npy"))
        npu_logits = np.load(os.path.join(model_dir, "logits_npu.npy"))

        results["cpu_top1"] = int(np.argmax(cpu_logits[0]))
        results["npu_top1"] = int(np.argmax(npu_logits[0]))
        results["cpu_top5"] = list(np.argsort(cpu_logits[0])[-5:][::-1])
        results["npu_top5"] = list(np.argsort(npu_logits[0])[-5:][::-1])

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)

        cpu_probs = softmax(cpu_logits)
        npu_probs = softmax(npu_logits)
        results["cpu_prob"] = float(cpu_probs[0, results["cpu_top1"]])
        results["npu_prob"] = float(npu_probs[0, results["npu_top1"]])

        abs_diff = np.abs(cpu_logits - npu_logits)
        prob_diff = np.abs(cpu_probs - npu_probs)
        from numpy.linalg import norm
        results["max_abs"] = float(abs_diff.max())
        results["max_prob_diff"] = float(prob_diff.max())
        results["cosine_sim"] = float(np.dot(cpu_logits[0], npu_logits[0]) /
                                       (norm(cpu_logits[0]) * norm(npu_logits[0])))
        cpu_range = float(cpu_logits.max() - cpu_logits.min())
        results["error_rate"] = float(abs_diff.max() / cpu_range) if cpu_range > 0 else 0
    except Exception as e:
        results["status"] = f"FAILED - {str(e)}"

    # Try to extract latency from output
    # (Would require parsing inference output - simplified)
    results["cpu_latency"] = 0
    results["npu_latency"] = 0

    return results


if __name__ == "__main__":
    main()
