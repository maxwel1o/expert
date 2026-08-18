#!/usr/bin/env python3
"""Generate a terminal-style screenshot for a model's inference results.

Usage:
  python3 generate_screenshot.py --model levit-128 --output screenshot.png
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCREENSHOT_SCRIPT = "/opt/atomgit/terminal_screenshot.py"


def main():
    parser = argparse.ArgumentParser(description="Generate terminal screenshot")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--output", default="screenshot.png", help="Output PNG path")
    args = parser.parse_args()

    model_name = args.model

    # Load results
    try:
        with open(f"/tmp/{model_name}_npu_results.json") as f:
            npu = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Run NPU inference first for {model_name}")
        sys.exit(1)

    cpu = None
    try:
        with open(f"/tmp/{model_name}_cpu_results.json") as f:
            cpu = json.load(f)
    except FileNotFoundError:
        pass

    # Build terminal text
    lines = [f"$ python3 inference.py --device npu --image test.jpg", ""]
    lines.append(f"Loading model from facebook/{model_name}...")
    lines.append(f"Loading image: test.jpg (224x224)")
    lines.append("")
    lines.append("=" * 55)
    lines.append(f"  {model_name} Inference Results (NPU)")
    lines.append("=" * 55)
    lines.append(f"  Average inference time: {npu['time_ms']:.2f} ms")
    probs = npu['probabilities'][0]
    top5 = sorted(range(len(probs)), key=lambda i: -probs[i])[:5]
    lines.append("  Top-5 predictions:")
    for i, idx in enumerate(top5, 1):
        lines.append(f"    {i}. class {idx}: {probs[idx]*100:.2f}%")

    if cpu:
        lines.append("")
        lines.append("$ python3 compare_cpu_npu.py")
        lines.append("")
        lines.append("=" * 55)
        lines.append("  CPU vs NPU Accuracy Comparison")
        lines.append("=" * 55)
        cpu_top1 = max(range(len(cpu["logits"][0])), key=lambda i: cpu["logits"][0][i])
        npu_top1 = max(range(len(npu["logits"][0])), key=lambda i: npu["logits"][0][i])

        try:
            with open(f"/tmp/{model_name}_comparison.json") as f:
                comp = json.load(f)
            lines.append(f"  Logits MAE:       {comp['logits_mae']:.8f}")
            lines.append(f"  Probs MaxAE:      {comp['probs_max_ae']:.6f} ({comp['probs_error_pct']:.4f}%)")
            lines.append(f"  Cosine Similarity: {comp['cosine_similarity']:.8f}")
            lines.append(f"  Top-1 Match:      {'YES' if comp['top1_match'] else 'NO'}")
            lines.append(f"  Top-5 Match:      {'YES' if comp['top5_match'] else 'NO'}")
            lines.append("")
            if comp['passed']:
                lines.append(f"  RESULT: PASS (error < 1%)")
            else:
                lines.append(f"  RESULT: FAIL (error >= 1%)")
        except (FileNotFoundError, KeyError):
            pass

    # Write temp file
    text_path = f"/tmp/{model_name}_screenshot_text.txt"
    Path(text_path).write_text("\n".join(lines))

    # Generate screenshot
    ret = subprocess.run(
        [sys.executable, SCREENSHOT_SCRIPT, "--input", text_path, "--output", args.output],
        capture_output=True, text=True
    )
    if ret.returncode == 0:
        print(f"Screenshot saved: {args.output}")
    else:
        print(f"ERROR: {ret.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
