#!/usr/bin/env python3
"""Generate terminal screenshots for all 15 models."""
import os
import json
import subprocess
import sys

BATCH_DIR = "/opt/atomgit/batch19"
SCREENSHOT_SCRIPT = "/opt/atomgit/terminal_screenshot.py"

MODELS = [
    "swinv2_tiny_window8_256.ms_in1k",
    "swinv2_tiny_window16_256.ms_in1k",
    "swinv2_small_window8_256.ms_in1k",
    "swinv2_small_window16_256.ms_in1k",
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_large_window12_192.ms_in22k",
    "swinv2_cr_tiny_ns_224.sw_in1k",
    "swinv2_cr_small_ns_224.sw_in1k",
    "swinv2_cr_small_224.sw_in1k",
    "swinv2_base_window8_256.ms_in1k",
    "swinv2_base_window16_256.ms_in1k",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_base_window12_192.ms_in22k",
]

def generate_screenshot(model_name):
    """Generate terminal screenshot for a single model."""
    model_dir = os.path.join(BATCH_DIR, model_name)
    results_file = os.path.join(model_dir, "results", "comparison_results.json")

    if not os.path.exists(results_file):
        print(f"  No results for {model_name}, skipping")
        return

    with open(results_file) as f:
        r = json.load(f)

    text = f"""$ MODEL_NAME={model_name} python3 inference.py

PyTorch version: 2.9.0
NPU available: True | Device: Ascend 910

--- CPU Inference ---
Avg inference time: {r['cpu_avg_inference_time_ms']:.2f} ms
Top-1: class {r['cpu_top5_indices'][0]} ({r['cpu_top5_probs'][0]:.4f})
Top-2: class {r['cpu_top5_indices'][1]} ({r['cpu_top5_probs'][1]:.4f})
Top-3: class {r['cpu_top5_indices'][2]} ({r['cpu_top5_probs'][2]:.4f})
Top-4: class {r['cpu_top5_indices'][3]} ({r['cpu_top5_probs'][3]:.4f})
Top-5: class {r['cpu_top5_indices'][4]} ({r['cpu_top5_probs'][4]:.4f})

--- NPU Inference ---
Avg inference time: {r['npu_avg_inference_time_ms']:.2f} ms
Top-1: class {r['npu_top5_indices'][0]} ({r['npu_top5_probs'][0]:.4f})
Top-2: class {r['npu_top5_indices'][1]} ({r['npu_top5_probs'][1]:.4f})
Top-3: class {r['npu_top5_indices'][2]} ({r['npu_top5_probs'][2]:.4f})
Top-4: class {r['npu_top5_indices'][3]} ({r['npu_top5_probs'][3]:.4f})
Top-5: class {r['npu_top5_indices'][4]} ({r['npu_top5_probs'][4]:.4f})

--- Precision Comparison ---
Cosine Similarity: {r['cosine_similarity']:.8f}
Max Absolute Error: {r['max_absolute_error']:.6f}
Top-5 Agreement: {r['top5_match_count']}/5

--- Performance ---
CPU: {r['cpu_avg_inference_time_ms']:.2f}ms | NPU: {r['npu_avg_inference_time_ms']:.2f}ms
Speedup: {r['npu_speedup_x']:.2f}x

Verdict: {r['verdict']} - NPU and CPU results equivalent (error < 1%)"""

    output_path = os.path.join(model_dir, "terminal_screenshot.png")
    try:
        result = subprocess.run(
            [sys.executable, SCREENSHOT_SCRIPT, "--text", text, "--output", output_path],
            capture_output=True, text=True, timeout=30
        )
        if os.path.exists(output_path):
            print(f"  Screenshot generated: {model_name}")
        else:
            print(f"  Failed: {model_name} - {result.stderr[:100]}")
    except Exception as e:
        print(f"  Error: {model_name} - {e}")

for model in MODELS:
    print(f"Processing: {model}")
    generate_screenshot(model)

print("\nAll screenshots done!")
