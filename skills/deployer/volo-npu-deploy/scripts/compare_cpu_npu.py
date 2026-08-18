#!/usr/bin/env python3
"""CPU vs NPU precision comparison for VOLO models - loading from local cache."""
import argparse
import json
import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from timm import create_model
from timm.data import resolve_data_config, create_transform

CACHE_DIR = "/opt/atomgit/.cache/modelscope/timm"
INDEX_FILE = "/opt/atomgit/.cache/modelscope/volo_models_index.json"

def get_weight_path(model_name):
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE) as f:
            idx = json.load(f)
        wp = idx.get(model_name)
        if wp and os.path.exists(wp):
            return wp
    local_dir = os.path.join(CACHE_DIR, model_name)
    for fn in ["model.safetensors", "pytorch_model.bin"]:
        fp = os.path.join(local_dir, fn)
        if os.path.exists(fp):
            return fp
    return None

def load_model_local(model_name, device):
    print(f"  Loading model on {device}...", flush=True)
    model = create_model(model_name, pretrained=False)
    weight_path = get_weight_path(model_name)
    if weight_path is None:
        raise FileNotFoundError(f"No weight file found for {model_name}")
    if weight_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, model):
    img = Image.open(image_path).convert('RGB')
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor

def run_inference(model, input_tensor, device, num_runs=10):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _ = model(input_tensor)
        start = time.time()
        for _ in range(num_runs):
            output = model(input_tensor)
        total_time = time.time() - start
    return output.cpu(), total_time / num_runs

def compute_metrics(cpu_logits, npu_logits):
    cpu_probs = F.softmax(cpu_logits, dim=1).numpy()
    npu_probs = F.softmax(npu_logits, dim=1).numpy()
    cpu_np = cpu_logits.numpy()
    npu_np = npu_logits.numpy()

    logits_diff = np.abs(cpu_np - npu_np)
    max_logits_error = float(logits_diff.max())
    mean_logits_error = float(logits_diff.mean())

    from numpy.linalg import norm
    cos_sim_logits = float(np.dot(cpu_np.flatten(), npu_np.flatten()) /
                          (norm(cpu_np.flatten()) * norm(npu_np.flatten())))

    prob_diff = np.abs(cpu_probs - npu_probs)
    max_prob_error = float(prob_diff.max())
    mean_prob_error = float(prob_diff.mean())

    cpu_top1 = int(np.argmax(cpu_probs, axis=1)[0])
    npu_top1 = int(np.argmax(npu_probs, axis=1)[0])
    top1_match = cpu_top1 == npu_top1

    cpu_top5 = set(np.argsort(cpu_probs[0])[-5:])
    npu_top5 = set(np.argsort(npu_probs[0])[-5:])
    top5_overlap = len(cpu_top5 & npu_top5)

    return {
        "max_logits_error": max_logits_error,
        "mean_logits_error": mean_logits_error,
        "cosine_similarity_logits": cos_sim_logits,
        "max_prob_error": max_prob_error,
        "mean_prob_error": mean_prob_error,
        "top1_match": top1_match,
        "cpu_top1": cpu_top1,
        "npu_top1": npu_top1,
        "top5_overlap": top5_overlap,
    }

def main():
    parser = argparse.ArgumentParser(description='VOLO CPU vs NPU Comparison')
    parser.add_argument('--model', type=str, required=True, help='timm model name')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--num_runs', type=int, default=10, help='Inference runs for timing')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"CPU vs NPU Precision Comparison")
    print(f"Model: {args.model}")
    print(f"{'='*60}")

    if args.image is None:
        from urllib.request import urlretrieve
        test_img_path = "/tmp/test_volo_compare.jpg"
        try:
            urlretrieve(
                "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01882714_koala.JPEG",
                test_img_path
            )
        except:
            img = Image.new('RGB', (224, 224), color=(100, 100, 200))
            img.save(test_img_path)
        args.image = test_img_path
    print(f"Test image: {args.image}")

    print("\n[1/4] Loading model on CPU...")
    cpu_model = load_model_local(args.model, 'cpu')
    input_tensor = preprocess_image(args.image, cpu_model)

    print("[2/4] Running CPU inference...")
    cpu_logits, cpu_avg_time = run_inference(cpu_model, input_tensor, 'cpu', args.num_runs)
    print(f"  CPU avg inference time: {cpu_avg_time*1000:.2f} ms")

    del cpu_model
    import gc
    gc.collect()

    print("\n[3/4] Loading model on NPU...")
    npu_model = load_model_local(args.model, 'npu:0')
    input_tensor_npu = preprocess_image(args.image, npu_model)

    print("[4/4] Running NPU inference...")
    npu_logits, npu_avg_time = run_inference(npu_model, input_tensor_npu, 'npu:0', args.num_runs)
    print(f"  NPU avg inference time: {npu_avg_time*1000:.2f} ms")

    print("\nComputing precision metrics...")
    metrics = compute_metrics(cpu_logits, npu_logits)

    print(f"\n{'='*60}")
    print(f"PRECISION COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"\n--- Logits ---")
    print(f"  Max absolute error:      {metrics['max_logits_error']:.8f}")
    print(f"  Mean absolute error:     {metrics['mean_logits_error']:.8f}")
    print(f"  Cosine similarity:       {metrics['cosine_similarity_logits']:.10f}")

    print(f"\n--- Probabilities ---")
    print(f"  Max absolute error:      {metrics['max_prob_error']:.8f}")
    print(f"  Mean absolute error:     {metrics['mean_prob_error']:.8f}")

    print(f"\n--- Top-1 Prediction ---")
    print(f"  CPU top-1 class:         {metrics['cpu_top1']}")
    print(f"  NPU top-1 class:         {metrics['npu_top1']}")
    print(f"  Match:                   {'YES' if metrics['top1_match'] else 'NO'}")

    print(f"\n--- Top-5 ---")
    print(f"  Overlap count:           {metrics['top5_overlap']}/5")

    speedup = cpu_avg_time / npu_avg_time if npu_avg_time > 0 else 0
    print(f"\n--- Performance ---")
    print(f"  CPU avg inference time:  {cpu_avg_time*1000:.2f} ms")
    print(f"  NPU avg inference time:  {npu_avg_time*1000:.2f} ms")
    print(f"  NPU speedup vs CPU:      {speedup:.2f}x")

    passed = metrics['max_prob_error'] < 0.01
    print(f"\n{'='*60}")
    print(f"CONCLUSION: {'PASSED' if passed else 'FAILED'}")
    if passed:
        print(f"NPU and CPU inference error < 1% (max prob error = {metrics['max_prob_error']:.6f})")
    else:
        print(f"NPU and CPU inference error >= 1% (max prob error = {metrics['max_prob_error']:.6f})")
    print(f"{'='*60}")

    results = {
        "model": args.model,
        "test_image": args.image,
        "cpu_avg_time_ms": round(cpu_avg_time * 1000, 2),
        "npu_avg_time_ms": round(npu_avg_time * 1000, 2),
        "npu_speedup": round(speedup, 2),
        "metrics": metrics,
        "conclusion": "PASSED" if passed else "FAILED",
    }

    with open("compare_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to compare_results.json")

    del npu_model
    gc.collect()
    try:
        torch.npu.empty_cache()
    except:
        pass

    return results

if __name__ == "__main__":
    main()
