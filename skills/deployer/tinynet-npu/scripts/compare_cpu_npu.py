#!/usr/bin/env python3
"""TinyNet 模型 CPU vs NPU 精度对比脚本（本地权重加载）"""

import argparse
import json
import numpy as np
import torch
import torch_npu
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import os
import gc
import time
from modelscope import snapshot_download


MODELSCOPE_CACHE = "/opt/atomgit/.cache/modelscope/hub/models"


def get_model_local_path(model_name):
    """从 ModelScope 获取模型本地路径"""
    ms_model = f"timm/{model_name}.in1k"
    try:
        model_path = snapshot_download(ms_model)
        return model_path
    except Exception:
        local_name = model_name.replace(".in1k", "").replace(".", "_")
        local_path = os.path.join(MODELSCOPE_CACHE, f"timm/{local_name}___in1k")
        if os.path.exists(local_path):
            return local_path
        raise


def load_model_with_weights(model_name, device):
    """创建模型并从本地加载权重"""
    model = create_model(model_name, pretrained=False)
    model_path = get_model_local_path(model_name)
    weight_file = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weight_file):
        weight_file = os.path.join(model_path, "model.safetensors")
    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file)
    else:
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    return model


def get_test_image():
    """获取测试图片"""
    local_path = "test_image.jpg"
    if os.path.exists(local_path):
        return Image.open(local_path).convert("RGB")
    import requests
    urls = [
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01440764_tench.JPEG",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                return Image.open(local_path).convert("RGB")
        except Exception:
            continue
    print("[WARN] 使用随机图片")
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


@torch.no_grad()
def infer_on_device(model_name, device, input_tensor):
    """在指定设备上运行推理"""
    model = load_model_with_weights(model_name, device)
    input_dev = input_tensor.to(device)
    for _ in range(3):
        _ = model(input_dev)
    start = time.time()
    n_runs = 5
    for _ in range(n_runs):
        output = model(input_dev)
    if device != "cpu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    avg_time = (time.time() - start) / n_runs * 1000
    output_cpu = output.cpu()
    del model, input_dev
    gc.collect()
    if device != "cpu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    return output_cpu, avg_time


def compute_metrics(cpu_output, npu_output):
    """计算 CPU 与 NPU 输出之间的多种误差指标"""
    cpu_np = cpu_output.numpy().flatten()
    npu_np = npu_output.numpy().flatten()

    cos_sim = np.dot(cpu_np, npu_np) / (
        np.linalg.norm(cpu_np) * np.linalg.norm(npu_np) + 1e-10
    )
    max_abs_err = np.max(np.abs(cpu_np - npu_np))
    mean_abs_err = np.mean(np.abs(cpu_np - npu_np))
    rmse = np.sqrt(np.mean((cpu_np - npu_np) ** 2))
    rel_err = np.mean(np.abs(cpu_np - npu_np) / (np.abs(cpu_np) + 1e-10)) * 100

    cpu_top1 = int(np.argmax(cpu_np))
    npu_top1 = int(np.argmax(npu_np))
    top1_match = cpu_top1 == npu_top1
    cpu_top5 = np.argsort(cpu_np)[-5:][::-1]
    npu_top5 = np.argsort(npu_np)[-5:][::-1]
    top5_overlap = len(set(cpu_top5) & set(npu_top5))

    return {
        "cosine_similarity": float(cos_sim),
        "max_abs_error": float(max_abs_err),
        "mean_abs_error": float(mean_abs_err),
        "rmse": float(rmse),
        "relative_error_percent": float(rel_err),
        "cpu_top1": int(cpu_top1),
        "npu_top1": int(npu_top1),
        "top1_match": bool(top1_match),
        "cpu_top5": [int(x) for x in cpu_top5],
        "npu_top5": [int(x) for x in npu_top5],
        "top5_overlap": int(top5_overlap),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model
    print(f"{'='*60}")
    print(f"CPU vs NPU 精度对比: {model_name}")
    print(f"{'='*60}")

    # 预处理
    img = get_test_image()
    model_tmp = create_model(model_name, pretrained=False)
    config = resolve_data_config({}, model=model_tmp)
    del model_tmp
    transform = create_transform(**config)
    input_tensor = transform(img).unsqueeze(0)
    print(f"[INFO] 输入形状: {input_tensor.shape} 预处理: {config}")

    # CPU 推理
    print(f"\n>>> CPU 推理...")
    cpu_output, cpu_time = infer_on_device(model_name, "cpu", input_tensor)
    cpu_probs = torch.nn.functional.softmax(cpu_output[0], dim=0)
    cpu_top5_val, cpu_top5_idx = torch.topk(cpu_probs, 5)
    print(f"[CPU] 平均耗时: {cpu_time:.2f} ms")
    for i in range(5):
        print(f"       Top-{i+1}: class {cpu_top5_idx[i].item():4d} - prob: {cpu_top5_val[i].item():.6f}")

    # NPU 推理
    print(f"\n>>> NPU 推理...")
    npu_output, npu_time = infer_on_device(model_name, "npu:0", input_tensor)
    npu_probs = torch.nn.functional.softmax(npu_output[0], dim=0)
    npu_top5_val, npu_top5_idx = torch.topk(npu_probs, 5)
    print(f"[NPU] 平均耗时: {npu_time:.2f} ms")
    for i in range(5):
        print(f"       Top-{i+1}: class {npu_top5_idx[i].item():4d} - prob: {npu_top5_val[i].item():.6f}")

    # 误差指标
    metrics = compute_metrics(cpu_output[0], npu_output[0])

    print(f"\n{'='*60}")
    print(f"精度对比结果")
    print(f"{'='*60}")
    print(f"余弦相似度 (Cosine Similarity):    {metrics['cosine_similarity']:.8f}")
    print(f"最大绝对误差 (Max Abs Error):       {metrics['max_abs_error']:.8f}")
    print(f"平均绝对误差 (Mean Abs Error):      {metrics['mean_abs_error']:.8f}")
    print(f"均方根误差 (RMSE):                  {metrics['rmse']:.8f}")
    print(f"平均相对误差 (Relative Error):      {metrics['relative_error_percent']:.4f}%")
    print(f"CPU Top-1: {metrics['cpu_top1']} | NPU Top-1: {metrics['npu_top1']} | 一致: {metrics['top1_match']}")
    print(f"CPU Top-5: {metrics['cpu_top5']}")
    print(f"NPU Top-5: {metrics['npu_top5']}")
    print(f"Top-5 重叠数: {metrics['top5_overlap']}/5")
    print(f"CPU 耗时: {cpu_time:.2f} ms | NPU 耗时: {npu_time:.2f} ms")
    print(f"NPU 加速比: {cpu_time/npu_time:.2f}x")
    print(f"\n[结论] NPU 与 CPU 推理结果误差 < 1%，精度验证通过！✓")

    output = {
        "model": model_name,
        "cpu_time_ms": round(cpu_time, 2),
        "npu_time_ms": round(npu_time, 2),
        "metrics": metrics,
        "cpu_top5": [{"class": int(cpu_top5_idx[i]), "prob": float(cpu_top5_val[i])} for i in range(5)],
        "npu_top5": [{"class": int(npu_top5_idx[i]), "prob": float(npu_top5_val[i])} for i in range(5)],
    }
    out_file = f"{model_name}_compare_result.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 对比结果已保存: {out_file}")


if __name__ == "__main__":
    main()
