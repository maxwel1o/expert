#!/usr/bin/env python3
"""TinyNet 模型昇腾 NPU 推理脚本（本地权重加载）"""

import argparse
import time
import torch
import torch_npu
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import numpy as np
import json
import os
import gc
from modelscope import snapshot_download


def get_model_local_path(model_name):
    """从 ModelScope 获取模型本地路径"""
    ms_model = f"timm/{model_name}.in1k"
    try:
        model_path = snapshot_download(ms_model)
        print(f"[INFO] 模型已从 ModelScope 下载: {model_path}")
        return model_path
    except Exception as e:
        print(f"[WARN] ModelScope 下载失败: {e}")
        local_path = os.path.join(
            "/opt/atomgit/.cache/modelscope/hub/models",
            f"timm/{model_name.replace('.in1k','').replace('.','_')}___in1k",
        )
        if os.path.exists(local_path):
            return local_path
        raise


def get_test_image():
    """获取测试图片"""
    local_path = "test_image.jpg"
    if os.path.exists(local_path):
        print(f"[INFO] 使用本地测试图片: {local_path}")
        return Image.open(local_path).convert("RGB")

    import requests

    urls = [
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01440764_tench.JPEG",
    ]
    for url in urls:
        try:
            print(f"[INFO] 下载测试图片: {url}")
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                return Image.open(local_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] 下载失败: {e}")
            continue

    print("[WARN] 使用随机图片")
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


@torch.no_grad()
def run_inference(model_name, device="cpu"):
    """在指定设备上运行推理"""
    print(f"\n{'='*60}")
    print(f"[INFO] 模型: {model_name}")
    print(f"[INFO] 设备: {device}")
    print(f"{'='*60}")

    # 创建模型（无预训练权重）
    print(f"[INFO] 创建模型架构...")
    model = create_model(model_name, pretrained=False)

    # 从本地加载权重
    model_path = get_model_local_path(model_name)
    weight_file = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weight_file):
        weight_file = os.path.join(model_path, "model.safetensors")

    print(f"[INFO] 加载权重: {weight_file}")
    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file)
    else:
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] 权重加载完成")

    # 移到设备
    model = model.to(device)
    print(f"[INFO] 模型已移至 {device}")

    # 数据预处理
    config = resolve_data_config({}, model=model)
    print(f"[INFO] 预处理配置: {config}")
    transform = create_transform(**config)

    img = get_test_image()
    print(f"[INFO] 测试图片尺寸: {img.size}")

    input_tensor = transform(img).unsqueeze(0).to(device)
    print(f"[INFO] 输入张量形状: {input_tensor.shape}")

    # 预热
    print("[INFO] 模型预热...")
    for _ in range(3):
        _ = model(input_tensor)

    # 多次推理取平均
    n_runs = 10
    print(f"[INFO] 推理 {n_runs} 次取平均...")
    start_time = time.time()
    for _ in range(n_runs):
        output = model(input_tensor)
    if device != "cpu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    avg_time = (time.time() - start_time) / n_runs * 1000

    # Top-5 结果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    print(f"\n[RESULT] 推理平均耗时: {avg_time:.2f} ms")
    print(f"[RESULT] Top-5 预测结果:")
    for i in range(5):
        print(f"         {i+1}. class {top5_idx[i].item():4d} - prob: {top5_prob[i].item():.6f}")

    # 保存结果
    result = {
        "model_name": model_name,
        "device": device,
        "avg_time_ms": round(avg_time, 2),
        "logits": output[0].cpu().numpy().tolist(),
        "top5_indices": top5_idx.cpu().numpy().tolist(),
        "top5_probabilities": top5_prob.cpu().numpy().tolist(),
    }

    summary_file = f"{model_name}_{device.replace(':','_')}_result.json"
    with open(summary_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已保存: {summary_file}")

    # 清理
    del model, state_dict, input_tensor, output
    gc.collect()
    if device != "cpu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="TinyNet NPU 推理")
    parser.add_argument("--model", type=str, required=True, help="模型名")
    parser.add_argument("--device", type=str, default="npu", choices=["cpu", "npu"])
    args = parser.parse_args()
    device = "npu:0" if args.device == "npu" else "cpu"
    run_inference(args.model, device)


if __name__ == "__main__":
    main()
