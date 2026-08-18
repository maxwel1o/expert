#!/usr/bin/env python3
"""timm SE 系列模型 NPU 推理脚本 (Batch 24)"""
import argparse
import json
import time
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

def get_model_info(model_name: str):
    """获取模型信息"""
    try:
        metadata = timm.models.get_pretrained_config(model_name)
        if hasattr(metadata, 'to_dict'):
            metadata = metadata.to_dict()
        return metadata
    except:
        return {"note": "metadata not available"}

def load_model(model_name: str, device: str = "cpu"):
    """加载模型到指定设备"""
    print(f"[INFO] Loading model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded on {device}")
    return model

def preprocess_image(image_path: str, model_name: str, model=None):
    """预处理图片"""
    img = Image.open(image_path).convert("RGB")
    config = resolve_data_config({}, model=model or model_name)
    transform = create_transform(**config)
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, img

@torch.no_grad()
def inference(model, input_tensor, device: str):
    """执行推理"""
    input_tensor = input_tensor.to(device)
    start = time.time()
    output = model(input_tensor)
    elapsed = time.time() - start
    return output, elapsed

def main():
    parser = argparse.ArgumentParser(description="timm SE 模型 NPU 推理")
    parser.add_argument("--model", type=str, required=True, help="timm 模型名称")
    parser.add_argument("--image", type=str, default="/opt/atomgit/batch24/test_input.jpg",
                        help="输入图片路径")
    parser.add_argument("--device", type=str, choices=["cpu", "npu"], default="cpu",
                        help="推理设备")
    parser.add_argument("--topk", type=int, default=5, help="Top-K 结果数")
    args = parser.parse_args()

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Image: {args.image}")

    # 获取模型信息
    info = get_model_info(args.model)
    print(f"[INFO] Model info: {json.dumps(info, indent=2)[:200]}")

    # 加载模型
    model = load_model(args.model, args.device)

    # 预处理
    input_tensor, orig_img = preprocess_image(args.image, args.model, model=model)
    print(f"[INFO] Input shape: {input_tensor.shape}")

    # 推理
    output, elapsed = inference(model, input_tensor, args.device)
    print(f"[INFO] Output shape: {output.shape}")
    print(f"[INFO] Inference time: {elapsed*1000:.2f} ms")

    # Top-K 结果
    probs = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probs, args.topk)
    print(f"\n[RESULTS] Top-{args.topk} predictions:")
    for i in range(args.topk):
        print(f"  class_{top_indices[0][i].item()}: {top_probs[0][i].item():.4f}")

    # 输出 logits 统计
    print(f"\n[STATS] Logits - mean: {output.mean().item():.4f}, "
          f"std: {output.std().item():.4f}, "
          f"max: {output.max().item():.4f}, "
          f"min: {output.min().item():.4f}")

    # 保存输出
    torch.save(output.cpu(), f"output_{args.device}.pt")
    print(f"[INFO] Output saved to output_{args.device}.pt")

if __name__ == "__main__":
    main()
