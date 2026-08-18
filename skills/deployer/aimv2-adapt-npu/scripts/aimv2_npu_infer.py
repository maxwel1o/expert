#!/usr/bin/env python3
"""
Apple AIMv2 适配版昇腾 NPU 推理脚本
支持 224-distilled / 336-distilled / 224-lit / native 四种变体
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from convert_weights import convert_state_dict, needs_conversion

NPU_DEVICE = "npu:0"
DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16}

MODEL_INFO = {
    "224-distilled": {
        "name": "apple/aimv2-large-patch14-224-distilled",
        "image_size": 224, "type": "vision",
    },
    "336-distilled": {
        "name": "apple/aimv2-large-patch14-336-distilled",
        "image_size": 336, "type": "vision",
    },
    "224-lit": {
        "name": "apple/aimv2-large-patch14-224-lit",
        "image_size": 224, "type": "lit",
    },
    "native": {
        "name": "apple/aimv2-large-patch14-native",
        "image_size": 224, "type": "vision",
    },
}

CONFIG_CLASSES = {
    "224-distilled": "AIMv2DistilledConfig", "336-distilled": "AIMv2DistilledConfig",
    "224-lit": "AIMv2LitConfig", "native": "AIMv2NativeConfig",
}

MODEL_CLASSES = {
    "224-distilled": "AIMv2DistilledModel", "336-distilled": "AIMv2DistilledModel",
    "224-lit": "AIMv2LitModel", "native": "AIMv2Model",
}

CONFIG_FILES = {
    "224-distilled": "configuration_aimv2_distilled.py",
    "336-distilled": "configuration_aimv2_distilled.py",
    "224-lit": "configuration_aimv2_lit.py",
    "native": "configuration_aimv2_native.py",
}

MODEL_FILES = {
    "224-distilled": "modeling_aimv2_distilled.py",
    "336-distilled": "modeling_aimv2_distilled.py",
    "224-lit": "modeling_aimv2_lit.py",
    "native": "modeling_aimv2_native.py",
}


def _import_model_files(variant):
    config_file = _HERE / CONFIG_FILES[variant]
    model_file = _HERE / MODEL_FILES[variant]

    if not config_file.exists() or not model_file.exists():
        print(f"错误：找不到模型文件 {config_file} 或 {model_file}")
        print("请确保 modeling_* 和 configuration_* 文件在 scripts/ 目录下")
        sys.exit(1)

    import importlib.util
    cfg_spec = importlib.util.spec_from_file_location("aimv2_config", config_file)
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)
    ConfigClass = getattr(cfg_mod, CONFIG_CLASSES[variant])

    mod_spec = importlib.util.spec_from_file_location("aimv2_model", model_file)
    mod_mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod_mod)
    ModelClass = getattr(mod_mod, MODEL_CLASSES[variant])

    return ConfigClass, ModelClass


def load_model(variant, model_dir, device, dtype):
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    cfg.pop("auto_map", None)

    ConfigClass, ModelClass = _import_model_files(variant)
    config = ConfigClass(**cfg)

    from safetensors.torch import load_file
    sd_path = model_dir / "model.safetensors"
    if not sd_path.exists():
        print(f"错误：找不到 {sd_path}")
        print(f"请先下载模型权重到 {model_dir}")
        sys.exit(1)

    sd = load_file(str(sd_path))
    if needs_conversion(sd):
        print("检测到 Timm 格式权重，执行 Key 转换...")
        sd = convert_state_dict(variant, sd)

    model = ModelClass(config)
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def infer(model, pixel_values, input_ids=None, attention_mask=None):
    if input_ids is not None:
        return model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    return model(pixel_values=pixel_values)


def main():
    parser = argparse.ArgumentParser(description="AIMv2 NPU 推理")
    parser.add_argument("--model", required=True, choices=list(MODEL_INFO.keys()))
    parser.add_argument("--image", type=str, help="图片路径（可选）")
    parser.add_argument("--text", default="a photo of a cat", help="文本 prompt（仅 lit）")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = parser.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    info = MODEL_INFO[args.model]

    cache_dir = Path.home() / ".cache" / "modelscope" / "hub"
    model_dir = cache_dir / "models" / "apple" / info["name"]

    print(f"变体: {args.model}")
    print(f"加载模型: {model_dir}")
    model = load_model(args.model, model_dir, NPU_DEVICE, dtype)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型加载完成: {num_params/1e6:.2f}M params")

    image_size = info["image_size"]
    if args.image:
        from PIL import Image
        from torchvision import transforms
        img = Image.open(args.image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        pixel_values = transform(img).unsqueeze(0).to(device=NPU_DEVICE, dtype=dtype)
    else:
        pixel_values = torch.randn(1, 3, image_size, image_size).to(device=NPU_DEVICE, dtype=dtype)

    if info["type"] == "lit":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("apple/aimv2-large-patch14-224-lit")
        tokens = tokenizer(args.text, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"].to(NPU_DEVICE)
        attention_mask = tokens["attention_mask"].to(NPU_DEVICE)
        output = infer(model, pixel_values, input_ids, attention_mask)
        key = "logits_per_image" if hasattr(output, "logits_per_image") else 0
        if isinstance(key, str):
            print(f"相似度 (image-text): {output[key].item():.4f}")
        else:
            print(f"Output: {output[0].shape}")
    else:
        output = infer(model, pixel_values)
        last_hidden = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output[0]
        print(f"Output shape: {last_hidden.shape}")
        print(f"Output stats: mean={last_hidden.mean().item():.4f}, std={last_hidden.std().item():.4f}")

    print("推理完成！")


if __name__ == "__main__":
    main()
