"""
Apple AIMv2 权重 Key 转换工具
==============================
将 safetensors 中的 Timm/HF 风格 key 映射到建模代码格式。

Timm 格式 (distilled/native):
  embeddings.patch_embed.weight → preprocessor.patchifier.proj.weight
  encoder.layers.{i}.attention.{q,k,v}_proj → trunk.blocks.{i}.attn.qkv (merged)

Lit 格式:
  vision_model.embeddings.* → image_encoder.preprocessor.*
  vision_model.encoder.layers.{i}.attention.{q,k,v}_proj → image_encoder.trunk.blocks.{i}.attn.qkv
  text_model.embeddings.* → text_encoder.preprocessor.*
  text_model.encoder.layers.{i}.* → text_encoder.trunk.blocks.{i}.*
"""

import json
from pathlib import Path
import torch


def needs_conversion(state_dict: dict) -> bool:
    """检查权重是否需要 key 转换"""
    return any(
        k.startswith("embeddings.") or k.startswith("encoder.") or
        k.startswith("vision_model.") or k.startswith("text_model.")
        for k in state_dict.keys()
    )


def convert_state_dict(model_key: str, state_dict: dict) -> dict:
    """根据模型类型转换权重 key"""
    if model_key in ("224-distilled", "336-distilled"):
        return _convert_timm_distilled(state_dict)
    elif model_key == "native":
        return _convert_timm_native(state_dict)
    elif model_key == "224-lit":
        return _convert_timm_lit(state_dict)
    raise ValueError(f"Unknown model: {model_key}")


def _convert_timm_encoder_block(sd: dict, src_prefix: str, dst_prefix: str, num_layers: int):
    """转换 encoder 层的 key (distilled/native 使用)"""
    for i in range(num_layers):
        # QKV: merge q_proj, k_proj, v_proj
        q = sd.pop(f"{src_prefix}.{i}.attention.q_proj.weight", None)
        k = sd.pop(f"{src_prefix}.{i}.attention.k_proj.weight", None)
        v = sd.pop(f"{src_prefix}.{i}.attention.v_proj.weight", None)
        if q is not None and k is not None and v is not None:
            sd[f"{dst_prefix}.{i}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0)

        # QKV bias (distilled has qkv_bias=False, but handle anyway)
        qb = sd.pop(f"{src_prefix}.{i}.attention.q_proj.bias", None)
        kb = sd.pop(f"{src_prefix}.{i}.attention.k_proj.bias", None)
        vb = sd.pop(f"{src_prefix}.{i}.attention.v_proj.bias", None)
        if qb is not None and kb is not None and vb is not None:
            sd[f"{dst_prefix}.{i}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        # Out proj
        v = sd.pop(f"{src_prefix}.{i}.attention.out_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.attn.proj.weight"] = v
        vb = sd.pop(f"{src_prefix}.{i}.attention.out_proj.bias", None)
        if vb is not None:
            sd[f"{dst_prefix}.{i}.attn.proj.bias"] = vb

        # FFN: gate_proj → fc1, up_proj → fc3, down_proj → fc2
        v = sd.pop(f"{src_prefix}.{i}.ffn.gate_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.mlp.fc1.weight"] = v
        v = sd.pop(f"{src_prefix}.{i}.ffn.up_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.mlp.fc3.weight"] = v
        v = sd.pop(f"{src_prefix}.{i}.ffn.down_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.mlp.fc2.weight"] = v

        # RMS norms
        v = sd.pop(f"{src_prefix}.{i}.rms_norm1.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.norm_1.weight"] = v
        v = sd.pop(f"{src_prefix}.{i}.rms_norm2.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.norm_2.weight"] = v


def _convert_timm_distilled(sd: dict) -> dict:
    """转换 distilled 模型 key"""
    # Patch embed
    if "embeddings.patch_embed.weight" in sd:
        sd["preprocessor.patchifier.proj.weight"] = sd.pop("embeddings.patch_embed.weight")
    if "embeddings.patch_embed.bias" in sd:
        sd["preprocessor.patchifier.proj.bias"] = sd.pop("embeddings.patch_embed.bias")
    if "embeddings.rms_norm.weight" in sd:
        sd["preprocessor.patchifier.norm.weight"] = sd.pop("embeddings.rms_norm.weight")
    if "embeddings.position_embedding.weight" in sd:
        sd["preprocessor.pos_embed"] = sd.pop("embeddings.position_embedding.weight").unsqueeze(0)

    # Encoder layers (24 layers)
    _convert_timm_encoder_block(sd, "encoder.layers", "trunk.blocks", 24)

    # Post trunk norm
    if "rms_norm.weight" in sd:
        sd["trunk.post_trunk_norm.weight"] = sd.pop("rms_norm.weight")

    return sd


def _convert_timm_native(sd: dict) -> dict:
    """转换 native 模型 key (与 distilled 类似但无 position_embedding)"""
    if "embeddings.patch_embed.weight" in sd:
        sd["preprocessor.patchifier.proj.weight"] = sd.pop("embeddings.patch_embed.weight")
    if "embeddings.patch_embed.bias" in sd:
        sd["preprocessor.patchifier.proj.bias"] = sd.pop("embeddings.patch_embed.bias")
    if "embeddings.rms_norm.weight" in sd:
        sd["preprocessor.patchifier.norm.weight"] = sd.pop("embeddings.rms_norm.weight")

    _convert_timm_encoder_block(sd, "encoder.layers", "trunk.blocks", 24)

    if "rms_norm.weight" in sd:
        sd["trunk.post_trunk_norm.weight"] = sd.pop("rms_norm.weight")

    return sd


def _convert_timm_lit_encoder_block(sd: dict, src_prefix: str, dst_prefix: str, num_layers: int):
    """转换 lit 模型的 encoder 层 (与 timm 相同但 prefix 不同)"""
    for i in range(num_layers):
        q = sd.pop(f"{src_prefix}.{i}.attention.q_proj.weight", None)
        k = sd.pop(f"{src_prefix}.{i}.attention.k_proj.weight", None)
        v = sd.pop(f"{src_prefix}.{i}.attention.v_proj.weight", None)
        if q is not None and k is not None and v is not None:
            sd[f"{dst_prefix}.{i}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0)

        qb = sd.pop(f"{src_prefix}.{i}.attention.q_proj.bias", None)
        kb = sd.pop(f"{src_prefix}.{i}.attention.k_proj.bias", None)
        vb = sd.pop(f"{src_prefix}.{i}.attention.v_proj.bias", None)
        if qb is not None and kb is not None and vb is not None:
            sd[f"{dst_prefix}.{i}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        v = sd.pop(f"{src_prefix}.{i}.attention.out_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.attn.proj.weight"] = v
        vb = sd.pop(f"{src_prefix}.{i}.attention.out_proj.bias", None)
        if vb is not None:
            sd[f"{dst_prefix}.{i}.attn.proj.bias"] = vb

        v = sd.pop(f"{src_prefix}.{i}.ffn.gate_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.mlp.fc1.weight"] = v
        v = sd.pop(f"{src_prefix}.{i}.ffn.up_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.mlp.fc3.weight"] = v
        v = sd.pop(f"{src_prefix}.{i}.ffn.down_proj.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.mlp.fc2.weight"] = v

        v = sd.pop(f"{src_prefix}.{i}.rms_norm1.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.norm_1.weight"] = v
        v = sd.pop(f"{src_prefix}.{i}.rms_norm2.weight", None)
        if v is not None:
            sd[f"{dst_prefix}.{i}.norm_2.weight"] = v


def _convert_timm_lit(sd: dict) -> dict:
    """转换 lit 模型 key"""
    # === Vision encoder ===
    # Patch embed
    if "vision_model.embeddings.patch_embed.weight" in sd:
        sd["image_encoder.preprocessor.patchifier.proj.weight"] = sd.pop("vision_model.embeddings.patch_embed.weight")
    if "vision_model.embeddings.patch_embed.bias" in sd:
        sd["image_encoder.preprocessor.patchifier.proj.bias"] = sd.pop("vision_model.embeddings.patch_embed.bias")
    if "vision_model.embeddings.rms_norm.weight" in sd:
        sd["image_encoder.preprocessor.patchifier.norm.weight"] = sd.pop("vision_model.embeddings.rms_norm.weight")
    if "vision_model.embeddings.position_embedding.weight" in sd:
        sd["image_encoder.preprocessor.pos_embed"] = sd.pop("vision_model.embeddings.position_embedding.weight").unsqueeze(0)

    # Vision encoder layers
    _convert_timm_lit_encoder_block(sd, "vision_model.encoder.layers", "image_encoder.trunk.blocks", 24)

    # Vision post trunk norm
    if "vision_model.rms_norm.weight" in sd:
        sd["image_encoder.trunk.post_trunk_norm.weight"] = sd.pop("vision_model.rms_norm.weight")

    # Vision head (attention pooling)
    if "vision_model.head.k_proj.weight" in sd:
        sd["image_encoder.head.k.weight"] = sd.pop("vision_model.head.k_proj.weight")
    if "vision_model.head.v_proj.weight" in sd:
        sd["image_encoder.head.v.weight"] = sd.pop("vision_model.head.v_proj.weight")
    if "vision_model.head.cls_token" in sd:
        sd["image_encoder.head.cls_token"] = sd.pop("vision_model.head.cls_token")
    if "vision_model.head.output_proj.weight" in sd:
        sd["image_encoder.head.linear.weight"] = sd.pop("vision_model.head.output_proj.weight")
    if "vision_model.head.output_proj.bias" in sd:
        sd["image_encoder.head.linear.bias"] = sd.pop("vision_model.head.output_proj.bias")

    # === Text encoder ===
    if "text_model.embeddings.token_embedding.weight" in sd:
        sd["text_encoder.preprocessor.text_embedding.weight"] = sd.pop("text_model.embeddings.token_embedding.weight")
    if "text_model.embeddings.position_embedding.weight" in sd:
        sd["text_encoder.preprocessor.positional_embedding"] = sd.pop("text_model.embeddings.position_embedding.weight")

    # Text encoder layers (12 layers)
    _convert_timm_lit_encoder_block(sd, "text_model.encoder.layers", "text_encoder.trunk.blocks", 12)

    # Text post trunk norm
    if "text_model.rms_norm.weight" in sd:
        sd["text_encoder.trunk.post_trunk_norm.weight"] = sd.pop("text_model.rms_norm.weight")

    # === Projections ===
    if "visual_projection.weight" in sd:
        sd["image_projector.weight"] = sd.pop("visual_projection.weight")
    if "text_projection.weight" in sd:
        sd["text_projector.weight"] = sd.pop("text_projection.weight")
    if "logit_scale" in sd:
        sd["log_logit_scale"] = sd.pop("logit_scale")

    return sd
