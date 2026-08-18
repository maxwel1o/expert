#!/usr/bin/env python3
"""Weight format converter: OpenMMLab -> timm for common CV architectures.

ModelScope downloads CV model weights in OpenMMLab format
(backbone.xxx, head.xxx keys), which need conversion to timm format
(blocks.xxx, fc.xxx keys) before loading.

Usage:
    from convert_state_dict import convert_state_dict
    checkpoint = torch.load("pytorch_model.pt", map_location="cpu")
    state_dict = convert_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=False)
"""

import re
from collections import OrderedDict


def convert_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Convert OpenMMLab format state dict to timm format.

    Supports ResNeSt, ViT/DeiT, ConvNeXt, BEiTv2, NextViT,
    TinyNAS, BNext architectures.
    """
    new_sd: OrderedDict = OrderedDict()

    for k, v in state_dict.items():
        # Remove "module." prefix (DataParallel wrapper)
        k = re.sub(r"^module\.", "", k)

        if k.startswith("backbone."):
            new_k = k[len("backbone."):]

            # ResNeSt/ViT/BEiT: backbone.layers.i.xxx -> blocks.i.xxx
            new_k = re.sub(r"^layers\.(\d+)\.", r"blocks.\1.", new_k)

            # backbone.cls_token -> cls_token
            # backbone.pos_embed -> pos_embed
            if new_k in ("cls_token", "pos_embed", "mask_token"):
                pass

            # backbone.patch_embed.projection -> patch_embed.proj
            new_k = new_k.replace("patch_embed.projection.", "patch_embed.proj.")

            # backbone.ln1 -> norm (ViT backbone norm)
            new_k = re.sub(r"^ln1\.", "norm.", new_k)
            new_k = re.sub(r"^ln_pre\.", "norm.", new_k)

            # ffn.layers.0.0.weight/bias -> mlp.fc1.weight/bias
            # ffn.layers.1.weight/bias -> mlp.fc2.weight/bias
            new_k = re.sub(
                r"^(blocks\.\d+\.)ffn\.layers\.0\.0\.",
                r"\1mlp.fc1.", new_k
            )
            new_k = re.sub(
                r"^(blocks\.\d+\.)ffn\.layers\.0\.",
                r"\1mlp.fc1.", new_k
            )
            new_k = re.sub(
                r"^(blocks\.\d+\.)ffn\.layers\.1\.",
                r"\1mlp.fc2.", new_k
            )
            # ffn.layers.1 (no numbered sub-layer) -> mlp.fc2
            new_k = re.sub(
                r"^(blocks\.\d+\.)ffn\.layers\.1\.",
                r"\1mlp.fc2.", new_k
            )

            # Remove attention.qkv.in_proj -> attention.qkv (for ViT)
            new_k = new_k.replace("attn.qkv.in_proj.", "attn.qkv.")

            # Remove attention.projection -> attention.proj
            new_k = new_k.replace("attn.projection.", "attn.proj.")

            # ConvNeXt: backbone.stages.N. -> stages.N.
            new_k = re.sub(r"^stages\.", "stages.", new_k)

            new_sd[new_k] = v

        elif k.startswith("head."):
            # head.layers.head -> head, head.fc -> head
            new_k = re.sub(r"^head\.layers\.head\.", "head.", k)
            new_k = re.sub(r"^head\.fc\.", "head.", new_k)

            # If after stripping it's just "head" + weight/bias, keep as is
            new_sd[new_k] = v

        elif any(k.startswith(p) for p in (
            "cls_token", "pos_embed", "mask_token", "patch_embed", "ln_pre", "norm"
        )):
            new_sd[k] = v

        else:
            new_sd[k] = v

    return new_sd
