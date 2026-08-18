from typing import Optional, Tuple, Union

import torch
import dataclasses
import math

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import ModelOutput

from configuration_aimv2_lit import AIMv2Config, AIMv2VisionConfig, AIMv2TextConfig
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel

__all__ = ["AIMv2VisionModel", "AIMv2TextModel", "AIMv2Model"]

AIMv2VisionOrTextConfig = Union[AIMv2VisionConfig, AIMv2TextConfig]


@dataclasses.dataclass
class AIMv2Output(ModelOutput):
    logits_per_image: torch.Tensor
    logits_per_text: Optional[torch.Tensor] = None
    image_features: Optional[torch.Tensor] = None
    text_features: Optional[torch.Tensor] = None
    vision_output: Optional[BaseModelOutputWithNoAttention] = None
    text_output: Optional[BaseModelOutputWithNoAttention] = None


class AIMv2TextPreprocessor(nn.Module):
    def __init__(self, config: AIMv2TextConfig):
        super().__init__()
        self.max_context_length = config.max_context_length
        self.eos_token_id = config.eos_token_id

        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Parameter(
            torch.zeros(self.max_context_length, config.hidden_size)
        )

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, N = input_ids.shape
        max_len = min(N, self.max_context_length)
        eos_token_mask = input_ids == self.eos_token_id
        tokens = self.text_embedding(input_ids)
        tokens = tokens[:, :max_len] + self.positional_embedding[:max_len].unsqueeze(0)
        return tokens, eos_token_mask


class AIMv2ExtractEOS(nn.Module):
    def forward(
        self, tokens: torch.Tensor, eos_token_mask: torch.Tensor
    ) -> torch.Tensor:
        B, _, D = tokens.shape
        eos_token_mask = torch.argmax(eos_token_mask.float(), dim=-1)
        assert eos_token_mask.shape == (B,)
        eos_token_mask = eos_token_mask.reshape(B, 1, 1).expand(B, 1, D)
        eos_token = torch.gather(tokens, 1, eos_token_mask)
        eos_token = eos_token.squeeze(1)
        return eos_token


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2VisionOrTextConfig):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


class AIMv2PatchEmbed(nn.Module):
    def __init__(self, config: AIMv2VisionOrTextConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AIMv2ViTPreprocessor(nn.Module):
    def __init__(self, config: AIMv2VisionConfig):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = AIMv2PatchEmbed(config)
        self.pos_embed = nn.Parameter(torch.zeros((1, num_patches, config.hidden_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(x)
        _, N, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)
        tokens = tokens + pos_embed[:, :N]
        return tokens


class AIMv2Attention(nn.Module):
    def __init__(self, config: AIMv2VisionOrTextConfig):
        super().__init__()
        dim = config.hidden_size

        self.num_heads = config.num_attention_heads
        self.is_causal = config.is_causal
        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj = nn.Linear(dim, dim, bias=config.use_bias)
        self.proj_drop = nn.Dropout(config.projection_dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if mask is None:
            x = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        else:
            mask_converter = AttentionMaskConverter(self.is_causal)
            mask = mask_converter.to_4d(
                mask, key_value_length=N, query_length=N, dtype=q.dtype
            )
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AIMv2Block(nn.Module):
    def __init__(self, config: AIMv2VisionOrTextConfig):
        super().__init__()
        self.attn = AIMv2Attention(config)
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config)
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class AIMv2AttentionPoolingHead(nn.Module):
    def __init__(self, config: AIMv2VisionConfig):
        super().__init__()
        dim = config.hidden_size
        qkv_bias = config.qkv_bias

        self.num_heads = config.num_attention_heads
        self.num_queries = config.num_queries

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, self.num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)
        return out


class AIMv2Transformer(nn.Module):
    def __init__(self, config: AIMv2VisionOrTextConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AIMv2Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_trunk_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            tokens = block(tokens, mask)
            if output_hidden_states:
                hidden_states += (tokens,)
        tokens = self.post_trunk_norm(tokens)
        return tokens, hidden_states


class AIMv2PretrainedModel(PreTrainedModel):
    base_model_prefix = "aimv2"
    _supports_sdpa = True


class AIMv2VisionModel(AIMv2PretrainedModel):
    config_class = AIMv2VisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["AIMv2ViTPreprocessor", "AIMv2Block"]

    def __init__(self, config: AIMv2VisionConfig):
        super().__init__(config)
        self.preprocessor = AIMv2ViTPreprocessor(config)
        self.trunk = AIMv2Transformer(config)
        self.head = AIMv2AttentionPoolingHead(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        x = self.preprocessor(pixel_values)
        x, hidden_states = self.trunk(
            x, mask, output_hidden_states=output_hidden_states
        )
        x = self.head(x)

        if not return_dict:
            res = (x,)
            res += (hidden_states,) if output_hidden_states else ()
            return res

        return BaseModelOutputWithNoAttention(
            last_hidden_state=x,
            hidden_states=hidden_states,
        )


class AIMv2TextModel(AIMv2PretrainedModel):
    config_class = AIMv2TextConfig
    main_input_name = "input_ids"
    _no_split_modules = ["AIMv2TextPreprocessor", "AIMv2Block"]

    def __init__(self, config: AIMv2TextConfig):
        super().__init__(config)
        self.preprocessor = AIMv2TextPreprocessor(config)
        self.trunk = AIMv2Transformer(config)
        self.head = AIMv2ExtractEOS()

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        x, eos_token_mask = self.preprocessor(pixel_values)
        x, hidden_states = self.trunk(
            x, mask, output_hidden_states=output_hidden_states
        )
        x = self.head(x, eos_token_mask)

        if not return_dict:
            res = (x,)
            res += (hidden_states,) if output_hidden_states else ()
            return res

        return BaseModelOutputWithNoAttention(
            last_hidden_state=x,
            hidden_states=hidden_states,
        )


class AIMv2Model(AIMv2PretrainedModel):
    config_class = AIMv2Config
    main_input_name = ["input_ids", "pixel_values"]
    _no_split_modules = ["AIMv2ViTPreprocessor", "AIMv2TextPreprocessor", "AIMv2Block"]

    def __init__(self, config: AIMv2Config):
        super().__init__(config)
        self.image_encoder = AIMv2VisionModel(config.vision_config)
        self.text_encoder = AIMv2TextModel(config.text_config)

        self.image_projector = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=False
        )
        self.text_projector = nn.Linear(
            config.text_config.hidden_size, config.projection_dim, bias=False
        )

        self.log_logit_scale = nn.Parameter(
            torch.full([], fill_value=math.log(1.0 / config.init_temperature))
        )
        self.max_log_logit_scale = math.log(config.max_logit_scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Union[Tuple[torch.Tensor, ...], BaseModelOutputWithNoAttention],
            Union[Tuple[torch.Tensor, ...], BaseModelOutputWithNoAttention],
        ],
        AIMv2Output,
    ]:
        if return_dict is None:
            return_dict = self.config.use_return_dict

        image_out = self.image_encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_features = image_out.last_hidden_state if return_dict else image_out[0]
        image_features = self.image_projector(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)

        text_out = self.text_encoder(
            input_ids,
            mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_features = text_out.last_hidden_state if return_dict else text_out[0]
        text_features = self.text_projector(text_features)
        text_features = F.normalize(text_features, p=2, dim=-1)

        logit_scale = self.log_logit_scale.clamp(0.0, self.max_log_logit_scale).exp()
        logits_per_text = (logit_scale * text_features) @ image_features.t()
        logits_per_image = logits_per_text.t()

        if not return_dict:
            return (
                logits_per_image,
                logits_per_text,
                image_features,
                text_features,
                image_out,
                text_out,
            )

        return AIMv2Output(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            image_features=image_features,
            text_features=text_features,
            vision_output=image_out,
            text_output=text_out,
        )

    def get_image_features(
        self,
        input_pixels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.image_encoder(input_pixels, mask=attention_mask, return_dict=True)
        image_features = self.image_projector(out.last_hidden_state)
        return image_features

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.text_encoder(input_ids, mask=attention_mask, return_dict=True)
        text_features = self.text_projector(out.last_hidden_state)
        return text_features
