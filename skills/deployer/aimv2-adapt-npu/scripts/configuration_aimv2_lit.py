from typing import Any, Dict, Optional, Union

from transformers.configuration_utils import PretrainedConfig

__all__ = ["AIMv2VisionConfig", "AIMv2TextConfig", "AIMv2Config"]


class AIMv2VisionConfig(PretrainedConfig):
    """This is the configuration class to store the configuration of an [`AIMv2VisionModel`].

    Instantiating a configuration with the defaults will yield a similar configuration
    to that of the [apple/aimv2-large-patch14-224-lit](https://huggingface.co/apple/aimv2-large-patch14-224-lit).

    Args:
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the SwiGLU representations.
        num_hidden_layers: Number of hidden layers in the Transformer.
        num_attention_heads: Number of attention heads for each attention layer
            in the Transformer.
        num_queries: Number of learnable queries for the attention-pooling head.
        num_channels: Number of input channels.
        image_size: Image size.
        patch_size: Patch size.
        rms_norm_eps: Epsilon value used for the RMS normalization layer.
        attention_dropout: Dropout ratio for attention probabilities.
        projection_dropout: Dropout ratio for the projection layer after the attention.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        use_bias: Whether to add a bias in the feed-forward and projection layers.
        kwargs: Keyword arguments for the [`PretrainedConfig`].
    """

    model_type: str = "aimv2"
    base_config_key: str = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_queries: int = 1,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps

        self.projection_dropout = projection_dropout
        self.qkv_bias = qkv_bias
        self.use_bias = use_bias
        self.is_causal = False


class AIMv2TextConfig(PretrainedConfig):
    """This is the configuration class to store the configuration of an [`AIMv2TextModel`].

    Instantiating a configuration with the defaults will yield a similar configuration
    to that of the [apple/aimv2-large-patch14-224-lit](https://huggingface.co/apple/aimv2-large-patch14-224-lit).

    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the SwiGLU representations.
        num_hidden_layers: Number of hidden layers in the Transformer.
        num_attention_heads: Number of attention heads for each attention layer
            in the Transformer.
        rms_norm_eps: Epsilon value used for the RMS normalization layer.
        attention_dropout: Dropout ratio for attention probabilities.
        projection_dropout: Dropout ratio for the projection layer after the attention.
        qkv_bias: Whether to add a bias to the queries, keys and values.
        use_bias: Whether to add a bias in the feed-forward and projection layers.
        eos_token_id: End-of-sequence token id.
        max_context_length: Maximum number of tokens for the context.
        kwargs: Keyword arguments for the [`PretrainedConfig`].
    """

    model_type: str = "aimv2"
    base_config_key: str = "text_config"

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        eos_token_id: int = 49407,
        max_context_length: int = 77,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps

        self.projection_dropout = projection_dropout
        self.qkv_bias = qkv_bias
        self.use_bias = use_bias
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.eos_token_id = eos_token_id
        self.is_causal = True


class AIMv2Config(PretrainedConfig):
    """This is the configuration class to store the configuration of an [`AIMv2Model`].

    Instantiating a configuration with the defaults will yield a similar configuration
    to that of the [apple/aimv2-large-patch14-224-lit](https://huggingface.co/apple/aimv2-large-patch14-224-lit).

    Args:
        vision_config: Vision config.
        text_config: Text config.
        projection_dim: Dimension of the image and text projection layers.
        kwargs: Keyword arguments for the [`PretrainedConfig`].
    """

    model_type = "aimv2"
    is_composition: bool = True
    sub_configs: Dict[str, PretrainedConfig] = {
        "vision_config": AIMv2VisionConfig,
        "text_config": AIMv2TextConfig,
    }

    def __init__(
        self,
        vision_config: Optional[Union[AIMv2VisionConfig, Dict[str, Any]]] = None,
        text_config: Optional[Union[AIMv2TextConfig, Dict[str, Any]]] = None,
        projection_dim: int = 768,
        init_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = AIMv2VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = AIMv2VisionConfig(**vision_config)

        if text_config is None:
            text_config = AIMv2TextConfig()
        elif isinstance(text_config, dict):
            text_config = AIMv2TextConfig(**text_config)

        self.vision_config = vision_config
        self.text_config = text_config
        self.projection_dim = projection_dim

        self.init_temperature = init_temperature
        self.max_logit_scale = max_logit_scale
