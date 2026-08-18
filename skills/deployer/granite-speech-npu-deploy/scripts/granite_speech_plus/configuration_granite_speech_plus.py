from transformers.models.granite_speech.configuration_granite_speech import (
    GraniteSpeechEncoderConfig,
    GraniteSpeechConfig,
)
from transformers.models.auto import CONFIG_MAPPING


class GraniteSpeechPlusEncoderConfig(GraniteSpeechEncoderConfig):
    r"""
    Configuration class for Granite Speech Plus Encoder.
    Extends GraniteSpeechEncoderConfig with cat_hidden_layers parameter.

    Args:
        cat_hidden_layers (list[int], *optional*, defaults to `[3]`):
            List of layer indices whose hidden states will be concatenated
            to the final encoder output. This increases the output dimension
            by `hidden_dim * len(cat_hidden_layers)`.
    """
    model_type = "granite_speech_plus_encoder"

    def __init__(
        self,
        cat_hidden_layers=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if cat_hidden_layers is None:
            cat_hidden_layers = [3]
        self.cat_hidden_layers = cat_hidden_layers


class GraniteSpeechPlusConfig(GraniteSpeechConfig):
    r"""
    Configuration class for Granite Speech Plus For Conditional Generation.
    Uses GraniteSpeechPlusEncoderConfig for the encoder.

    Args:
        encoder_config (`GraniteSpeechPlusEncoderConfig`, *optional*):
            The config object or dictionary of the Granite Speech Plus encoder.
    """
    model_type = "granite_speech_plus"
    sub_configs = {
        "text_config": CONFIG_MAPPING,
        "encoder_config": GraniteSpeechPlusEncoderConfig,
        "projector_config": CONFIG_MAPPING,
    }

    def __init__(
        self,
        encoder_config=None,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config = GraniteSpeechPlusEncoderConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = GraniteSpeechPlusEncoderConfig()

        super().__init__(encoder_config=encoder_config, **kwargs)


__all__ = ["GraniteSpeechPlusEncoderConfig", "GraniteSpeechPlusConfig"]
