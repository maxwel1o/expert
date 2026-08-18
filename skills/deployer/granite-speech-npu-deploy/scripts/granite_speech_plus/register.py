"""
Register granite_speech_plus model types with transformers.
Call register() before loading any granite_speech_plus model.
"""
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

from .configuration_granite_speech_plus import GraniteSpeechPlusEncoderConfig, GraniteSpeechPlusConfig
from .modeling_granite_speech_plus import GraniteSpeechPlusForConditionalGeneration


_registered = False


def register():
    global _registered
    if _registered:
        return
    _registered = True

    # Register configs
    AutoConfig.register("granite_speech_plus_encoder", GraniteSpeechPlusEncoderConfig, exist_ok=True)
    AutoConfig.register("granite_speech_plus", GraniteSpeechPlusConfig, exist_ok=True)

    # Register model name
    MODEL_MAPPING_NAMES["granite_speech_plus"] = "Granite Speech Plus"

    print("[granite_speech_plus] Registered model types: granite_speech_plus, granite_speech_plus_encoder")
