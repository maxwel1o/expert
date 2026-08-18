import torch
from torch import nn

from transformers.generation import GenerationMixin
from transformers.utils import logging, auto_docstring
from transformers.models.granite_speech.modeling_granite_speech import (
    GraniteSpeechPreTrainedModel,
    GraniteSpeechEncoderProjector,
    GraniteSpeechCausalLMOutputWithPast,
    GraniteSpeechConformerBlock,
)
from transformers.models.auto import AutoModelForCausalLM

from .configuration_granite_speech_plus import GraniteSpeechPlusConfig, GraniteSpeechPlusEncoderConfig


logger = logging.get_logger(__name__)


class GraniteSpeechPlusEncoder(nn.Module):
    """
    Granite Speech Plus encoder with support for cat_hidden_layers.
    Extends the base conformer encoder by concatenating intermediate
    hidden states to the final output.
    """

    def __init__(self, config: GraniteSpeechPlusEncoderConfig):
        super().__init__()
        self.config = config

        # Precompute clamped relative positional encoding distances
        seq = torch.arange(config.context_size)
        relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
        attention_dists = torch.clamp(relpos_dist, -config.context_size, config.context_size) + config.max_pos_emb
        self.register_buffer("attention_dists", attention_dists, persistent=False)
        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.layers = nn.ModuleList([GraniteSpeechConformerBlock(config) for _ in range(config.num_layers)])

        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim, bias=True)
        self.num_layers = config.num_layers
        self.cat_hidden_layers = sorted(config.cat_hidden_layers) if config.cat_hidden_layers else []

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.input_linear(hidden_states)
        cat_hidden_list = []
        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=self.attention_dists)

            if idx in self.cat_hidden_layers:
                cat_hidden_list.append(hidden_states.clone())

            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid = self.out(hidden_states_mid)
                hidden_states += self.out_mid(nn.Softmax(dim=-1)(hidden_states_mid))

        # Concatenate intermediate hidden states to the final output
        if cat_hidden_list:
            hidden_states = torch.cat([hidden_states] + cat_hidden_list, dim=-1)
        return hidden_states


@auto_docstring
class GraniteSpeechPlusPreTrainedModel(GraniteSpeechPreTrainedModel):
    config: GraniteSpeechPlusConfig

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    The Granite Speech Plus model, which consists of an audio encoder (with intermediate
    hidden state concatenation), projector, and language model.
    """
)
class GraniteSpeechPlusForConditionalGeneration(GraniteSpeechPlusPreTrainedModel, GenerationMixin):
    def __init__(self, config: GraniteSpeechPlusConfig):
        super().__init__(config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.encoder = GraniteSpeechPlusEncoder(config.encoder_config)
        self.projector = GraniteSpeechEncoderProjector(config)

        if config.has_lora_adapter:
            try:
                from peft import is_peft_available
                if not is_peft_available():
                    logger.warning(
                        "Config indicates that a lora adapter should be present, but "
                        "peft is not installed; this will cause the model to perform "
                        "incorrectly when audio inputs are provided. Please install "
                        "peft and reload the model!"
                    )
            except ImportError:
                pass

        self.post_init()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_audio_features(self, input_features: torch.Tensor) -> torch.Tensor:
        encoder_embeds = self.encoder(input_features)
        projected_embeds = self.projector(encoder_embeds)
        return projected_embeds

    @auto_docstring
    def forward(
        self,
        input_ids=None,
        input_features=None,
        input_features_mask=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **lm_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_features is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_features and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            is_audio_idx = input_ids == self.config.audio_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[is_audio_idx] = 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if input_features is not None:
            if input_features.dtype != self.dtype:
                input_features = input_features.to(self.dtype)
            audio_embeds = self.get_audio_features(input_features)

            inputs_embeds = self.get_merged_audio_embeddings(
                input_ids=input_ids,
                audio_features=audio_embeds,
                input_features_mask=input_features_mask,
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
        logits = outputs[0]

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1):].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GraniteSpeechCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        input_features=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position is not None and cache_position[0] == 0:
            model_inputs["input_features"] = input_features
        return model_inputs

    def get_merged_audio_embeddings(
        self, input_ids, audio_features, input_features_mask=None
    ):
        is_audio_index = input_ids == self.config.audio_token_id
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.language_model.get_input_embeddings()(llm_input_ids)

        special_audio_mask = is_audio_index.unsqueeze(-1)
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        if input_features_mask is not None:
            if torch.all(is_audio_index.int().sum(dim=1) != input_features_mask.int().sum(dim=1)).item():
                raise ValueError("Number of audio tokens does not match number of audio features")
            audio_features = audio_features[input_features_mask]

        inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)
        return inputs_embeds

    def generate(self, *args, **kwargs):
        input_features = kwargs.pop("input_features", None)
        try:
            from peft import is_peft_available
            if is_peft_available() and self._hf_peft_config_loaded:
                if input_features is not None:
                    self.enable_adapters()
                else:
                    self.disable_adapters()
        except (ImportError, AttributeError):
            pass
        return super().generate(*args, input_features=input_features, **kwargs)


__all__ = [
    "GraniteSpeechPlusEncoder",
    "GraniteSpeechPlusPreTrainedModel",
    "GraniteSpeechPlusForConditionalGeneration",
]
