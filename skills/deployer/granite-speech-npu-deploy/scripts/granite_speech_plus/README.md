# Granite Speech Plus - NPU Adaptation

`granite_speech_plus` is a transformers extension for `granite-speech-4.1-2b-plus`, adding the
`GraniteSpeechPlusEncoder` (with `cat_hidden_layers` support) and `GraniteSpeechPlusForConditionalGeneration`
model class.

## Usage

```python
from granite_speech_plus.register import register
register()

from granite_speech_plus import GraniteSpeechPlusForConditionalGeneration, GraniteSpeechPlusConfig

model = GraniteSpeechPlusForConditionalGeneration.from_pretrained(
    "/path/to/granite-speech-4.1-2b-plus",
    dtype=torch.bfloat16,
)

# Forward pass
outputs = model(input_ids=input_ids, input_features=input_features)
```

## Architecture

- **Encoder**: `GraniteSpeechPlusEncoder` - conformer encoder with `cat_hidden_layers=[3]`,
  concatenating layer 3's hidden state to the output (hidden_dim × 2 = 2048)
- **Projector**: QFormer-based projector with `encoder_hidden_size=2048`
- **Text Decoder**: `GraniteForCausalLM` (granite-4.0-1b-base, 40 layers)
