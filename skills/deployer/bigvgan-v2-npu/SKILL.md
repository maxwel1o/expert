---
name: bigvgan-v2-npu
description: Deploy BigVGAN v2 neural vocoder on Ascend NPU for mel-to-waveform inference. Covers model loading, NPU device detection, end-to-end audio synthesis, and performance benchmarking.
---

# BigVGAN v2 NPU Deployment

Deploy NVIDIA BigVGAN v2 neural vocoder on Huawei Ascend NPU with native PyTorch.

## When to Invoke

- User wants to run BigVGAN on Ascend NPU
- User needs a mel-to-waveform vocoder on NPU
- User asks about audio model deployment on Ascend
- User encounters CUDA-specific issues with BigVGAN

## Prerequisites

- Ascend NPU with CANN >= 8.0
- torch >= 2.5.0, torch-npu >= 2.5.0
- librosa, scipy, numpy

## Workflow

### Step 1: Download Weights

```bash
# HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download nvidia/bigvgan_v2_22khz_80band_256x \
  --local-dir ./bigvgan_v2_22khz_80band_256x \
  --include "bigvgan_generator.pt" "config.json"
```

### Step 2: NPU Inference Script

```python
import os, json, torch
import bigvgan
from env import AttrDict
from meldataset import get_mel_spectrogram

try:
    import torch_npu
    device = torch.device("npu:0") if torch_npu.npu.is_available() else torch.device("cpu")
except Exception:
    device = torch.device("cpu")

with open("config.json") as f:
    h = AttrDict(json.load(f))

model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
ckpt = torch.load("bigvgan_generator.pt", map_location="cpu")
try:
    model.load_state_dict(ckpt["generator"])
except RuntimeError:
    model.remove_weight_norm()
    model.load_state_dict(ckpt["generator"])
model.remove_weight_norm()
model = model.eval().to(device)

# Random mel input
mel = torch.randn(1, h.num_mels, 100, device=device)
with torch.inference_mode():
    wav = model(mel)
```

### Step 3: End-to-End from Audio

```python
import librosa
wav, sr = librosa.load("input.wav", sr=h.sampling_rate, mono=True)
wav_t = torch.FloatTensor(wav).unsqueeze(0)
mel = get_mel_spectrogram(wav_t, h).to(device)
with torch.inference_mode():
    wav_gen = model(mel)
```

## Key Points

- Always set `use_cuda_kernel=False` on NPU
- Model uses standard PyTorch ops; no custom operator replacement needed
- If `torch.stft` fails on NPU, compute mel on CPU then move to NPU
- Typical NPU memory: ~450 MB, RTF > 5x for 100-frame mel

## Verification Checklist

- [ ] Model loads without CUDA errors
- [ ] Random mel inference produces waveform
- [ ] Audio-to-audio pipeline runs end-to-end
- [ ] Output wav shape matches expected `[1, 1, T]`
