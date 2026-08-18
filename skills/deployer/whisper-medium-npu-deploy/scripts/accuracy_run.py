#!/usr/bin/env python3
"""whisper-medium 精度对比：NPU vs CPU"""
import os
import json
import numpy as np
import torch
import torch_npu  # noqa: F401
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_PATH = os.environ.get("WHISPER_MODEL_PATH", "./whisper-medium")
AUDIO_PATH = os.environ.get("TEST_AUDIO_PATH", "test_audio.wav")

device = torch.device("npu:0")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)

audio, sr = sf.read(AUDIO_PATH)
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

torch.manual_seed(42)

# ── CPU baseline ──
print("=" * 50)
print("CPU Baseline")
print("=" * 50)
model_cpu = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.float32, low_cpu_mem_usage=True,
)
model_cpu.eval()

forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
dec_ids = torch.tensor([[processor.tokenizer.bos_token_id]])
for _, tid in forced_ids:
    dec_ids = torch.cat([dec_ids, torch.tensor([[tid]])], dim=-1)

with torch.no_grad():
    out_cpu = model_cpu(input_features, decoder_input_ids=dec_ids, return_dict=True)
    gen_cpu = model_cpu.generate(
        input_features, language="english", task="transcribe",
        return_timestamps=False, max_length=448, num_beams=1,
    )

logits_cpu = out_cpu.logits.numpy()
trans_cpu = processor.batch_decode(gen_cpu, skip_special_tokens=True)[0]
print(f"Transcription: {trans_cpu}")
print(f"Logits shape: {logits_cpu.shape}")
del model_cpu

# ── NPU inference ──
print("\n" + "=" * 50)
print("NPU Inference")
print("=" * 50)
model_npu = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.float32, low_cpu_mem_usage=True,
).to(device)
model_npu.eval()

with torch.no_grad():
    out_npu = model_npu(
        input_features.to(device),
        decoder_input_ids=dec_ids.to(device),
        return_dict=True,
    )
    gen_npu = model_npu.generate(
        input_features.to(device), language="english", task="transcribe",
        return_timestamps=False, max_length=448, num_beams=1,
    )

torch.npu.synchronize()
logits_npu = out_npu.logits.cpu().numpy()
trans_npu = processor.batch_decode(gen_npu.cpu(), skip_special_tokens=True)[0]
print(f"Transcription: {trans_npu}")
print(f"Logits shape: {logits_npu.shape}")

# ── Comparison ──
print("\n" + "=" * 50)
print("Accuracy Comparison")
print("=" * 50)

abs_diff = np.abs(logits_cpu - logits_npu)
cos_sim = np.dot(logits_cpu.flatten(), logits_npu.flatten()) / (
    np.linalg.norm(logits_cpu.flatten()) * np.linalg.norm(logits_npu.flatten())
)
corr = np.corrcoef(logits_cpu.flatten(), logits_npu.flatten())[0, 1]


def softmax(x, axis=-1):
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


cpu_probs = softmax(logits_cpu)
npu_probs = softmax(logits_npu)
max_prob_diff = np.abs(cpu_probs - npu_probs).max()

cpu_preds = np.argmax(logits_cpu, axis=-1)
npu_preds = np.argmax(logits_npu, axis=-1)

results = {
    "max_abs_logit_diff": round(float(abs_diff.max()), 6),
    "mean_abs_logit_diff": round(float(abs_diff.mean()), 6),
    "cosine_similarity": round(float(cos_sim), 8),
    "correlation": round(float(corr), 8),
    "max_prob_diff_pct": round(float(max_prob_diff * 100), 4),
    "cpu_transcription": trans_cpu,
    "npu_transcription": trans_npu,
    "cpu_tokens": [int(t) for t in gen_cpu[0]],
    "npu_tokens": [int(t) for t in gen_npu[0]],
    "token_match": bool((gen_cpu[0] == gen_npu[0].cpu()).all()),
}

print(f"CPU Transcription:    {results['cpu_transcription']}")
print(f"NPU Transcription:    {results['npu_transcription']}")
print(f"Max abs logit diff:   {results['max_abs_logit_diff']}")
print(f"Mean abs logit diff:  {results['mean_abs_logit_diff']}")
print(f"Cosine similarity:    {results['cosine_similarity']}")
print(f"Correlation:          {results['correlation']}")
print(f"Max prob diff:        {results['max_prob_diff_pct']}%")
print(f"Token match:          {results['token_match']}")

print(f"\n{'='*50}")
print(f"OVERALL: {'PASS' if results['token_match'] and results['max_prob_diff_pct'] < 1.0 else 'FAIL'}")
print(f"{'='*50}")

with open("accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to accuracy_results.json")
