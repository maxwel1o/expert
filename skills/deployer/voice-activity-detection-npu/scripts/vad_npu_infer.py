#!/usr/bin/env python3
"""
基础推理脚本：pyannote/segmentation 在 NPU 上的前向推理验证。
"""

import warnings
import numpy as np
import torch
import torch_npu

warnings.filterwarnings("ignore")

device = torch.device("npu:0")
print(f"[INFO] Using device: {torch.npu.get_device_name(0)}")

# 加载模型
from pyannote.audio import Model

model = Model.from_pretrained("pyannote/segmentation", map_location=str(device))
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO] PyanNet ({total_params:,} params)")

# 生成测试信号
sr = 16000
duration = 2.0
t = torch.linspace(0, duration, int(duration * sr))

# 混合正弦波（模拟语音）
signal = torch.zeros_like(t)
for f, a in [(200, 0.5), (400, 0.3), (600, 0.2)]:
    signal += a * torch.sin(2 * np.pi * f * t)
signal = signal / signal.abs().max()

# 转为 (batch, channel, time) 格式
waveform = signal.unsqueeze(0).unsqueeze(0).float().to(device)

print(f"[INFO] Input shape: {waveform.shape}")
print(f"[INFO] Input duration: {duration}s")

# 前向推理
with torch.no_grad():
    output = model(waveform)

output_cpu = output.cpu().numpy()
vad_score = np.exp(output_cpu).max(axis=-1)

print(f"[INFO] Output shape: {output_cpu.shape}")
print(f"[INFO] Output range: [{output_cpu.min():.6f}, {output_cpu.max():.6f}]")
print(f"[INFO] VAD score range: [{vad_score.min():.4f}, {vad_score.max():.4f}]")
print(f"[INFO] Inference: SUCCESS")
