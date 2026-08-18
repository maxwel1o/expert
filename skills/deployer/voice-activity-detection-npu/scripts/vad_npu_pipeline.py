#!/usr/bin/env python3
"""
完整 VAD 管线脚本：在 NPU 上运行语音活动检测。
使用 pyannote.audio Inference 处理滑动窗口、Powerset 转换和 overlap-add 聚合。
"""

import warnings
import numpy as np
import torch
import torch_npu

warnings.filterwarnings("ignore")

# ============================================================
# 1. 环境准备
# ============================================================
device = torch.device("npu:0")
print(f"[INFO] NPU: {torch.npu.get_device_name(0)}")

from pyannote.audio import Model, Inference

# ============================================================
# 2. 模型加载
# ============================================================
model = Model.from_pretrained("pyannote/segmentation", map_location=str(device))
model.eval()
print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 3. 构建 VAD 推理管线
# ============================================================
# pre_aggregation_hook: max over speakers → VAD score
inference = Inference(
    model,
    window="sliding",
    pre_aggregation_hook=lambda scores: np.max(scores, axis=-1, keepdims=True),
    device=device,
)

# ============================================================
# 4. 生成测试音频（含语音/静音段）
# ============================================================
sr = 16000
duration = 5.0
t = torch.linspace(0, duration, int(duration * sr))
signal = torch.zeros_like(t)

segments = [
    (0.0, 1.2, 0.5, 300),     # 语音
    (1.2, 1.8, 0.02, 0),       # 静音+噪声
    (1.8, 3.0, 0.4, 350),      # 语音
    (3.0, 3.5, 0.01, 0),       # 静音+噪声
    (3.5, 4.5, 0.45, 250),     # 语音
    (4.5, 5.0, 0.015, 0),      # 静音+噪声
]
for start, end, amp, freq in segments:
    mask = (t >= start) & (t < end)
    if freq > 0:
        signal[mask] = amp * torch.sin(2 * np.pi * freq * t[mask]) * torch.sin(2 * np.pi * 3 * t[mask])
    else:
        signal[mask] = amp * torch.randn(mask.sum())

waveform = signal.unsqueeze(0).float()
print(f"[INFO] Audio: {waveform.size(-1)} samples ({duration:.1f}s)")

# ============================================================
# 5. 推理 + 后处理
# ============================================================
print("[INFO] Running VAD inference...")
file = {"waveform": waveform, "sample_rate": sr}
result = inference(file)
scores = result.data  # (num_frames, 1)

# 滞后阈值判决
def scores_to_segments(scores, onset=0.5, offset=0.5, min_duration_on=0.055, min_duration_off=0.098):
    scores = scores.squeeze(-1)
    num_frames = len(scores)
    if num_frames == 0:
        return []

    sec_per_frame = 0.029  # 模型 receptive field step

    # Hysteresis
    speech = np.zeros(num_frames, dtype=bool)
    in_speech = False
    for t in range(num_frames):
        if in_speech:
            if scores[t] < offset:
                in_speech = False
        else:
            if scores[t] > onset:
                in_speech = True
        speech[t] = in_speech

    # 按最短语音/静音段过滤
    segments_list = []
    i = 0
    while i < num_frames:
        if speech[i]:
            start = i * sec_per_frame
            while i < num_frames and speech[i]:
                i += 1
            end = i * sec_per_frame
            if end - start >= min_duration_on:
                segments_list.append((start, end))
        else:
            i += 1

    # 合并短静音
    if min_duration_off > 0 and segments_list:
        merged = [segments_list[0]]
        for seg in segments_list[1:]:
            gap = seg[0] - merged[-1][1]
            if gap < min_duration_off:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)
        segments_list = merged

    return segments_list

segments = scores_to_segments(scores)

# ============================================================
# 6. 输出结果
# ============================================================
print(f"[INFO] Scores: {scores.shape[0]} frames, [{scores.min():.4f}, {scores.max():.4f}]")
print(f"[INFO] Detected {len(segments)} speech segment(s):")
for i, (start, end) in enumerate(segments):
    print(f"  [{i}] {start:.2f}s - {end:.2f}s ({end - start:.2f}s)")

print("[INFO] VAD Pipeline: SUCCESS")
