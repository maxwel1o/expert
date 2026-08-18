#!/usr/bin/env python3
"""
Pyannote Speaker Diarization — NPU Inference Script
====================================================
Standalone inference for speaker-diarization/speaker-diarization-community-1
on Huawei Ascend NPU.

Usage:
    python3 inference.py <audio_file> [--model community-1|basic] [--output result.json]
"""

import os
import sys
import json
import warnings
from pathlib import Path

import torch
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# NPU Detection
# ============================================================
HAS_NPU = False
try:
    import torch_npu
    HAS_NPU = torch.npu.is_available()
except Exception:
    pass


# ============================================================
# NPU Adaptation Patches
# ============================================================
def _mock_torchcodec():
    import soundfile as sf
    import pyannote.audio.core.io as pyannote_io

    class MockAudioStreamMetadata:
        def __init__(self, filepath):
            info = sf.info(str(filepath))
            self.sample_rate = info.samplerate
            self.duration_seconds_from_header = info.duration
            self.num_frames_from_header = info.frames
            self.num_channels = info.channels

    class MockAudioSamples:
        def __init__(self, data, sample_rate):
            self.data = data
            self.sample_rate = sample_rate

    class MockAudioDecoder:
        def __init__(self, file):
            if isinstance(file, (str, Path)):
                data, sr = sf.read(str(file), always_2d=True)
            else:
                data, sr = sf.read(file, always_2d=True)
            data = torch.from_numpy(data.T).float()
            self._all_samples = MockAudioSamples(data, sr)
            self._metadata = MockAudioStreamMetadata(file)

        @property
        def metadata(self):
            return self._metadata

        def get_all_samples(self):
            return self._all_samples

        def get_samples_played_in_range(self, start_sec, end_sec):
            sr = self._metadata.sample_rate
            s = int(round(start_sec * sr))
            e = int(round(end_sec * sr))
            data = self._all_samples.data[:, s:e]
            return MockAudioSamples(data, sr)

    pyannote_io.AudioDecoder = MockAudioDecoder
    pyannote_io.AudioSamples = MockAudioSamples
    pyannote_io.AudioStreamMetadata = MockAudioStreamMetadata


def _patch_fbank_npu():
    import torch.nn.functional as F
    from pyannote.audio.models.embedding.wespeaker import BaseWeSpeakerResNet
    from pyannote.audio.utils.receptive_field import conv1d_num_frames

    def patched_compute_fbank(self, waveforms):
        waveforms = waveforms * (1 << 15)
        device = waveforms.device
        fft_device = torch.device("cpu") if device.type in ("mps", "npu") else device
        features = torch.vmap(self._fbank)(waveforms.to(fft_device)).to(device)
        if self.hparams.fbank_centering_span is None:
            return features - torch.mean(features, dim=1, keepdim=True)
        ws = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
        ss = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)
        ks = conv1d_num_frames(
            num_samples=int(self.hparams.fbank_centering_span * self.hparams.sample_rate),
            kernel_size=ws, stride=ss,
        )
        kernel = torch.full((1, 1, ks), 1.0 / ks, device=device)
        features = features.unsqueeze(dim=1)
        features = F.conv1d(features, kernel, padding="same")
        features = features.squeeze(dim=1)
        return features

    BaseWeSpeakerResNet.compute_fbank = patched_compute_fbank


def _patch_get_devices():
    from pyannote.audio.pipelines.utils import getter
    original = getter.get_devices

    def patched(needs=None):
        if HAS_NPU:
            nd = torch.npu.device_count()
            devices = [torch.device(f"npu:{i}") for i in range(nd)] if nd else [torch.device("cpu")]
            return devices if needs is None else [d for _, d in zip(range(needs), __import__('itertools').cycle(devices))]
        return original(needs)

    getter.get_devices = patched


def _patch_pipeline_cuda():
    from pyannote.audio.core.pipeline import Pipeline
    original = Pipeline.cuda

    def patched(self, device=None):
        if HAS_NPU:
            if device is None:
                return self.to(torch.device("npu"))
            if isinstance(device, int):
                return self.to(torch.device("npu", device))
        return original(self, device)

    Pipeline.cuda = patched


def apply_patches():
    """Apply all NPU adaptation patches."""
    print("[INFO] Applying NPU patches...")
    _mock_torchcodec()
    _patch_fbank_npu()
    _patch_get_devices()
    _patch_pipeline_cuda()

    import pyannote.audio.core.task
    torch.serialization.add_safe_globals([
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Problem,
    ])
    warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
    print("[INFO] NPU patches applied successfully")


# ============================================================
# Model Loading
# ============================================================
def load_pipeline(model="community-1", device="auto", models_dir="models"):
    """Load speaker diarization pipeline.

    Args:
        model: "community-1" or "basic"
        device: "auto", "npu", "cpu", "cuda"
        models_dir: directory containing model weights
    """
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    device_map = {
        "auto": torch.device("npu") if HAS_NPU else torch.device("cpu"),
        "npu": torch.device("npu"),
        "cpu": torch.device("cpu"),
        "cuda": torch.device("cuda"),
    }
    dev = device_map.get(device, torch.device(device))

    models_path = Path(models_dir)

    if model == "community-1":
        model_path = models_path / "speaker-diarization-community-1"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please download from https://gitcode.com/hf_mirrors/pyannote/speaker-diarization-community-1"
            )
        pipeline = Pipeline.from_pretrained(str(model_path))
    else:
        c1_path = models_path / "speaker-diarization-community-1"
        if not (c1_path / "segmentation").exists():
            raise FileNotFoundError(
                f"Community-1 sub-models not found at {c1_path}. "
                f"Please download from https://gitcode.com/hf_mirrors/pyannote/speaker-diarization-community-1"
            )
        pipeline = SpeakerDiarization(
            segmentation=str(c1_path / "segmentation"),
            embedding=str(c1_path / "embedding"),
            plda=str(c1_path / "plda"),
            clustering="VBxClustering",
            embedding_batch_size=32,
            segmentation_batch_size=32,
        )

    pipeline.to(dev)
    print(f"[INFO] Pipeline loaded on {dev}")
    return pipeline


# ============================================================
# Inference
# ============================================================
def run_diarization(pipeline, audio_file, num_speakers=None, min_speakers=None, max_speakers=None):
    """Run speaker diarization."""
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    output = pipeline(audio_file, **kwargs)

    segments = []
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
        })

    result = {
        "diarization": segments,
        "num_speakers": len(output.speaker_diarization.labels()),
    }
    if output.speaker_embeddings is not None:
        result["speaker_embeddings_shape"] = list(output.speaker_embeddings.shape)
    return result


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pyannote Speaker Diarization on NPU")
    parser.add_argument("audio_file", type=str, help="Path to audio file (.wav)")
    parser.add_argument("--model", type=str, default="community-1", choices=["community-1", "basic"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, npu, cpu, cuda")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with model weights")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)

    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"[ERROR] File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    apply_patches()
    pipeline = load_pipeline(model=args.model, device=args.device, models_dir=args.models_dir)

    print(f"[INFO] Running diarization on {args.audio_file}...")
    result = run_diarization(pipeline, args.audio_file, args.num_speakers, args.min_speakers, args.max_speakers)

    print(f"\n[RESULTS] Detected {result['num_speakers']} speaker(s)")
    for seg in result["diarization"]:
        print(f"  {seg['speaker']}: {seg['start']}s -> {seg['end']}s")
    if "speaker_embeddings_shape" in result:
        print(f"  Embedding shape: {result['speaker_embeddings_shape']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
