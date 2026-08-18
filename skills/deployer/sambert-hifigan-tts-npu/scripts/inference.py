#!/usr/bin/env python3
"""
Sambert-HifiGAN TTS Inference on Ascend NPU.

Direct weight loading (bypasses ModelScope pipeline and ttsfrd).
Supports CPU and Ascend NPU devices.

Usage:
  python3 inference.py --text "北京今天天气怎么样" --device cpu
  python3 inference.py --text "北京今天天气怎么样" --device npu
  python3 inference.py --text "Hello world" --model en-us_16k --voice andy --device cpu
"""

import os
import sys
import re
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Ensure ttsfrd stub is importable
STUB_PATH = "/tmp/ttsfrd_stub"
if STUB_PATH not in sys.path:
    sys.path.insert(0, STUB_PATH)

import ttsfrd
import yaml
import torch
import torch.nn.functional as F

from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit
from kantts.models.sambert.kantts_sambert import KanTtsSAMBERT
from kantts.models.hifigan.hifigan import Generator

# Default cache location
MODELSCOPE_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", ".cache", "modelscope", "hub", "models", "iic"
) if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")) else "/opt/atomgit/.cache/modelscope/hub/models/iic"

# Model name to path helper
MODEL_PATHS = {}


def discover_models():
    """Discover downloaded models."""
    models = {}
    if os.path.isdir(MODELSCOPE_CACHE):
        for name in os.listdir(MODELSCOPE_CACHE):
            voices_dir = os.path.join(MODELSCOPE_CACHE, name, "voices")
            if os.path.isdir(voices_dir):
                voices = sorted([
                    d for d in os.listdir(voices_dir)
                    if os.path.isdir(os.path.join(voices_dir, d))
                ])
                if voices:
                    models[name] = voices
    return models


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_config(am_config, dict_dir, am_ckpt_path=None):
    params = am_config["Model"]["KanTtsSAMBERT"]["params"]
    config = dict(params)
    for key, fname in [("sy", "sy_dict.txt"), ("tone", "tone_dict.txt"),
                       ("syllable_flag", "syllable_flag_dict.txt"),
                       ("word_segment", "word_segment_dict.txt"),
                       ("emotion", "emo_category_dict.txt"),
                       ("speaker", "speaker_dict.txt")]:
        path = os.path.join(dict_dir, fname)
        with open(path) as f:
            n = len([l for l in f if l.strip()])
        config[key] = n + 3
    config["FP"] = False
    config["MAS"] = False
    # Auto-adjust embedding dims to match checkpoint (handles en-us models
    # that use +2 special tokens instead of +3)
    if am_ckpt_path and os.path.exists(am_ckpt_path):
        ckpt = torch.load(am_ckpt_path, map_location="cpu", weights_only=True)
        emb_keys = {
            "text_encoder.sy_emb.weight": "sy",
            "text_encoder.tone_emb.weight": "tone",
            "text_encoder.syllable_flag_emb.weight": "syllable_flag",
            "text_encoder.ws_emb.weight": "word_segment",
            "spk_tokenizer.weight": "speaker",
            "emo_tokenizer.weight": "emotion",
        }
        for ckpt_key, config_key in emb_keys.items():
            ckpt_size = ckpt["model"][ckpt_key].size(0)
            if ckpt_size != config[config_key]:
                print(f"  Adjusting {config_key}: {config[config_key]} -> {ckpt_size} (checkpoint)")
                config[config_key] = ckpt_size
    return config


@torch.no_grad()
def sambert_infer(model, ling_features, emotion_id, speaker_id, device):
    model.eval()
    tensors = []
    for f in ling_features:
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f.copy()).long()
        tensors.append(f.to(device))
    inputs_ling = torch.stack(tensors, dim=-1).unsqueeze(0)
    T = inputs_ling.size(1)
    input_lengths = torch.tensor([T], device=device, dtype=torch.long)
    inputs_emotion = torch.full((1, T), emotion_id, device=device, dtype=torch.long)
    inputs_speaker = torch.full((1, T), speaker_id, device=device, dtype=torch.long)
    output = model(
        inputs_ling=inputs_ling,
        inputs_emotion=inputs_emotion,
        inputs_speaker=inputs_speaker,
        input_lengths=input_lengths,
    )
    return output["postnet_outputs"]


@torch.no_grad()
def hifigan_infer(model, mel, device):
    model.eval()
    return model(mel.transpose(1, 2))


def load_model(model_dir, voice, device):
    """Load Sambert + HifiGAN models from checkpoint files."""
    voice_dir = os.path.join(model_dir, "voices", voice)
    am_ckpt = os.path.join(voice_dir, "am", "ckpt", "checkpoint_0.pth")
    voc_ckpt = os.path.join(voice_dir, "voc", "ckpt", "checkpoint_0.pth")
    am_cfg = os.path.join(voice_dir, "am", "config.yaml")
    voc_cfg = os.path.join(voice_dir, "voc", "config.yaml")
    dict_dir = os.path.join(voice_dir, "dict")

    am_config = load_config(am_cfg)
    voc_config = load_config(voc_cfg)
    model_config = build_model_config(am_config, dict_dir, am_ckpt)

    # Linguistic unit
    ling_unit = KanTtsLinguisticUnit(am_config)

    # Sambert
    sambert = KanTtsSAMBERT(model_config)
    ckpt = torch.load(am_ckpt, map_location="cpu", weights_only=True)
    sambert.load_state_dict(ckpt["model"], strict=False)
    sambert.to(device)

    # HifiGAN
    gp = voc_config["Model"]["Generator"]["params"]
    hifigan = Generator(
        in_channels=gp["in_channels"], out_channels=gp["out_channels"],
        channels=gp["channels"], kernel_size=gp["kernel_size"],
        upsample_scales=gp["upsample_scales"],
        upsample_kernal_sizes=gp["upsample_kernal_sizes"],
        resblock_kernel_sizes=gp["resblock_kernel_sizes"],
        resblock_dilations=gp["resblock_dilations"],
        bias=gp["bias"], causal=gp["causal"],
        nonlinear_activation=gp["nonlinear_activation"],
        nonlinear_activation_params=gp["nonlinear_activation_params"],
        use_weight_norm=gp["use_weight_norm"],
    )
    voc_data = torch.load(voc_ckpt, map_location="cpu", weights_only=True)
    hifigan.load_state_dict(voc_data["model"]["generator"])
    hifigan.to(device)

    return sambert, hifigan, ling_unit


def text_to_features(text, model_dir, ling_unit):
    """Convert text to linguistic features."""
    engine = ttsfrd.TtsFrontendEngine()
    engine.initialize(model_dir)
    engine.set_lang_type("PinYin")
    res = engine.gen_tacotron_symbols(text)
    parts = res.strip().split("\t")
    symbols = parts[1] if len(parts) == 2 else parts[0]
    encoded = ling_unit.encode_symbol_sequence(symbols)
    return encoded


def synthesize(text, model_dir, voice, speaker, emotion, device):
    """Full TTS synthesis: text -> audio."""
    sambert, hifigan, ling_unit = load_model(model_dir, voice, device)
    encoded = text_to_features(text, model_dir, ling_unit)
    spk_id = ling_unit._speaker_to_id[speaker]
    emo_id = ling_unit._emo_category_to_id[emotion]
    mel = sambert_infer(sambert, encoded, emo_id, spk_id, device)
    audio = hifigan_infer(hifigan, mel, device)
    return audio.squeeze().cpu().numpy()


def main():
    # Discover available models
    available = discover_models()
    model_names = list(available.keys())

    parser = argparse.ArgumentParser(description="Sambert-HifiGAN TTS Inference")
    parser.add_argument("--model", type=str, default="speech_sambert-hifigan_tts_zh-cn_16k",
                        choices=model_names if model_names else None,
                        help="Model name")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Full path to model directory (overrides --model)")
    parser.add_argument("--voice", type=str, default=None,
                        help="Voice name")
    parser.add_argument("--text", type=str, default="北京今天天气怎么样",
                        help="Input text")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"],
                        help="Device")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio path")
    parser.add_argument("--speaker", type=str, default="F7",
                        help="Speaker ID")
    parser.add_argument("--emotion", type=str, default="emotion_neutral",
                        help="Emotion category")
    args = parser.parse_args()

    # Resolve model directory
    if args.model_dir:
        model_dir = args.model_dir
    elif args.model in available:
        model_dir = os.path.join(MODELSCOPE_CACHE, args.model)
    else:
        print(f"Model '{args.model}' not found. Available: {model_names}")
        sys.exit(1)

    # Auto-detect voice if not specified
    voice = args.voice
    if voice is None:
        voices_dir = os.path.join(model_dir, "voices")
        if os.path.isdir(voices_dir):
            voices = sorted([
                d for d in os.listdir(voices_dir)
                if os.path.isdir(os.path.join(voices_dir, d))
            ])
            if voices:
                voice = voices[0]
                print(f"Auto-selected voice: {voice}")
    if not voice:
        print("No voice found. Specify --voice")
        sys.exit(1)

    # Setup device
    if args.device == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            print("torch_npu not available, falling back to CPU")
            args.device = "cpu"
        device = torch.device("npu:0" if args.device == "npu" else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Synthesizing: '{args.text}'")
    print(f"Model: {os.path.basename(model_dir)}, Voice: {voice}, Device: {args.device}")

    audio = synthesize(args.text, model_dir, voice, args.speaker, args.emotion, device)

    import soundfile as sf
    sf.write(args.output, audio, 16000)
    duration = len(audio) / 16000.0
    print(f"Output: {args.output} ({duration:.2f}s, {len(audio)} samples)")


if __name__ == "__main__":
    main()
