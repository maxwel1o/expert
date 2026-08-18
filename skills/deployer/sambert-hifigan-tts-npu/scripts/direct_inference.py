#!/usr/bin/env python3
"""
Direct Sambert-HifiGAN inference without ModelScope pipeline.
Loads weights directly from checkpoint files.

Usage:
  python3 direct_inference.py --model-dir <path> --voice zhitian_emo --text "你好世界" --device cpu
  python3 direct_inference.py --model-dir <path> --voice zhitian_emo --text "你好世界" --device npu
"""

import os
import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, "/tmp/ttsfrd_stub")

import ttsfrd
import yaml
import torch
import torch.nn.functional as F

from kantts.utils.ling_unit.ling_unit import KanTtsLinguisticUnit
from kantts.models.sambert.kantts_sambert import KanTtsSAMBERT
from kantts.models.hifigan.hifigan import Generator
from kantts.models.utils import get_mask_from_lengths


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_sambert_model_config(am_config, dict_dir, am_ckpt_path=None):
    """Build model config dict matching KanTtsSAMBERT.__init__ expectations."""
    model_params = am_config["Model"]["KanTtsSAMBERT"]["params"]
    config = dict(model_params)

    def count_dict(path):
        with open(path, "r") as f:
            return len([l for l in f if l.strip()])

    n_sy = count_dict(os.path.join(dict_dir, "sy_dict.txt"))
    n_tone = count_dict(os.path.join(dict_dir, "tone_dict.txt"))
    n_syllable = count_dict(os.path.join(dict_dir, "syllable_flag_dict.txt"))
    n_wordseg = count_dict(os.path.join(dict_dir, "word_segment_dict.txt"))
    n_emo = count_dict(os.path.join(dict_dir, "emo_category_dict.txt"))
    n_spk = count_dict(os.path.join(dict_dir, "speaker_dict.txt"))

    config["sy"] = n_sy + 3
    config["tone"] = n_tone + 3
    config["syllable_flag"] = n_syllable + 3
    config["word_segment"] = n_wordseg + 3
    config["emotion"] = n_emo + 3
    config["speaker"] = n_spk + 3
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
    """Run Sambert inference: text -> mel spectrogram.

    Uses KanTtsSAMBERT.forward() directly, which handles:
    text_encoder -> variance_adaptor (with duration/pitch/energy prediction)
    -> LFR -> mel_decoder -> postnet
    """
    model.eval()

    # Stack features: list of [T] arrays -> [1, T, n_feat_types]
    ling_tensors = []
    for f in ling_features:
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f.copy()).long()
        ling_tensors.append(f.to(device))
    inputs_ling = torch.stack(ling_tensors, dim=-1).unsqueeze(0)
    T = inputs_ling.size(1)
    input_lengths = torch.tensor([T], device=device, dtype=torch.long)

    # Expand emotion/speaker to full sequence length (model expects [B, T])
    inputs_emotion = torch.full([1, T], emotion_id, device=device, dtype=torch.long)
    inputs_speaker = torch.full([1, T], speaker_id, device=device, dtype=torch.long)

    # Full forward pass (inference mode: no targets provided)
    output = model(
        inputs_ling=inputs_ling,
        inputs_emotion=inputs_emotion,
        inputs_speaker=inputs_speaker,
        input_lengths=input_lengths,
    )

    mel = output["postnet_outputs"]  # [1, T_mel, 80]
    return mel


@torch.no_grad()
def hifigan_infer(model, mel, device):
    """Run HifiGAN vocoder: mel -> audio waveform."""
    model.eval()
    audio = model(mel.transpose(1, 2))  # [B, 80, T] -> [B, 1, T_audio]
    return audio


def main():
    parser = argparse.ArgumentParser(description="Direct Sambert-HifiGAN inference")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Model directory containing voices/ subdir")
    parser.add_argument("--voice", type=str, default="zhitian_emo",
                        help="Voice name (subdir under voices/)")
    parser.add_argument("--text", type=str, default="北京今天天气怎么样",
                        help="Input text")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "npu"],
                        help="Device to run inference on")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--speaker", type=str, default="F7",
                        help="Speaker ID")
    parser.add_argument("--emotion", type=str, default="emotion_neutral",
                        help="Emotion category")
    args = parser.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
        device = torch.device("npu:0")
    else:
        device = torch.device("cpu")

    voice_dir = os.path.join(args.model_dir, "voices", args.voice)
    am_ckpt = os.path.join(voice_dir, "am", "ckpt", "checkpoint_0.pth")
    voc_ckpt = os.path.join(voice_dir, "voc", "ckpt", "checkpoint_0.pth")
    am_config_path = os.path.join(voice_dir, "am", "config.yaml")
    voc_config_path = os.path.join(voice_dir, "voc", "config.yaml")
    dict_dir = os.path.join(voice_dir, "dict")

    for p in [am_ckpt, voc_ckpt, am_config_path, voc_config_path, dict_dir]:
        assert os.path.exists(p), f"Path not found: {p}"

    print(f"[INFO] Loading configs...")
    am_config = load_config(am_config_path)
    voc_config = load_config(voc_config_path)
    model_config = build_sambert_model_config(am_config, dict_dir, am_ckpt)

    print(f"[INFO] Initializing linguistic unit...")
    ling_unit = KanTtsLinguisticUnit(am_config)

    print(f"[INFO] Initializing ttsfrd engine...")
    engine = ttsfrd.TtsFrontendEngine()
    engine.initialize(args.model_dir)
    engine.set_lang_type("PinYin")

    print(f"[INFO] Processing text: {args.text}")
    res = engine.gen_tacotron_symbols(args.text)
    parts = res.strip().split("\t")
    symbols = parts[1] if len(parts) == 2 else parts[0]
    print(f"  Symbol string: {symbols[:120]}...")

    encoded = ling_unit.encode_symbol_sequence(symbols)

    speaker_id = ling_unit._speaker_to_id[args.speaker]
    emotion_id = ling_unit._emo_category_to_id[args.emotion]
    print(f"  Speaker '{args.speaker}' -> id {speaker_id}")
    print(f"  Emotion '{args.emotion}' -> id {emotion_id}")
    for lfeat_type, arr in zip(ling_unit._lfeat_type_list, encoded):
        print(f"  {lfeat_type}: shape={arr.shape}")

    # Build Sambert
    print(f"[INFO] Building Sambert model...")
    sambert = KanTtsSAMBERT(model_config)
    print(f"  Loading checkpoint: {am_ckpt}")
    ckpt = torch.load(am_ckpt, map_location="cpu", weights_only=True)
    # strict=False: position_enc and inv_timescales buffers are deterministically
    # computed in __init__ and not stored in the checkpoint
    sambert.load_state_dict(ckpt["model"], strict=False)
    sambert.to(device)
    n_sambert = sum(p.numel() for p in sambert.parameters())
    print(f"  Sambert params: {n_sambert:,}")

    # Build HifiGAN Generator
    print(f"[INFO] Building HifiGAN Generator...")
    gp = voc_config["Model"]["Generator"]["params"]
    hifigan = Generator(
        in_channels=gp["in_channels"],
        out_channels=gp["out_channels"],
        channels=gp["channels"],
        kernel_size=gp["kernel_size"],
        upsample_scales=gp["upsample_scales"],
        upsample_kernal_sizes=gp["upsample_kernal_sizes"],
        resblock_kernel_sizes=gp["resblock_kernel_sizes"],
        resblock_dilations=gp["resblock_dilations"],
        bias=gp["bias"],
        causal=gp["causal"],
        nonlinear_activation=gp["nonlinear_activation"],
        nonlinear_activation_params=gp["nonlinear_activation_params"],
        use_weight_norm=gp["use_weight_norm"],
    )
    print(f"  Loading checkpoint: {voc_ckpt}")
    voc_data = torch.load(voc_ckpt, map_location="cpu", weights_only=True)
    hifigan.load_state_dict(voc_data["model"]["generator"])
    hifigan.to(device)
    n_hifi = sum(p.numel() for p in hifigan.parameters())
    print(f"  HifiGAN params: {n_hifi:,}")

    # Run Sambert
    print(f"[INFO] Running Sambert on {args.device}...")
    mel = sambert_infer(sambert, encoded, emotion_id, speaker_id, device)
    print(f"  Output mel shape: {mel.shape}")

    # Run HifiGAN
    print(f"[INFO] Running HifiGAN on {args.device}...")
    audio = hifigan_infer(hifigan, mel, device)
    print(f"  Output audio shape: {audio.shape}")

    # Save audio
    audio_np = audio.squeeze().cpu().numpy()
    import soundfile as sf
    sf.write(args.output, audio_np, 16000)
    file_size = os.path.getsize(args.output)
    duration = len(audio_np) / 16000.0
    print(f"[INFO] Audio saved to {args.output}")
    print(f"  Duration: {duration:.2f}s, Size: {file_size:,} bytes")


if __name__ == "__main__":
    main()
