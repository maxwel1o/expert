#!/usr/bin/env python3
"""
Compare Sambert-HifiGAN inference results between CPU and NPU.
Runs on both devices and reports metrics.
"""

import os
import sys
import argparse
import warnings
import time
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


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_config(am_config, dict_dir, am_ckpt_path=None):
    params = am_config["Model"]["KanTtsSAMBERT"]["params"]
    config = dict(params)

    def count_dict(path):
        with open(path) as f:
            return len([l for l in f if l.strip()])

    config["sy"] = count_dict(os.path.join(dict_dir, "sy_dict.txt")) + 3
    config["tone"] = count_dict(os.path.join(dict_dir, "tone_dict.txt")) + 3
    config["syllable_flag"] = count_dict(os.path.join(dict_dir, "syllable_flag_dict.txt")) + 3
    config["word_segment"] = count_dict(os.path.join(dict_dir, "word_segment_dict.txt")) + 3
    config["emotion"] = count_dict(os.path.join(dict_dir, "emo_category_dict.txt")) + 3
    config["speaker"] = count_dict(os.path.join(dict_dir, "speaker_dict.txt")) + 3
    config["FP"] = False
    config["MAS"] = False
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
    ling_tensors = []
    for f in ling_features:
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f.copy()).long()
        ling_tensors.append(f.to(device))
    inputs_ling = torch.stack(ling_tensors, dim=-1).unsqueeze(0)
    T = inputs_ling.size(1)
    input_lengths = torch.tensor([T], device=device, dtype=torch.long)
    inputs_emotion = torch.full([1, T], emotion_id, device=device, dtype=torch.long)
    inputs_speaker = torch.full([1, T], speaker_id, device=device, dtype=torch.long)
    output = model(inputs_ling=inputs_ling, inputs_emotion=inputs_emotion,
                   inputs_speaker=inputs_speaker, input_lengths=input_lengths)
    return output["postnet_outputs"]


@torch.no_grad()
def hifigan_infer(model, mel, device):
    model.eval()
    return model(mel.transpose(1, 2))


def run_inference(device_str, model_dir, voice, text, speaker, emotion):
    if device_str == "npu":
        import torch_npu  # noqa: F401
        device = torch.device("npu:0")
    else:
        device = torch.device("cpu")

    voice_dir = os.path.join(model_dir, "voices", voice)
    am_ckpt = os.path.join(voice_dir, "am", "ckpt", "checkpoint_0.pth")
    voc_ckpt = os.path.join(voice_dir, "voc", "ckpt", "checkpoint_0.pth")
    am_cfg = os.path.join(voice_dir, "am", "config.yaml")
    voc_cfg = os.path.join(voice_dir, "voc", "config.yaml")
    dict_dir = os.path.join(voice_dir, "dict")

    am_config = load_config(am_cfg)
    voc_config = load_config(voc_cfg)
    model_config = build_model_config(am_config, dict_dir, am_ckpt)

    ling_unit = KanTtsLinguisticUnit(am_config)
    engine = ttsfrd.TtsFrontendEngine()
    engine.initialize(model_dir)
    engine.set_lang_type("PinYin")

    res = engine.gen_tacotron_symbols(text)
    parts = res.strip().split("\t")
    symbols = parts[1] if len(parts) == 2 else parts[0]
    encoded = ling_unit.encode_symbol_sequence(symbols)
    spk_id = ling_unit._speaker_to_id[speaker]
    emo_id = ling_unit._emo_category_to_id[emotion]

    sambert = KanTtsSAMBERT(model_config)
    ckpt = torch.load(am_ckpt, map_location="cpu", weights_only=True)
    sambert.load_state_dict(ckpt["model"], strict=False)
    sambert.to(device)

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

    t0 = time.time()
    mel = sambert_infer(sambert, encoded, emo_id, spk_id, device)
    t_sambert = time.time() - t0

    t0 = time.time()
    audio = hifigan_infer(hifigan, mel, device)
    t_hifigan = time.time() - t0

    audio_np = audio.squeeze().cpu().numpy()
    return audio_np, mel, t_sambert, t_hifigan


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs NPU inference")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--voice", type=str, default="zhitian_emo")
    parser.add_argument("--text", type=str, default="北京今天天气怎么样")
    parser.add_argument("--speaker", type=str, default="F7")
    parser.add_argument("--emotion", type=str, default="emotion_neutral")
    parser.add_argument("--output", type=str, default="compare_report.txt")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Voice: {args.voice}")
    print(f"Text: {args.text}")
    print("=" * 60)

    print("\n[1/2] Running inference on CPU...")
    cpu_audio, cpu_mel, cpu_ts, cpu_th = run_inference(
        "cpu", args.model_dir, args.voice, args.text, args.speaker, args.emotion
    )
    print(f"  Sambert: {cpu_ts:.2f}s, HifiGAN: {cpu_th:.2f}s, Audio: {len(cpu_audio)} samples")

    print("\n[2/2] Running inference on NPU...")
    npu_audio, npu_mel, npu_ts, npu_th = run_inference(
        "npu", args.model_dir, args.voice, args.text, args.speaker, args.emotion
    )
    print(f"  Sambert: {npu_ts:.2f}s, HifiGAN: {npu_th:.2f}s, Audio: {len(npu_audio)} samples")

    # Compare
    min_len = min(len(cpu_audio), len(npu_audio))
    cpu_a, npu_a = cpu_audio[:min_len], npu_audio[:min_len]

    diff = np.abs(cpu_a - npu_a)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    rmse = float(np.sqrt(np.mean((cpu_a - npu_a) ** 2)))
    signal_power = float(np.mean(cpu_a ** 2))
    noise_power = float(np.mean((cpu_a - npu_a) ** 2))
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    corr = float(np.corrcoef(cpu_a, npu_a)[0, 1])

    cpu_range = float(cpu_a.max() - cpu_a.min())
    diff_ratio = max_diff / cpu_range * 100 if cpu_range > 0 else 0

    speedup_sambert = cpu_ts / npu_ts if npu_ts > 0 else 0
    speedup_hifigan = cpu_th / npu_th if npu_th > 0 else 0

    report = f"""============================================================
CPU vs NPU Comparison Report
============================================================
Model:     {args.model_dir}
Voice:     {args.voice}
Text:      {args.text}
Speaker:   {args.speaker}
Emotion:   {args.emotion}

Audio Length:
  CPU: {len(cpu_audio)} samples ({len(cpu_audio)/16000:.2f}s)
  NPU: {len(npu_audio)} samples ({len(npu_audio)/16000:.2f}s)

Waveform Comparison:
  Max Absolute Diff:   {max_diff:.6f}
  Mean Absolute Diff:  {mean_diff:.6f}
  RMSE:                {rmse:.6f}
  Max Diff / Range:    {diff_ratio:.2f}%
  SNR:                 {snr:.2f} dB
  Correlation:         {corr:.6f}

Performance:
  Sambert CPU:  {cpu_ts:.2f}s | NPU: {npu_ts:.2f}s | Speedup: {speedup_sambert:.2f}x
  HifiGAN CPU:  {cpu_th:.2f}s | NPU: {npu_th:.2f}s | Speedup: {speedup_hifigan:.2f}x

Verification: {'PASS' if snr > 20 else 'CHECK'} (SNR > 20 dB)
============================================================
"""
    print("\n" + report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
