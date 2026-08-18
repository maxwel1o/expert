#!/usr/bin/env python3
"""
English NeuTTS-Nano NPU Inference Script for Ascend 910B4.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    python3 inference.py --text "Your text." --output output.wav
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*owner does not match.*")
warnings.filterwarnings("ignore", message=".*Permission mismatch.*")
warnings.filterwarnings("ignore", message=".*Skipping import.*")

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch_npu  # noqa: F401 -- registers npu device

MODEL_ID = "neuphonic/neutts-nano"
LANGUAGE = "en-us"


def patch_phonemizer():
    from phonemizer.backend.espeak.base import BaseEspeakBackend
    BaseEspeakBackend.is_available = classmethod(lambda cls: True)

    import phonemizer.backend.espeak.wrapper

    def _stub_init(self):
        self._version = (0, 0, 0)
        self._data_path = "/tmp"
        self._voice = None
        self._espeak = None
        self._libc_ = None
        self._tempfile_ = None

    phonemizer.backend.espeak.wrapper.EspeakWrapper.__init__ = _stub_init
    phonemizer.backend.espeak.wrapper.EspeakWrapper.version = property(
        lambda self: (0, 0, 0)
    )
    phonemizer.backend.espeak.wrapper.EspeakWrapper.library_path = property(
        lambda self: "(stub)"
    )
    phonemizer.backend.espeak.wrapper.EspeakWrapper.set_voice = lambda self, v: None
    phonemizer.backend.espeak.wrapper.EspeakWrapper.text_to_phonemes = (
        lambda self, text, tie=False: text
    )

    from phonemizer.backend.espeak.espeak import EspeakBackend

    def _stub_ebackend_init(self, language, **kwargs):
        self._espeak = None
        self._with_stress = kwargs.get("with_stress", False)
        self._tie = False
        self._lang_switch = None
        self._words_mismatch = None
        self.code = language
        self._language = language
        self._logger = None
        self._preserve_punctuation = kwargs.get("preserve_punctuation", False)
        self._punctuator = None

    EspeakBackend.__init__ = _stub_ebackend_init

    import neutts.phonemizers as pm

    class _SimplePhonemizer:
        def __init__(self, language_code="en-us"):
            self.code = language_code

        def phonemize(self, text):
            single = isinstance(text, str)
            if single:
                text = [text]
            return text[0].lower().strip() if single else [t.lower().strip() for t in text]

        def version(self):
            return (0, 0, 0)

    pm.BasePhonemizer = _SimplePhonemizer
    pm.CUSTOM_PHONEMIZERS.clear()
    pm.CUSTOM_PHONEMIZERS.update(
        {
            "en-us": _SimplePhonemizer("en-us"),
            "de": _SimplePhonemizer("de"),
            "fr-fr": _SimplePhonemizer("fr-fr"),
            "es": _SimplePhonemizer("es"),
        }
    )
    pm.FrenchPhonemizer = _SimplePhonemizer
    import neutts.neutts as nm

    nm.BasePhonemizer = _SimplePhonemizer


def main():
    parser = argparse.ArgumentParser(description="English NeuTTS-Nano NPU Inference")
    parser.add_argument(
        "--model", default=None, help="Local model path (default: HF download)"
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref_audio", help="Reference WAV for voice cloning")
    parser.add_argument("--ref_text", default="", help="Transcript of reference audio")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    args = parser.parse_args()

    backbone_repo = args.model if args.model else MODEL_ID

    print(f"=== NeuTTS-Nano (English) NPU Inference ===")
    print(f"  Backbone: {backbone_repo}")
    print(f"  NPU: {torch_npu.npu.get_device_name(0)}")
    print(f"  Text: {args.text[:80]}...")

    patch_phonemizer()
    from neutts import NeuTTS

    print("[NPU] Loading model...")
    t0 = time.time()
    tts = NeuTTS(
        backbone_repo=backbone_repo,
        backbone_device="npu",
        codec_repo="neuphonic/neucodec",
        codec_device="npu",
        language=LANGUAGE,
    )
    print(f"[NPU] Loaded in {time.time() - t0:.1f}s")
    print(f"[NPU] Backbone: {next(tts.backbone.parameters()).device}")
    print(f"[NPU] Codec: {next(tts.codec.parameters()).device}")

    if args.ref_audio and os.path.exists(args.ref_audio):
        ref_codes = tts.encode_reference(args.ref_audio)
        print(f"[NPU] Reference encoded: {ref_codes.shape}")
    else:
        print("[NPU] No reference audio, using test mode")
        ref_codes = torch.zeros(50, dtype=torch.long)

    print("[NPU] Running inference...")
    t0 = time.time()
    wav = tts.infer(args.text, ref_codes, args.ref_text)
    elapsed = time.time() - t0
    duration = len(wav) / 24000
    print(
        f"[NPU] Inference: {elapsed:.2f}s -> {duration:.1f}s audio "
        f"(RTF={elapsed / duration:.2f})"
    )

    import soundfile as sf

    sf.write(args.output, wav, 24000)
    print(f"[NPU] Saved: {args.output}")


if __name__ == "__main__":
    main()
