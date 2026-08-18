"""
Deterministic accuracy verification: NPU vs CPU logit comparison.

NeuTTS uses stochastic sampling (do_sample=True, temperature=1.0) for each
inference, so waveform comparison is meaningless. Instead, we compare the
backbone model's forward pass (logits) given the same input tokens.

This confirms that the NPU computation is within numerical tolerance of CPU.
"""

import os
import sys
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*owner does not match.*")
warnings.filterwarnings("ignore", message=".*Permission mismatch.*")
warnings.filterwarnings("ignore", message=".*Skipping import.*")

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch_npu  # noqa: F401

MODEL_BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_LANGUAGES = {
    "neutts-nano": "en-us",
    "neutts-nano-german": "de",
    "neutts-nano-french": "fr-fr",
    "neutts-nano-spanish": "es",
}


def patch_phonemizer():
    """Same patches as npu_adapter.py."""
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
    def _stub_version(self):
        return (0, 0, 0)
    phonemizer.backend.espeak.wrapper.EspeakWrapper.version = property(_stub_version)
    def _stub_library_path(self):
        return "(stub)"
    phonemizer.backend.espeak.wrapper.EspeakWrapper.library_path = property(_stub_library_path)
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
    pm.CUSTOM_PHONEMIZERS.update({
        "en-us": _SimplePhonemizer("en-us"),
        "de": _SimplePhonemizer("de"),
        "fr-fr": _SimplePhonemizer("fr-fr"),
        "es": _SimplePhonemizer("es"),
    })
    pm.FrenchPhonemizer = _SimplePhonemizer
    import neutts.neutts as nm
    nm.BasePhonemizer = _SimplePhonemizer
    from neutts import BACKBONE_LANGUAGE_MAP
    for name, lang in MODEL_LANGUAGES.items():
        BACKBONE_LANGUAGE_MAP[os.path.join(MODEL_BASE, name)] = lang


def compute_logit_metrics(logits_npu: torch.Tensor, logits_cpu: torch.Tensor) -> dict:
    """Compare two logit tensors (1D: [vocab_size])."""
    diff = (logits_npu.float() - logits_cpu.float()).abs()
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    cpu_abs = logits_cpu.float().abs()
    cpu_mean = float(cpu_abs.mean())
    rel_err = (mean_diff / cpu_mean * 100) if cpu_mean > 1e-10 else 0.0

    # Top-1 agreement: do both agree on the most likely token?
    top1_npu = int(torch.argmax(logits_npu))
    top1_cpu = int(torch.argmax(logits_cpu))
    top1_agree = 100.0 if top1_npu == top1_cpu else 0.0

    # Top-5 overlap
    top5_npu = set(torch.topk(logits_npu.float(), 5).indices.tolist())
    top5_cpu = set(torch.topk(logits_cpu.float(), 5).indices.tolist())
    top5_overlap = len(top5_npu & top5_cpu) / 5 * 100

    return {
        "max_abs_error": max_diff,
        "mean_abs_error": mean_diff,
        "relative_error_pct": rel_err,
        "top1_agreement_pct": top1_agree,
        "top5_overlap_pct": top5_overlap,
    }


def verify_model(model_name="neutts-nano", tolerance=1.0):
    print(f"\n{'='*60}")
    print(f"DETERMINISTIC VERIFY: {model_name}")
    print(f"{'='*60}")

    patch_phonemizer()
    from neutts import NeuTTS

    bb_path = os.path.join(MODEL_BASE, model_name)
    language = MODEL_LANGUAGES.get(model_name, "en-us")

    # ── Create a reference input ──────────────────────────────────
    print("\n[Setup] Creating reference input...")
    tts = NeuTTS(
        backbone_repo=bb_path, backbone_device="cpu",
        codec_repo="neuphonic/neucodec", codec_device="cpu",
        language=language,
    )
    # Build a prompt
    text = "Hello world test"
    ref_codes = torch.zeros(50, dtype=torch.long)
    prompt_ids = tts._apply_chat_template(ref_codes, "", text)
    input_ids = torch.tensor(prompt_ids).unsqueeze(0)  # [1, seq_len]
    print(f"  Input shape: {input_ids.shape}")
    del tts

    # ── CPU logits ────────────────────────────────────────────────
    print("\n[CPU] Computing logits...")
    from transformers import AutoModelForCausalLM
    torch.manual_seed(42)
    cpu_model = AutoModelForCausalLM.from_pretrained(bb_path).to("cpu")
    cpu_model.eval()
    t0 = time.time()
    with torch.no_grad():
        cpu_out = cpu_model(input_ids)
    cpu_logits = cpu_out.logits[0, -1, :].cpu()  # Last token logits
    cpu_time = time.time() - t0
    print(f"[CPU] Time: {cpu_time:.2f}s, Logits shape: {cpu_logits.shape}")
    del cpu_model

    # ── NPU logits ────────────────────────────────────────────────
    print("\n[NPU] Computing logits...")
    torch.manual_seed(42)
    npu_model = AutoModelForCausalLM.from_pretrained(bb_path).to("npu")
    npu_model.eval()
    npu_input = input_ids.to("npu")
    t0 = time.time()
    with torch.no_grad():
        npu_out = npu_model(npu_input)
    npu_logits = npu_out.logits[0, -1, :].cpu()
    npu_time = time.time() - t0
    print(f"[NPU] Time: {npu_time:.2f}s, Logits shape: {npu_logits.shape}")
    del npu_model

    # ── Compare ───────────────────────────────────────────────────
    metrics = compute_logit_metrics(npu_logits, cpu_logits)
    metrics["cpu_time_s"] = cpu_time
    metrics["npu_time_s"] = npu_time
    metrics["speedup"] = cpu_time / npu_time if npu_time > 0 else float("inf")
    metrics["passed"] = metrics["relative_error_pct"] < tolerance

    print(f"\n[Results: {model_name}]")
    print(f"  Speedup: {metrics['speedup']:.2f}x")
    print(f"  Max abs error:  {metrics['max_abs_error']:.6f}")
    print(f"  Mean abs error: {metrics['mean_abs_error']:.6f}")
    print(f"  Relative error: {metrics['relative_error_pct']:.4f}%")
    print(f"  Top-1 agreement: {metrics['top1_agreement_pct']:.1f}%")
    status = "PASS" if metrics["passed"] else "FAIL"
    print(f"  STATUS: {status}")

    return metrics


def main():
    models = ["neutts-nano", "neutts-nano-german", "neutts-nano-french", "neutts-nano-spanish"]
    results = {}
    for m in models:
        try:
            results[m] = verify_model(m)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[m] = {"error": str(e), "passed": False}

    print(f"\n{'='*60}")
    print("SUMMARY (Logit-level deterministic comparison)")
    print(f"{'='*60}")
    all_ok = True
    for m, r in results.items():
        ok = r.get("passed", False)
        info = f"{r.get('relative_error_pct', r.get('error', 'N/A'))}%"
        top1 = r.get("top1_agreement_pct", "N/A")
        spd = r.get("speedup", "N/A")
        print(f"  {m}: {'PASS' if ok else 'FAIL'} | err={info} | top1={top1}% | speedup={spd}x")
        if not ok:
            all_ok = False
    print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILED'}")

    # JSON-safe report
    safe_results = {}
    for m, r in results.items():
        if "error" in r:
            safe_results[m] = {"error": r["error"], "passed": r["passed"]}
        else:
            safe_results[m] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                              for k, v in r.items()}
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "npu": torch_npu.npu.get_device_name(0),
        "tolerance_pct": 1.0,
        "method": "logit-level comparison (deterministic)",
        "results": safe_results,
        "all_pass": all_ok,
    }
    with open("verification_report_deterministic.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nReport: verification_report_deterministic.json")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
