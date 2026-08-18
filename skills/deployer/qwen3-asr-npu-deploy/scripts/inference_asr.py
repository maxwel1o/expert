"""
Qwen3-ASR-0.6B Inference on Ascend NPU
========================================
Usage:
    python inference_asr.py --audio <path> [--model-path <path>] [--language <lang>]
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import torch, torch_npu

# Step 1: Register custom model classes
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

from qwen_asr import Qwen3ASRModel


def load_model(model_path, device="npu:0"):
    """Load model on NPU (device_map=None + manual .to() because
    transformers does not support device_map='npu:0')."""
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=None, low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    asr = Qwen3ASRModel(
        backend="transformers", model=model, processor=processor,
        max_inference_batch_size=-1, max_new_tokens=512,
    )
    asr.device = torch.device(device)
    asr.dtype = torch.bfloat16
    return asr


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR-0.6B NPU Inference")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--model-path", default="./Qwen3-ASR-0.6B",
                        help="Model path or HF repo ID")
    parser.add_argument("--language", default=None,
                        help="Language hint (e.g. 'Chinese', 'English'). Auto-detect if omitted.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path)

    print(f"Transcribing {args.audio} ...")
    results = model.transcribe(audio=args.audio, language=args.language)

    for r in results:
        print(f"\nLanguage: {r.language}")
        print(f"Text:     {r.text}")


if __name__ == "__main__":
    main()
