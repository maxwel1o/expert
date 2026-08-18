"""
Qwen3-ForcedAligner-0.6B Inference on Ascend NPU
==================================================
Usage:
    python inference_aligner.py --audio <path> --text <transcript> --language <lang>
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

from qwen_asr import Qwen3ForcedAligner


def load_model(model_path, device="npu:0"):
    model = Qwen3ForcedAligner.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=None,
    )
    model.model = model.model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ForcedAligner-0.6B NPU Inference")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--text", required=True, help="Transcript text for alignment")
    parser.add_argument("--language", required=True, help="Language (e.g. 'Chinese', 'English')")
    parser.add_argument("--model-path", default="./Qwen3-ForcedAligner-0.6B",
                        help="Model path or HF repo ID")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path)

    print(f"Aligning {args.audio} ...")
    results = model.align(audio=args.audio, text=args.text, language=args.language)

    print(f"\nTimestamps ({len(results[0])} tokens):")
    for item in results[0]:
        print(f"  {item.text:>8s}: {item.start_time:.3f}s - {item.end_time:.3f}s")


if __name__ == "__main__":
    main()
