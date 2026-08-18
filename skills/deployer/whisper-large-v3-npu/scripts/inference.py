#!/usr/bin/env python3
"""
whisper-large-v3 昇腾 NPU 推理脚本
支持音频转文本 (transcribe) 和翻译 (translate) 两种模式。

用法:
    python inference.py --audio input.wav
    python inference.py --audio input.wav --task translate --language zh
    python inference.py --audio input.wav --max_tokens 256 --model /path/to/model
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["HCCL_CONNECT_TIMEOUT"] = "1800"

import numpy as np
import soundfile as sf
import torch
import torch_npu  # noqa: F401 - 注册 NPU 设备


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """加载音频文件并重采样到目标采样率."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in (".wav", ".flac", ".ogg"):
        audio, sr = sf.read(audio_path)
    else:
        import librosa
        audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def check_env():
    """检查 NPU 环境是否可用."""
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU 不可用，请检查 CANN 和 torch_npu 安装。")
    print(f"[环境] NPU: {torch.npu.get_device_name(0)}")
    print(f"[环境] PyTorch: {torch.__version__}")
    print(f"[环境] torch_npu: {torch_npu.__file__.split('site-packages/')[1]}")


def main():
    parser = argparse.ArgumentParser(
        description="whisper-large-v3 Ascend NPU 推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python inference.py --audio speech.wav
  python inference.py --audio speech.wav --task translate --language zh
  python inference.py --audio speech.mp3 --language en
        """,
    )
    parser.add_argument("--audio", required=True, help="输入音频文件路径 (wav/flac/mp3/ogg)")
    parser.add_argument(
        "--model",
        default="/opt/atomgit/whisper-large-v3",
        help="whisper-large-v3 模型目录路径",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="任务类型: transcribe(转录) 或 translate(翻译为英文)",
    )
    parser.add_argument("--language", default="en", help="源语言代码 (en/zh/ja/ko/...)")
    parser.add_argument("--max_tokens", type=int, default=256, help="最大输出 token 数")
    parser.add_argument(
        "--attn",
        default="eager",
        choices=["eager", "sdpa"],
        help="注意力实现方式 (NPU 推荐 eager)",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在: {args.audio}")
        sys.exit(1)

    print("=" * 60)
    print("Whisper-large-v3 Ascend NPU 推理")
    print("=" * 60)
    check_env()

    # 加载模型
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = "npu"

    if not os.path.exists(args.model):
        # Try HF cache fallback
        import glob
        candidates = glob.glob("/opt/atomgit/.cache/huggingface/models--*whisper*--*/snapshots/*/")
        if candidates:
            args.model = candidates[0]
            print(f"[模型] 自动发现: {args.model}")
        else:
            print(f"错误: 模型路径不存在: {args.model}")
            print("请从以下地址下载模型:")
            print("  https://gitcode.com/hf_mirrors/openai/whisper-large-v3")
            sys.exit(1)

    print(f"[模型] 加载: {args.model}")
    t0 = time.time()
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    print(f"[模型] 加载完成 ({time.time() - t0:.1f}s)")

    processor = WhisperProcessor.from_pretrained(args.model)

    # 加载并处理音频
    print(f"[音频] 加载: {args.audio}")
    audio = load_audio(args.audio)
    print(f"[音频] {len(audio)} 采样点, {len(audio)/16000:.1f}s, 最大振幅 {audio.max():.3f}")

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=dtype)
    print(f"[音频] 梅尔特征: {input_features.shape}")

    # 推理
    print(f"[推理] 任务: {args.task}, 语言: {args.language}")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_features,
            max_new_tokens=args.max_tokens,
            task=args.task,
            language=args.language,
        )
    torch.npu.synchronize()
    elapsed = time.time() - t0

    text = processor.decode(outputs[0], skip_special_tokens=True)
    n_tokens = len(outputs[0])

    print()
    print("=" * 60)
    print(f"结果: {text}")
    print("=" * 60)
    print()
    print(f"[性能] 耗时: {elapsed:.2f}s")
    print(f"[性能] Token 数: {n_tokens}")
    print(f"[性能] 速度: {1000 * elapsed / max(n_tokens, 1):.1f}ms/token")

    return text


if __name__ == "__main__":
    main()
