#!/usr/bin/env python3
"""
Granite Speech 4.1 2B - NPU Inference Script
Supports both base (granite-speech-4.1-2b) and plus (granite-speech-4.1-2b-plus) variants.
"""
import argparse
import os
import sys
import time
import torch
import torch_npu
import numpy as np


def setup_env():
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TASK_QUEUE_ENABLE'] = '1'


def load_model(model_path, dtype=torch.bfloat16):
    """Load model on NPU."""
    from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
    from transformers import AutoProcessor

    # Check if this is the plus variant
    import json
    with open(os.path.join(model_path, 'config.json')) as f:
        cfg = json.load(f)

    model_type = cfg.get('model_type', '')
    if model_type == 'granite_speech_plus':
        sys.path.insert(0, '/opt/atomgit')
        from granite_speech_plus.register import register
        register()
        from granite_speech_plus import GraniteSpeechPlusForConditionalGeneration as ModelClass
    else:
        ModelClass = GraniteSpeechForConditionalGeneration

    print(f"Loading model from {model_path}...")
    model = ModelClass.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True,
    )
    model = model.npu()
    model.eval()
    print(f"Model loaded on NPU: {next(model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"Processor: {type(processor).__name__}")

    return model, processor


def preprocess_audio(audio_path, processor, device='npu'):
    """Load audio and extract features."""
    import soundfile as sf

    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    print(f"  Audio shape: {audio.shape}, sample_rate: {sr}")

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Extract features
    inputs = processor(audio=audio, sampling_rate=16000, return_tensors='pt')
    input_features = inputs['input_features'].to(dtype=torch.bfloat16).to(device)
    print(f"  Input features shape: {input_features.shape}")

    return input_features


def prepare_text_input(text, processor, model_config, device='npu'):
    """Prepare text input with audio token placeholder."""
    audio_token = model_config.audio_token_id

    # Tokenize text
    encoded = processor.tokenizer(text, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Replace audio token placeholder (<|audio|>) with actual audio token ID
    # The tokenizer will have its own representation for <|audio|>
    return input_ids, attention_mask


def generate_with_audio(model, processor, text, input_features, max_new_tokens=128):
    """Generate text conditioned on audio input."""
    # Format the prompt - the text should contain <|audio|> token
    # Insert audio features into the model
    device = next(model.parameters()).device

    # Tokenize the text prompt
    tokens = processor.tokenizer(text, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Replace audio token placeholder with the actual audio token ID
    audio_token_str = processor.audio_token if hasattr(processor, 'audio_token') else '<|audio|>'
    audio_token_id = model.config.audio_token_id

    # Check if tokenizer has audio token
    audio_tok_id = processor.tokenizer.convert_tokens_to_ids(audio_token_str)
    if audio_tok_id != model.config.audio_token_id:
        # Map the tokenizer's audio token ID to the model's audio token ID
        input_ids = torch.where(input_ids == audio_tok_id, audio_token_id, input_ids)

    # Generate
    print(f"Generating (max {max_new_tokens} tokens)...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    elapsed = time.time() - start_time
    generated = outputs[0][input_ids.shape[1]:]

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Generated {len(generated)} tokens in {elapsed:.2f}s ({len(generated)/elapsed:.2f} tok/s)")
    print(f"Output: {text}")

    return text


def forward_pass_benchmark(model, input_ids, input_features, num_warmup=5, num_runs=50):
    """Benchmark forward pass performance."""
    print(f"\nBenchmarking forward pass ({num_runs} runs, {num_warmup} warmup)...")

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_ids=input_ids, input_features=input_features)

    # Benchmark
    torch.npu.synchronize()
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids, input_features=input_features)
    torch.npu.synchronize()
    elapsed = time.time() - start

    avg_time = elapsed / num_runs * 1000  # ms
    print(f"  Average forward pass time: {avg_time:.2f} ms")
    print(f"  Total time: {elapsed:.2f} s")

    return {
        'avg_forward_ms': avg_time,
        'total_time_s': elapsed,
        'num_runs': num_runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/opt/atomgit/granite-speech-4.1-2b',
                        help='Path to model directory')
    parser.add_argument('--audio-path', default=None,
                        help='Path to audio file (WAV)')
    parser.add_argument('--text', default='<|audio|> Transcribe the audio.',
                        help='Input text with <|audio|> placeholder')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16'])
    args = parser.parse_args()

    setup_env()
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    model, processor = load_model(args.model_path, dtype=dtype)

    if args.benchmark:
        # Create dummy inputs for benchmark
        batch_size, seq_len = 1, 64
        vocab_size = model.config.text_config.vocab_size
        device = next(model.parameters()).device
        input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len), device=device)
        input_ids[0, 20:25] = model.config.audio_token_id
        input_features = torch.randn(batch_size, 200, 160, dtype=dtype, device=device)

        results = forward_pass_benchmark(model, input_ids, input_features)
        print(f"\nResults: {results}")

    if args.audio_path:
        input_features = preprocess_audio(args.audio_path, processor)
        generate_with_audio(model, processor, args.text, input_features,
                          max_new_tokens=args.max_new_tokens)


if __name__ == '__main__':
    main()
