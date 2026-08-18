#!/usr/bin/env python3
"""
Granite Speech NPU Performance Benchmark
Measures forward pass time, throughput, and accuracy for both models.
"""
import json
import os
import sys
import time
import torch
import torch_npu
import numpy as np


def setup_env():
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TASK_QUEUE_ENABLE'] = '1'


def get_npu_info():
    """Get NPU device info."""
    info = {
        'npu_count': torch.npu.device_count(),
        'npu_name': torch.npu.get_device_name(0) if torch.npu.is_available() else 'N/A',
        'npu_memory': torch.npu.get_device_properties(0).total_memory if torch.npu.is_available() else 0,
    }
    # Try npu-smi
    try:
        import subprocess
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=5)
        info['npu_smi'] = result.stdout
    except:
        info['npu_smi'] = 'N/A'
    return info


def load_model(model_path):
    """Load model on NPU."""
    import json as _json
    with open(os.path.join(model_path, 'config.json')) as f:
        cfg = _json.load(f)

    model_type = cfg.get('model_type', '')
    model_name = cfg.get('_name_or_path', model_path)

    if model_type == 'granite_speech_plus':
        sys.path.insert(0, '/opt/atomgit')
        from granite_speech_plus.register import register
        register()
        from granite_speech_plus import GraniteSpeechPlusForConditionalGeneration as ModelClass
    else:
        from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration as ModelClass

    dtype = torch.bfloat16
    model = ModelClass.from_pretrained(model_path, dtype=dtype, low_cpu_mem_usage=True)
    model = model.npu()
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model: {model_name}")
    print(f"  Type: {type(model).__name__}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Device: {next(model.parameters()).device}")

    return model, model_type


def benchmark_forward(model, input_seq_lens=[32, 64, 128], audio_frames=200, num_runs=50):
    """Benchmark forward pass at different sequence lengths."""
    device = next(model.parameters()).device
    vocab_size = model.config.text_config.vocab_size
    results = {}

    for seq_len in input_seq_lens:
        print(f"\n  Benchmarking seq_len={seq_len}...")

        input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len), device=device)
        audio_start = seq_len // 3
        audio_end = audio_start + 5
        input_ids[0, audio_start:audio_end] = model.config.audio_token_id

        input_features = torch.randn(1, audio_frames, 160, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids=input_ids, input_features=input_features)

        # Benchmark
        torch.npu.synchronize()
        times = []
        for _ in range(num_runs):
            torch.npu.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = model(input_ids=input_ids, input_features=input_features)
            torch.npu.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000)

        avg_ms = np.mean(times)
        median_ms = np.median(times)
        p99_ms = np.percentile(times, 99)
        std_ms = np.std(times)

        # Throughput: tokens per second (just forward pass)
        tokens_per_sec = seq_len / (avg_ms / 1000)

        results[seq_len] = {
            'avg_ms': round(avg_ms, 2),
            'median_ms': round(median_ms, 2),
            'p99_ms': round(p99_ms, 2),
            'std_ms': round(std_ms, 2),
            'tokens_per_sec': round(tokens_per_sec, 2),
            'output_logits_shape': list(out.logits.shape),
        }
        print(f"    Avg: {avg_ms:.2f}ms | Median: {median_ms:.2f}ms | P99: {p99_ms:.2f}ms | "
              f"Throughput: {tokens_per_sec:.2f} tok/s")

    return results


def benchmark_generate(model, input_seq_lens=[32, 64], max_new_tokens=[32, 64], audio_frames=200):
    """Benchmark generation (autoregressive decode)."""
    device = next(model.parameters()).device
    vocab_size = model.config.text_config.vocab_size
    results = {}

    for seq_len in input_seq_lens:
        for new_tokens in max_new_tokens:
            print(f"\n  Benchmarking generate: input={seq_len}, output={new_tokens}...")

            input_ids = torch.randint(0, min(vocab_size, 1000), (1, seq_len), device=device)
            input_ids[0, seq_len//3:seq_len//3+5] = model.config.audio_token_id
            attention_mask = torch.ones(1, seq_len, device=device)
            input_features = torch.randn(1, audio_frames, 160, dtype=torch.bfloat16, device=device)

            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    _ = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_features=input_features,
                        max_new_tokens=new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )

            # Benchmark
            torch.npu.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                    max_new_tokens=new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            torch.npu.synchronize()
            elapsed = time.time() - t0

            output_len = out.shape[1] - seq_len
            key = f"in{seq_len}_out{new_tokens}"
            results[key] = {
                'total_time_s': round(elapsed, 3),
                'output_tokens': output_len,
                'tokens_per_sec': round(output_len / elapsed, 2),
            }
            print(f"    Time: {elapsed:.3f}s | Output: {output_len} tok | "
                  f"Speed: {output_len/elapsed:.2f} tok/s")

    return results


def benchmark():
    """Run full benchmark for both models."""
    setup_env()
    results = {}

    base_path = '/opt/atomgit/granite-speech-4.1-2b'
    plus_path = '/opt/atomgit/granite-speech-4.1-2b-plus'

    # NPU info
    results['npu_info'] = get_npu_info()
    print(f"NPU: {results['npu_info']['npu_name']} x{results['npu_info']['npu_count']}")

    for model_name, model_path in [("granite-speech-4.1-2b", base_path),
                                    ("granite-speech-4.1-2b-plus", plus_path)]:
        print(f"\n{'='*60}")
        print(f"Benchmark: {model_name}")
        print('='*60)

        model, model_type = load_model(model_path)
        model_results = {}

        # Forward pass benchmark
        print(f"\n--- Forward Pass Benchmark ---")
        model_results['forward'] = benchmark_forward(
            model, input_seq_lens=[32, 64, 128], num_runs=50
        )

        # Generation benchmark
        print(f"\n--- Generation Benchmark ---")
        model_results['generate'] = benchmark_generate(
            model, input_seq_lens=[32, 64], max_new_tokens=[32, 64]
        )

        # Memory usage
        torch.npu.synchronize()
        memory_allocated = torch.npu.memory_allocated()
        memory_reserved = torch.npu.memory_reserved()
        model_results['memory'] = {
            'allocated_mb': round(memory_allocated / 1024**2, 2),
            'reserved_mb': round(memory_reserved / 1024**2, 2),
        }
        print(f"\n--- Memory ---")
        print(f"  Allocated: {memory_allocated/1024**2:.2f} MB")
        print(f"  Reserved: {memory_reserved/1024**2:.2f} MB")

        results[model_name] = model_results
        torch.npu.empty_cache()

    # Save results
    output_path = '/opt/atomgit/deliverables/benchmark_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    benchmark()
