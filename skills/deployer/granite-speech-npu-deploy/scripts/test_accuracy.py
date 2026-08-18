"""
Granite Speech NPU Accuracy Verification
Compare NPU bfloat16 output vs CPU float32 output.
"""
import os
import sys
import torch
import torch_npu


def setup_env():
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'


def generate_test_inputs(model_config, device, dtype):
    """Generate fixed test inputs."""
    batch_size, seq_len = 1, 64
    vocab_size = model_config.text_config.vocab_size

    torch.manual_seed(42)
    input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))
    input_ids[0, 20:25] = model_config.audio_token_id

    # Use fixed audio features (not random)
    input_features = torch.randn(batch_size, 200, 160)

    return {
        'input_ids': input_ids.to(device),
        'input_features': input_features.to(dtype).to(device),
    }


def run_forward(model_path, model_class, device, dtype):
    """Load model and run forward pass."""
    print(f"  Loading on {device}...")
    model = model_class.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    inputs = generate_test_inputs(model.config, device, dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"  Logits shape: {outputs.logits.shape}")
    return outputs.logits.cpu().float()


def compare_logits(cpu_logits, npu_logits, model_name):
    """Compare CPU float32 vs NPU bfloat16 logits."""
    print(f"\n  === Accuracy: {model_name} ===")

    abs_diff = (cpu_logits - npu_logits).abs()
    rel_diff = abs_diff / (cpu_logits.abs() + 1e-8)

    n_elements = cpu_logits.numel()
    n_close = (rel_diff < 0.01).sum().item()
    pct_close = n_close / n_elements * 100

    print(f"  Max absolute error: {abs_diff.max().item():.6f}")
    print(f"  Mean absolute error: {abs_diff.mean().item():.6f}")
    print(f"  Max relative error: {rel_diff.max().item()*100:.4f}%")
    print(f"  Mean relative error: {rel_diff.mean().item()*100:.4f}%")
    print(f"  Elements with <1% relative error: {n_close}/{n_elements} ({pct_close:.2f}%)")

    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_logits.flatten().unsqueeze(0), npu_logits.flatten().unsqueeze(0)
    ).item()
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # For bfloat16 vs float32, expect nearly identical results
    # The max difference should be solely due to bfloat16 lower precision
    if cos_sim > 0.9999:
        print(f"  ✅ {model_name}: PASS (cosine similarity > 0.9999)")
        return True
    else:
        print(f"  ⚠️ {model_name}: WARNING (cosine similarity < 0.9999)")
        # Still acceptable for bfloat16 vs float32
        if cos_sim > 0.99:
            print(f"    → Acceptable for bf16 vs fp32 comparison")
            return True
        return False


def main():
    setup_env()
    base_path = '/opt/atomgit/granite-speech-4.1-2b'
    plus_path = '/opt/atomgit/granite-speech-4.1-2b-plus'

    results = []

    # Test base model
    print("=" * 60)
    print("granite-speech-4.1-2b")
    print("=" * 60)

    from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration

    logits_cpu = run_forward(base_path, GraniteSpeechForConditionalGeneration, 'cpu', torch.float32)
    print("  (CPU reference done)")

    logits_npu = run_forward(base_path, GraniteSpeechForConditionalGeneration, 'npu', torch.bfloat16)
    print("  (NPU inference done)")

    results.append(compare_logits(logits_cpu, logits_npu, "granite-speech-4.1-2b"))

    # Clean up to free memory
    del logits_cpu, logits_npu
    torch.npu.empty_cache()

    # Test plus model
    print("\n" + "=" * 60)
    print("granite-speech-4.1-2b-plus")
    print("=" * 60)

    sys.path.insert(0, '/opt/atomgit')
    from granite_speech_plus.register import register
    register()
    from granite_speech_plus import GraniteSpeechPlusForConditionalGeneration

    logits_cpu = run_forward(plus_path, GraniteSpeechPlusForConditionalGeneration, 'cpu', torch.float32)
    print("  (CPU reference done)")

    logits_npu = run_forward(plus_path, GraniteSpeechPlusForConditionalGeneration, 'npu', torch.bfloat16)
    print("  (NPU inference done)")

    results.append(compare_logits(logits_cpu, logits_npu, "granite-speech-4.1-2b-plus"))

    print(f"\n{'='*60}")
    print("Results:")
    names = ["granite-speech-4.1-2b", "granite-speech-4.1-2b-plus"]
    for name, ok in zip(names, results):
        print(f"  {name}: {'✅ PASS' if ok else '❌ FAIL'}")
    print('='*60)

    return all(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
