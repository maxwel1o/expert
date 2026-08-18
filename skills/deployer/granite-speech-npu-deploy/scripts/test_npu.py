"""
Granite Speech NPU Adaptation - Verification Script
Tests both base and plus models on Ascend NPU with accuracy validation.
"""
import argparse
import os
import sys
import torch
import torch_npu


def setup_env():
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'


def test_base_model(device='cpu'):
    """Test granite-speech-4.1-2b base model."""
    print(f"\n{'='*60}")
    print(f"Testing base model on {device}")
    print('='*60)

    from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration
    model_path = '/opt/atomgit/granite-speech-4.1-2b'

    dtype = torch.bfloat16
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")

    # Forward pass test
    batch_size, seq_len = 1, 32
    vocab_size = model.config.text_config.vocab_size

    torch.manual_seed(42)
    input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len), device=device)
    input_ids[0, 10:15] = model.config.audio_token_id
    input_features = torch.randn(batch_size, 200, 160, dtype=dtype, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, input_features=input_features)

    print(f"  Output logits shape: {outputs.logits.shape}")
    print(f"  Output logits range: [{outputs.logits.min().item():.4f}, {outputs.logits.max().item():.4f}]")
    print(f"  Output logits mean: {outputs.logits.mean().item():.6f}")
    print(f"  Output logits std: {outputs.logits.std().item():.6f}")
    print(f"  ✅ Base model on {device} - SUCCESS")
    return outputs.logits


def test_plus_model(device='cpu'):
    """Test granite-speech-4.1-2b-plus model."""
    print(f"\n{'='*60}")
    print(f"Testing plus model on {device}")
    print('='*60)

    sys.path.insert(0, '/opt/atomgit')
    from granite_speech_plus.register import register
    register()
    from granite_speech_plus import GraniteSpeechPlusForConditionalGeneration

    model_path = '/opt/atomgit/granite-speech-4.1-2b-plus'

    dtype = torch.bfloat16
    model = GraniteSpeechPlusForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    print(f"  Encoder: {type(model.encoder).__name__}")
    print(f"  cat_hidden_layers: {model.encoder.cat_hidden_layers}")

    # Forward pass test
    batch_size, seq_len = 1, 32
    vocab_size = model.config.text_config.vocab_size

    torch.manual_seed(42)
    input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len), device=device)
    input_ids[0, 10:15] = model.config.audio_token_id
    input_features = torch.randn(batch_size, 200, 160, dtype=dtype, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, input_features=input_features)

    print(f"  Output logits shape: {outputs.logits.shape}")
    print(f"  Output logits range: [{outputs.logits.min().item():.4f}, {outputs.logits.max().item():.4f}]")
    print(f"  Output logits mean: {outputs.logits.mean().item():.6f}")
    print(f"  Output logits std: {outputs.logits.std().item():.6f}")
    print(f"  ✅ Plus model on {device} - SUCCESS")
    return outputs.logits


def compare_cpu_npu(model_name, cpu_logits, npu_logits):
    """Compare CPU vs NPU logits with tolerance check."""
    print(f"\n  --- Accuracy comparison: {model_name} ---")

    cpu_logits = cpu_logits.float().cpu()
    npu_logits = npu_logits.float().cpu()

    # Compute error metrics
    abs_diff = (cpu_logits - npu_logits).abs()
    rel_diff = abs_diff / (cpu_logits.abs() + 1e-10)

    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()
    max_rel_err = rel_diff.max().item() * 100
    mean_rel_err = rel_diff.mean().item() * 100

    print(f"    Max absolute error: {max_abs_err:.6f}")
    print(f"    Mean absolute error: {mean_abs_err:.6f}")
    print(f"    Max relative error: {max_rel_err:.4f}%")
    print(f"    Mean relative error: {mean_rel_err:.4f}%")

    # Cosine similarity of entire logits
    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_logits.flatten().unsqueeze(0),
        npu_logits.flatten().unsqueeze(0)
    ).item()
    print(f"    Cosine similarity: {cos_sim:.8f}")

    if max_rel_err < 1.0:
        print(f"    ✅ {model_name}: Accuracy PASS (max_rel_err={max_rel_err:.4f}% < 1%)")
        return True
    else:
        print(f"    ❌ {model_name}: Accuracy FAIL (max_rel_err={max_rel_err:.4f}% >= 1%)")
        return False


def test_accuracy():
    """Run accuracy comparison for both models."""
    print("=" * 60)
    print("Granite Speech NPU Accuracy Verification")
    print("=" * 60)

    # Test base model on CPU and NPU
    if torch.cuda.is_available():
        cpu_device = 'cuda'
    else:
        cpu_device = 'cpu'

    # We use float32 on CPU for reference, bfloat16 on NPU
    # Actually for fair comparison, use the same dtype
    print("\n[Phase 1] Testing base model accuracy...")

    cpu_logits_base = test_base_model('cpu')
    npu_logits_base = test_base_model('npu')

    base_ok = compare_cpu_npu("granite-speech-4.1-2b", cpu_logits_base, npu_logits_base)

    # Test plus model on CPU and NPU
    print("\n[Phase 2] Testing plus model accuracy...")

    cpu_logits_plus = test_plus_model('cpu')
    npu_logits_plus = test_plus_model('npu')

    plus_ok = compare_cpu_npu("granite-speech-4.1-2b-plus", cpu_logits_plus, npu_logits_plus)

    print(f"\n{'='*60}")
    print("Results:")
    print(f"  granite-speech-4.1-2b:      {'✅ PASS' if base_ok else '❌ FAIL'}")
    print(f"  granite-speech-4.1-2b-plus:  {'✅ PASS' if plus_ok else '❌ FAIL'}")
    print('='*60)

    return base_ok and plus_ok


def test_smoke():
    """Quick smoke test - run both models on NPU only."""
    print("=" * 60)
    print("Granite Speech NPU Smoke Test")
    print("=" * 60)

    test_base_model('npu')
    test_plus_model('npu')

    print("\n✅ All smoke tests passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'accuracy'], default='smoke')
    parser.add_argument('--model', choices=['base', 'plus', 'all'], default='all')
    args = parser.parse_args()

    setup_env()

    if args.mode == 'accuracy':
        success = test_accuracy()
        sys.exit(0 if success else 1)
    else:
        test_smoke()
