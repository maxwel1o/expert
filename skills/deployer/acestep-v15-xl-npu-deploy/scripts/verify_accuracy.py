#!/usr/bin/env python3
"""ACE-Step v15 XL NPU vs CPU accuracy verification script."""
import argparse
import torch
import torch_npu
from transformers import AutoConfig, AutoModel

def load_model(model_path, device, dtype):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    return model

def run_forward(model, inputs, silence_latent, device):
    model_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    silence = silence_latent.to(device)
    with torch.no_grad():
        outputs = model.training_losses(
            hidden_states=model_inputs["src_latents"],
            attention_mask=torch.ones(1, 250, dtype=torch.bfloat16, device=device),
            chunk_masks=model_inputs["chunk_masks"],
            silence_latent=silence,
            cfg_ratio=0.15,
            **model_inputs,
        )
    return outputs["diffusion_loss"].item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    inputs = {
        "text_hidden_states": torch.randn(1, 77, 1024, dtype=torch.bfloat16),
        "text_attention_mask": torch.ones(1, 77, dtype=torch.bfloat16),
        "lyric_hidden_states": torch.randn(1, 123, 1024, dtype=torch.bfloat16),
        "lyric_attention_mask": torch.ones(1, 123, dtype=torch.bfloat16),
        "refer_audio_acoustic_hidden_states_packed": torch.randn(2, 750, 64, dtype=torch.bfloat16),
        "refer_audio_order_mask": torch.LongTensor([0, 0]),
        "src_latents": torch.randn(1, 250, 64, dtype=torch.bfloat16),
        "chunk_masks": torch.ones(1, 250, 64, dtype=torch.bfloat16),
        "is_covers": torch.tensor([0], dtype=torch.long),
    }
    silence_latent = torch.randn(1, 250, 64, dtype=torch.bfloat16)

    # NPU inference
    print(f"Loading model on NPU from {args.model_path}...")
    model_npu = load_model(args.model_path, "npu", torch.bfloat16)
    loss_npu = run_forward(model_npu, inputs, silence_latent, "npu")
    print(f"NPU Loss: {loss_npu:.6f}")

    del model_npu
    torch.npu.empty_cache()

    if args.skip_cpu:
        print(f"NPU inference OK (--skip-cpu, no CPU comparison)")
        print("Result: PASSED (NPU only)")
        return

    # CPU inference
    print(f"Loading model on CPU from {args.model_path}...")
    model_cpu = load_model(args.model_path, "cpu", torch.bfloat16)
    loss_cpu = run_forward(model_cpu, inputs, silence_latent, "cpu")
    print(f"CPU Loss: {loss_cpu:.6f}")

    rel_err = abs(loss_npu - loss_cpu) / (abs(loss_cpu) + 1e-8) * 100
    print(f"Relative Error: {rel_err:.4f}%")
    print(f"Threshold: {args.threshold}%")

    if rel_err <= args.threshold:
        print("Result: PASSED")
    else:
        print("Result: FAILED")
        exit(1)

if __name__ == "__main__":
    main()
