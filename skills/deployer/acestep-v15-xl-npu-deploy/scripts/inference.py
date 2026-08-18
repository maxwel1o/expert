#!/usr/bin/env python3
"""ACE-Step v15 XL NPU inference script."""
import argparse
import torch
import torch_npu
from transformers import AutoConfig, AutoModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="npu", choices=["npu", "cpu"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mode", default="both", choices=["loss", "gen", "both"])
    parser.add_argument("--infer-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if args.device == "npu":
        model = model.npu()
    model.eval()

    inputs = {
        "text_hidden_states": torch.randn(1, 77, 1024, dtype=torch.bfloat16, device=args.device),
        "text_attention_mask": torch.ones(1, 77, dtype=torch.bfloat16, device=args.device),
        "lyric_hidden_states": torch.randn(1, 123, 1024, dtype=torch.bfloat16, device=args.device),
        "lyric_attention_mask": torch.ones(1, 123, dtype=torch.bfloat16, device=args.device),
        "refer_audio_acoustic_hidden_states_packed": torch.randn(2, 750, 64, dtype=torch.bfloat16, device=args.device),
        "refer_audio_order_mask": torch.LongTensor([0, 0]).to(args.device),
        "src_latents": torch.randn(1, 250, 64, dtype=torch.bfloat16, device=args.device),
        "chunk_masks": torch.ones(1, 250, 64, dtype=torch.bfloat16, device=args.device),
        "is_covers": torch.tensor([0], dtype=torch.long, device=args.device),
    }

    with torch.no_grad():
        if args.mode in ("loss", "both"):
            outputs = model.training_losses(
                hidden_states=inputs["src_latents"],
                attention_mask=torch.ones(1, 250, dtype=torch.bfloat16, device=args.device),
                chunk_masks=inputs["chunk_masks"],
                silence_latent=torch.randn(1, 250, 64, dtype=torch.bfloat16, device=args.device),
                cfg_ratio=0.15,
                **inputs,
            )
            loss = outputs["diffusion_loss"]
            print(f"Loss: {loss.item():.6f}")

        if args.mode in ("gen", "both"):
            gen_outputs = model.generate_audio(
                silence_latent=torch.randn(1, 250, 64, dtype=torch.bfloat16, device=args.device),
                seed=args.seed,
                infer_steps=args.infer_steps,
                diffusion_guidance_scale=args.guidance_scale,
                **inputs,
            )
            latents = gen_outputs["target_latents"]
            print(f"Generated latents shape: {latents.shape}")

    torch.npu.empty_cache()
    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
