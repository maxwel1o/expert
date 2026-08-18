"""
OpenJourney 昇腾 NPU 精度验证脚本
测试 NPU 确定性与单步 UNet 精度
"""
import torch
import torch_npu
import argparse


def test_determinism(pipe):
    """NPU 确定性: 同 seed 应产生像素完全一致的图像"""
    prompt = "mdjrnm-syle portrait photo of a girl"
    gen1 = torch.Generator(device="npu").manual_seed(42)
    img1 = pipe(prompt, num_inference_steps=5, generator=gen1).images[0]

    gen2 = torch.Generator(device="npu").manual_seed(42)
    img2 = pipe(prompt, num_inference_steps=5, generator=gen2).images[0]

    import numpy as np
    diff = np.abs(np.array(img1).astype(np.float32) - np.array(img2).astype(np.float32))
    max_diff = diff.max()
    print(f"Determinism test: max pixel diff = {max_diff}")
    print(f"  {'PASS' if max_diff == 0 else 'FAIL'}")
    return max_diff == 0


def test_unet_accuracy():
    """NPU float32 vs CPU float32 单步 UNet 精度"""
    from diffusers import StableDiffusionPipeline

    # NPU forward
    pipe_npu = StableDiffusionPipeline.from_pretrained(
        "./models/openjourney",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("npu")

    torch.manual_seed(42)
    latent = torch.randn(1, 4, 64, 64).to("npu")
    timestep = torch.tensor([5]).to("npu")
    enc = torch.randn(1, 77, 768).to("npu")

    with torch.no_grad():
        out_npu = pipe_npu.unet(latent, timestep, encoder_hidden_states=enc).sample.cpu()
    del pipe_npu
    torch.npu.empty_cache()

    # CPU forward
    import diffusers.utils.import_utils as iu
    iu.is_torch_npu_available = lambda: False

    pipe_cpu = StableDiffusionPipeline.from_pretrained(
        "./models/openjourney",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cpu")

    torch.manual_seed(42)
    latent_cpu = torch.randn(1, 4, 64, 64)
    enc_cpu = torch.randn(1, 77, 768)

    with torch.no_grad():
        out_cpu = pipe_cpu.unet(latent_cpu, timestep, encoder_hidden_states=enc_cpu).sample
    del pipe_cpu

    diff = (out_npu - out_cpu).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()
    signal = out_cpu.abs().mean().item()
    rel_err = mae / max(signal, 1e-8) * 100

    print(f"\nSingle-step UNet accuracy (NPU fp32 vs CPU fp32):")
    print(f"  MAE:       {mae:.8f}")
    print(f"  Max Err:   {max_err:.8f}")
    print(f"  Rel Err:   {rel_err:.4f}%")
    print(f"  Threshold: < 1%")
    print(f"  {'PASS' if rel_err < 1.0 else 'FAIL'}")

    return rel_err


def main():
    parser = argparse.ArgumentParser(description="Accuracy verification")
    parser.add_argument("--model", default="./models/openjourney", help="Model path")
    args = parser.parse_args()

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("npu")

    test_determinism(pipe)
    del pipe
    torch.npu.empty_cache()
    test_unet_accuracy()

    print("\nDone.")


if __name__ == "__main__":
    main()
