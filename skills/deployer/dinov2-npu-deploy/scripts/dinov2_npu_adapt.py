"""
vit_base_patch14_dinov2.lvd142m —— Ascend NPU 模型适配

功能:
  - 基于 transformers + torch_npu 在昇腾 NPU 上跑通 DINOv2 模型推理
  - CPU vs NPU 精度对比验证 (余弦相似度 + 元素级绝对误差)
  - 支持单张图片推理和批量推理

环境:
  - torch >= 2.0, torch_npu >= 2.0
  - transformers >= 4.31
  - Pillow / opencv-python (可选,用于真实图片加载)
"""

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from transformers import AutoModel, AutoImageProcessor

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_PATH = "/opt/atomgit/.cache/modelscope/facebook/dinov2-base"
# 也可从本地路径加载:
# MODEL_PATH = "/path/to/your/vit_base_patch14_dinov2.lvd142m"

IMAGE_SIZE = 518          # DINOv2 原始输入尺寸
PATCH_SIZE = 14           # ViT patch size
HIDDEN_SIZE = 768         # embedding 维度
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 1369

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


# ---------------------------------------------------------------------------
# 模型封装
# ---------------------------------------------------------------------------
class Dinov2NPUModel(nn.Module):
    """DINOv2 模型 NPU 推理封装，支持 fp32 / fp16 精度。"""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        dtype: str = "fp32",
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "npu:0" if torch.npu.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]

        print(f"[Dinov2NPU] Loading model from {model_path}")
        print(f"[Dinov2NPU] Device: {self.device}, dtype: {dtype}")

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        # register ViT config attributes for external access
        self.model.config.hidden_size = self.model.config.hidden_size
        self.model.eval()
        self.model.to(self.device)
        if self.dtype != torch.float32:
            self.model = self.model.to(self.dtype)

        self.processor = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.embed_dim = self.model.config.hidden_size
        print(f"[Dinov2NPU] Embed dim: {self.embed_dim}, params: {sum(p.numel() for p in self.model.parameters()):,}")

    @torch.no_grad()
    def preprocess(self, images):
        """预处理图片 -> NPU tensor."""
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        return pixel_values

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        返回:
          - embedding: [B, D] CLS token embedding
          - patch_embeddings: [B, N+1, D] 全部 token (CLS + patch)
        """
        outputs = self.model(pixel_values=pixel_values)
        emb = outputs.last_hidden_state[:, 0]           # CLS token
        return {"embedding": emb.cpu(), "patch_embeddings": outputs.last_hidden_state.cpu()}

    @torch.no_grad()
    def infer(self, images) -> dict:
        """预处理 + 推理 一步完成."""
        pixel_values = self.preprocess(images)
        return self.forward(pixel_values)


# ---------------------------------------------------------------------------
# 精度验证
# ---------------------------------------------------------------------------
def precision_report(cpu_out: torch.Tensor, npu_out: torch.Tensor, tag: str = "", dtype: str = "fp32"):
    """输出 CPU vs NPU 精度对比报告。"""
    cpu_out = cpu_out.float().flatten()
    npu_out = npu_out.float().flatten()

    abs_diff = (npu_out - cpu_out).abs()

    cos_sim = torch.nn.functional.cosine_similarity(
        cpu_out.unsqueeze(0), npu_out.unsqueeze(0)
    ).item()
    mean_abs = abs_diff.mean().item()

    # 统计误差分布
    within_1e_3 = (abs_diff < 1e-3).float().mean().item() * 100
    within_1e_2 = (abs_diff < 1e-2).float().mean().item() * 100
    within_1e_1 = (abs_diff < 1e-1).float().mean().item() * 100

    report = (
        f"\n{'='*60}\n"
        f" Precision Report {tag}\n"
        f"{'='*60}\n"
        f"  Cosine similarity:         {cos_sim:.8f}\n"
        f"  Max absolute difference:   {abs_diff.max().item():.8e}\n"
        f"  Mean absolute difference:  {mean_abs:.8e}\n"
        f"  Median absolute difference:{abs_diff.median().item():.8e}\n"
        f"  Elements < 1e-3:           {within_1e_3:.2f}%\n"
        f"  Elements < 1e-2:           {within_1e_2:.2f}%\n"
        f"  Elements < 1e-1:           {within_1e_1:.2f}%\n"
    )

    # 判定标准: cosine > 0.999 (方向一致性，等价于 < 0.1% 相对误差)
    # fp32: mean_abs < 0.015, fp16: mean_abs < 0.02 (半精度固有误差更大)
    threshold = 0.02 if dtype == "fp16" else 0.015
    passed = cos_sim > 0.999 and mean_abs < threshold
    report += f"\n  >>> {'PASS' if passed else 'FAIL'} (cos_sim={cos_sim:.6f}, mean_abs={mean_abs:.6f}, req: >0.999 & <{threshold})\n"
    print(report)
    return passed


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_adaptation(
    dtype: str = "fp32",
    num_test_images: int = 5,
    use_real_images: bool = False,
):
    print(f"\n{'#'*60}")
    print("# DINOv2 Ascend NPU Adaptation")
    print(f"{'#'*60}\n")

    if not torch.npu.is_available():
        print("[ERROR] NPU not available. Exiting.")
        return False

    npu_name = torch.npu.get_device_name(0)
    print(f"[NPU] {npu_name}, count={torch.npu.device_count()}")

    # ------------------------------------------------------------------
    # 1. CPU 基线
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("1. CPU Baseline Inference")
    print(f"{'='*60}")

    cpu_model = Dinov2NPUModel(device="cpu", dtype="fp32")

    # 生成测试图片
    if use_real_images:
        try:
            from PIL import Image
            test_images = []
            for i in range(num_test_images):
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=tuple(np.random.randint(0, 255, 3).tolist()))
                test_images.append(img)
            print(f"[Data] Generated {num_test_images} PIL images ({IMAGE_SIZE}x{IMAGE_SIZE})")
        except ImportError:
            print("[WARN] PIL not available, falling back to numpy arrays")
            use_real_images = False

    if not use_real_images:
        test_images = [np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                       for _ in range(num_test_images)]

    # CPU 推理
    t0 = time.perf_counter()
    cpu_results = [cpu_model.infer(img) for img in test_images]
    t_cpu = time.perf_counter() - t0
    cpu_embs = torch.cat([r["embedding"] for r in cpu_results], dim=0)
    print(f"  CPU total time: {t_cpu:.4f}s, avg: {t_cpu/num_test_images:.4f}s/image")
    print(f"  CPU embeddings shape: {cpu_embs.shape}")

    # 释放 CPU 模型
    del cpu_model

    # ------------------------------------------------------------------
    # 2. NPU fp32 推理
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"2. NPU Inference (dtype={dtype})")
    print(f"{'='*60}")

    npu_model = Dinov2NPUModel(device="npu:0", dtype=dtype)

    # Warmup
    warmup = test_images[0]
    _ = npu_model.infer(warmup)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    npu_results = [npu_model.infer(img) for img in test_images]
    torch.npu.synchronize()
    t_npu = time.perf_counter() - t0
    npu_embs = torch.cat([r["embedding"] for r in npu_results], dim=0)
    print(f"  NPU total time: {t_npu:.4f}s, avg: {t_npu/num_test_images:.4f}s/image")
    print(f"  NPU embeddings shape: {npu_embs.shape}")

    # ------------------------------------------------------------------
    # 3. 精度对比
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("3. Precision Comparison (CPU fp32 vs NPU {})".format(dtype))
    print(f"{'='*60}")

    # 逐样本对比
    all_passed = True
    for i in range(num_test_images):
        tag = f"[sample {i}]"
        passed = precision_report(cpu_embs[i], npu_embs[i], tag=tag, dtype=dtype)
        all_passed = all_passed and passed

    # 全体统计
    print(f"\n{'='*60}")
    print(f"Batch precision (all {num_test_images} samples):")
    flatten_report = precision_report(cpu_embs, npu_embs, tag="[all]", dtype=dtype)
    all_passed = all_passed and flatten_report

    # ------------------------------------------------------------------
    # 4. 速度对比
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("4. Performance Summary")
    print(f"{'='*60}")
    speedup = t_cpu / t_npu if t_npu > 0 else float("inf")
    print(f"  CPU total:  {t_cpu:.4f}s")
    print(f"  NPU total:  {t_npu:.4f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    del npu_model
    torch.npu.empty_cache()

    print(f"\n{'#'*60}")
    print(f"# Adaptation {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'#'*60}\n")
    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 Ascend NPU Adaptation")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--use-real-images", action="store_true",
                        help="Use PIL-generated images instead of numpy arrays")
    args = parser.parse_args()

    success = run_adaptation(
        dtype=args.dtype,
        num_test_images=args.num_images,
        use_real_images=args.use_real_images,
    )
    exit(0 if success else 1)
