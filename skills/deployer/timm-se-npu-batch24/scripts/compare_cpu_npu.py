#!/usr/bin/env python3
"""timm SE 系列模型 CPU vs NPU 精度对比脚本 (Batch 24)"""
import argparse
import torch
import numpy as np

def compute_metrics(cpu_output: torch.Tensor, npu_output: torch.Tensor):
    """计算 CPU 和 NPU 输出之间的各项误差指标"""
    cpu_np = cpu_output.numpy().flatten()
    npu_np = npu_output.numpy().flatten()

    # 最大绝对误差
    max_abs_error = np.max(np.abs(cpu_np - npu_np))
    # 平均绝对误差
    mean_abs_error = np.mean(np.abs(cpu_np - npu_np))
    # 均方误差
    mse = np.mean((cpu_np - npu_np) ** 2)
    # 余弦相似度
    cos_sim = np.dot(cpu_np, npu_np) / (np.linalg.norm(cpu_np) * np.linalg.norm(npu_np) + 1e-10)
    # 相对误差（避免除零）
    rel_error = np.mean(np.abs(cpu_np - npu_np) / (np.abs(cpu_np) + 1e-8))

    # Top-1 一致性
    cpu_top1 = np.argmax(cpu_np)
    npu_top1 = np.argmax(npu_np)
    top1_match = cpu_top1 == npu_top1

    # Top-5 一致性
    cpu_top5 = np.argsort(cpu_np)[-5:]
    npu_top5 = np.argsort(npu_np)[-5:]
    top5_overlap = len(set(cpu_top5) & set(npu_top5))

    # Softmax 概率差异
    cpu_probs = torch.nn.functional.softmax(torch.from_numpy(cpu_output.numpy()), dim=1)
    npu_probs = torch.nn.functional.softmax(torch.from_numpy(npu_output.numpy()), dim=1)
    prob_diff = torch.max(torch.abs(cpu_probs - npu_probs)).item()

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "mse": mse,
        "cosine_similarity": cos_sim,
        "relative_error": rel_error,
        "top1_match": top1_match,
        "cpu_top1_label": int(cpu_top1),
        "npu_top1_label": int(npu_top1),
        "top5_overlap": top5_overlap,
        "max_prob_diff": prob_diff,
    }

def main():
    parser = argparse.ArgumentParser(description="CPU vs NPU 精度对比")
    parser.add_argument("--cpu-output", type=str, default="output_cpu.pt", help="CPU 输出文件")
    parser.add_argument("--npu-output", type=str, default="output_npu.pt", help="NPU 输出文件")
    args = parser.parse_args()

    print("=" * 60)
    print("CPU vs NPU 精度对比")
    print("=" * 60)

    # 加载输出
    cpu_output = torch.load(args.cpu_output, map_location="cpu", weights_only=True)
    npu_output = torch.load(args.npu_output, map_location="cpu", weights_only=True)

    print(f"CPU output shape: {cpu_output.shape}")
    print(f"NPU output shape: {npu_output.shape}")

    # 计算指标
    metrics = compute_metrics(cpu_output, npu_output)

    print("\n--- 精度指标 ---")
    print(f"最大绝对误差 (Max Abs Error): {metrics['max_abs_error']:.6e}")
    print(f"平均绝对误差 (Mean Abs Error): {metrics['mean_abs_error']:.6e}")
    print(f"均方误差 (MSE):                {metrics['mse']:.6e}")
    print(f"余弦相似度 (Cosine Similarity): {metrics['cosine_similarity']:.10f}")
    print(f"相对误差 (Relative Error):      {metrics['relative_error']:.6e}")
    print(f"最大概率差异 (Max Prob Diff):   {metrics['max_prob_diff']:.6e}")
    print(f"Top-1 一致:                    {'是' if metrics['top1_match'] else '否'}")
    print(f"CPU Top-1 标签:                {metrics['cpu_top1_label']}")
    print(f"NPU Top-1 标签:                {metrics['npu_top1_label']}")
    print(f"Top-5 重叠数:                  {metrics['top5_overlap']}/5")

    # 判断精度是否合格
    max_rel = metrics['relative_error'] * 100
    print(f"\n--- 结论 ---")
    print(f"NPU 与 CPU 推理结果相对误差: {max_rel:.4f}%")
    if metrics['cosine_similarity'] > 0.9999:
        print("精度判断: 通过 (余弦相似度 ≈ 1.0)")
    elif max_rel < 1.0:
        print(f"精度判断: 通过 (相对误差 < 1%)")
    else:
        print(f"精度判断: 注意 (相对误差 = {max_rel:.4f}%, 建议检查)")

    # 汇总一行
    print(f"\n[SUMMARY] max_err={metrics['max_abs_error']:.6e} "
          f"cos_sim={metrics['cosine_similarity']:.6f} "
          f"rel_err={max_rel:.4f}% "
          f"top1_match={metrics['top1_match']}")

if __name__ == "__main__":
    main()
