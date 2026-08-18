#!/usr/bin/env python3
"""精度对比脚本：对比 CPU 与 NPU 推理结果"""
import pickle
import numpy as np

def load_result(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

cpu_result = load_result('/opt/atomgit/vit-base-patch16-224-npu/scripts/cpu_output.pkl')
npu_result = load_result('/opt/atomgit/vit-base-patch16-224-npu/scripts/npu_output.pkl')

cpu_logits = cpu_result['logits']
npu_logits = npu_result['logits']

# 绝对误差
abs_diff = np.abs(cpu_logits - npu_logits)
max_abs_diff = np.max(abs_diff)
mean_abs_diff = np.mean(abs_diff)

# 相对误差（避免除0）
rel_diff = abs_diff / (np.abs(cpu_logits) + 1e-8)
max_rel_diff = np.max(rel_diff)
mean_rel_diff = np.mean(rel_diff)

# 分类结果一致性
cpu_top1 = np.argmax(cpu_logits, axis=-1)
npu_top1 = np.argmax(npu_logits, axis=-1)
top1_match = np.sum(cpu_top1 == npu_top1) / len(cpu_top1) * 100

print("=" * 60)
print("精度对比结果 (CPU vs NPU)")
print("=" * 60)
print(f"Max Absolute Difference:  {max_abs_diff:.6e}")
print(f"Mean Absolute Difference: {mean_abs_diff:.6e}")
print(f"Max Relative Difference:  {max_rel_diff:.6e}")
print(f"Mean Relative Difference: {mean_rel_diff:.6e}")
print(f"Top-1 Match Rate:         {top1_match:.2f}%")

# 判断是否通过 1% 误差阈值
# 使用相对误差 < 1% 作为标准
pass_threshold = mean_rel_diff < 0.01
print("-" * 60)
if pass_threshold:
    print("结论: 精度验证通过 (Mean Relative Diff < 1%)")
else:
    print("结论: 精度验证未通过 (Mean Relative Diff >= 1%)")
print("=" * 60)
