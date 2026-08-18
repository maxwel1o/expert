#!/usr/bin/env python3
"""验证数据集顺序一致性：检查子集是否保持原始数据集的相对顺序"""

import json
import sys
from pathlib import Path

def check_order(source_path: str, subset_path: str):
    """检查 subset 是否保持 source 的相对顺序"""
    with open(source_path, 'r') as f:
        source = json.load(f)
    with open(subset_path, 'r') as f:
        subset = json.load(f)
    
    # Build source index map (content -> position)
    src_map = {}
    for i, item in enumerate(source):
        content = item.get("input", "")
        if content not in src_map:  # first occurrence wins for duplicates
            src_map[content] = i
    
    # Map subset items to source positions
    indices = []
    missing = 0
    for item in subset:
        content = item.get("input", "")
        if content in src_map:
            indices.append(src_map[content])
        else:
            missing += 1
    
    # Check monotonicity
    ordered_pairs = sum(1 for i in range(len(indices)-1) if indices[i] < indices[i+1])
    total_pairs = len(indices) - 1
    is_ordered = ordered_pairs == total_pairs
    
    print(f"Source: {len(source)} items")
    print(f"Subset: {len(subset)} items")
    print(f"Mapped: {len(indices)}, Missing: {missing}")
    print(f"Ordered pairs: {ordered_pairs}/{total_pairs} ({ordered_pairs/total_pairs*100:.1f}%)")
    print(f"Preserves order: {'✅ YES' if is_ordered else '❌ NO'}")
    
    return is_ordered

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source.json> <subset.json>")
        sys.exit(1)
    check_order(sys.argv[1], sys.argv[2])
