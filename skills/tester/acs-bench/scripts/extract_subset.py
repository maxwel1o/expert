#!/usr/bin/env python3
"""
从大型JSON数组文件中流式提取前N条记录，生成子集数据集。
适用于内存不足无法加载整个文件的场景（如3.4GB RAM + 2.2GB JSON）。

用法:
  python3 scripts/extract_subset.py \
    --src $PROF_ROOT/dataset/fixed_length/in10240_n10000_dsv30324/10240.json/10240.json \
    --dst $PROF_ROOT/dataset/fixed_length/in10240_n50_dsv30324/10240.json/10240.json \
    --count 50

原理:
  使用 json.JSONDecoder.raw_decode() 逐条解析JSON数组元素，
  仅在内存中保留当前解析的entry，不加载整个文件。
  通过逐步扩大读取窗口 + raw_decode 定位每条记录的边界。
"""

import argparse
import json
import os
import sys
import time


def extract_subset(src_path: str, dst_path: str, count: int):
    """流式提取JSON数组前N条记录"""
    file_size = os.path.getsize(src_path)
    print(f"源文件: {src_path} ({file_size / 1024**3:.2f} GB)")
    print(f"目标: {dst_path} (前 {count} 条)")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    decoder = json.JSONDecoder()
    entries = []
    buf = ""
    buf_start_pos = 0
    chunk_size = 4 * 1024 * 1024  # 4MB chunks
    total_parsed = 0
    t0 = time.time()

    with open(src_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        while first_char.strip() == '':
            first_char = f.read(1)
        if first_char != '[':
            raise ValueError(f"期望JSON数组开头 '['，实际为 '{first_char}'")

        while total_parsed < count:
            while True:
                if not buf:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    buf = chunk
                    buf_start_pos = f.tell() - len(chunk)

                stripped = buf.lstrip()
                if stripped == buf:
                    break
                if stripped:
                    consumed = len(buf) - len(stripped)
                    buf = stripped
                    buf_start_pos += consumed
                    break
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buf += chunk

            if not buf:
                break

            try:
                obj, end_idx = decoder.raw_decode(buf)
                entries.append(obj)
                total_parsed += 1

                remaining = buf[end_idx:]
                consumed = len(buf) - len(remaining)
                buf_start_pos += consumed
                buf = remaining

                if total_parsed % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  已提取 {total_parsed}/{count} 条 ({elapsed:.1f}s)")

            except json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    print(f"  文件结束，共提取 {total_parsed} 条")
                    break
                buf += chunk

    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    dst_size = os.path.getsize(dst_path)
    elapsed = time.time() - t0
    print(f"\n完成！提取 {total_parsed} 条 → {dst_path}")
    print(f"  子集大小: {dst_size / 1024**2:.1f} MB")
    print(f"  耗时: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="流式提取JSON数组前N条记录")
    parser.add_argument("--src", required=True, help="源JSON数组文件路径")
    parser.add_argument("--dst", required=True, help="目标子集文件路径")
    parser.add_argument("--count", type=int, required=True, help="提取条数")
    args = parser.parse_args()
    extract_subset(args.src, args.dst, args.count)


if __name__ == "__main__":
    main()
