#!/usr/bin/env python3
"""
压测注入数据清理脚本

功能：清理已使用的注入数据集，释放磁盘空间（源数据集保留）。

用法：
  # 清理指定轮次的shared数据
  python3 cleanup_injected_data.py --round 1 --prefix-mode shared

  # 清理指定轮次的unique数据
  python3 cleanup_injected_data.py --round 2 --prefix-mode unique

  # 清理指定轮次的shared和unique数据
  python3 cleanup_injected_data.py --round 1 --prefix-mode both

  # 清理多个轮次
  python3 cleanup_injected_data.py --round 1,2,3 --prefix-mode shared

  # 清理所有注入数据（shared + unique，所有轮次）
  python3 cleanup_injected_data.py --prefix-mode both --all

  # 干跑模式（仅列出，不删除）
  python3 cleanup_injected_data.py --prefix-mode both --all --dry-run

  # 查看当前注入数据占用
  python3 cleanup_injected_data.py --stats

参数：
  --round        轮次号，逗号分隔（如 1,2,3）
  --prefix-mode  前缀模式：shared / unique / both，默认 both
  --source       源数据集基名，默认 data_n3838_avg11944
  --dataset-dir  数据集目录，默认 dataset/mt_dataset/
  --all          清理所有轮次的注入数据
  --dry-run      干跑模式，仅列出不删除
  --stats        仅显示当前注入数据统计
"""

import argparse
import os
import glob


def get_injected_files(dataset_dir: str, source_base: str, round_num: int = None,
                        prefix_mode: str = 'both') -> list:
    """获取匹配的注入数据文件列表"""
    files = []
    
    patterns = []
    if prefix_mode in ('shared', 'both'):
        if round_num is not None:
            patterns.append(f"{source_base}_r{round_num:02d}.json")
        else:
            patterns.append(f"{source_base}_r[0-9][0-9].json")
    
    if prefix_mode in ('unique', 'both'):
        if round_num is not None:
            patterns.append(f"{source_base}_r{round_num:02d}_uid.json")
        else:
            patterns.append(f"{source_base}_r[0-9][0-9]_uid.json")
    
    for pattern in patterns:
        matched = glob.glob(os.path.join(dataset_dir, pattern))
        files.extend(matched)
    
    return sorted(set(files))


def get_stats(dataset_dir: str, source_base: str) -> dict:
    """统计当前注入数据"""
    all_files = get_injected_files(dataset_dir, source_base, prefix_mode='both')
    
    stats = {'total_files': 0, 'total_size_mb': 0.0, 'shared': [], 'unique': []}
    for f in all_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        stats['total_files'] += 1
        stats['total_size_mb'] += size_mb
        
        basename = os.path.basename(f)
        if '_uid.json' in basename:
            stats['unique'].append((basename, size_mb))
        else:
            stats['shared'].append((basename, size_mb))
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='压测注入数据清理')
    parser.add_argument('--round', type=str, default=None,
                        help='轮次号，逗号分隔（如 1,2,3）')
    parser.add_argument('--prefix-mode', default='both', choices=['shared', 'unique', 'both'],
                        help='前缀模式：shared/unique/both，默认both')
    parser.add_argument('--source', default='data_n3838_avg11944',
                        help='源数据集基名')
    parser.add_argument('--dataset-dir', default='dataset/mt_dataset/',
                        help='数据集目录')
    parser.add_argument('--all', action='store_true',
                        help='清理所有轮次')
    parser.add_argument('--dry-run', action='store_true',
                        help='干跑模式，仅列出不删除')
    parser.add_argument('--stats', action='store_true',
                        help='仅显示统计信息')
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_stats(args.dataset_dir, args.source)
        print(f"📊 注入数据统计")
        print(f"   总文件数: {stats['total_files']}")
        print(f"   总大小: {stats['total_size_mb']:.1f} MB")
        if stats['shared']:
            print(f"\n   shared文件:")
            for name, size in stats['shared']:
                print(f"     {name} ({size:.1f} MB)")
        if stats['unique']:
            print(f"\n   unique文件:")
            for name, size in stats['unique']:
                print(f"     {name} ({size:.1f} MB)")
        return
    
    # 确定要清理的轮次
    if args.all:
        rounds = None  # 清理所有
    elif args.round:
        rounds = [int(r.strip()) for r in args.round.split(',')]
    else:
        print("❌ 必须指定 --round 或 --all")
        return
    
    # 收集要删除的文件
    files_to_delete = []
    if rounds is None:
        files_to_delete = get_injected_files(args.dataset_dir, args.source,
                                              prefix_mode=args.prefix_mode)
    else:
        for r in rounds:
            files_to_delete.extend(
                get_injected_files(args.dataset_dir, args.source,
                                   round_num=r, prefix_mode=args.prefix_mode))
    
    files_to_delete = sorted(set(files_to_delete))
    
    if not files_to_delete:
        print("没有匹配的注入数据文件")
        return
    
    # 计算总大小
    total_size = sum(os.path.getsize(f) for f in files_to_delete) / (1024 * 1024)
    
    print(f"🗑️  将清理 {len(files_to_delete)} 个文件，释放 {total_size:.1f} MB:")
    for f in files_to_delete:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"   {os.path.basename(f)} ({size_mb:.1f} MB)")
    
    if args.dry_run:
        print("\n⚠️ 干跑模式，未实际删除")
        return
    
    # 执行删除
    for f in files_to_delete:
        os.remove(f)
    
    print(f"\n✅ 已清理 {len(files_to_delete)} 个文件，释放 {total_size:.1f} MB")


if __name__ == '__main__':
    main()
