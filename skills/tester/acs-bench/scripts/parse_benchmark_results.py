#!/usr/bin/env python3
"""
压测结果结构化解析脚本

解析acs-bench输出的CSV文件，匹配阶段映射，过滤异常结果，
支持shared/unique双场景对比。

用法：
  # 仅解析shared/unique模式
  python3 parse_benchmark_results.py --today --mode shared
  python3 parse_benchmark_results.py --today --mode unique

  # A vs B对比表（前缀匹配 vs 前缀不匹配）
  python3 parse_benchmark_results.py --today --compare --max-fail 0.05
"""

import csv
import os
import sys
import argparse
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 默认阶段映射：时间键(YYYY-MM-DD_HH_MM) → (阶段名, 前缀模式, c, r)
# 时间键来自文件名: summary_mt_test_..._2026-05-08_11_05_33.csv → 2026-05-08_11_05
DEFAULT_STAGE_MAP = {
    # A场景(shared) - 2026-05-08
    '2026-05-08_10_57': ('A-P0冒烟',   'shared', 10,   5),
    '2026-05-08_11_00': ('A-P1天花板', 'shared', 1000, 50),
    '2026-05-08_11_05': ('A-P2≤16s',   'shared', 1000, 48),
    '2026-05-08_11_12': ('A-P3≤17s',   'shared', 1000, 49),
    '2026-05-08_11_17': ('A-P4≤18s',   'shared', 1050, 53),
    '2026-05-08_11_21': ('A-P5≤19s',   'shared', 1050, 55),
    '2026-05-08_11_23': ('B-P0冒烟',   'unique', 10,   5),
    # B场景(unique) - 2026-05-08 S3摸高
    '2026-05-08_14_29': ('B-P1天花板', 'unique', 400,  20),
    '2026-05-08_14_35': ('B-P1天花板', 'unique', 600,  30),
    '2026-05-08_14_41': ('B-P1天花板', 'unique', 800,  40),
    '2026-05-08_14_55': ('B-P2≤16s',   'unique', 600,  22),
    '2026-05-08_15_09': ('B-P3≤18s',   'unique', 600,  25),
    '2026-05-08_15_26': ('B-P4≤17s',   'unique', 600,  24),
    '2026-05-08_15_53': ('B-P5≤16s',   'unique', 600,  23),
    '2026-05-08_16_24': ('B-P6≤19s',   'unique', 600,  26),
}


def parse_csv_file(filepath: str) -> Optional[Dict]:
    """解析单个CSV文件，提取关键指标"""
    try:
        with open(filepath) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = next(reader)
    except (StopIteration, FileNotFoundError, PermissionError):
        return None

    # Build column index map
    col_map = {name.strip(): i for i, name in enumerate(header)}

    def get_col(name: str, default=None):
        idx = col_map.get(name)
        if idx is not None and idx < len(data):
            val = data[idx].strip()
            return val if val else default
        return default

    concurrency = int(get_col('Concurrency', 0))
    qps = float(get_col('QPS', 0))
    avg_e2e = float(get_col('AVG_E2E(s)', 0))
    tp99_e2e = float(get_col('TP99_E2E(s)', 0))
    avg_ttft = float(get_col('AVG_TTFT(s)', 0))
    avg_tpot = float(get_col('AVG_TPOT(s)', 0))
    output_tps = float(get_col('Output_Token_Throughput(tokens/s)', 0))
    fail_rate = float(get_col('Fail_Rate', 0))
    avg_completion = float(get_col('AVG_COMPLETION_TOKENS', 0))

    return {
        'concurrency': concurrency,
        'qps': qps,
        'avg_e2e': avg_e2e,
        'tp99_e2e': tp99_e2e,
        'avg_ttft': avg_ttft,
        'avg_tpot': avg_tpot,
        'output_tps': output_tps,
        'fail_rate': fail_rate,
        'avg_completion': avg_completion,
        'filepath': filepath,
    }


def find_stage(filename: str, concurrency: int, stage_map: Dict) -> Optional[tuple]:
    """从文件名提取时间键，匹配阶段映射"""
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})\.csv$', filename)
    if m:
        time_key = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        if time_key in stage_map:
            return stage_map[time_key]

    # Fallback: try to match by concurrency
    for key, val in stage_map.items():
        if val[2] == concurrency:
            return val
    return None


def print_results(results: List[Dict], title: str = "Results"):
    """打印结果表格"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    print(f"{'阶段':<12} {'前缀':<7} {'c':>5} {'r':>3} {'QPS':>7} {'E2E':>8} {'TP99':>8} {'TTFT':>8} {'OutTput':>9} {'Fail':>6}")
    print("-" * 80)
    for r in results:
        stage = r.get('stage', '')
        mode = r.get('mode', '')
        c = r.get('concurrency', 0)
        rate = r.get('rate', 0)
        qps = r.get('qps', 0)
        e2e = r.get('avg_e2e', 0)
        tp99 = r.get('tp99_e2e', 0)
        ttft = r.get('avg_ttft', 0)
        outtps = r.get('output_tps', 0)
        fail = r.get('fail_rate', 0)
        print(f"{stage:<12} {mode:<7} {c:>5} {rate:>3} {qps:>7.2f} {e2e:>7.2f}s {tp99:>7.2f}s {ttft:>7.2f}s {outtps:>9.1f} {fail:>5.1%}")


def print_compare(shared_results: List[Dict], unique_results: List[Dict]):
    """打印A vs B对比表"""
    print(f"\n{'='*80}")
    print(f" A vs B 对比（前缀匹配 vs 前缀不匹配）")
    print(f"{'='*80}")
    print(f"{'阶段':<12} {'A-QPS':>7} {'B-QPS':>7} {'ΔQPS':>7} {'A-E2E':>8} {'B-E2E':>8} {'ΔE2E':>8}")
    print("-" * 80)
    for a in shared_results:
        for b in unique_results:
            if a.get('stage') == b.get('stage'):
                stage = a['stage']
                aqps = a['qps']
                bqps = b['qps']
                ae2e = a['avg_e2e']
                be2e = b['avg_e2e']
                print(f"{stage:<12} {aqps:>7.2f} {bqps:>7.2f} {aqps-bqps:>+7.2f} {ae2e:>7.2f}s {be2e:>7.2f}s {ae2e-be2e:>+7.2f}s")


def main():
    parser = argparse.ArgumentParser(description="压测结果结构化解析")
    parser.add_argument('--today', action='store_true', help='仅解析今天的结果')
    parser.add_argument('--date', help='指定日期 YYYY-MM-DD')
    parser.add_argument('--csv-dir', default='./result/csv/', help='CSV目录')
    parser.add_argument('--mode', default='all', choices=['shared', 'unique', 'all'], help='过滤模式')
    parser.add_argument('--compare', action='store_true', help='输出A vs B对比表')
    parser.add_argument('--max-fail', type=float, default=1.0, help='最大Fail_Rate阈值')
    parser.add_argument('--stage-map', default=None, help='自定义阶段映射JSON文件')
    args = parser.parse_args()

    # Load stage map
    import json
    stage_map = DEFAULT_STAGE_MAP.copy()
    if args.stage_map:
        with open(args.stage_map) as f:
            stage_map.update(json.load(f))

    # Find CSV files
    csv_dir = args.csv_dir
    if not os.path.isdir(csv_dir):
        print(f"Error: {csv_dir} not found", file=sys.stderr)
        sys.exit(1)

    today_str = datetime.now().strftime('%Y-%m-%d')
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

    if args.today:
        csv_files = [f for f in csv_files if today_str in f]
    elif args.date:
        csv_files = [f for f in csv_files if args.date in f]

    # Parse all files
    all_results = []
    for fname in csv_files:
        filepath = os.path.join(csv_dir, fname)
        parsed = parse_csv_file(filepath)
        if parsed is None:
            continue

        # Filter by fail rate
        if parsed['fail_rate'] > args.max_fail:
            continue

        # Match stage
        stage_info = find_stage(fname, parsed['concurrency'], stage_map)
        if stage_info:
            parsed['stage'] = stage_info[0]
            parsed['mode'] = stage_info[1]
            parsed['rate'] = stage_info[3]
        else:
            parsed['stage'] = '?'
            parsed['mode'] = '?'
            parsed['rate'] = 0

        # Filter by mode
        if args.mode != 'all' and parsed.get('mode') != args.mode:
            continue

        all_results.append(parsed)

    if not all_results:
        print("No results found", file=sys.stderr)
        sys.exit(1)

    # Print results
    shared = [r for r in all_results if r.get('mode') == 'shared']
    unique = [r for r in all_results if r.get('mode') == 'unique']

    if args.mode == 'shared' or args.mode == 'all':
        if shared:
            print_results(shared, "场景A：前缀匹配（shared）")

    if args.mode == 'unique' or args.mode == 'all':
        if unique:
            print_results(unique, "场景B：前缀不匹配（unique）")

    if args.compare and shared and unique:
        print_compare(shared, unique)


if __name__ == '__main__':
    main()
