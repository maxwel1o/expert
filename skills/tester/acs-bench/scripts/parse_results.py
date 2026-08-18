#!/usr/bin/env python3
"""Parse acs-bench CSV results and generate a Markdown report."""

import csv, os, glob, sys
from datetime import datetime

def parse_results(results_dir):
    """Parse all summary CSVs into structured data."""
    results = []
    for f in sorted(glob.glob(os.path.join(results_dir, '*', 'summary_*.csv'))):
        tag = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                results.append({
                    'tag': tag,
                    'length': tag.split('_')[0],
                    'throughput': float(row.get('Total_Token_Throughput(tokens/s)', 0)),
                    'out_throughput': float(row.get('Output_Token_Throughput(tokens/s)', 0)),
                    'ttft': float(row.get('AVG_TTFT(s)', 0)),
                    'tpot': float(row.get('AVG_TPOT(s)', 0)),
                    'e2e': float(row.get('AVG_E2E(s)', 0)),
                    'qps': float(row.get('QPS', 0)),
                    'fail': float(row.get('Fail_Rate', 0)),
                })
    return results

def analyze(results):
    """Produce analysis dict from results."""
    by_length = {}
    for r in results:
        l = r['length']
        by_length.setdefault(l, []).append(r)

    analysis = {}
    for length, runs in sorted(by_length.items()):
        # Single concurrency baseline
        cc1_ok = [r for r in runs if '_cc1' in r['tag'] and r['fail'] == 0]
        all_runs = runs
        total = len(all_runs)
        failed = sum(1 for r in all_runs if r['fail'] >= 1.0)
        partial_fail = sum(1 for r in all_runs if 0 < r['fail'] < 1.0)

        baseline = None
        if cc1_ok:
            baseline = {
                'throughput': sum(r['throughput'] for r in cc1_ok) / len(cc1_ok),
                'ttft': sum(r['ttft'] for r in cc1_ok) / len(cc1_ok),
                'e2e': sum(r['e2e'] for r in cc1_ok) / len(cc1_ok),
                'qps': sum(r['qps'] for r in cc1_ok) / len(cc1_ok),
            }

        # Find first concurrency with failures
        fail_start = None
        for cc in [1, 2, 4, 8]:
            cc_runs = [r for r in runs if f'_cc{cc}' in r['tag']]
            if any(r['fail'] > 0 for r in cc_runs):
                fail_start = cc
                break

        analysis[length] = {
            'total_groups': total,
            'all_failed': failed,
            'partial_fail': partial_fail,
            'fully_failed': failed == total,
            'baseline': baseline,
            'fail_start_cc': fail_start,
        }

    return analysis

def generate_report(results, analysis, output_path, ttft_slo=None, tpot_slo=None):
    """Write Markdown report."""
    lines = [
        f"# ACS-Bench 压测报告",
        f"",
        f"**生成时间**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        f"",
        f"## 汇总",
        f"",
    ]

    # Summary table
    lines.append("| 上下文 | 可用 | 基准吞吐 | 基准TTFT | 基准E2E | 限流起始并发 |")
    lines.append("|--------|------|---------|---------|---------|------------|")
    for length, a in sorted(analysis.items()):
        if a['fully_failed']:
            lines.append(f"| {length} | ❌ | - | - | - | - |")
        elif a['baseline']:
            b = a['baseline']
            cc = a['fail_start_cc']
            cc_str = str(cc) if cc else "无"
            lines.append(f"| {length} | ✅ | {b['throughput']:.1f} tok/s | {b['ttft']:.1f}s | {b['e2e']:.1f}s | {cc_str} |")
        else:
            lines.append(f"| {length} | ⚠️ | - | - | - | - |")

    lines.append("")
    lines.append("## 详细结果")
    lines.append("")
    lines.append("| Tag | 吞吐(tok/s) | 输出吞吐 | TTFT(s) | E2E(s) | QPS | 失败率 |")
    lines.append("|-----|-----------|---------|---------|--------|-----|-------|")
    for r in results:
        lines.append(f"| {r['tag']} | {r['throughput']:.2f} | {r['out_throughput']:.2f} | {r['ttft']:.2f} | {r['e2e']:.2f} | {r['qps']:.3f} | {r['fail']:.0%} |")

    # SLO compliance
    if ttft_slo or tpot_slo:
        lines.append("")
        lines.append("## SLO 达标判定")
        lines.append("")
        slo_headers = "| Tag | TTFT | TPOT | 达标 |"
        slo_divider = "|-----|------|------|------|"
        if ttft_slo:
            slo_headers = slo_headers.replace("TTFT", f"TTFT (≤{ttft_slo}s)")
        if tpot_slo:
            slo_headers = slo_headers.replace("TPOT", f"TPOT (≤{tpot_slo*1000:.0f}ms)")
        lines.append(slo_headers)
        lines.append(slo_divider)
        slo_pass = 0
        for r in results:
            if r['fail'] >= 1.0:
                lines.append(f"| {r['tag']} | - | - | ❌ |")
                continue
            ttft_ok = (r['ttft'] <= ttft_slo) if ttft_slo else True
            tpot_ok = (r['tpot'] <= tpot_slo) if tpot_slo else True
            ok = ttft_ok and tpot_ok
            if ok:
                slo_pass += 1
            ttft_str = f"{r['ttft']:.2f}s {'✅' if ttft_ok else '❌'}" if ttft_slo else "-"
            tpot_str = f"{r['tpot']*1000:.1f}ms {'✅' if tpot_ok else '❌'}" if tpot_slo else "-"
            lines.append(f"| {r['tag']} | {ttft_str} | {tpot_str} | {'✅' if ok else '❌'} |")
        lines.append("")
        lines.append(f"**SLO 达标率: {slo_pass}/{len(results)}**")

    lines.append("")
    lines.append("## 建议")
    lines.append("")
    for length, a in sorted(analysis.items()):
        if a['fully_failed']:
            lines.append(f"- **{length}**: 服务端不支持，需联系服务方确认")
        elif a['fail_start_cc']:
            lines.append(f"- **{length}**: 建议并发 ≤{a['fail_start_cc']-1}，或申请提升 QPS 限额")
        else:
            lines.append(f"- **{length}**: 性能稳定，可按需调整并发")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path

if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else './results'
    output_path = sys.argv[2] if len(sys.argv) > 2 else './压测报告.md'
    ttft_slo = float(sys.argv[3]) if len(sys.argv) > 3 else None
    tpot_slo = float(sys.argv[4]) if len(sys.argv) > 4 else None

    results = parse_results(results_dir)
    if not results:
        print("未找到结果文件")
        sys.exit(1)

    analysis = analyze(results)
    report = generate_report(results, analysis, output_path, ttft_slo, tpot_slo)
    print(f"报告已生成: {report}")
    print(f"共 {len(results)} 组结果")
