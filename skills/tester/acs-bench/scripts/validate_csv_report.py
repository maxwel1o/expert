#!/usr/bin/env python3
"""
acs-bench CSV 报告校验脚本

校验压测输出CSV文件的完整性，支持两种模式：
1. 报告模式（默认）：校验 summary_csv.py 生成的汇总报告CSV（result/report/）
   - 支持19列（标准）和20列（含前缀模式）两种格式
   - 支持13列（摸高简报）格式
2. 原始模式：校验 acs-bench 直接输出的74列原始CSV（result/csv/）

校验项：
- 文件存在且非空
- 表头行存在且列数在预期范围内
- 数据行存在且列数与表头一致
- 关键列存在且数据非空
- 数值合理性

用法:
  # 报告模式（默认，校验汇总报告）
  python3 scripts/validate_csv_report.py --report
  python3 scripts/validate_csv_report.py --report --today
  python3 scripts/validate_csv_report.py result/report/S3_shared_benchmark_20260508.csv

  # 原始模式（校验74列原始CSV）
  python3 scripts/validate_csv_report.py --raw
  python3 scripts/validate_csv_report.py --raw --today
  python3 scripts/validate_csv_report.py result/csv/summary_xxx.csv

  # 严格模式：-1占位值也报错
  python3 scripts/validate_csv_report.py --report --strict
"""

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

# ==================== 报告CSV定义 ====================
# 19列标准格式（summary_csv.py 旧版）
REPORT_19_CRITICAL = [
    "执行时间", "场景", "请求数", "输出长度", "最大并发",
    "AVG_TTFT(s)", "AVG_TPOT(s)", "AVG_E2E(s)",
    "输出TPS", "总TPS", "实际QPS", "RPM",
]
REPORT_19_OPTIONAL = [
    "输入长度", "压测QPS", "TP90_TTFT(s)", "TP90_TPOT(s)",
    "TP90_E2E(s)", "输入TPS", "total_time",
]
REPORT_19_NUMERIC = {
    "实际QPS": (0, 10000, "实际QPS应在0~10000范围"),
    "RPM": (0, 600000, "RPM应在0~600000范围"),
    "AVG_E2E(s)": (0, 600, "AVG_E2E应在0~600s范围"),
    "AVG_TTFT(s)": (0, 300, "AVG_TTFT应在0~300s范围"),
    "最大并发": (1, 100000, "最大并发应>0"),
    "请求数": (1, 1000000, "请求数应>0"),
    "输出TPS": (0, 1000000, "输出TPS应≥0"),
    "总TPS": (0, 1000000, "总TPS应≥0"),
}

# 20列格式（summary_csv.py 新版，含前缀模式列）
REPORT_20_CRITICAL = REPORT_19_CRITICAL  # 前缀模式为可选（旧数据无此列数据）
REPORT_20_OPTIONAL = REPORT_19_OPTIONAL + ["前缀模式"]
REPORT_20_NUMERIC = REPORT_19_NUMERIC

# 13列摸高简报格式（parse_benchmark_results.py 生成）
# 摸高简报含分隔行/汇总行，部分行关键列可能为空，仅校验表头列存在
REPORT_13_CRITICAL = []  # 不校验每行数据非空（摸高过程有占位行）
REPORT_13_OPTIONAL = [
    "场景", "阶段", "轮次", "c", "r", "QPS", "AVG_E2E(s)", "AVG_TTFT(s)",
    "TPOT(s/token)", "AVG_COMPLETION", "Fail_Rate(%)", "concurrency_min", "备注",
]
REPORT_13_NUMERIC = {
    "QPS": (0, 10000, "QPS应在0~10000范围"),
    "AVG_E2E(s)": (0, 600, "AVG_E2E应在0~600s范围"),
    "AVG_TTFT(s)": (0, 300, "AVG_TTFT应在0~300s范围"),
    "c": (1, 100000, "并发c应>0"),
    "r": (1, 10000, "速率r应>0"),
    "Fail_Rate(%)": (0, 100, "Fail_Rate应在0~100%"),
}

# 报告格式识别表：(列数, 类型名, critical, optional, numeric)
REPORT_FORMATS = {
    19: ("报告19列", REPORT_19_CRITICAL, REPORT_19_OPTIONAL, REPORT_19_NUMERIC),
    20: ("报告20列(含前缀模式)", REPORT_20_CRITICAL, REPORT_20_OPTIONAL, REPORT_20_NUMERIC),
    13: ("摸高简报13列", REPORT_13_CRITICAL, REPORT_13_OPTIONAL, REPORT_13_NUMERIC),
}
REPORT_VALID_COL_COUNTS = set(REPORT_FORMATS.keys())

# ==================== 原始CSV（74列）定义 ====================
RAW_EXPECTED_COLUMNS = 74

RAW_CRITICAL_COLUMNS = [
    "Execution_Time", "Num_Requests", "Output_Length", "Concurrency",
    "Total_Token_Throughput(tokens/s)", "Output_Token_Throughput(tokens/s)",
    "AVG_TTFT(s)", "AVG_TPOT(s)", "AVG_E2E(s)", "TP99_E2E(s)",
    "Total_Time(s)", "QPS", "Fail_Rate",
    "AVG_COMPLETION_TOKENS", "AVG_PROMPT_TOKENS",
]

RAW_OPTIONAL_COLUMNS = [
    "TP75_SERVER_TTFT(s)", "TP90_SERVER_TTFT(s)", "TP95_SERVER_TTFT(s)",
    "TP99_SERVER_TTFT(s)", "MAX_SERVER_TTFT(s)", "AVG_SERVER_TTFT(s)",
    "TP75_SERVER_TPOT(s)", "TP90_SERVER_TPOT(s)", "TP95_SERVER_TPOT(s)",
    "TP99_SERVER_TPOT(s)", "MAX_SERVER_TPOT(s)", "AVG_SERVER_TPOT(s)",
    "TP75_SERVER_E2E(s)", "TP90_SERVER_E2E(s)", "TP95_SERVER_E2E(s)",
    "TP99_SERVER_E2E(s)", "MAX_SERVER_E2E(s)", "AVG_SERVER_E2E(s)",
    "Input_Length",
]

RAW_NUMERIC_CHECKS = {
    "QPS": (0, 10000, "QPS应在0~10000范围"),
    "Fail_Rate": (0, 1, "Fail_Rate应在0~1范围"),
    "AVG_E2E(s)": (0, 600, "AVG_E2E应在0~600s范围"),
    "Concurrency": (1, 100000, "Concurrency应>0"),
}

# ==================== 颜色 ====================
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def info(msg):
    print(f"{BLUE}[INFO]{NC} {msg}")


def ok(msg):
    print(f"{GREEN}[OK]{NC} {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{NC} {msg}")


def err(msg):
    print(f"{RED}[ERROR]{NC} {msg}")


def validate_csv_file(
    csv_path: str,
    expected_col_count: int,
    critical_columns: list,
    optional_columns: list,
    numeric_checks: dict,
    strict: bool = False,
    csv_type: str = "report",
) -> dict:
    """校验单个CSV文件，返回校验结果dict"""
    result = {
        "file": csv_path,
        "type": csv_type,
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # 1. 文件存在且非空
    if not os.path.exists(csv_path):
        result["valid"] = False
        result["errors"].append("文件不存在")
        return result

    file_size = os.path.getsize(csv_path)
    if file_size == 0:
        result["valid"] = False
        result["errors"].append("文件为空")
        return result

    # 2. 读取CSV
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"CSV读取失败: {e}")
        return result

    if len(rows) < 1:
        result["valid"] = False
        result["errors"].append("CSV无表头行")
        return result

    # 3. 表头校验
    header = rows[0]
    actual_col_count = len(header)
    result["stats"]["column_count"] = actual_col_count

    if actual_col_count != expected_col_count:
        result["valid"] = False
        result["errors"].append(
            f"列数={actual_col_count}，预期={expected_col_count}"
        )

    # 4. 数据行校验
    if len(rows) < 2:
        result["valid"] = False
        result["errors"].append("CSV无数据行")
        return result

    bad_rows = []
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != actual_col_count:
            bad_rows.append((i, len(row)))
    if bad_rows:
        result["valid"] = False
        for row_num, col_count in bad_rows[:3]:
            result["errors"].append(f"第{row_num}行列数={col_count}，表头={actual_col_count}")
        if len(bad_rows) > 3:
            result["errors"].append(f"...共{len(bad_rows)}行列数不一致")

    # 5. 关键列数据校验
    header_stripped = [h.strip() for h in header]
    col_map = {h: i for i, h in enumerate(header_stripped)}

    missing_critical = [c for c in critical_columns if c not in col_map]
    if missing_critical:
        result["valid"] = False
        result["errors"].append(f"关键列缺失: {', '.join(missing_critical)}")

    empty_critical_rows = {}
    for col_name in critical_columns:
        if col_name not in col_map:
            continue
        col_idx = col_map[col_name]
        for i, row in enumerate(rows[1:], start=2):
            if col_idx >= len(row):
                continue
            val = row[col_idx].strip()
            if not val:
                empty_critical_rows.setdefault(col_name, []).append(i)
            elif val in ("-1", "-1.0") and strict:
                empty_critical_rows.setdefault(f"{col_name}=-1", []).append(i)

    if empty_critical_rows:
        result["valid"] = False
        for col_name, row_nums in empty_critical_rows.items():
            if len(row_nums) <= 3:
                rows_str = ','.join(str(r) for r in row_nums)
            else:
                rows_str = f"{row_nums[0]}~{row_nums[-1]}({len(row_nums)}行)"
            result["errors"].append(f"关键列数据为空: {col_name} 行={rows_str}")

    # 可选列空值统计
    empty_optional_count = 0
    for col_name in optional_columns:
        if col_name not in col_map:
            continue
        col_idx = col_map[col_name]
        for row in rows[1:]:
            if col_idx >= len(row):
                continue
            val = row[col_idx].strip()
            if not val or val in ("-1", "-1.0"):
                empty_optional_count += 1

    if empty_optional_count > 0:
        result["warnings"].append(f"可选列数据为空/占位: {empty_optional_count}处")

    # 6. 数值合理性校验（第1行数据行）
    data_row = rows[1]
    for col_name, (lo, hi, msg) in numeric_checks.items():
        if col_name in col_map:
            col_idx = col_map[col_name]
            if col_idx < len(data_row):
                val = data_row[col_idx].strip()
                if val:
                    try:
                        v = float(val)
                        if not (lo <= v <= hi):
                            result["warnings"].append(f"{col_name}={v}，{msg}")
                    except ValueError:
                        result["warnings"].append(f"{col_name}非数值: {val}")

    # 7. 汇总统计
    result["stats"]["file_size"] = file_size
    result["stats"]["data_rows"] = len(rows) - 1
    result["stats"]["empty_critical"] = len(empty_critical_rows)
    result["stats"]["empty_optional"] = empty_optional_count

    return result


def print_result(r: dict):
    """打印单个校验结果"""
    fname = os.path.basename(r["file"])
    type_tag = f"[{r['type']}]" if r.get("type") else ""
    if r["valid"]:
        ok(f"✅ {fname} {type_tag}")
    else:
        err(f"❌ {fname} {type_tag}")

    for e in r["errors"]:
        err(f"   {e}")
    for w in r["warnings"]:
        warn(f"   {w}")

    if r["stats"]:
        s = r["stats"]
        info(
            f"   列数={s.get('column_count', '?')}, "
            f"数据行={s.get('data_rows', '?')}, "
            f"大小={s.get('file_size', 0)}B"
        )


def main():
    parser = argparse.ArgumentParser(
        description="acs-bench CSV报告校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 校验汇总报告（默认，支持19/20/13列自动识别）
  python3 scripts/validate_csv_report.py --report
  python3 scripts/validate_csv_report.py --report --today

  # 校验74列原始CSV
  python3 scripts/validate_csv_report.py --raw
  python3 scripts/validate_csv_report.py --raw --today

  # 校验单个文件（自动识别类型）
  python3 scripts/validate_csv_report.py result/report/S3_benchmark_20260508.csv
  python3 scripts/validate_csv_report.py result/csv/summary_xxx.csv
"""
    )
    parser.add_argument("csv_file", nargs="?", help="单个CSV文件路径（自动识别报告/原始类型）")
    parser.add_argument("--report", action="store_true", help="报告模式：校验汇总报告CSV（默认）")
    parser.add_argument("--raw", action="store_true", help="原始模式：校验74列原始CSV")
    parser.add_argument("--dir", help="CSV目录（默认: 报告→result/report/，原始→result/csv/）")
    parser.add_argument("--today", action="store_true", help="仅校验今天的CSV文件")
    parser.add_argument("--strict", action="store_true", help="严格模式：-1占位值也报错")
    args = parser.parse_args()

    mode = "raw" if args.raw else "report"

    csv_files = []

    if args.csv_file:
        csv_files = [args.csv_file]
    else:
        if mode == "report":
            default_dir = "./result/report/"
            file_pattern = "*_benchmark_*.csv"  # 仅最终交付报告，排除摸高简报(*_summary_*.csv)
        else:
            default_dir = "./result/csv/"
            file_pattern = "summary_*.csv"

        csv_dir = args.dir or default_dir
        if not os.path.isdir(csv_dir):
            err(f"目录不存在: {csv_dir}")
            sys.exit(1)

        today_str = date.today().strftime("%Y-%m-%d")
        for f in sorted(Path(csv_dir).glob(file_pattern)):
            if args.today and today_str not in f.name:
                continue
            csv_files.append(str(f))

    if not csv_files:
        warn(f"未找到CSV文件（模式={mode}）")
        sys.exit(0)

    info(f"校验 {len(csv_files)} 个CSV文件（模式={mode}）...")
    print()

    all_valid = True
    results = []

    for f in csv_files:
        if mode == "raw":
            r = validate_csv_file(
                f,
                expected_col_count=RAW_EXPECTED_COLUMNS,
                critical_columns=RAW_CRITICAL_COLUMNS,
                optional_columns=RAW_OPTIONAL_COLUMNS,
                numeric_checks=RAW_NUMERIC_CHECKS,
                strict=args.strict,
                csv_type="原始74列",
            )
        else:
            # 报告模式：先读表头识别列数，再选对应格式
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    reader = csv.reader(fh)
                    header = next(reader)
                actual_cols = len(header)
            except Exception:
                actual_cols = 0

            if actual_cols in REPORT_FORMATS:
                fmt_name, crit, optl, numc = REPORT_FORMATS[actual_cols]
            else:
                # 未知列数，用最接近的格式
                fmt_name = f"报告{actual_cols}列(未知格式)"
                crit = REPORT_19_CRITICAL
                optl = REPORT_19_OPTIONAL
                numc = REPORT_19_NUMERIC

            r = validate_csv_file(
                f,
                expected_col_count=actual_cols if actual_cols in REPORT_FORMATS else 19,
                critical_columns=crit,
                optional_columns=optl,
                numeric_checks=numc,
                strict=args.strict,
                csv_type=fmt_name,
            )

        results.append(r)
        print_result(r)
        if not r["valid"]:
            all_valid = False

    print()
    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = len(results) - valid_count

    if invalid_count == 0:
        ok(f"全部通过: {valid_count}/{len(results)}")
    else:
        err(f"校验失败: {invalid_count}/{len(results)} 个文件不通过")
        sys.exit(1)


if __name__ == "__main__":
    main()
