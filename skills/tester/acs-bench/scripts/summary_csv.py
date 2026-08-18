#!/usr/bin/env python3
"""
Benchmark Result Summary CSV Generator.

Parses all benchmark CSV files in a directory and produces a unified summary CSV
with user-specified columns: 执行时间, 场景, 请求数, 输入长度, AVG_CONTENT_TOKENS, 最大并发,
压测QPS, AVG_TTFT(s), TP90_TTFT(s), AVG_TPOT(s), TP90_TPOT(s), AVG_E2E(s),
TP90_E2E(s), 输入TPS, 输出TPS, 总TPS, total_time, 实际QPS, RPM

Usage:
    python summary_csv.py --dir $PROF_ROOT/result/csv/ -o $PROF_ROOT/result/report/summary.csv
    python summary_csv.py --dir $PROF_ROOT/result/csv/                  # stdout
    python summary_csv.py --dir $PROF_ROOT/result/csv/ --sort qps       # sort by 实际QPS desc
    python summary_csv.py --dir $PROF_ROOT/result/csv/ --sort time      # sort by 执行时间 asc
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# CSV column indices (0-based) from acs-bench output
# ---------------------------------------------------------------------------
COL_EXECUTION_TIME = 0
COL_NUM_REQUESTS = 1
COL_EPOCHS = 2
COL_INPUT_LENGTH = 3
COL_OUTPUT_LENGTH = 4
COL_CONCURRENCY = 5
COL_TOTAL_TOKEN_THROUGHPUT = 6
COL_OUTPUT_TOKEN_THROUGHPUT = 7
COL_TP90_TTFT = 9
COL_AVG_TTFT = 13
COL_TP90_TPOT = 15
COL_AVG_TPOT = 19
COL_TP90_E2E = 44
COL_AVG_E2E = 48
COL_TOTAL_TIME = 55
COL_QPS = 56
COL_FAIL_RATE = 57
COL_AVG_PROMPT_TOKENS = 73
COL_AVG_CONTENT_TOKENS = 72

# Summary CSV output columns
SUMMARY_COLUMNS = [
    "执行时间",
    "场景",
    "前缀模式",
    "请求数",
    "输入长度",
    "AVG_CONTENT_TOKENS",
    "最大并发",
    "压测QPS",
    "AVG_TTFT(s)",
    "TP90_TTFT(s)",
    "AVG_TPOT(s)",
    "TP90_TPOT(s)",
    "AVG_E2E(s)",
    "TP90_E2E(s)",
    "输入TPS",
    "输出TPS",
    "总TPS",
    "total_time",
    "实际QPS",
    "RPM",
]

# Default stage map: timestamp_key(YYYY-MM-DD_HH_MM) -> (stage, prefix_mode, c, r)
# Used to add 前缀模式 column to summary CSV
DEFAULT_STAGE_MAP = {
    '2026-05-08_10_57': ('A-P0冒烟',   'shared(前缀匹配)', 10,   5),
    '2026-05-08_11_00': ('A-P1天花板', 'shared(前缀匹配)', 1000, 50),
    '2026-05-08_11_05': ('A-P2≤16s',   'shared(前缀匹配)', 1000, 48),
    '2026-05-08_11_12': ('A-P3≤17s',   'shared(前缀匹配)', 1000, 49),
    '2026-05-08_11_17': ('A-P4≤18s',   'shared(前缀匹配)', 1050, 53),
    '2026-05-08_11_21': ('A-P5≤19s',   'shared(前缀匹配)', 1050, 55),
    '2026-05-08_11_23': ('B-P0冒烟',   'unique(前缀不匹配)', 10,   5),
    # B场景(unique) - 2026-05-08 S3摸高
    '2026-05-08_14_29': ('B-P1天花板', 'unique(前缀不匹配)', 400,  20),
    '2026-05-08_14_35': ('B-P1天花板', 'unique(前缀不匹配)', 600,  30),
    '2026-05-08_14_41': ('B-P1天花板', 'unique(前缀不匹配)', 800,  40),
    '2026-05-08_14_55': ('B-P2≤16s',   'unique(前缀不匹配)', 600,  22),
    '2026-05-08_15_09': ('B-P3≤18s',   'unique(前缀不匹配)', 600,  25),
    '2026-05-08_15_26': ('B-P4≤17s',   'unique(前缀不匹配)', 600,  24),
    '2026-05-08_15_53': ('B-P5≤16s',   'unique(前缀不匹配)', 600,  23),
    '2026-05-08_16_24': ('B-P6≤19s',   'unique(前缀不匹配)', 600,  26),
}


def match_prefix_mode(filename, stage_map):
    """Match CSV filename to stage map and return prefix mode.
    Filename format: summary_mt_test_..._2026-05-08_11_05_33.csv
    """
    m = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})\.csv$', filename)
    if m:
        time_key = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        if time_key in stage_map:
            return stage_map[time_key][1]  # Return prefix_mode
    return ""


def safe_float(val, default=None):
    """Convert to float, return default on failure."""
    if val is None:
        return default
    try:
        v = float(str(val).strip())
        return v if v != -1.0 else default  # -1.0 is sentinel for unavailable
    except (ValueError, TypeError):
        return default


def safe_int(val, default=None):
    """Convert to int, return default on failure."""
    f = safe_float(val, default)
    if f is None:
        return default
    try:
        return int(f)
    except (ValueError, TypeError):
        return default


def match_log_file(execution_time_str, concurrency=None, log_dir=os.environ.get("PROF_ROOT", "/root/prof") + "/log"):
    """
    Find the corresponding log file by fuzzy matching execution time and concurrency.
    Returns (request_rate, is_fixedlen, prefix_mode) or (None, None, None) if not found.
    
    prefix_mode is inferred from log filename:
    - Contains '_uid' → 'unique(前缀不匹配)'
    - Otherwise → 'shared(前缀匹配)'
    
    Log filename patterns:
    - peak_{dataset}_r{N}_uid_c{concurrency}_r{rate}_{timestamp}.log  (unique)
    - run_peak_{dataset}_c{concurrency}_r{rate}_{timestamp}.log  (shared)
    - run_stability_{dataset}_c{concurrency}_r{rate}_run{N}_{timestamp}.log
    - run_in{n}_n{m}_c{concurrency}_r{rate}_{timestamp}.log  (fixed-length)
    - log_mt_concurrency_{c}_rate_{r}.log  (old format, no timestamp)
    
    Matching strategy:
    1. Parse log filename timestamp and compare with CSV Execution_Time
    2. Log timestamp = benchmark START time, CSV timestamp = benchmark END time
    3. Accept matches within ±60 minutes window
    4. If concurrency also matches, boost confidence
    """
    if not os.path.isdir(log_dir):
        return None, None, None

    # Parse execution time (benchmark end time)
    try:
        dt_end = datetime.strptime(execution_time_str.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None, None, None

    try:
        log_files = os.listdir(log_dir)
    except OSError:
        return None, None, None

    # Parse all log files and score matches
    candidates = []
    for fname in log_files:
        if not fname.endswith(".log"):
            continue

        # Extract timestamp from filename: YYYYMMDD_HHMM
        ts_match = re.search(r"(\d{8}_\d{4})", fname)
        if not ts_match:
            continue

        try:
            dt_log = datetime.strptime(ts_match.group(1), "%Y%m%d_%H%M")
        except ValueError:
            continue

        # Time difference in minutes (log = start, csv = end, so log should be BEFORE csv)
        diff_minutes = (dt_end - dt_log).total_seconds() / 60

        # Must be within 0~120 minutes (log starts before csv finishes)
        if diff_minutes < -5 or diff_minutes > 120:
            continue

        # Extract concurrency from filename: _c{concurrency}_
        c_match = re.search(r"_c(\d+)_", fname)
        log_concurrency = int(c_match.group(1)) if c_match else None

        # Extract request-rate: prefer _c{N}_r{rate}_ (rate after concurrency)
        # Fallback to last _r{rate}_ occurrence (avays matching dataset round _r01)
        rate_match = re.search(r"_c\d+_r(\d+)_", fname)
        if not rate_match:
            # Fallback: find all _r{N}_ and take the last one (dataset round _r01 comes first)
            all_rates = list(re.finditer(r"_r(\d+)_", fname))
            rate_match = all_rates[-1] if all_rates else None
        rate = int(rate_match.group(1)) if rate_match else None

        # Detect fixed-length: filename contains in{n}_n{m} pattern
        is_fixedlen = bool(re.search(r"_in\d+[kK]?_n\d+_", fname))

        # Infer prefix mode from filename: _uid → unique, else → shared
        prefix_mode = "unique(前缀不匹配)" if "_uid" in fname else "shared(前缀匹配)"

        # Score: lower diff is better, concurrency match is a strong signal
        score = diff_minutes  # base: time proximity
        if concurrency is not None and log_concurrency == concurrency:
            score -= 100  # strong bonus for concurrency match
        elif concurrency is not None and log_concurrency is not None and log_concurrency != concurrency:
            score += 200  # penalty for concurrency mismatch

        candidates.append((score, rate, is_fixedlen, prefix_mode, fname))

    if not candidates:
        return None, None, None

    # Sort by score (lower is better), pick best match
    candidates.sort(key=lambda x: x[0])
    _, rate, is_fixedlen, prefix_mode, fname = candidates[0]

    return rate, is_fixedlen, prefix_mode


def load_history_rates(history_path="~/.bash_history"):
    """
    Parse bash history for acs-bench commands and extract (concurrency, num_requests, 
    output_length, request_rate) tuples. Used as fallback when log filename matching fails.
    
    Returns list of dicts: [{concurrency, num_requests, output_length, request_rate, input_path}, ...]
    """
    path = os.path.expanduser(history_path)
    if not os.path.isfile(path):
        return []

    results = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "acs-bench prof" not in line or "--request-rate" not in line:
                    continue

                c = re.search(r"--concurrency\s+(\d+)", line)
                r = re.search(r"--request-rate\s+(\d+)", line)
                n = re.search(r"--num-requests\s+(\d+)", line)
                o = re.search(r"--output-length\s+(\d+)", line)
                ip = re.search(r"--input-path\s+\"?([^\"\s]+)\"?", line)

                if c and r:
                    results.append({
                        "concurrency": int(c.group(1)),
                        "request_rate": int(r.group(1)),
                        "num_requests": int(n.group(1)) if n else None,
                        "output_length": int(o.group(1)) if o else None,
                        "input_path": ip.group(1) if ip else None,
                    })
    except Exception:
        pass

    return results


def match_rate_from_history(concurrency, num_requests, output_length, history_entries):
    """
    Match a CSV result to history commands by (concurrency, num_requests, output_length).
    Returns the most common request_rate for matching entries, or None.
    """
    if not history_entries:
        return None

    matches = []
    for h in history_entries:
        # Must match concurrency
        if h["concurrency"] != concurrency:
            continue
        # Match num_requests if available
        if num_requests is not None and h["num_requests"] is not None and h["num_requests"] != num_requests:
            continue
        # Match output_length if available
        if output_length is not None and h["output_length"] is not None and h["output_length"] != output_length:
            continue
        matches.append(h["request_rate"])

    if not matches:
        return None

    # Return most common rate
    from collections import Counter
    return Counter(matches).most_common(1)[0][0]


def infer_scene(input_length, avg_prompt_tokens, is_fixedlen_from_log=None):
    """
    Infer scene type:
    1. If log file indicates fixed-length → 定长
    2. If Input_Length is non-empty → 定长
    3. Otherwise → 混长
    """
    # Priority 1: from log file
    if is_fixedlen_from_log is True:
        return "定长"
    
    # Priority 2: from CSV Input_Length field
    il = safe_float(input_length)
    if il is not None and il > 0:
        return "定长"
    
    return "混长"


def parse_csv_file(filepath, log_dir=os.environ.get("PROF_ROOT", "/root/prof") + "/log", history_entries=None):
    """Parse a single benchmark CSV file and return summary dict."""
    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Warning: Failed to read {filepath}: {e}", file=sys.stderr)
        return None

    if len(rows) < 2:
        return None

    row = rows[1]
    n = len(row)

    def get_col(idx, default=None):
        if 0 <= idx < n:
            return row[idx].strip() if row[idx].strip() else default
        return default

    # Extract raw values
    execution_time = get_col(COL_EXECUTION_TIME, "")
    num_requests = safe_int(get_col(COL_NUM_REQUESTS))
    input_length = get_col(COL_INPUT_LENGTH)  # may be empty string
    output_length = safe_int(get_col(COL_OUTPUT_LENGTH))
    avg_content_tokens = safe_float(get_col(COL_AVG_CONTENT_TOKENS))
    if avg_content_tokens is not None:
        avg_content_tokens = int(round(avg_content_tokens))
    concurrency = safe_int(get_col(COL_CONCURRENCY))
    total_tps = safe_float(get_col(COL_TOTAL_TOKEN_THROUGHPUT))
    output_tps = safe_float(get_col(COL_OUTPUT_TOKEN_THROUGHPUT))
    avg_ttft = safe_float(get_col(COL_AVG_TTFT))
    tp90_ttft = safe_float(get_col(COL_TP90_TTFT))
    avg_tpot = safe_float(get_col(COL_AVG_TPOT))
    tp90_tpot = safe_float(get_col(COL_TP90_TPOT))
    avg_e2e = safe_float(get_col(COL_AVG_E2E))
    tp90_e2e = safe_float(get_col(COL_TP90_E2E))
    total_time = safe_float(get_col(COL_TOTAL_TIME))
    actual_qps = safe_float(get_col(COL_QPS))
    fail_rate = safe_float(get_col(COL_FAIL_RATE))
    avg_prompt_tokens = safe_float(get_col(COL_AVG_PROMPT_TOKENS))

    # Match log file for request-rate, scene detection, and prefix mode
    request_rate, is_fixedlen, log_prefix_mode = match_log_file(execution_time, concurrency, log_dir)

    # Fallback: try bash history if log matching failed
    if request_rate is None and history_entries is not None:
        request_rate = match_rate_from_history(concurrency, num_requests, output_length, history_entries)

    # Infer scene
    scene = infer_scene(input_length, avg_prompt_tokens, is_fixedlen)

    # Input length: use Input_Length if available, else AVG_PROMPT_TOKENS
    input_len_val = safe_float(input_length)
    if input_len_val is None or input_len_val <= 0:
        input_len_val = avg_prompt_tokens
    if input_len_val is not None:
        input_len_val = int(round(input_len_val))

    # Input TPS = Total TPS - Output TPS
    input_tps = None
    if total_tps is not None and output_tps is not None:
        input_tps = round(total_tps - output_tps, 2)

    # RPM = actual QPS × 60
    rpm = None
    if actual_qps is not None:
        rpm = round(actual_qps * 60, 2)

    return {
        "执行时间": execution_time,
        "场景": scene,
        "前缀模式": log_prefix_mode or "",  # Filled by caller via match_prefix_mode as override
        "请求数": num_requests,
        "输入长度": input_len_val,
        "AVG_CONTENT_TOKENS": avg_content_tokens,
        "最大并发": concurrency,
        "压测QPS": request_rate,
        "AVG_TTFT(s)": avg_ttft,
        "TP90_TTFT(s)": tp90_ttft,
        "AVG_TPOT(s)": avg_tpot,
        "TP90_TPOT(s)": tp90_tpot,
        "AVG_E2E(s)": avg_e2e,
        "TP90_E2E(s)": tp90_e2e,
        "输入TPS": input_tps,
        "输出TPS": output_tps,
        "总TPS": total_tps,
        "total_time": total_time,
        "实际QPS": actual_qps,
        "RPM": rpm,
        "_source": os.path.basename(filepath),
        "_mtime": os.path.getmtime(filepath),
    }


def format_val(v):
    """Format a value for CSV output."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Benchmark result summary CSV generator")
    parser.add_argument("--dir", required=True, help="Directory containing benchmark CSV files")
    parser.add_argument("-o", "--output", help="Output CSV file path (default: stdout)")
    parser.add_argument("--sort", choices=["time", "qps", "rpm"], default="time",
                        help="Sort order: time (asc), qps (desc), rpm (desc)")
    parser.add_argument("--log-dir", default=os.environ.get("PROF_ROOT", "/root/prof") + "/log",
                        help="Log directory for request-rate extraction")
    parser.add_argument("--history", default="~/.bash_history",
                        help="Bash history file for request-rate fallback extraction")
    parser.add_argument("--stage-map", default=None,
                        help="Stage map JSON file for prefix mode matching (default: built-in DEFAULT_STAGE_MAP)")
    args = parser.parse_args()

    # Load bash history for request-rate fallback
    history_entries = load_history_rates(args.history)
    if history_entries:
        print(f"Loaded {len(history_entries)} history commands with request-rate", file=sys.stderr)

    # Find all CSV files
    csv_dir = args.dir
    if not os.path.isdir(csv_dir):
        print(f"Error: {csv_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(
        [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")],
        key=lambda p: os.path.getmtime(p),
    )

    if not csv_files:
        print("No CSV files found", file=sys.stderr)
        sys.exit(1)

    # Parse all files
    # Load stage map
    import json
    stage_map = DEFAULT_STAGE_MAP.copy()
    if args.stage_map:
        with open(args.stage_map, encoding='utf-8') as f:
            stage_map.update(json.load(f))

    results = []
    for f in csv_files:
        r = parse_csv_file(f, args.log_dir, history_entries)
        if r is not None:
            # Fill prefix mode: stage_map overrides log-inferred mode
            stage_prefix = match_prefix_mode(os.path.basename(f), stage_map)
            if stage_prefix:  # stage_map has explicit mapping, use it
                r["前缀模式"] = stage_prefix
            # else: keep log-inferred prefix_mode from parse_csv_file
            results.append(r)

    if not results:
        print("No valid results parsed", file=sys.stderr)
        sys.exit(1)

    # Sort
    if args.sort == "time":
        results.sort(key=lambda r: r.get("执行时间", ""))
    elif args.sort == "qps":
        results.sort(key=lambda r: r.get("实际QPS") or 0, reverse=True)
    elif args.sort == "rpm":
        results.sort(key=lambda r: r.get("RPM") or 0, reverse=True)

    # Write CSV
    output_cols = [c for c in SUMMARY_COLUMNS]

    # Integer column indices (0-based) in SUMMARY_COLUMNS: values formatted as X.000 should be stripped to X
    INT_COL_INDICES = {3, 4, 5, 6, 7, 14, 15, 16, 17, 19}

    # utf-8-sig: 写文件时加BOM头，Excel才能正确识别UTF-8中文，否则乱码
    out_fh = open(args.output, "w", newline="", encoding='utf-8-sig') if args.output else sys.stdout
    try:
        writer = csv.writer(out_fh)
        writer.writerow(output_cols)
        for r in results:
            row = [format_val(r.get(c)) for c in output_cols]
            # Post-process integer columns: strip trailing .000
            for idx in INT_COL_INDICES:
                if idx < len(row) and isinstance(row[idx], str) and row[idx].endswith('.000'):
                    row[idx] = row[idx][:-4]
            writer.writerow(row)
    finally:
        if args.output:
            out_fh.close()

    # Summary stats
    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"Total CSV files found: {len(csv_files)}", file=sys.stderr)
    print(f"Successfully parsed: {len(results)}", file=sys.stderr)
    if args.output:
        print(f"Output saved to: {args.output}", file=sys.stderr)

    qps_values = [r.get("实际QPS") for r in results if r.get("实际QPS") is not None]
    if qps_values:
        print(f"QPS range: {min(qps_values):.2f} ~ {max(qps_values):.2f}", file=sys.stderr)
    rpm_values = [r.get("RPM") for r in results if r.get("RPM") is not None]
    if rpm_values:
        print(f"RPM range: {min(rpm_values):.0f} ~ {max(rpm_values):.0f}", file=sys.stderr)

    scenes = {}
    for r in results:
        s = r.get("场景", "unknown")
        scenes[s] = scenes.get(s, 0) + 1
    print(f"Scene breakdown: {scenes}", file=sys.stderr)

    rr_found = sum(1 for r in results if r.get("压测QPS") is not None)
    print(f"Request-rate matched from logs: {rr_found}/{len(results)}", file=sys.stderr)


if __name__ == "__main__":
    main()
