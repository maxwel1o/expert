#!/usr/bin/env python3
"""
从CSV提取messages字段，构造acs-bench标准数据集格式。

用法:
  python3 extract_messages_from_csv.py \
    -i $PROF_ROOT/dataset/mt_dataset/prompt_long_utf8_xxx.csv \
    -o $PROF_ROOT/dataset/mt_dataset/data_n17829_avg3894.json

CSV格式: 单列prompt，每行是JSON含messages字段
输出格式: [{id: int, input: str}] — acs-bench标准数据集
"""

import csv, json, argparse, os, sys

def main():
    parser = argparse.ArgumentParser(description='从CSV提取messages字段构造acs-bench数据集')
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--field', default='messages', help='JSON中messages字段名 (默认: messages)')
    parser.add_argument('--csv-col', default=0, type=int, help='CSV列索引 (默认: 0)')
    args = parser.parse_args()

    result = []
    errors = 0

    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for i, row in enumerate(reader):
            try:
                data = json.loads(row[args.csv_col])
                messages = data.get(args.field, [])
                result.append({
                    "id": i,
                    "input": json.dumps(messages, ensure_ascii=False)
                })
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error at row {i}: {e}", file=sys.stderr)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    # Stats
    input_lens = [len(item['input']) for item in result]
    avg_len = sum(input_lens) / len(input_lens) if input_lens else 0
    print(f"Total: {len(result)}, Errors: {errors}")
    print(f"Avg input len: {avg_len:.0f}, Min: {min(input_lens)}, Max: {max(input_lens)}")
    print(f"Output: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f}MB)")

if __name__ == '__main__':
    main()
