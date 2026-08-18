#!/usr/bin/env python3
"""
压测轮次数据注入脚本

功能：基于源数据集，在每条数据的 input content 最前面注入唯一轮次标识，
生成新一轮压测用的数据集，确保每轮数据不同（避免KV Cache命中）。

支持两种前缀模式：
  shared  — 所有数据共用相同前缀：[模型_场景_轮次_时间]
            示例：[DSV3_S3_R01_20260508_0930]
  unique  — 每条数据使用不同前缀：[模型_场景_轮次_时间_数据id]
            示例：[DSV3_S3_R01_20260508_0930_0000]
            示例：[DSV3_S3_R01_20260508_0930_3837]

用法：
  # shared模式（默认）— 所有数据相同前缀
  python3 inject_round_identifier.py \
    --source dataset/mt_dataset/data_n3838_avg11944.json \
    --model DSV3 --scene S3 --round 1 \
    --prefix-mode shared

  # unique模式 — 每条数据不同前缀（含数据id）
  python3 inject_round_identifier.py \
    --source dataset/mt_dataset/data_n3838_avg11944.json \
    --model DSV3 --scene S3 --round 1 \
    --prefix-mode unique

参数：
  --source       源数据集JSON路径（必需）
  --model        模型简称，如 DSV3（必需）
  --scene        场景简称，如 S3（必需）
  --round        轮次号，整数（必需）
  --prefix-mode  前缀模式：shared（相同前缀）或 unique（每条不同，含数据id），默认 shared
  --id-width     unique模式下数据id的零填充宽度，默认4（即0000~9999）
  --output       输出数据集路径（可选，默认自动生成）
  --ts           时间戳，格式YYYYMMDD_HHMM（可选，默认当前时间）
"""

import json
import argparse
import os
from datetime import datetime


def inject_identifier(source_path: str, model: str, scene: str, round_num: int,
                      prefix_mode: str = 'shared', id_width: int = 4,
                      output_path: str = None, ts: str = None) -> str:
    """注入轮次标识并生成新数据集
    
    Args:
        prefix_mode: 'shared' — 所有数据共用相同前缀 [模型_场景_轮次_时间]
                     'unique' — 每条数据不同前缀 [模型_场景_轮次_时间_数据id]
        id_width: unique模式下数据id的零填充宽度
    """
    
    if prefix_mode not in ('shared', 'unique'):
        raise ValueError(f"prefix_mode 必须为 'shared' 或 'unique'，收到: {prefix_mode}")
    
    # 生成时间戳
    if ts is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 构造基础标识（不含数据id）
    base_identifier = f"{model}_{scene}_R{round_num:02d}_{ts}"
    
    # 加载源数据
    with open(source_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    print(f"源数据集: {source_path}")
    print(f"总条数: {total}")
    print(f"前缀模式: {prefix_mode}")
    
    if prefix_mode == 'shared':
        identifier = f"[{base_identifier}]"
        print(f"轮次标识: {identifier}（所有数据共用）")
    else:
        print(f"标识前缀: [{base_identifier}_XXXX]（每条数据不同）")
    
    # 注入标识到每条数据的 input content 最前面
    modified_count = 0
    for idx, item in enumerate(data):
        input_str = item['input']
        messages = json.loads(input_str)
        
        # 构造本条数据的标识
        if prefix_mode == 'shared':
            identifier = f"[{base_identifier}]"
        else:  # unique
            data_id = str(idx).zfill(id_width)
            identifier = f"[{base_identifier}_{data_id}]"
        
        # 在第一条 user 消息的 content 最前面插入标识
        for msg in messages:
            if msg.get('role') == 'user':
                msg['content'] = identifier + "\n" + msg['content']
                modified_count += 1
                break
        
        item['input'] = json.dumps(messages, ensure_ascii=False)
    
    print(f"注入条数: {modified_count}")
    
    # 自动生成输出路径
    if output_path is None:
        base, ext = os.path.splitext(source_path)
        suffix = f"_r{round_num:02d}" if prefix_mode == 'shared' else f"_r{round_num:02d}_uid"
        output_path = f"{base}{suffix}{ext}"
    
    # 写入新数据集
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"输出数据集: {output_path}")
    
    # 验证
    with open(output_path, 'r', encoding='utf-8') as f:
        verify_data = json.load(f)
    assert len(verify_data) == total, "条数不一致！"
    
    # 抽查验证
    first_msgs = json.loads(verify_data[0]['input'])
    first_content = first_msgs[0]['content']
    
    if prefix_mode == 'shared':
        expected = f"[{base_identifier}]"
        assert first_content.startswith(expected), f"首条标识不匹配: {first_content[:100]}"
        print(f"✅ 验证通过：{total}条，shared模式，首条标识: {expected}")
    else:
        expected_first = f"[{base_identifier}_{'0'.zfill(id_width)}]"
        expected_last_prefix = f"[{base_identifier}_{str(total-1).zfill(id_width)}]"
        last_msgs = json.loads(verify_data[-1]['input'])
        last_content = last_msgs[0]['content']
        assert first_content.startswith(expected_first), f"首条标识不匹配: {first_content[:120]}"
        assert last_content.startswith(expected_last_prefix), f"末条标识不匹配: {last_content[:120]}"
        print(f"✅ 验证通过：{total}条，unique模式")
        print(f"   首条标识: {expected_first}")
        print(f"   末条标识: {expected_last_prefix}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='压测轮次数据注入')
    parser.add_argument('--source', required=True, help='源数据集JSON路径')
    parser.add_argument('--model', required=True, help='模型简称（如 DSV3）')
    parser.add_argument('--scene', required=True, help='场景简称（如 S3）')
    parser.add_argument('--round', type=int, required=True, help='轮次号')
    parser.add_argument('--prefix-mode', default='shared', choices=['shared', 'unique'],
                        help='前缀模式：shared=相同前缀, unique=每条不同(含数据id)，默认shared')
    parser.add_argument('--id-width', type=int, default=4,
                        help='unique模式下数据id零填充宽度，默认4')
    parser.add_argument('--output', default=None, help='输出路径（默认自动生成）')
    parser.add_argument('--ts', default=None, help='时间戳 YYYYMMDD_HHMM（默认当前时间）')
    
    args = parser.parse_args()
    inject_identifier(args.source, args.model, args.scene, args.round,
                      args.prefix_mode, args.id_width, args.output, args.ts)


if __name__ == '__main__':
    main()
