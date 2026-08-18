#!/usr/bin/env python3
"""生成模拟终端截图（读取日志文件，生成终端风格截图）"""
import sys
import os


def wrap_terminal_output(text, output_path, title="Terminal Output"):
    lines = text.strip().split('\n')
    width = min(max(len(l) for l in lines) + 8, 120)
    title_line = f" {title} "

    result = []
    result.append('┌' + '─' * (width - 2) + '┐')
    result.append('│' + title_line.center(width - 2) + '│')
    result.append('├' + '─' * (width - 2) + '┤')
    for line in lines:
        clean = line.rstrip()
        if len(clean) > width - 5:
            clean = clean[:width-8] + '...'
        result.append('│  ' + clean.ljust(width - 5) + '│')
    result.append('└' + '─' * (width - 2) + '┘')

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(result))
    print(f"[INFO] Screenshot saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 gen_screenshot.py <input_file> <output_file>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        text = f.read()
    title = sys.argv[3] if len(sys.argv) > 3 else "Terminal Output"
    wrap_terminal_output(text, sys.argv[2], title)
