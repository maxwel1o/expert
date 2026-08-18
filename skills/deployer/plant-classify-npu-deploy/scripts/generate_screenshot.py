#!/usr/bin/env python3
"""
生成模拟终端输出截图
用于模型 README 中的推理结果展示
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def create_terminal_screenshot(lines, output_path, title="terminal"):
    """Create a terminal-style screenshot with given text lines."""
    font_size = 10
    bg = '#1e1e1e'
    fg = '#d4d4d4'
    tb = '#323233'
    lh = font_size * 1.5
    cw = font_size * 0.55
    pd = 25
    th = 30

    if isinstance(lines, str):
        lines = lines.split('\n')

    ml = max(len(l) for l in lines) if lines else 80
    tw = ml * cw
    fh = (len(lines) * lh + pd * 2 + th) / 100
    fw = max(7, (tw + pd * 2) / 100)

    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Title bar
    trec = mpatches.FancyBboxPatch(
        (0, 1 - th / (fh * 100)), 1, th / (fh * 100),
        boxstyle='round,pad=0', facecolor=tb, edgecolor='none'
    )
    ax.add_patch(trec)
    ax.text(0.5, 1 - th / (2 * fh * 100), title,
            ha='center', va='center', fontsize=font_size + 1,
            color='#ffffff', fontweight='bold')

    # Control dots
    for j, c in enumerate(['#ff5f56', '#ffbd2e', '#27c93f']):
        dot = mpatches.Circle(
            (20 / (fw * 100) + j * 10 / (fw * 100),
             1 - th / (fh * 100) - 8 / (fh * 100)),
            3 / (fh * 100), color=c
        )
        ax.add_patch(dot)

    ys = 1 - (th + pd) / (fh * 100)
    colors_map = {
        'ERROR': '#f44747', 'FAIL': '#f44747', '✗': '#f44747',
        'PASS': '#6a9955', 'pass': '#6a9955', '✓': '#6a9955',
        '===': '#569cd6', '>>>': '#4ec9b0',
        '[INFO]': '#4ec9b0', '[WARN]': '#ce9178', '[ERROR]': '#f44747',
    }

    for i, ln in enumerate(lines):
        y = ys - (i * lh) / (fh * 100)
        color = fg
        for k, v in colors_map.items():
            if k in ln:
                color = v
                break
        ax.text(pd / (fw * 100), y, ln, va='top', fontsize=font_size,
                color=color, fontfamily='DejaVu Sans Mono')

    ax.axis('off')
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=bg)
    plt.close()
    print(f"Screenshot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate terminal screenshot")
    parser.add_argument("--title", default="terminal@npu", help="Terminal window title")
    parser.add_argument("--output", default="screenshot_inference.png", help="Output path")
    parser.add_argument("--lines", nargs="+", help="Text lines for screenshot")
    parser.add_argument("--lines-file", help="File containing text lines")

    args = parser.parse_args()

    if args.lines_file:
        with open(args.lines_file, 'r') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]
    elif args.lines:
        lines = args.lines
    else:
        # Default demo lines
        lines = [
            "=" * 60,
            f"  {args.title} - Ascend NPU Inference",
            "=" * 60,
            "",
            "  >>> Running NPU inference...",
            "  [INFO] Model loaded successfully",
            "  [NPU] Inference time: 13.29 ms",
            "",
            "=" * 60,
            "  Top-5 Predictions:",
            "=" * 60,
            "    1. paddy     (id=18): 15.49%",
            "    2. aloevera  (id= 0): 13.46%",
            "    3. waterapple(id=28):  8.71%",
            "    4. eggplant  (id= 9):  7.45%",
            "    5. tea       (id=27):  7.39%",
            "=" * 60,
            "  [PASS] Precision check OK",
            "=" * 60,
        ]

    create_terminal_screenshot(lines, args.output, title=args.title)


if __name__ == "__main__":
    main()
