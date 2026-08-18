#!/usr/bin/env python3
"""Generate terminal-style screenshot HTML for Xception models."""
import os
import html
import argparse

STYLE = """
<style>
body { background: #1e1e1e; display: flex; justify-content: center; padding: 20px; margin: 0; font-family: 'Courier New', monospace; }
.terminal { background: #2d2d2d; border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.5); width: 900px; overflow: hidden; }
.titlebar { background: #1a1a1a; padding: 12px 16px; display: flex; align-items: center; gap: 8px; }
.circle { width: 12px; height: 12px; border-radius: 50%; }
.circle.red { background: #ff5f56; }
.circle.yellow { background: #ffbd2e; }
.circle.green { background: #27c93f; }
.title { color: #888; font-size: 13px; margin-left: 12px; }
.content { padding: 20px; color: #d4d4d4; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }
.success { color: #27c93f; }
.error { color: #f44747; }
.highlight { color: #569cd6; }
</style>
"""

def generate(log_path, output_html):
    with open(log_path, 'r') as f:
        content = f.read()

    escaped = html.escape(content)
    html_out = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NPU Inference Terminal</title>{STYLE}</head><body>
<div class="terminal">
<div class="titlebar">
<div class="circle red"></div>
<div class="circle yellow"></div>
<div class="circle green"></div>
<div class="title">Xception - NPU Inference</div>
</div>
<div class="content">"""
    for line in escaped.split('\n'):
        css = "success" if "PASS" in line else ("error" if "FAIL" in line or "Error" in line or "error" in line else ("highlight" if line.startswith('===') else "output"))
        html_out += f'<div class="{css}">{line}</div>\n'

    html_out += """</div></div></body></html>"""

    with open(output_html, 'w') as f:
        f.write(html_out)
    print(f"Screenshot generated: {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to inference log")
    parser.add_argument("--output", default="screenshot.html", help="Output HTML path")
    args = parser.parse_args()
    generate(args.log, args.output)
