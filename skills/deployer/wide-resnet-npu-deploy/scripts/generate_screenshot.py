#!/usr/bin/env python3
"""Generate terminal-style screenshot HTML files from comparison logs."""
import os
import html

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
.prompt { color: #6a9955; }
.output { color: #d4d4d4; }
.highlight { color: #569cd6; }
.success { color: #27c93f; }
.error { color: #f44747; }
</style>
"""

def generate_screenshot(model_name, log_path, output_html):
    if not os.path.exists(log_path):
        print(f"No log found at {log_path}, skipping")
        return
    with open(log_path, 'r') as f:
        log_content = f.read()

    escaped = html.escape(log_content)
    title = f"{model_name} - NPU Inference Terminal"

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>{STYLE}</head><body>
<div class="terminal">
<div class="titlebar">
<div class="circle red"></div>
<div class="circle yellow"></div>
<div class="circle green"></div>
<div class="title">{model_name} - NPU Inference</div>
</div>
<div class="content">"""
    for line in escaped.split('\n'):
        if 'PASS' in line or 'PASSED' in line:
            html_content += f'<div class="success">{line}</div>'
        elif 'FAIL' in line or 'Error' in line or 'error' in line:
            html_content += f'<div class="error">{line}</div>'
        elif line.startswith('==='):
            html_content += f'<div class="highlight">{line}</div>'
        elif line.startswith('$') or line.startswith('#'):
            html_content += f'<div class="prompt">{line}</div>'
        else:
            html_content += f'<div class="output">{line}</div>'

    html_content += f"""</div></div></body></html>"""

    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Screenshot HTML generated: {output_html}")

if __name__ == "__main__":
    base = "/opt/atomgit/batch3_wide_resnet"
    models = [
        "wide_resnet50_2.racm_in1k",
        "wide_resnet50_2.tv2_in1k",
        "wide_resnet50_2.tv_in1k",
        "wide_resnet101_2.tv_in1k",
        "wide_resnet101_2.tv2_in1k",
    ]

    for m in models:
        d = os.path.join(base, m)
        log = os.path.join(d, "compare_full.log")
        out = os.path.join(d, "screenshot.html")
        generate_screenshot(m, log, out)
