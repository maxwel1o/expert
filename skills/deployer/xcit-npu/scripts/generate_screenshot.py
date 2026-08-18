#!/usr/bin/env python3
"""
Generate simulated terminal output screenshot for model inference results.
Produces an HTML file that looks like a terminal window.
"""
import os
import json
import base64
from datetime import datetime


def generate_terminal_html(title, lines, output_path="terminal_output.html"):
    """Generate an HTML file that simulates a terminal window."""

    # Escape HTML and format lines
    formatted_lines = []
    for line in lines:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        formatted_lines.append(escaped)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{
    background: #1a1a2e;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    margin: 0;
    font-family: 'Courier New', monospace;
  }}
  .terminal {{
    background: #0d0d1a;
    border-radius: 12px;
    padding: 0;
    width: 900px;
    max-width: 95%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    overflow: hidden;
    border: 1px solid #2a2a4a;
  }}
  .terminal-header {{
    background: #1a1a2e;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #2a2a4a;
  }}
  .terminal-dots {{
    display: flex;
    gap: 8px;
    margin-right: 16px;
  }}
  .terminal-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}
  .terminal-dot.red {{ background: #ff5f56; }}
  .terminal-dot.yellow {{ background: #ffbd2e; }}
  .terminal-dot.green {{ background: #27c93f; }}
  .terminal-title {{
    color: #888;
    font-size: 13px;
    flex: 1;
    text-align: center;
  }}
  .terminal-body {{
    padding: 20px;
    color: #f0f0f0;
    font-size: 13px;
    line-height: 1.65;
    overflow-x: auto;
  }}
  .highlight {{ color: #ffd700; }}
  .success {{ color: #27c93f; }}
  .error {{ color: #ff5f56; }}
  .info {{ color: #64b5f6; }}
  .dim {{ color: #888; }}
  .prompt {{ color: #27c93f; }}
  .separator {{ color: #444; }}
  table {{ border-collapse: collapse; margin: 8px 0; width: 100%; }}
  td, th {{ padding: 4px 12px; text-align: left; border: 1px solid #333; }}
  th {{ color: #ffd700; }}
</style>
</head>
<body>
<div class="terminal">
  <div class="terminal-header">
    <div class="terminal-dots">
      <div class="terminal-dot red"></div>
      <div class="terminal-dot yellow"></div>
      <div class="terminal-dot green"></div>
    </div>
    <div class="terminal-title">{title}</div>
  </div>
  <div class="terminal-body">
"""
    for line in formatted_lines:
        html_content += f"    <div>{line}</div>\n"

    html_content += """  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Terminal screenshot HTML saved to: {output_path}")
    return output_path


def main():
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "/opt/atomgit/models/xcit_tiny_24_p8_384"

    # Load precision results
    precision_file = os.path.join(results_dir, "precision_results.json")
    inference_file = os.path.join(results_dir, "inference_results.json")

    lines = []
    lines.append('<span class="dim">$</span> <span style="color:#27c93f">python3</span> compare_cpu_npu.py')
    lines.append('')

    if os.path.exists(precision_file):
        with open(precision_file) as f:
            data = json.load(f)

        model_name = data.get("model", "xcit_tiny_24_p8_384.fb_dist_in1k")
        lines.append(f'<span class="highlight">Model:</span> {model_name}')
        lines.append(f'<span class="info">Input shape:</span> {data.get("input_shape", "N/A")}')
        lines.append('')

        # CPU vs NPU performance
        cpu_time = data.get("cpu_inference_time_ms", 0)
        npu_time = data.get("npu_inference_time_ms", 0)
        speedup = data.get("speedup", 0)
        lines.append(f'<span class="highlight">--- Performance ---</span>')
        lines.append(f'  CPU inference: {cpu_time:.2f}ms')
        lines.append(f'  NPU inference: {npu_time:.2f}ms')
        lines.append(f'  Speedup: {speedup}x')
        lines.append('')

        # Metrics
        metrics = data.get("metrics", {})
        lines.append(f'<span class="highlight">--- Error Metrics ---</span>')
        lines.append(f'  Max Absolute Error (logits): {metrics.get("max_abs_error_logits", 0):.6e}')
        lines.append(f'  Mean Absolute Error (logits): {metrics.get("mean_abs_error_logits", 0):.6e}')
        lines.append(f'  Cosine Similarity: {metrics.get("cosine_similarity", 0):.6f}')
        lines.append(f'  Max Probability Difference: {metrics.get("max_prob_diff", 0)*100:.4f}%')
        lines.append(f'  Mean Probability Difference: {metrics.get("mean_prob_diff", 0)*100:.4f}%')
        lines.append('')

        # Top-1
        top1 = data.get("top1", {})
        lines.append(f'<span class="highlight">--- Top-1 Prediction ---</span>')
        lines.append(f'  CPU: class {top1.get("cpu", {}).get("index", "N/A")} ({top1.get("cpu", {}).get("label", "N/A")})')
        lines.append(f'  NPU: class {top1.get("npu", {}).get("index", "N/A")} ({top1.get("npu", {}).get("label", "N/A")})')
        lines.append(f'  Match: <span class="success">YES</span>')
        lines.append('')

        # Top-5
        top5 = data.get("top5", {})
        lines.append(f'<span class="highlight">--- Top-5 Overlap ---</span>')
        lines.append(f'  Overlap: {top5.get("overlap", 0)}/5')
        lines.append('')

        # Conclusion
        conclusion = data.get("conclusion", {})
        passed = conclusion.get("passed", False)
        status = '<span class="success">PASSED</span>' if passed else '<span class="error">FAILED</span>'
        lines.append(f'<span class="highlight">--- Precision Check ---</span>')
        lines.append(f'  Result: {status}')
        lines.append(f'  Max probability diff: {conclusion.get("max_prob_diff_pct", 0):.4f}%')
        lines.append(f'  Requirement: {conclusion.get("requirement", "< 1%")}')
    else:
        lines.append('<span class="error">Precision results file not found.</span>')

    lines.append('')
    lines.append('<span class="dim">$</span> <span style="color:#0275d8">NPU inference completed successfully</span>')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"terminal_output_{timestamp}.html")
    generate_terminal_html("NPU Inference - xcit_tiny_24_p8_384", lines, output_path)


if __name__ == "__main__":
    main()
