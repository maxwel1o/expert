#!/usr/bin/env python3
"""
替代 vllm bench latency 的手动延迟测量脚本。
直接向已运行的 vLLM API 服务发送请求，测量 TTFT 和总延迟。
"""

import json
import statistics
import time
import requests
import sys


def measure(model_name, host="127.0.0.1", port=8000, input_len=128, output_len=128, runs=5):
    url = f"http://{host}:{port}/v1/chat/completions"

    # 构造一个大致达到 input_len 的 prompt
    base_prompt = "请解释以下概念：" + "人工智能" * (input_len // 4)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": base_prompt[:input_len * 4]}
        ],
        "temperature": 0,
        "max_tokens": output_len,
    }

    ttfts = []
    totals = []

    print(f"测量 {runs} 次延迟 (input≈{input_len}, output={output_len}) ...")
    print("-" * 50)

    for i in range(runs):
        start = time.time()
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        resp.raise_for_status()

        # TTFT: 收到第一个字节的时间
        first_byte_time = None
        for chunk in resp.iter_content(chunk_size=1):
            if first_byte_time is None:
                first_byte_time = time.time()
            break

        # 读取剩余内容
        for chunk in resp.iter_content(chunk_size=8192):
            pass

        total_time = time.time() - start
        ttft = (first_byte_time - start) * 1000 if first_byte_time else 0
        ttfts.append(ttft)
        totals.append(total_time)
        print(f"Run {i+1}: TTFT={ttft:.1f}ms, Total={total_time:.2f}s")
        time.sleep(1)

    print("-" * 50)
    print(f"Mean TTFT:   {statistics.mean(ttfts):.1f} ms")
    print(f"Median TTFT: {statistics.median(ttfts):.1f} ms")
    print(f"P99 TTFT:    {sorted(ttfts)[int(len(ttfts)*0.99)] if len(ttfts)>1 else ttfts[0]:.1f} ms")
    print(f"Mean Total:  {statistics.mean(totals):.2f} s")
    print(f"TPOT (est):  {(statistics.mean(totals)*1000 - statistics.mean(ttfts))/output_len:.1f} ms")

    return {
        "input_len": input_len,
        "output_len": output_len,
        "runs": runs,
        "ttft_ms": {"mean": statistics.mean(ttfts), "median": statistics.median(ttfts), "p99": sorted(ttfts)[int(len(ttfts)*0.99)] if len(ttfts)>1 else ttfts[0]},
        "total_s": {"mean": statistics.mean(totals)},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure vLLM API latency manually")
    parser.add_argument("--model", default="qwen3-4b-thinking-2507-fp8", help="Model name")
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--input-len", type=int, default=128, help="Input length")
    parser.add_argument("--output-len", type=int, default=128, help="Output length")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    args = parser.parse_args()

    try:
        result = measure(args.model, args.host, args.port, args.input_len, args.output_len, args.runs)
        with open("/tmp/npu_latency_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n结果已保存: /tmp/npu_latency_result.json")
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 http://{args.host}:{args.port}")
        print("请确保 vLLM 服务已启动。")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
