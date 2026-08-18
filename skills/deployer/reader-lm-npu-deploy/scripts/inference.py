#!/usr/bin/env python3
"""
reader-lm-0.5b NPU Inference Script
====================================
Inference on Ascend NPU via vLLM-Ascend.

Requirements:
  pip install vllm vllm-ascend transformers requests

Usage:
  # Start server
  python3 inference.py serve --port 8000

  # Run inference
  python3 inference.py run --port 8000 --prompt "<html><body><h1>Hello</h1></body></html>"
"""
import argparse
import json
import os
import subprocess
import sys
import time
import requests


def start_server(args):
    """Start vLLM serving on Ascend NPU."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        args.model_path,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory),
        "--trust-remote-code",
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")

    print(f"Starting vLLM server for {args.model_path} on port {args.port}...")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["ASCEND_RUNTIME_OPTIONS"] = ""

    process = subprocess.Popen(cmd, env=env)

    # Wait for server to be ready
    for i in range(60):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{args.port}/v1/models", timeout=5)
            if r.status_code == 200:
                print(f"Server ready after {i * 2 + 2}s")
                break
        except requests.exceptions.ConnectionError:
            pass
        if i == 30:
            print("Server still starting...")

    print(f"Server PID: {process.pid}")
    print(f"API endpoint: http://127.0.0.1:{args.port}/v1")
    process.wait()


def run_inference(args):
    """Run inference against a running vLLM server."""
    base_url = f"http://127.0.0.1:{args.port}/v1"
    model = args.model_path

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
    }
    if args.logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5

    print(f"Prompt: {args.prompt[:100]}...")
    start = time.time()

    resp = requests.post(f"{base_url}/chat/completions", json=payload)
    elapsed = time.time() - start
    data = resp.json()

    output = data["choices"][0]["message"]["content"]
    usage = data["usage"]

    print(f"\nOutput ({usage['completion_tokens']} tokens, {elapsed:.2f}s):")
    print(output)
    print(f"\nUsage: {json.dumps(usage, indent=2)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Full response saved to {args.output}")


def benchmark(args):
    """Run a simple benchmark against the server."""
    base_url = f"http://127.0.0.1:{args.port}/v1"
    model = args.model_path

    prompts = [
        "<html><body><h1>Breaking News</h1><p>AI advances in 2025.</p></body></html>",
        "Hello, how are you doing today?",
        "<html><body><p>Short paragraph.</p></body></html>",
        "<html><body><div><h2>Title</h2><p>Content here with <b>bold</b> text.</p></div></body></html>",
        "<html><body><ul><li>Item one</li><li>Item two</li><li>Item three</li></ul></body></html>",
    ]

    print(f"Benchmarking {model} on {base_url}")
    print(f"{'#':<5} {'Prompt':<50} {'Tokens':<8} {'Time(s)':<10} {'TPS':<10}")
    print("-" * 83)

    total_tokens = 0
    total_time = 0

    for i, prompt in enumerate(prompts):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        start = time.time()
        resp = requests.post(f"{base_url}/chat/completions", json=payload)
        elapsed = time.time() - start
        data = resp.json()
        usage = data["usage"]
        output = data["choices"][0]["message"]["content"]

        tokens = usage["completion_tokens"]
        tps = tokens / elapsed if elapsed > 0 else 0
        total_tokens += tokens
        total_time += elapsed

        print(f"{i:<5} {prompt[:48]:<50} {tokens:<8} {elapsed:<10.2f} {tps:<10.1f}")

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print("-" * 83)
    print(f"{'Total':<5} {'':<50} {total_tokens:<8} {total_time:<10.2f} {avg_tps:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description="reader-lm NPU Inference")
    parser.add_argument("model_path", nargs="?", default="/opt/atomgit/models/reader-lm-0.5b",
                        help="Model path (default: /opt/atomgit/models/reader-lm-0.5b)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start vLLM server")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    serve_parser.add_argument("--max-model-len", type=int, default=4096)
    serve_parser.add_argument("--max-num-seqs", type=int, default=8)
    serve_parser.add_argument("--gpu-memory", type=float, default=0.9)
    serve_parser.add_argument("--enforce-eager", action="store_true", default=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run inference")
    run_parser.add_argument("--port", type=int, default=8000)
    run_parser.add_argument("--prompt", type=str,
                            default="<html><body><h1>Hello</h1><p>World</p></body></html>")
    run_parser.add_argument("--max-tokens", type=int, default=128)
    run_parser.add_argument("--temperature", type=float, default=0.1)
    run_parser.add_argument("--top-p", type=float, default=0.9)
    run_parser.add_argument("--logprobs", action="store_true")
    run_parser.add_argument("--output", type=str, help="Save response to file")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("--port", type=int, default=8000)
    bench_parser.add_argument("--max-tokens", type=int, default=64)

    args = parser.parse_args()

    if args.command == "serve":
        start_server(args)
    elif args.command == "run":
        run_inference(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
