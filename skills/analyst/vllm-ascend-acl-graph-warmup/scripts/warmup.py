#!/usr/bin/env python3
"""
vLLM-Ascend ACL Graph warmup script.
Sends a single request to trigger ACL Graph compilation before real traffic.

Usage:
    python warmup.py \
        --api-url http://127.0.0.1:8000/v1/chat/completions \
        --model-name gemma-3-270m-it \
        --timeout 120
"""

import argparse
import time
import sys

import requests


def wait_for_service(api_url: str, max_wait: int = 60):
    """Poll the /v1/models endpoint until the service is ready."""
    print("Waiting for vLLM service to be ready...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(api_url.replace("/v1/chat/completions", "/v1/models"), timeout=5)
            if resp.status_code == 200:
                print("Service is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("Timeout: service did not become ready.")
    return False


def warmup(api_url: str, model_name: str, max_tokens: int = 16, timeout: int = 120):
    """Send a warmup request with a long timeout."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    print(f"Sending warmup request (timeout={timeout}s)...")
    start = time.perf_counter()
    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        end = time.perf_counter()
        elapsed = end - start
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"Warmup complete in {elapsed:.1f}s")
        print(f"Response: {content[:100]}")
        if elapsed > 10:
            print("Note: High latency detected, likely due to ACL Graph compilation.")
            print("      Subsequent requests should be faster.")
        return True
    except requests.exceptions.Timeout:
        print(f"Warmup request timed out after {timeout}s.")
        print("Suggestion: increase --timeout or check vLLM logs for compilation errors.")
        return False
    except Exception as e:
        print(f"Warmup request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="vLLM-Ascend ACL Graph warmup")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1/chat/completions",
                        help="OpenAI-compatible chat completions endpoint")
    parser.add_argument("--model-name", required=True, help="Model name (served-model-name)")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max tokens for warmup request")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--wait-timeout", type=int, default=60, help="Max seconds to wait for service")
    args = parser.parse_args()

    if not wait_for_service(args.api_url, max_wait=args.wait_timeout):
        sys.exit(1)

    success = warmup(args.api_url, args.model_name, max_tokens=args.max_tokens, timeout=args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
