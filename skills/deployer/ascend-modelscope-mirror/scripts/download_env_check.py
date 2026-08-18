#!/usr/bin/env python3
"""
模型下载环境检查脚本。
检测网络连通性、环境变量配置、缓存目录状态。

Usage:
    python download_env_check.py
"""

import os
import socket
import sys
import urllib.request


def check_connectivity(host: str, port: int = 443, timeout: int = 5) -> bool:
    """检查 TCP 连通性。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def check_http(url: str, timeout: int = 10) -> tuple[bool, int]:
    """检查 HTTP 可达性，返回 (ok, status_code)。"""
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, resp.status
    except Exception as e:
        return False, 0


def check_env_var(name: str, recommended: str = None):
    """检查环境变量状态。"""
    value = os.environ.get(name)
    if value:
        print(f"[OK]   {name} = {value}")
    elif recommended:
        print(f"[WARN] {name} 未设置 (推荐: {recommended})")
    else:
        print(f"[INFO] {name} 未设置")


def check_dir(path: str, label: str):
    """检查目录存在性、权限和剩余空间。"""
    if not os.path.exists(path):
        print(f"[INFO] {label} 不存在: {path}")
        return
    if not os.path.isdir(path):
        print(f"[WARN] {label} 不是目录: {path}")
        return

    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    perm = "RW" if (readable and writable) else ("R" if readable else "-")

    try:
        stat = os.statvfs(path)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        print(f"[OK]   {label}: {path} (权限={perm}, 剩余={free_gb:.1f}GB)")
    except Exception as e:
        print(f"[WARN] {label}: {path} (权限={perm}, 无法获取空间: {e})")


def main():
    print("=== NPU 模型下载环境检查 ===")
    print()

    print("--- 网络连通性 ---")
    sites = [
        ("huggingface.co", "HuggingFace 官方"),
        ("www.modelscope.cn", "ModelScope"),
        ("hf-mirror.com", "HF-Mirror"),
    ]
    for host, label in sites:
        ok = check_connectivity(host)
        status = "[OK]" if ok else "[FAIL]"
        print(f"{status} {label}: {host}:443")

    print()
    print("--- HTTP 可达性 ---")
    urls = [
        ("https://www.modelscope.cn", "ModelScope"),
        ("https://hf-mirror.com", "HF-Mirror"),
    ]
    for url, label in urls:
        ok, code = check_http(url)
        status = "[OK]" if ok else "[FAIL]"
        print(f"{status} {label}: {url} (HTTP {code})")

    print()
    print("--- 环境变量 ---")
    check_env_var("VLLM_USE_MODELSCOPE", recommended="true")
    check_env_var("HF_ENDPOINT", recommended="https://hf-mirror.com")
    check_env_var("HF_HOME")
    check_env_var("MODELSCOPE_CACHE")
    check_env_var("HF_HUB_ENABLE_HF_TRANSFER")

    print()
    print("--- 缓存目录 ---")
    check_dir(os.path.expanduser("~/.cache/huggingface"), "HF 默认缓存")
    check_dir(os.path.expanduser("~/.cache/modelscope"), "MS 默认缓存")
    if os.environ.get("HF_HOME"):
        check_dir(os.environ.get("HF_HOME"), "HF_HOME")
    if os.environ.get("MODELSCOPE_CACHE"):
        check_dir(os.environ.get("MODELSCOPE_CACHE"), "MODELSCOPE_CACHE")


if __name__ == "__main__":
    main()
