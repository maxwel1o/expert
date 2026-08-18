#!/usr/bin/env python3
"""
vLLM-Ascend 环境信息收集脚本

自动收集 vLLM-Ascend 运行环境的关键信息，用于故障排查。
输出结构化的环境报告，可直接用于排障分析。

用法:
    python collect_vllm_ascend_env.py
    python collect_vllm_ascend_env.py --output env_report.json
    python collect_vllm_ascend_env.py --output env_report.json --model-path /path/to/model
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime


def run_cmd(cmd, timeout=15):
    """Run a shell command and return stdout, stderr, returncode."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except Exception as e:
        return "", str(e), -1


def get_os_info():
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
    }


def get_npu_info():
    info = {"available": False, "count": 0, "devices": [], "smi_output": ""}

    # npu-smi info (truncate to max 120 lines)
    stdout, stderr, rc = run_cmd("npu-smi info")
    if rc == 0:
        lines = stdout.split("\n")
        if len(lines) > 120:
            info["smi_output"] = "\n".join(lines[:120]) + f"\n... (truncated, {len(lines)} total lines)"
        else:
            info["smi_output"] = stdout
        # Parse device count from npu-smi table
        device_ids = set()
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                device_ids.add(parts[0])
        info["count"] = len(device_ids)
        info["available"] = info["count"] > 0
    else:
        info["smi_error"] = stderr or "npu-smi not found"

    # torch.npu info
    try:
        import torch
        import torch_npu

        info["torch_npu_available"] = torch.npu.is_available()
        info["torch_npu_device_count"] = (
            torch.npu.device_count() if torch.npu.is_available() else 0
        )
        if torch.npu.is_available() and info["torch_npu_device_count"] > 0:
            try:
                info["device_name"] = torch.npu.get_device_name(0)
                info["device_capability"] = torch.npu.get_device_capability(0)
            except Exception:
                pass
    except ImportError as e:
        info["torch_npu_error"] = str(e)

    return info


def get_driver_info():
    info = {}
    version_file = "/usr/local/Ascend/driver/version.cfg"
    if os.path.exists(version_file):
        with open(version_file) as f:
            info["version_cfg"] = f.read().strip()

    stdout, _, rc = run_cmd("cat /usr/local/Ascend/driver/version.cfg 2>/dev/null")
    if rc == 0:
        info["driver_version_raw"] = stdout

    return info


def get_cann_info():
    info = {}
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    info["ASCEND_HOME_PATH_set"] = bool(ascend_home)

    if not ascend_home:
        ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
        info["ASCEND_TOOLKIT_HOME_set"] = bool(ascend_home)

    if ascend_home and os.path.isdir(ascend_home):
        version_file = os.path.join(ascend_home, "version.cfg")
        if os.path.exists(version_file):
            with open(version_file) as f:
                info["cann_version"] = f.read().strip()

        lib_path = os.path.join(ascend_home, "lib64")
        info["lib64_exists"] = os.path.isdir(lib_path)
        info["libascend_hal_exists"] = os.path.exists(
            os.path.join(lib_path, "libascend_hal.so")
        )
    else:
        info["error"] = "ASCEND_HOME_PATH not set or invalid"

    return info


def get_python_packages():
    packages = {}
    pkg_names = [
        "torch",
        "torch_npu",
        "vllm",
        "vllm_ascend",
        "transformers",
        "accelerate",
        "safetensors",
        "tokenizers",
        "numpy",
        "Pillow",
        "timm",
        "peft",
        "deepspeed",
    ]
    for name in pkg_names:
        try:
            import importlib.metadata

            # Try different package name conventions
            for pkg_name in [name, name.replace("_", "-")]:
                try:
                    ver = importlib.metadata.version(pkg_name)
                    packages[name] = ver
                    break
                except importlib.metadata.PackageNotFoundError:
                    continue
            else:
                # Fallback: try importing and checking __version__
                try:
                    mod = importlib.import_module(name)
                    packages[name] = getattr(mod, "__version__", "installed(no version)")
                except ImportError:
                    packages[name] = "not installed"
        except Exception:
            packages[name] = "error"

    # Special: check if torch has CUDA (indicates wrong torch version)
    try:
        import torch

        packages["torch_cuda_available"] = torch.cuda.is_available()
        packages["torch_npu_support"] = hasattr(torch, "npu")
    except ImportError:
        pass

    return packages


def _contains_keyword(val, keywords):
    """Check if a value contains any of the given keywords (case-insensitive)."""
    val_lower = val.lower()
    return any(kw.lower() in val_lower for kw in keywords)


def get_env_vars():
    """Collect relevant environment variables. Sensitive paths are sanitized."""
    # Boolean-only vars: only report whether set, not the value
    boolean_only = {"ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_AICPU_PATH"}

    # Long path vars: only report keyword presence, not full value
    long_path_vars = {"LD_LIBRARY_PATH", "PATH", "PYTHONPATH"}

    # Simple vars: report value as-is
    simple_vars = [
        "PYTORCH_NPU_ALLOC_CONF",
        "TASK_QUEUE_ENABLE",
        "CPU_AFFINITY_CONF",
        "COMBINED_ENABLE",
        "ASCEND_SLOG_PRINT_TO_STDOUT",
        "ASCEND_GLOBAL_LOG_LEVEL",
        "HCCL_IF_IP",
        "HCCL_CONNECT_TIMEOUT",
        "HCCL_DEBUG",
        "HCCL_DEBUG_SUBSYS",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "OMPI_COMM_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ]

    env = {}

    # Boolean-only: just whether set
    for var in boolean_only:
        val = os.environ.get(var)
        if val is not None:
            env[var] = "SET"
        else:
            env[var] = "NOT SET"

    # Long path vars: only report keyword presence
    for var in long_path_vars:
        val = os.environ.get(var)
        if val is not None:
            env[var] = {
                "set": True,
                "contains_ascend": _contains_keyword(val, ["ascend"]),
                "contains_vllm": _contains_keyword(val, ["vllm"]),
                "length": len(val),
            }
        else:
            env[var] = {"set": False}

    # Simple vars: report value
    for var in simple_vars:
        val = os.environ.get(var)
        if val is not None:
            env[var] = val

    return env


def get_system_resources():
    info = {}

    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["total_ram_kb"] = int(line.split()[1])
                    info["total_ram_gb"] = round(info["total_ram_kb"] / 1024 / 1024, 1)
                    break
    except Exception:
        pass

    # CPU
    info["cpu_count"] = os.cpu_count()
    info["cpu_arch"] = platform.machine()

    # Disk
    try:
        total, used, free = shutil.disk_usage("/")
        info["disk_total_gb"] = round(total / 1024**3, 1)
        info["disk_free_gb"] = round(free / 1024**3, 1)
    except Exception:
        pass

    return info


def check_model_path(model_path):
    """Check model directory for required files."""
    if not model_path:
        return {"provided": False}

    info = {
        "provided": True,
        "path": model_path,
        "exists": os.path.isdir(model_path),
    }

    if not info["exists"]:
        return info

    # Check required files
    required = ["config.json"]
    recommended = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]

    info["files"] = {}
    for f in required + recommended:
        path = os.path.join(model_path, f)
        info["files"][f] = {
            "exists": os.path.exists(path),
            "required": f in required,
        }

    # Check weight files
    import glob

    weight_files = []
    for ext in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        weight_files.extend(glob.glob(os.path.join(model_path, ext)))

    if weight_files:
        total_size = sum(os.path.getsize(f) for f in weight_files)
        info["weight_files"] = {
            "count": len(weight_files),
            "total_size_gb": round(total_size / 1024**3, 2),
            "files": [os.path.basename(f) for f in sorted(weight_files)],
        }

    # Try reading config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            info["model_config"] = {
                "model_type": config.get("model_type", "unknown"),
                "architectures": config.get("architectures", []),
                "hidden_size": config.get("hidden_size"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "vocab_size": config.get("vocab_size"),
                "torch_dtype": config.get("torch_dtype"),
                "quantization_config": config.get("quantization_config") is not None,
            }
        except Exception as e:
            info["config_parse_error"] = str(e)

    return info


def run_torch_npu_test():
    """Run a basic torch_npu functionality test."""
    result = {"tested": False}
    try:
        import torch
        import torch_npu

        result["tested"] = True

        # Test 1: basic tensor
        x = torch.randn(2, 3).npu()
        result["basic_tensor"] = "PASS"

        # Test 2: matmul
        a = torch.randn(64, 64).npu()
        b = torch.randn(64, 64).npu()
        c = torch.mm(a, b)
        result["matmul"] = "PASS" if c.shape == (64, 64) else "FAIL"

        # Test 3: memory info
        if torch.npu.is_available():
            mem = torch.npu.memory_stats()
            result["memory_allocated_mb"] = round(
                mem.get("allocated_bytes.all.current", 0) / 1024**2, 1
            )

        result["overall"] = "PASS"

    except Exception as e:
        result["overall"] = "FAIL"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect vLLM-Ascend environment information for troubleshooting"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON file path (default: print to stdout)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Model path to check (optional)",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        default=True,
        help="Run torch_npu basic test (default: True)",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip torch_npu basic test",
    )
    args = parser.parse_args()

    report = {
        "collected_at": datetime.now().isoformat(),
        "script_version": "1.0.0",
    }

    print("=" * 60)
    print("vLLM-Ascend Environment Collection")
    print("=" * 60)

    # 1. OS info
    print("\n[1/8] Collecting OS info...")
    report["os"] = get_os_info()

    # 2. NPU info
    print("[2/8] Collecting NPU info...")
    report["npu"] = get_npu_info()

    # 3. Driver info
    print("[3/8] Collecting driver info...")
    report["driver"] = get_driver_info()

    # 4. CANN info
    print("[4/8] Collecting CANN info...")
    report["cann"] = get_cann_info()

    # 5. Python packages
    print("[5/8] Collecting Python packages...")
    report["packages"] = get_python_packages()

    # 6. Environment variables
    print("[6/8] Collecting environment variables...")
    report["env_vars"] = get_env_vars()

    # 7. System resources
    print("[7/8] Collecting system resources...")
    report["system"] = get_system_resources()

    # 8. Model path check
    if args.model_path:
        print("[8/8] Checking model path...")
        report["model"] = check_model_path(args.model_path)
    else:
        print("[8/8] Skipping model path check (not provided)")

    # 9. Basic test
    if args.run_test and not args.no_test:
        print("\n[TEST] Running torch_npu basic test...")
        report["torch_npu_test"] = run_torch_npu_test()

    # Quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"  OS:             {report['os']['system']} {report['os']['machine']}")
    print(f"  Python:         {report['os']['python_version']}")
    print(f"  NPU available:  {report['npu'].get('available', 'unknown')}")
    print(f"  NPU count:      {report['npu'].get('count', 'unknown')}")
    print(f"  CANN:           {report['cann'].get('cann_version', 'unknown')}")
    print(f"  torch:          {report['packages'].get('torch', 'unknown')}")
    print(f"  torch_npu:      {report['packages'].get('torch_npu', 'unknown')}")
    print(f"  vllm:           {report['packages'].get('vllm', 'unknown')}")
    print(f"  vllm_ascend:    {report['packages'].get('vllm_ascend', 'unknown')}")
    print(f"  transformers:   {report['packages'].get('transformers', 'unknown')}")

    if "torch_npu_test" in report:
        test = report["torch_npu_test"]
        print(f"  Basic test:     {test.get('overall', 'skipped')}")

    if "model" in report and report["model"].get("provided"):
        m = report["model"]
        print(f"  Model path:     {m.get('path', '')}")
        print(f"  Model exists:   {m.get('exists', 'unknown')}")
        if "model_config" in m:
            print(f"  Model type:     {m['model_config'].get('model_type', 'unknown')}")
            print(f"  Architecture:   {m['model_config'].get('architectures', [])}")

    # Potential issues quick check
    print("\n" + "=" * 60)
    print("POTENTIAL ISSUES")
    print("=" * 60)
    issues = []

    if not report["npu"].get("available"):
        issues.append("[BLOCKING] NPU not available")

    if report["packages"].get("torch_npu") == "not installed":
        issues.append("[BLOCKING] torch_npu not installed")

    if report["packages"].get("torch_cuda_available"):
        issues.append(
            "[WARNING] PyTorch has CUDA support - ensure you're using NPU-specific torch"
        )

    if not report["cann"].get("ASCEND_HOME_PATH_set"):
        issues.append("[WARNING] ASCEND_HOME_PATH not set")

    if "error" in report["cann"]:
        issues.append(f"[WARNING] CANN: {report['cann']['error']}")

    if report["packages"].get("torch") != "not installed" and report["packages"].get(
        "torch_npu"
    ) != "not installed":
        # Check version match
        torch_ver = report["packages"].get("torch", "").split("+")[0]
        npu_ver = report["packages"].get("torch_npu", "").split("+")[0].split(".post")[0]
        if torch_ver and npu_ver and torch_ver != npu_ver:
            issues.append(
                f"[WARNING] torch ({torch_ver}) and torch_npu ({npu_ver}) version mismatch"
            )

    if "torch_npu_test" in report and report["torch_npu_test"].get("overall") == "FAIL":
        issues.append(
            f"[BLOCKING] torch_npu basic test failed: {report['torch_npu_test'].get('error', 'unknown')}"
        )

    if not issues:
        print("  No obvious issues detected.")

    for issue in issues:
        print(f"  {issue}")

    # Output
    print()
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Full report saved to: {args.output}")
    else:
        print("Full report (JSON):")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
