#!/usr/bin/env python3
"""
EngineCore NPU 初始化失败诊断脚本。
一键收集环境信息，定位根因。
"""

import json
import os
import subprocess
import sys


def run_cmd(cmd, timeout=10):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1


def check_npu_smi():
    stdout, stderr, rc = run_cmd("npu-smi info 2>/dev/null")
    if rc != 0:
        return {"available": False, "error": stderr or "npu-smi not found"}
    lines = stdout.splitlines()
    devices = []
    for line in lines:
        if line.startswith("|") and "Ascend" in line and "NPU" not in line.split("|")[1]:
            parts = line.split("|")
            if len(parts) >= 3:
                devices.append(parts[1].strip())
    return {"available": True, "raw": stdout, "devices": devices}


def check_davinci_devices():
    stdout, _, rc = run_cmd("ls -la /dev/davinci* 2>/dev/null")
    if rc != 0:
        return []
    devices = []
    for line in stdout.splitlines():
        if line.startswith("c"):
            parts = line.split()
            name = parts[-1] if parts else ""
            devices.append(name)
    return devices


def check_torch_npu():
    try:
        import torch
        import torch_npu
        available = torch_npu.npu.is_available()
        count = torch_npu.npu.device_count() if available else 0
        current = torch_npu.npu.current_device() if available else -1
        version = torch_npu.__version__
        return {
            "import_ok": True,
            "available": available,
            "count": count,
            "current_device": current,
            "version": version,
        }
    except Exception as e:
        return {"import_ok": False, "error": str(e)}


def check_multiprocessing():
    try:
        import torch.multiprocessing as mp
        method = mp.get_start_method(allow_none=True) or "fork (default)"
        return {"method": method}
    except Exception as e:
        return {"method": "unknown", "error": str(e)}


def check_env_vars():
    return {
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(not set)"),
        "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "(not set)"),
        "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS", "(not set)"),
        "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF", "(not set)"),
        "ASCEND_TOOLKIT_HOME": os.environ.get("ASCEND_TOOLKIT_HOME", "(not set)"),
        "TASK_QUEUE_ENABLE": os.environ.get("TASK_QUEUE_ENABLE", "(not set)"),
    }


def check_cann_log_dir():
    user = os.environ.get("USER", os.environ.get("LOGNAME", "unknown"))
    log_dir = f"/home/{user}/ascend/log"
    if os.path.isdir(log_dir):
        return {"path": log_dir, "exists": True, "writable": os.access(log_dir, os.W_OK)}
    return {"path": log_dir, "exists": False, "writable": False}


def check_vllm_versions():
    versions = {}
    for pkg in ["vllm", "vllm_ascend"]:
        stdout, _, rc = run_cmd(f"pip show {pkg} 2>/dev/null | grep Version")
        if rc == 0:
            versions[pkg] = stdout.replace("Version:", "").strip()
        else:
            versions[pkg] = "(not installed)"
    return versions


def analyze(diagnosis):
    issues = []

    if not diagnosis["torch_npu"]["import_ok"]:
        issues.append("torch_npu 导入失败，CANN 或 torch_npu 未正确安装")
    elif not diagnosis["torch_npu"]["available"]:
        issues.append("NPU 不可用，驱动或设备有问题")

    mp_method = diagnosis["multiprocessing"]["method"]
    if "fork" in mp_method.lower():
        issues.append(f"multiprocessing 启动方式为 fork，子进程可能无法继承 NPU 上下文")

    env = diagnosis["env_vars"]
    if env["VLLM_WORKER_MULTIPROC_METHOD"] == "(not set)":
        issues.append("VLLM_WORKER_MULTIPROC_METHOD 未设置，EngineCore 默认使用 fork")

    visible = env["ASCEND_RT_VISIBLE_DEVICES"]
    if visible != "(not set)":
        issues.append(f"ASCEND_RT_VISIBLE_DEVICES={visible}，单卡容器环境可能因设备映射冲突导致子进程失败")

    log_dir = diagnosis["cann_log_dir"]
    if not log_dir["writable"]:
        issues.append(f"CANN 日志目录 {log_dir['path']} 不可写")

    if not diagnosis["davinci_devices"]:
        issues.append("未找到 /dev/davinci* 设备文件")

    # 确定根因和推荐方案
    root_cause = "unknown"
    recommended_fix = ""

    if not issues:
        root_cause = "none"
        recommended_fix = "none"
    elif any("fork" in i for i in issues) and any("VISIBLE_DEVICES" in i for i in issues):
        root_cause = "fork multiprocessing + ASCEND_RT_VISIBLE_DEVICES conflict"
        recommended_fix = "A+B"
    elif any("fork" in i for i in issues):
        root_cause = "fork multiprocessing method"
        recommended_fix = "A"
    elif any("VISIBLE_DEVICES" in i for i in issues):
        root_cause = "ASCEND_RT_VISIBLE_DEVICES conflict"
        recommended_fix = "B"
    elif any("日志" in i or "log" in i for i in issues):
        root_cause = "CANN log directory permission"
        recommended_fix = "C"
    else:
        root_cause = "multiple environment issues"
        recommended_fix = "A+B+C"

    return root_cause, recommended_fix, issues


def main():
    print("=" * 60)
    print("  vLLM-Ascend EngineCore NPU 初始化失败诊断工具")
    print("=" * 60)
    print()

    diagnosis = {
        "npu_smi": check_npu_smi(),
        "davinci_devices": check_davinci_devices(),
        "torch_npu": check_torch_npu(),
        "multiprocessing": check_multiprocessing(),
        "env_vars": check_env_vars(),
        "cann_log_dir": check_cann_log_dir(),
        "vllm_versions": check_vllm_versions(),
    }

    root_cause, recommended_fix, issues = analyze(diagnosis)
    diagnosis["root_cause"] = root_cause
    diagnosis["recommended_fix"] = recommended_fix
    diagnosis["issues_found"] = issues
    diagnosis["status"] = "diagnosed"

    # 打印摘要
    print("【NPU 状态】")
    print(f"  npu-smi: {'可用' if diagnosis['npu_smi']['available'] else '不可用'}")
    if diagnosis['npu_smi']['available']:
        print(f"  设备: {diagnosis['npu_smi']['devices']}")
    print(f"  /dev/davinci*: {diagnosis['davinci_devices'] or '未找到'}")

    tnp = diagnosis["torch_npu"]
    print(f"\n【torch_npu】")
    print(f"  导入: {'成功' if tnp['import_ok'] else '失败'}")
    if tnp['import_ok']:
        print(f"  可用: {tnp['available']}, 数量: {tnp['count']}, 当前设备: {tnp['current_device']}")
        print(f"  版本: {tnp['version']}")

    print(f"\n【Multiprocessing】")
    print(f"  启动方式: {diagnosis['multiprocessing']['method']}")

    print(f"\n【环境变量】")
    for k, v in diagnosis["env_vars"].items():
        print(f"  {k}: {v}")

    print(f"\n【vLLM 版本】")
    for k, v in diagnosis["vllm_versions"].items():
        print(f"  {k}: {v}")

    print(f"\n【CANN 日志目录】")
    log = diagnosis["cann_log_dir"]
    print(f"  路径: {log['path']}")
    print(f"  存在: {log['exists']}, 可写: {log['writable']}")

    print(f"\n{'=' * 60}")
    print(f"【诊断结论】")
    print(f"  根因: {root_cause}")
    print(f"  推荐修复方案: {recommended_fix}")
    if issues:
        print(f"\n  发现问题 ({len(issues)} 个):")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print(f"\n  未发现明显问题")
    print(f"{'=' * 60}")

    # 保存 JSON 报告
    report_path = "/tmp/npu_diagnosis_report.json"
    with open(report_path, "w") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
    print(f"\n详细报告已保存: {report_path}")

    # 返回码：0=无问题，1=有问题但可修复，2=严重问题
    if not issues:
        return 0
    elif tnp.get("available") and diagnosis["davinci_devices"]:
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
