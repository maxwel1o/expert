#!/usr/bin/env python3
"""
EngineCore NPU 初始化失败自动修复脚本。
根据 diagnose.py 的报告，自动应用推荐的修复方案。
"""

import json
import os
import subprocess
import sys


def run_diagnose():
    """运行诊断获取当前状态。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diag_script = os.path.join(script_dir, "diagnose.py")
    result = subprocess.run([sys.executable, diag_script], capture_output=True, text=True)
    report_path = "/tmp/npu_diagnosis_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            return json.load(f)
    return None


def apply_fix_a():
    """方案 A: 设置 VLLM_WORKER_MULTIPROC_METHOD=spawn"""
    print("[Fix A] 设置 VLLM_WORKER_MULTIPROC_METHOD=spawn ...")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # 也写入 bashrc/zshrc 以便持久化
    home = os.path.expanduser("~")
    for rc_file in [".bashrc", ".zshrc"]:
        rc_path = os.path.join(home, rc_file)
        if os.path.exists(rc_path):
            with open(rc_path, "r") as f:
                content = f.read()
            if "VLLM_WORKER_MULTIPROC_METHOD" not in content:
                with open(rc_path, "a") as f:
                    f.write("\nexport VLLM_WORKER_MULTIPROC_METHOD=spawn\n")
                print(f"  已追加到 {rc_path}")
    return {"fix": "A", "action": "set VLLM_WORKER_MULTIPROC_METHOD=spawn", "status": "applied"}


def apply_fix_b():
    """方案 B: 取消不当的 ASCEND_RT_VISIBLE_DEVICES"""
    print("[Fix B] 检查并调整 ASCEND_RT_VISIBLE_DEVICES ...")
    current = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if current:
        # 只有在单卡场景才建议取消
        try:
            import torch
            import torch_npu
            count = torch_npu.npu.device_count() if torch_npu.npu.is_available() else 0
        except Exception:
            count = 1  # 保守假设

        if count == 1 and "," not in current:
            del os.environ["ASCEND_RT_VISIBLE_DEVICES"]
            return {"fix": "B", "action": f"unset ASCEND_RT_VISIBLE_DEVICES (was {current})", "status": "applied"}
        else:
            return {"fix": "B", "action": f"保留 ASCEND_RT_VISIBLE_DEVICES={current} (多卡场景)", "status": "skipped"}
    return {"fix": "B", "action": "ASCEND_RT_VISIBLE_DEVICES 未设置，无需调整", "status": "skipped"}


def apply_fix_c():
    """方案 C: 修复 CANN 日志目录权限"""
    print("[Fix C] 修复 CANN 日志目录 ...")
    user = os.environ.get("USER", os.environ.get("LOGNAME", "unknown"))
    log_dir = f"/home/{user}/ascend/log"
    try:
        os.makedirs(log_dir, exist_ok=True)
        return {"fix": "C", "action": f"mkdir -p {log_dir}", "status": "applied"}
    except Exception as e:
        return {"fix": "C", "action": f"mkdir failed: {e}", "status": "failed"}


def apply_fix_d(model_path=None):
    """方案 D: 生成绕过子进程的替代命令 (bench 场景)"""
    print("[Fix D] 生成 bench 场景替代方案 ...")
    port = os.environ.get("VLLM_PORT", "8000")
    model = model_path or os.environ.get("MODEL_PATH", "/path/to/model")
    model_name = os.path.basename(model.rstrip("/"))

    serve_cmd = f"""VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve {model} \\
  --host 0.0.0.0 --port {port} \\
  --tensor-parallel-size 1 \\
  --trust-remote-code \\
  --max-model-len 65536 \\
  --gpu-memory-utilization 0.85"""

    bench_cmd = f"""vllm bench serve \\
  --model {model_name} \\
  --tokenizer {model} \\
  --host 127.0.0.1 --port {port} \\
  --dataset-name random \\
  --random-input 128 --random-output 128 \\
  --num-prompts 32 --request-rate 4"""

    print(f"\n  步骤1 - 启动服务:")
    print(f"    {serve_cmd}")
    print(f"\n  步骤2 - 压测 (另开终端):")
    print(f"    {bench_cmd}")
    print(f"\n  步骤3 - 测量延迟 (另开终端):")
    print(f"    python3 {os.path.dirname(__file__)}/measure_latency.py")

    return {"fix": "D", "action": "generated alternative bench commands", "status": "applied"}


def main():
    print("=" * 60)
    print("  vLLM-Ascend EngineCore NPU 初始化失败自动修复工具")
    print("=" * 60)
    print()

    # 读取诊断报告
    diagnosis = run_diagnose()
    if not diagnosis:
        print("错误: 无法读取诊断报告。请先运行 diagnose.py")
        sys.exit(1)

    recommended = diagnosis.get("recommended_fix", "")
    if recommended == "none":
        print("诊断显示当前环境无已知问题，无需修复。")
        sys.exit(0)

    print(f"诊断推荐修复方案: {recommended}")
    print()

    fixes = []

    if "A" in recommended:
        fixes.append(apply_fix_a())
    if "B" in recommended:
        fixes.append(apply_fix_b())
    if "C" in recommended:
        fixes.append(apply_fix_c())

    # 如果推荐中包含 bench 相关场景，也生成方案 D
    if any("bench" in i.lower() for i in diagnosis.get("issues_found", [])):
        fixes.append(apply_fix_d())

    # 保存修复报告
    report = {
        "fixes_applied": fixes,
        "verification": {
            "env_changes": {
                "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "(not set)"),
                "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "(not set)"),
            }
        },
        "status": "fixed"
    }

    report_path = "/tmp/npu_fix_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("【修复结果】")
    for fix in fixes:
        icon = "✅" if fix["status"] == "applied" else ("⚠️" if fix["status"] == "skipped" else "❌")
        print(f"  {icon} 方案 {fix['fix']}: {fix['action']} ({fix['status']})")
    print(f"{'=' * 60}")
    print(f"\n修复报告已保存: {report_path}")
    print("\n下一步验证:")
    print("  1. 重新启动 vLLM 服务")
    print("  2. 运行: curl -sf http://127.0.0.1:8000/v1/models")


if __name__ == "__main__":
    main()
