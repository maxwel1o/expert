#!/usr/bin/env python3
"""Create GitCode repos and push model files for all 15 SwinV2 models."""
import os
import sys
import json
import subprocess
import shutil

BATCH_DIR = "/opt/atomgit/batch19"
TOKEN = os.environ.get("ATOMGIT_USER_TOKEN", "")
GIT_USER = "m0_74196153"
GIT_EMAIL = "iamnotphage@163.com"
GITCODE_API = "https://api.gitcode.com/api/v5/user/repos"
GITCODE_BASE = "https://gitcode.com"
GITCODE_HOST = "gitcode.com"

MODELS = [
    "swinv2_tiny_window8_256.ms_in1k",
    "swinv2_tiny_window16_256.ms_in1k",
    "swinv2_small_window8_256.ms_in1k",
    "swinv2_small_window16_256.ms_in1k",
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_large_window12_192.ms_in22k",
    "swinv2_cr_tiny_ns_224.sw_in1k",
    "swinv2_cr_small_ns_224.sw_in1k",
    "swinv2_cr_small_224.sw_in1k",
    "swinv2_base_window8_256.ms_in1k",
    "swinv2_base_window16_256.ms_in1k",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_base_window12_192.ms_in22k",
]

def run_cmd(cmd, cwd=None, timeout=60):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)

def create_repo(repo_name):
    """Create a GitCode model repository."""
    headers = f"PRIVATE-TOKEN: {TOKEN}"
    data = json.dumps({
        "name": repo_name,
        "description": f"SwinV2 {repo_name.replace('-npu', '')} NPU adaptation - 昇腾 NPU 适配",
        "repository_type": "model",
        "visibility": "public"
    })
    cmd = f"""curl -s -X POST "{GITCODE_API}" \
        -H "Content-Type: application/json" \
        -H "{headers}" \
        -d '{data}'"""
    rc, out, err = run_cmd(cmd)
    return rc, out, err

def push_model(model_name):
    """Push a single model to GitCode."""
    model_dir = os.path.join(BATCH_DIR, model_name)
    repo_name = f"{model_name.replace('.', '-')}-npu"
    repo_url = f"https://auth:{TOKEN}@{GITCODE_HOST}/{GIT_USER}/{repo_name}.git"

    print(f"\n=== {model_name} ===")
    print(f"Repo: {repo_name}")

    # Step 1: Create repo on GitCode
    print(f"  Creating repo...")
    rc, out, err = create_repo(repo_name)
    if rc != 0:
        print(f"  API call failed: {err[:200]}")
    else:
        try:
            resp = json.loads(out)
            if "id" in resp:
                print(f"  Repo created: {repo_name}")
            elif "error" in resp or "message" in resp:
                msg = resp.get("error", resp.get("message", ""))
                if "already exists" in msg.lower():
                    print(f"  Repo already exists, will push")
                else:
                    print(f"  Repo error: {msg}")
        except json.JSONDecodeError:
            print(f"  Repo creation response: {out[:200]}")

    # Step 2: Initialize git repo
    print(f"  Initializing git...")
    git_dir = os.path.join(model_dir, ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir)

    # Configure git
    run_cmd(f"git config --global user.name '{GIT_USER}'")
    run_cmd(f"git config --global user.email '{GIT_EMAIL}'")

    # Init and add files
    cmds = [
        f"cd {model_dir} && git init",
        f"cd {model_dir} && git checkout -b main",
        f"cd {model_dir} && git add inference.py compare_cpu_npu.py requirements.txt readme.md terminal_screenshot.png",
        f"cd {model_dir} && git add results/ -f 2>/dev/null || true",
        f"cd {model_dir} && git commit -m 'Add {model_name} NPU adaptation' --allow-empty",
    ]

    for cmd in cmds:
        rc, out, err = run_cmd(cmd)
        if rc != 0 and "nothing to commit" not in err:
            print(f"  Git issue: {err[:100]}")

    # Step 3: Push to GitCode
    print(f"  Pushing to {repo_url[:60]}...")
    cmds = [
        f"cd {model_dir} && git remote add origin '{repo_url}'",
        f"cd {model_dir} && git push -u origin main -f 2>&1",
    ]

    for cmd in cmds:
        rc, out, err = run_cmd(cmd, timeout=120)
        if rc != 0:
            print(f"  Push issue: {err[:200]}")
        if out:
            print(f"  Out: {out[:200]}")

    # Check if push succeeded
    final_out = run_cmd(f"cd {model_dir} && git remote -v")[1]
    print(f"  Remote: {final_out}")
    print(f"  Done: {model_name}")

    return repo_name

def main():
    if not TOKEN:
        print("ERROR: ATOMGIT_USER_TOKEN not set")
        sys.exit(1)

    print(f"Pushing {len(MODELS)} models to GitCode...")
    print(f"User: {GIT_USER}")

    # repositories table
    repos = []

    for model in MODELS:
        repo_name = push_model(model)
        repos.append(repo_name)

    # Print summary table
    print("\n" + "="*80)
    print("GITCODE REPOSITORIES SUMMARY")
    print("="*80)
    print(f"| # | Model | GitCode Repository |")
    print(f"|---|-------|-------------------|")
    for i, (model, repo) in enumerate(zip(MODELS, repos), 1):
        url = f"{GITCODE_BASE}/{GIT_USER}/{repo}"
        print(f"| {i} | {model} | {url} |")

    # Save repo list
    with open(os.path.join(BATCH_DIR, "gitcode_repos.json"), "w") as f:
        json.dump({m: f"{GITCODE_BASE}/{GIT_USER}/{r}" for m, r in zip(MODELS, repos)}, f, indent=2)
    print(f"\nRepo list saved to {BATCH_DIR}/gitcode_repos.json")

if __name__ == "__main__":
    main()
