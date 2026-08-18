#!/usr/bin/env python3
"""Create AtomGit/GitCode model repos and push files for all 68 models."""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

BASE_DIR = "/opt/atomgit/batch_14/models"
TOKEN = os.environ.get("ATOMGIT_USER_TOKEN", "1HjZjFiwBmp9fUKuYu7zTA4i")
USERNAME = "m0_74196153"
API_BASE = "https://api.gitcode.com/api/v5"

# Files to push (relative paths within each model dir)
FILES_TO_PUSH = [
    "inference.py",
    "compare_cpu_npu.py",
    "requirements.txt",
    "README.md",
    "terminal_screenshot.png",
    "inference_cpu.log",
    "inference_npu.log",
    "compare.log",
    "results_cpu.json",
    "results_npu.json",
    "compare_results.json",
]


def create_repo(model_name):
    """Create a GitCode model repo if it doesn't exist."""
    repo_name = f"{model_name}-npu"
    url = f"{API_BASE}/user/repos"
    data = json.dumps({
        "name": repo_name,
        "description": f"Ascend NPU adapted model: {model_name}",
        "repository_type": "model",
    }).encode()

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            print(f"  [OK] Repo created: {result.get('full_name')}")
            return result.get("id")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if "already exists" in body:
            print(f"  [OK] Repo already exists: {USERNAME}/{repo_name}")
            # Try to get repo id
            get_url = f"{API_BASE}/repos/{USERNAME}/{repo_name}"
            get_req = urllib.request.Request(get_url)
            get_req.add_header("Authorization", f"Bearer {TOKEN}")
            try:
                with urllib.request.urlopen(get_req) as resp:
                    result = json.loads(resp.read())
                    return result.get("id")
            except:
                pass
            return None
        else:
            print(f"  [ERR] Failed to create repo: {body}")
            return None


def push_model(model_name):
    """Initialize git repo and push files."""
    model_dir = os.path.join(BASE_DIR, model_name)
    repo_name = f"{model_name}-npu"
    remote_url = f"https://auth:{TOKEN}@gitcode.com/{USERNAME}/{repo_name}.git"

    if not os.path.isdir(model_dir):
        print(f"  [SKIP] Model directory not found: {model_dir}")
        return False

    try:
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=model_dir,
                       capture_output=True, timeout=30)
        subprocess.run(["git", "branch", "-M", "main"], cwd=model_dir,
                       capture_output=True, timeout=30)

        # Add .gitignore
        gitignore = os.path.join(model_dir, ".gitignore")
        if not os.path.exists(gitignore):
            with open(gitignore, "w") as f:
                f.write("*.pt\n__pycache__/\n")

        # Copy shared scripts if missing
        for script in ["inference.py", "compare_cpu_npu.py"]:
            script_path = os.path.join(model_dir, script)
            if not os.path.exists(script_path):
                src = os.path.join("/opt/atomgit/batch_14", script)
                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, script_path)

        # Copy requirements.txt if missing
        req_path = os.path.join(model_dir, "requirements.txt")
        if not os.path.exists(req_path):
            with open(req_path, "w") as f:
                f.write("torch>=2.0.0\ntimm>=0.9.0\nmodelscope>=1.0.0\n")
                f.write("torch_npu\nPillow>=9.0.0\n")

        # Add all files
        subprocess.run(["git", "add", "-A"], cwd=model_dir,
                       capture_output=True, timeout=30)

        # Check if there's anything to commit
        result = subprocess.run(["git", "status", "--porcelain"],
                                cwd=model_dir, capture_output=True, text=True, timeout=30)
        if not result.stdout.strip():
            print(f"  [SKIP] No changes to commit for {model_name}")
            # Still try to push if remote exists
            remotes = subprocess.run(["git", "remote", "-v"], cwd=model_dir,
                                     capture_output=True, text=True, timeout=30)
            if "origin" in remotes.stdout:
                subprocess.run(["git", "push", "-u", "origin", "main", "--force"],
                               cwd=model_dir, capture_output=True, timeout=120)
                print(f"  [OK] Pushed (existing repo): {model_name}")
                return True
            return True

        # Commit
        subprocess.run(["git", "commit", "-m", f"Add NPU adaptation for {model_name}"],
                       cwd=model_dir, capture_output=True, timeout=30)

        # Add remote and push
        subprocess.run(["git", "remote", "add", "origin", remote_url],
                       cwd=model_dir, capture_output=True, timeout=30)

        push_result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"],
                                     cwd=model_dir, capture_output=True, text=True, timeout=120)
        if push_result.returncode != 0:
            # Try to set URL with token embedded
            subprocess.run(["git", "remote", "set-url", "origin", remote_url],
                           cwd=model_dir, capture_output=True, timeout=30)
            push_result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"],
                                         cwd=model_dir, capture_output=True, text=True, timeout=120)
            if push_result.returncode != 0:
                print(f"  [ERR] Push failed: {push_result.stderr[:200]}")
                return False

        print(f"  [OK] Pushed: {model_name}")
        return True

    except Exception as e:
        print(f"  [ERR] Exception: {e}")
        return False


def main():
    # Get model list sorted
    model_names = sorted(os.listdir(BASE_DIR))
    # Filter only directories
    model_names = [m for m in model_names if os.path.isdir(os.path.join(BASE_DIR, m))]
    # Sort to match ALL_MODELS order
    from run_batch import ALL_MODELS
    # Filter ALL_MODELS to only those with directories
    model_names = [m for m in ALL_MODELS if m in model_names]

    print(f"Total models to publish: {len(model_names)}")

    results = []
    for i, model_name in enumerate(model_names, 1):
        print(f"\n[{i}/{len(model_names)}] {model_name}")
        print(f"  Creating repo...")
        repo_id = create_repo(model_name)
        if repo_id is None and "already exists" not in "":
            results.append((model_name, "FAIL: repo creation failed"))
            continue

        print(f"  Pushing files...")
        success = push_model(model_name)
        status = "OK" if success else "FAIL"
        results.append((model_name, status))
        print(f"  Status: {status}")

        # Small delay between models
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print("Publication Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  {status}: {name}")
    ok_count = sum(1 for _, s in results if s == "OK")
    print(f"\nTotal: {len(results)}, Success: {ok_count}, Failed: {len(results) - ok_count}")
    print(f"Repo URL pattern: https://gitcode.com/{USERNAME}/{{model_name}}-npu")


if __name__ == "__main__":
    main()
