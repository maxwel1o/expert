#!/usr/bin/env python3
"""Create and push a GitCode model repository for a LeViT NPU adaptation.

Usage:
  export ATOMGIT_USER_TOKEN="your_token"
  python3 publish_repo.py --model levit-128 --files ./levit-128-npu/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def create_repo(api_token, repo_name):
    cmd = (
        f'curl -s --location "https://api.gitcode.com/api/v5/user/repos" '
        f'--header "PRIVATE-TOKEN: {api_token}" '
        f'--header "Content-Type: application/json" '
        f'--data \'{{"name": "{repo_name}", "repository_type": "model", "private": false}}\''
    )
    ret, out, err = run_cmd(cmd)
    data = json.loads(out) if out else {}
    if "id" in data:
        print(f"  Created repo: {data['web_url']}")
        return data
    elif "error_message" in data and "already exists" in data.get("error_message", ""):
        print(f"  Repo already exists: {repo_name}")
        return data
    else:
        print(f"  Error creating repo: {data.get('error_message', out)}")
        return None


def push_files(api_token, repo_name, files_dir):
    repo_url = f"https://auth:{api_token}@gitcode.com/gcw_C8PI9e90/{repo_name}.git"
    files_dir = Path(files_dir)

    if not files_dir.exists():
        print(f"  Directory not found: {files_dir}")
        return False

    # Init git
    run_cmd("git init", cwd=files_dir)
    run_cmd("git checkout -b main", cwd=files_dir)

    # Add files
    run_cmd("git add -A", cwd=files_dir)

    # Commit
    ret, out, err = run_cmd(
        f'git commit -m "Add {repo_name} NPU adaptation"',
        cwd=files_dir
    )
    if ret != 0 and "nothing to commit" not in err:
        print(f"  Commit error: {err}")
        return False

    # Add remote and push
    run_cmd(f"git remote remove origin", cwd=files_dir)
    run_cmd(f"git remote add origin {repo_url}", cwd=files_dir)
    ret, out, err = run_cmd("git push -u origin main", cwd=files_dir)

    if "rejected" in err:
        print("  Push rejected, force pushing...")
        ret, out, err = run_cmd("git push -f origin main", cwd=files_dir)

    if ret == 0:
        print(f"  Pushed to {repo_url}")
        return True
    else:
        print(f"  Push error: {err[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Publish LeViT NPU model repo")
    parser.add_argument("--model", required=True, help="Model name (e.g. levit-128)")
    parser.add_argument("--files", default=None, help="Directory with model files")
    parser.add_argument("--token", default=None, help="GitCode API token (default: ATOMGIT_USER_TOKEN env)")
    args = parser.parse_args()

    api_token = args.token or os.environ.get("ATOMGIT_USER_TOKEN") or os.environ.get("TOMGIT_USER_TOKEN")
    if not api_token:
        print("ERROR: No API token found. Set ATOMGIT_USER_TOKEN env var.")
        sys.exit(1)

    repo_name = f"{args.model}-npu"
    files_dir = args.files or f"./output/{repo_name}"

    print(f"Publishing {repo_name}...")

    # Create repo
    create_repo(api_token, repo_name)

    # Push files
    push_files(api_token, repo_name, files_dir)

    print(f"\nRepo URL: https://gitcode.com/gcw_C8PI9e90/{repo_name}")


if __name__ == "__main__":
    main()
