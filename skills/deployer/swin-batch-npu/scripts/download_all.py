#!/usr/bin/env python3
"""Pre-download all Swin models from ModelScope and populate HF cache."""
import hashlib
import os
import shutil
import sys

from modelscope.hub.snapshot_download import snapshot_download

MODELS = [
    "swin_tiny_patch4_window7_224.ms_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k",
    "swin_small_patch4_window7_224.ms_in1k",
    "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_small_patch4_window7_224.ms_in22k",
    "swin_s3_tiny_224.ms_in1k",
    "swin_s3_small_224.ms_in1k",
    "swin_s3_base_224.ms_in1k",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_large_patch4_window7_224.ms_in22k",
    "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
    "swin_large_patch4_window12_384.ms_in22k",
    "swin_base_patch4_window7_224.ms_in22k",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_base_patch4_window7_224.ms_in1k",
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "swin_base_patch4_window12_384.ms_in22k",
    "swin_base_patch4_window12_384.ms_in1k",
]

CACHE_DIR = "/opt/atomgit/.cache/modelscope"
HF_BASE = "/opt/atomgit/.cache/huggingface/hub"


def populate_hf_cache(model_name: str):
    """Download from ModelScope, then symlink into HF cache."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {model_name}")
    print(f"{'=' * 60}")

    # Download from ModelScope
    ms_path = snapshot_download(f"timm/{model_name}", cache_dir=CACHE_DIR)
    print(f"  Downloaded to: {ms_path}")

    # Find the actual model files directory (ModelScope creates symlinks)
    if os.path.islink(ms_path):
        real_path = os.readlink(ms_path)
        if not os.path.isabs(real_path):
            real_path = os.path.join(os.path.dirname(ms_path), real_path)
        ms_model_dir = real_path
    else:
        ms_model_dir = ms_path

    print(f"  Real path: {ms_model_dir}")

    # HF cache directory for this model
    hf_model_dir = os.path.join(HF_BASE, f"models--timm--{model_name.replace('/', '--')}")
    hf_blobs_dir = os.path.join(hf_model_dir, "blobs")
    hf_refs_dir = os.path.join(hf_model_dir, "refs")
    hf_snapshots_dir = os.path.join(hf_model_dir, "snapshots")

    os.makedirs(hf_blobs_dir, exist_ok=True)
    os.makedirs(hf_refs_dir, exist_ok=True)
    os.makedirs(hf_snapshots_dir, exist_ok=True)

    # Write a dummy ref
    commit_hash = "0" * 40
    with open(os.path.join(hf_refs_dir, "main"), "w") as f:
        f.write(commit_hash)

    snapshot_dir = os.path.join(hf_snapshots_dir, commit_hash)
    os.makedirs(snapshot_dir, exist_ok=True)

    # Symlink all model files from ModelScope to HF snapshot
    for fname in os.listdir(ms_model_dir):
        if fname.startswith('.'):
            continue
        src = os.path.join(ms_model_dir, fname)
        if os.path.isdir(src):
            continue
        dst = os.path.join(snapshot_dir, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"  Linked: {fname}")

        # Also compute sha256 and put in blobs for safetensors
        if fname.endswith('.safetensors') or fname.endswith('.bin'):
            with open(src, 'rb') as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            blob_path = os.path.join(hf_blobs_dir, sha256)
            if not os.path.exists(blob_path):
                shutil.copy2(src, blob_path)
                print(f"  Blob copied: {sha256[:16]}...")

    print(f"  HF cache ready for: {model_name}")


def main():
    n = len(MODELS)
    for i, model_name in enumerate(MODELS, 1):
        print(f"\n[{i}/{n}] {model_name}")
        populate_hf_cache(model_name)

    print(f"\nAll {n} models cached successfully!")


if __name__ == "__main__":
    main()
