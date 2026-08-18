#!/usr/bin/env python3
"""Verify every bundled Skill file against manifests/skills.sha256."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "skills.sha256"


def main() -> int:
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch: {relative}")
        checked += 1
    actual_files = sum(1 for p in (ROOT / "skills").rglob("*") if p.is_file())
    if checked != actual_files:
        raise SystemExit(
            f"checksum manifest covers {checked} files, directory contains {actual_files}"
        )
    print(f"skill checksums passed: {checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
