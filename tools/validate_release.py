#!/usr/bin/env python3
"""Validate the public five-role Hermes team release."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("leader", "deployer", "tester", "profiler", "analyst")
WORKERS = ROLES[1:]
EXPECTED_SKILLS = {
    "leader": 1,
    "deployer": 127,
    "tester": 19,
    "profiler": 5,
    "analyst": 49,
}
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env|credentials?|.*\.(db|sqlite|sqlite3))$", re.I)
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style key": re.compile(r"(?i)sk-(?:proj-)?(?!x{20,}\b)[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_roles() -> None:
    actual = sorted(p.parent.name for p in (ROOT / "roles").glob("*/SOUL.md"))
    if actual != sorted(ROLES):
        fail(f"role set mismatch: {actual}")


def read_manifest() -> list[dict[str, str]]:
    path = ROOT / "manifests" / "skills.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sum(EXPECTED_SKILLS.values()):
        fail(f"expected 201 manifest rows, found {len(rows)}")
    return rows


def validate_skills(rows: list[dict[str, str]]) -> None:
    counts = {role: 0 for role in ROLES}
    for row in rows:
        role = row.get("role", "")
        rel = row.get("skill_name", "")
        if role not in counts:
            fail(f"unknown manifest role: {role!r}")
        skill_file = ROOT / "skills" / role / rel / "SKILL.md"
        if not skill_file.is_file():
            fail(f"missing Skill entry point: {skill_file.relative_to(ROOT)}")
        counts[role] += 1
    if counts != EXPECTED_SKILLS:
        fail(f"skill counts mismatch: {counts}")


def validate_hygiene() -> None:
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if "\ufffd" in rel:
            fail(f"Unicode replacement character in path: {rel}")
        if rel == "tests/team_progress/test_security.py":
            continue
        if rel != ".env.example" and FORBIDDEN_NAMES.search(rel):
            fail(f"forbidden runtime/secret file: {rel}")
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                fail(f"possible {label} in {rel}")


def main() -> int:
    validate_roles()
    rows = read_manifest()
    validate_skills(rows)
    validate_hygiene()
    print("release validation passed")
    print("roles: 5 (1 Leader + 4 independent Workers)")
    print("skills: 201")
    for role in ROLES:
        print(f"  {role}: {EXPECTED_SKILLS[role]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
