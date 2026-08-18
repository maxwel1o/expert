import argparse
import os
import tempfile
from pathlib import Path

import yaml


COMMAND = "/opt/data/team-progress/bin/hermes-kanban-progress-hook"
EVENTS = ("kanban_task_completed", "kanban_task_blocked")
HOOK = {"command": COMMAND, "timeout": 15}


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Hermes config root must be a mapping")
    return value


def _write_atomic(path: Path, value: dict) -> None:
    stat = path.stat()
    rendered = yaml.safe_dump(
        value,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    fd, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.st_mode & 0o777)
        if hasattr(os, "chown"):
            os.chown(temporary, stat.st_uid, stat.st_gid)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_hooks(value: dict) -> dict:
    hooks = value.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Hermes hooks config must be a mapping")
    for event in EVENTS:
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            raise ValueError(f"hooks.{event} must be a list")
        retained = [
            item
            for item in entries
            if not (
                isinstance(item, dict)
                and item.get("command") == COMMAND
            )
        ]
        retained.append(dict(HOOK))
        hooks[event] = retained
    return value


def remove_hooks(value: dict) -> dict:
    hooks = value.get("hooks")
    if not isinstance(hooks, dict):
        return value
    for event in EVENTS:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue
        retained = [
            item
            for item in entries
            if not (
                isinstance(item, dict)
                and item.get("command") == COMMAND
            )
        ]
        if retained:
            hooks[event] = retained
        else:
            hooks.pop(event, None)
    if not hooks:
        value.pop("hooks", None)
    return value


def verify_hooks(value: dict) -> None:
    hooks = value.get("hooks")
    if not isinstance(hooks, dict):
        raise ValueError("Hermes hooks config is missing")
    for event in EVENTS:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            raise ValueError(f"hooks.{event} is missing")
        matches = [
            item
            for item in entries
            if isinstance(item, dict) and item.get("command") == COMMAND
        ]
        if matches != [HOOK]:
            raise ValueError(
                f"hooks.{event} must contain one exact team-progress hook"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merge-hermes-hooks.py")
    parser.add_argument(
        "action", choices=("apply", "remove", "verify", "check")
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        value = _load(args.config)
        if args.action == "apply":
            _write_atomic(args.config, apply_hooks(value))
            print(f"CONFIG_UPDATED path={args.config}")
        elif args.action == "remove":
            _write_atomic(args.config, remove_hooks(value))
            print(f"CONFIG_UPDATED path={args.config}")
        elif args.action == "verify":
            verify_hooks(value)
            print(f"CONFIG_VERIFIED path={args.config}")
        else:
            print(f"CONFIG_VALID path={args.config}")
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
