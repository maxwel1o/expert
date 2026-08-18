import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..adapters.hermes import reconcile_hermes_task
from ..model import Role
from ..store import ProgressStore


MAX_PAYLOAD_BYTES = 1024 * 1024
ALLOWED_EVENTS = frozenset(
    {"kanban_task_completed", "kanban_task_blocked"}
)
ALLOWED_ROLES = frozenset(
    {Role.DEPLOYER, Role.TESTER, Role.PROFILER, Role.ANALYST}
)
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class HookIgnored(ValueError):
    """The hook payload does not describe a supported registered transition."""


@dataclass(frozen=True)
class HermesHookEvent:
    event_name: str
    task_id: str
    role: Role


def parse_hook_payload(payload: object) -> HermesHookEvent:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HookIgnored("invalid_json") from exc
    if not isinstance(payload, dict):
        raise HookIgnored("payload_not_object")
    event_name = payload.get("hook_event_name")
    if event_name not in ALLOWED_EVENTS:
        raise HookIgnored("unsupported_event")
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise HookIgnored("missing_extra")
    task_id = extra.get("task_id")
    if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
        raise HookIgnored("invalid_task_id")
    try:
        role = Role(str(extra.get("assignee", "")).lower())
    except ValueError as exc:
        raise HookIgnored("invalid_role") from exc
    if role not in ALLOWED_ROLES:
        raise HookIgnored("invalid_role")
    return HermesHookEvent(event_name, task_id, role)


def _append_log(path: Path, code: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    safe_message = " ".join(str(message).splitlines())[:500]
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} code={code} message={safe_message}\n")


def handle_hook(
    stdin_text: str,
    progress_db: Path,
    source_db: Path,
    diagnostic_log: Path,
) -> int:
    if len(stdin_text.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        _append_log(diagnostic_log, "payload_too_large", "stdin exceeded 1 MiB")
        return 2
    try:
        event = parse_hook_payload(stdin_text)
    except HookIgnored:
        return 0
    try:
        reconcile_hermes_task(
            ProgressStore(progress_db),
            source_db,
            event.task_id,
            expected_role=event.role,
        )
        return 0
    except Exception as exc:
        code = getattr(exc, "code", exc.__class__.__name__)
        _append_log(diagnostic_log, str(code), str(exc))
        return 1


def main() -> int:
    progress_db = Path(
        os.environ.get(
            "TEAM_PROGRESS_DB",
            "/opt/data/team-progress/state/progress.db",
        )
    )
    source_db = Path(
        os.environ.get("TEAM_PROGRESS_HERMES_DB", "/opt/data/kanban.db")
    )
    diagnostic_log = Path(
        os.environ.get(
            "TEAM_PROGRESS_HOOK_LOG",
            "/opt/data/team-progress/state/hermes-hook.log",
        )
    )
    stdin_text = sys.stdin.read(MAX_PAYLOAD_BYTES + 1)
    return handle_hook(
        stdin_text,
        progress_db,
        source_db,
        diagnostic_log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
