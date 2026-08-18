import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..model import EventKind, Role, TaskStatus
from ..reconciliation import ReconciliationResult
from ..security import UnsafeTextError, validate_safe_text
from ..store import ProgressStore


@dataclass(frozen=True)
class HermesTaskRef:
    task_id: str
    role: Role
    status: str


ACTIVE = frozenset({"todo", "queued", "running", "blocked"})
WORKERS = frozenset(
    {Role.DEPLOYER, Role.TESTER, Role.PROFILER, Role.ANALYST}
)
QUEUED_STATES = frozenset({"triage", "todo", "scheduled", "ready"})
RUNNING_STATES = frozenset({"running", "review"})
_ARTIFACT_KEYS = frozenset({"artifact_refs", "artifacts", "outputs"})
_PATH_TOKEN = re.compile(
    r"(?<![\w])(?:[A-Za-z]:[\\/]|/|\./|\.\./)?"
    r"[\w.@%+~\\/:-]+\.(?:md|jsonl?|csv|txt|html|trace)(?![\w])",
    re.IGNORECASE,
)


class HermesAdapterError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class HermesTaskSnapshot:
    task_id: str
    role: Role
    source_status: str
    canonical_status: TaskStatus | None
    message: str
    artifact_refs: tuple[str, ...]
    source_timestamp: str
    source_run_id: str
    source_version: str


@dataclass(frozen=True)
class HermesReconcileReport:
    job_id: str
    observed: int
    changed: int
    unchanged: int
    diagnostics: int
    finalized: bool


def canonicalize_hermes_status(
    source_status: str, latest_outcome: str | None
) -> TaskStatus | None:
    status = str(source_status).strip().lower()
    outcome = (
        str(latest_outcome).strip().lower()
        if latest_outcome is not None
        else None
    )
    if status in QUEUED_STATES:
        return TaskStatus.QUEUED
    if status in RUNNING_STATES:
        return TaskStatus.RUNNING
    if status == "blocked":
        return TaskStatus.BLOCKED
    if status == "done":
        return TaskStatus.COMPLETED
    if status == "archived" and outcome == "completed":
        return TaskStatus.COMPLETED
    return None


def _schema_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _validate_schema(conn: sqlite3.Connection) -> None:
    required = {
        "tasks": {
            "id",
            "assignee",
            "status",
            "result",
            "started_at",
            "completed_at",
            "current_run_id",
        },
        "task_runs": {
            "id",
            "task_id",
            "status",
            "outcome",
            "summary",
            "ended_at",
        },
        "task_events": {
            "id",
            "task_id",
            "kind",
            "payload",
            "created_at",
        },
    }
    missing = {
        table: sorted(columns - _schema_columns(conn, table))
        for table, columns in required.items()
    }
    missing = {table: columns for table, columns in missing.items() if columns}
    if missing:
        raise HermesAdapterError(
            "schema_incompatible",
            "Hermes Kanban schema is missing required lifecycle fields",
        )


def _json_value(raw: object) -> object | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _artifact_values(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    result: list[str] = []
    for key in _ARTIFACT_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str):
            result.append(candidate)
        elif isinstance(candidate, list):
            result.extend(item for item in candidate if isinstance(item, str))
    return result


def _safe_message(raw: object, source_status: str) -> str:
    parsed = _json_value(raw)
    if isinstance(parsed, dict):
        candidate = parsed.get("message", parsed.get("summary", ""))
    else:
        candidate = raw
    text = str(candidate or f"Hermes task reached {source_status}")
    try:
        return validate_safe_text(text[:2000], "Hermes summary")
    except UnsafeTextError:
        return f"Hermes task reached {source_status}; summary redacted"


def _safe_artifacts(*raw_values: object) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw in raw_values:
        parsed = _json_value(raw)
        candidates.extend(_artifact_values(parsed))
        if isinstance(raw, str) and parsed is None:
            candidates.extend(match.group(0) for match in _PATH_TOKEN.finditer(raw))
    result: list[str] = []
    for candidate in candidates:
        try:
            safe = validate_safe_text(candidate, "artifact")
        except UnsafeTextError:
            continue
        if safe and safe not in result:
            result.append(safe)
    return tuple(result)


def _read_latest_run(
    conn: sqlite3.Connection, task_id: str, current_run_id: object
) -> sqlite3.Row | None:
    if current_run_id is not None:
        row = conn.execute(
            "SELECT id,status,outcome,summary,ended_at FROM task_runs "
            "WHERE id=? AND task_id=?",
            (current_run_id, task_id),
        ).fetchone()
        if row:
            return row
    return conn.execute(
        "SELECT id,status,outcome,summary,ended_at FROM task_runs "
        "WHERE task_id=? ORDER BY COALESCE(ended_at,'' ) DESC,id DESC LIMIT 1",
        (task_id,),
    ).fetchone()


def _read_latest_event(
    conn: sqlite3.Connection, task_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id,kind,payload,created_at FROM task_events "
        "WHERE task_id=? ORDER BY COALESCE(created_at,'' ) DESC,id DESC LIMIT 1",
        (task_id,),
    ).fetchone()


def read_task_snapshots(
    source_db: Path, task_ids: Sequence[str]
) -> dict[str, HermesTaskSnapshot]:
    path = Path(source_db)
    if not path.is_file():
        raise HermesAdapterError(
            "source_db_missing", f"Hermes Kanban database not found: {path}"
        )
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        raise HermesAdapterError("source_db_unreadable", str(exc)) from exc
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        _validate_schema(conn)
        unique_ids = tuple(dict.fromkeys(str(item) for item in task_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        rows = conn.execute(
            "SELECT id,assignee,status,result,started_at,completed_at,"
            f"current_run_id FROM tasks WHERE id IN ({placeholders})",
            unique_ids,
        )
        result: dict[str, HermesTaskSnapshot] = {}
        for task in rows:
            try:
                role = Role(str(task["assignee"]).lower())
            except ValueError:
                continue
            run = _read_latest_run(conn, task["id"], task["current_run_id"])
            event = _read_latest_event(conn, task["id"])
            outcome = run["outcome"] if run else None
            canonical = canonicalize_hermes_status(task["status"], outcome)
            summary = run["summary"] if run and run["summary"] else task["result"]
            event_payload = event["payload"] if event else None
            timestamp = (
                task["completed_at"]
                or (run["ended_at"] if run else None)
                or (event["created_at"] if event else None)
                or task["started_at"]
                or "none"
            )
            run_id = str(run["id"]) if run else "none"
            event_id = str(event["id"]) if event else "none"
            source_status = str(task["status"]).lower()
            result[task["id"]] = HermesTaskSnapshot(
                task_id=task["id"],
                role=role,
                source_status=source_status,
                canonical_status=canonical,
                message=_safe_message(summary, source_status),
                artifact_refs=_safe_artifacts(summary, event_payload),
                source_timestamp=str(timestamp),
                source_run_id=run_id,
                source_version=(
                    f"{source_status}:{run_id}:{event_id}:{timestamp}"
                ),
            )
        return result
    except HermesAdapterError:
        raise
    except sqlite3.Error as exc:
        raise HermesAdapterError("source_db_busy", str(exc)) from exc
    finally:
        conn.close()


def reconcile_hermes_job(
    store: ProgressStore, source_db: Path, job_id: str
) -> HermesReconcileReport:
    registered = store.status(job_id, include_archived=True)
    if not registered["jobs"]:
        raise KeyError(f"unknown job: {job_id}")
    task_rows = registered["tasks"]
    snapshots = read_task_snapshots(
        source_db, [row["task_id"] for row in task_rows]
    )
    changed = 0
    unchanged = 0
    diagnostics = 0
    for row in task_rows:
        task_id = row["task_id"]
        snapshot = snapshots.get(task_id)
        if snapshot is None:
            store.record_adapter_diagnostic(
                job_id,
                "hermes",
                "source_task_missing",
                f"Hermes source task is missing: {task_id}",
                f"adapter:hermes:{job_id}:source_task_missing:{task_id}",
            )
            diagnostics += 1
            continue
        if snapshot.canonical_status is None:
            store.record_adapter_diagnostic(
                job_id,
                "hermes",
                "source_status_ambiguous",
                (
                    f"Hermes source status cannot be mapped: "
                    f"{task_id}={snapshot.source_status}"
                ),
                (
                    f"adapter:hermes:{job_id}:source_status_ambiguous:"
                    f"{task_id}:{snapshot.source_version}"
                ),
            )
            diagnostics += 1
            continue
        result = store.reconcile_task_snapshot(
            task_id=snapshot.task_id,
            source="hermes",
            source_status=snapshot.source_status,
            canonical_status=snapshot.canonical_status,
            role=snapshot.role,
            message=snapshot.message,
            artifact_refs=snapshot.artifact_refs,
            source_version=snapshot.source_version,
        )
        if result.changed:
            changed += 1
        else:
            unchanged += 1
    store.maybe_finalize(job_id)
    final = store.final_status(job_id)
    return HermesReconcileReport(
        job_id=job_id,
        observed=len(snapshots),
        changed=changed,
        unchanged=unchanged,
        diagnostics=diagnostics,
        finalized=bool(final["has_final_summary"]),
    )


def reconcile_hermes_task(
    store: ProgressStore,
    source_db: Path,
    task_id: str,
    expected_role: Role | None = None,
) -> ReconciliationResult | None:
    matches = [
        row
        for row in store.status(None, include_archived=True)["tasks"]
        if row["task_id"] == task_id
    ]
    if not matches:
        return None
    registered = matches[0]
    role = Role(registered["role"])
    if expected_role is not None and role is not Role(expected_role):
        store.record_adapter_diagnostic(
            registered["job_id"],
            "hermes",
            "source_role_mismatch",
            f"Hermes hook role does not match registered task: {task_id}",
            (
                f"adapter:hermes:{registered['job_id']}:"
                f"source_role_mismatch:{task_id}:{expected_role}"
            ),
        )
        return None
    snapshot = read_task_snapshots(source_db, [task_id]).get(task_id)
    if snapshot is None:
        store.record_adapter_diagnostic(
            registered["job_id"],
            "hermes",
            "source_task_missing",
            f"Hermes source task is missing: {task_id}",
            (
                f"adapter:hermes:{registered['job_id']}:"
                f"source_task_missing:{task_id}"
            ),
        )
        return None
    if snapshot.role is not role:
        store.record_adapter_diagnostic(
            registered["job_id"],
            "hermes",
            "source_role_mismatch",
            f"Hermes source role does not match registered task: {task_id}",
            (
                f"adapter:hermes:{registered['job_id']}:"
                f"source_role_mismatch:{task_id}:{snapshot.role.value}"
            ),
        )
        return None
    if snapshot.canonical_status is None:
        store.record_adapter_diagnostic(
            registered["job_id"],
            "hermes",
            "source_status_ambiguous",
            f"Hermes source status cannot be mapped: {task_id}",
            (
                f"adapter:hermes:{registered['job_id']}:"
                f"source_status_ambiguous:{task_id}:{snapshot.source_version}"
            ),
        )
        return None
    return store.reconcile_task_snapshot(
        task_id=snapshot.task_id,
        source="hermes",
        source_status=snapshot.source_status,
        canonical_status=snapshot.canonical_status,
        role=snapshot.role,
        message=snapshot.message,
        artifact_refs=snapshot.artifact_refs,
        source_version=snapshot.source_version,
    )


def extract_safe_tasks(payload: object) -> list[HermesTaskRef]:
    if isinstance(payload, dict):
        rows = payload.get("tasks", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Hermes payload must be a list or contain a tasks list")
    if not isinstance(rows, list):
        raise ValueError("Hermes tasks must be a list")
    result: list[HermesTaskRef] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).lower()
        assignee = row.get("assignee", row.get("profile", ""))
        try:
            role = Role(str(assignee).lower())
        except ValueError:
            continue
        task_id = row.get("id", row.get("task_id"))
        if task_id and status in ACTIVE and role in WORKERS:
            result.append(HermesTaskRef(str(task_id), role, status))
    return result


def register_active_tasks(
    store: ProgressStore, payload: object, recovery_job_id: str
) -> int:
    refs = extract_safe_tasks(payload)
    for ref in refs:
        initial = (
            TaskStatus.BLOCKED
            if ref.status == "blocked"
            else TaskStatus.QUEUED
        )
        store.add_task(recovery_job_id, ref.task_id, ref.role, status=initial)
        if ref.status == "running":
            store.record(
                recovery_job_id,
                ref.task_id,
                ref.role,
                EventKind.STARTED,
                TaskStatus.RUNNING,
                "legacy-import",
                None,
                "imported active task",
            )
        elif ref.status == "blocked":
            store.record(
                recovery_job_id,
                ref.task_id,
                ref.role,
                EventKind.BLOCKED,
                TaskStatus.BLOCKED,
                "legacy-import",
                None,
                "imported blocked task",
            )
        else:
            store.record(
                recovery_job_id,
                ref.task_id,
                ref.role,
                EventKind.QUEUED,
                TaskStatus.QUEUED,
                "legacy-import",
                None,
                "imported queued task",
            )
    return len(refs)
