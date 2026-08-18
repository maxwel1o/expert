from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    LEADER = "leader"
    DEPLOYER = "deployer"
    TESTER = "tester"
    PROFILER = "profiler"
    ANALYST = "analyst"


class EventKind(StrEnum):
    QUEUED = "queued"
    STARTED = "started"
    MILESTONE = "milestone"
    HEARTBEAT = "heartbeat"
    BLOCKED = "blocked"
    FAILED = "failed"
    RETRYING = "retrying"
    COMPLETED = "completed"
    STALE = "stale"
    RECONCILED = "reconciled"
    ADAPTER_ERROR = "adapter_error"
    FINAL_SUMMARY = "final_summary"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"
    STALE = "stale"


TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.BLOCKED,
        TaskStatus.FAILED,
        TaskStatus.COMPLETED,
        TaskStatus.STALE,
    }
)


@dataclass(frozen=True)
class ProgressEvent:
    seq: int
    event_id: str
    job_id: str
    task_id: str
    parent_id: str | None
    role: Role
    kind: EventKind
    status: TaskStatus
    phase: str
    percent: float | None
    message: str
    artifact_refs: tuple[str, ...]
    important: bool
    created_at: str
