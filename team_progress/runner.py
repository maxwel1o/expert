import subprocess
import time
from dataclasses import dataclass

from .model import EventKind, Role, TaskStatus
from .store import ProgressStore


@dataclass(frozen=True)
class Identity:
    job_id: str
    task_id: str
    role: Role
    phase: str


def wait_for_lock(
    store: ProgressStore,
    identity: Identity,
    resources: tuple[str, ...],
    heartbeat_seconds: float = 300,
    lock_timeout_seconds: float | None = None,
) -> bool:
    deadline = (
        None
        if lock_timeout_seconds is None
        else time.monotonic() + lock_timeout_seconds
    )
    last_queue_heartbeat = 0.0
    while True:
        acquired, conflict = store.acquire(identity.task_id, resources)
        if acquired:
            return True
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            return False
        if now - last_queue_heartbeat >= heartbeat_seconds:
            store.record(
                identity.job_id,
                identity.task_id,
                identity.role,
                EventKind.HEARTBEAT,
                TaskStatus.QUEUED,
                identity.phase,
                None,
                f"still queued for {conflict}",
            )
            last_queue_heartbeat = now
        time.sleep(min(2.0, heartbeat_seconds))


def run_command(
    store: ProgressStore,
    identity: Identity,
    command: list[str],
    resources: tuple[str, ...],
    heartbeat_seconds: float = 300,
    lock_timeout_seconds: float | None = None,
) -> int:
    if not command:
        raise ValueError("command is required")
    if not wait_for_lock(
        store,
        identity,
        resources,
        heartbeat_seconds=heartbeat_seconds,
        lock_timeout_seconds=lock_timeout_seconds,
    ):
        return 75
    store.record(
        identity.job_id,
        identity.task_id,
        identity.role,
        EventKind.STARTED,
        TaskStatus.RUNNING,
        identity.phase,
        None,
        "command started",
    )
    process = None
    try:
        process = subprocess.Popen(command)
        while True:
            try:
                code = process.wait(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                store.renew(identity.task_id)
                store.record(
                    identity.job_id,
                    identity.task_id,
                    identity.role,
                    EventKind.HEARTBEAT,
                    TaskStatus.RUNNING,
                    identity.phase,
                    None,
                    "still running",
                )
        kind = EventKind.COMPLETED if code == 0 else EventKind.FAILED
        status = TaskStatus.COMPLETED if code == 0 else TaskStatus.FAILED
        store.record(
            identity.job_id,
            identity.task_id,
            identity.role,
            kind,
            status,
            identity.phase,
            100.0 if code == 0 else None,
            "command completed" if code == 0 else f"command exited {code}",
        )
        store.maybe_finalize(identity.job_id)
        return code
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
        raise
    finally:
        store.release(identity.task_id)

