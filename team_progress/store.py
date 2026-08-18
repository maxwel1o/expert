import json
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .model import EventKind, ProgressEvent, Role, TaskStatus, TERMINAL_STATUSES
from .reconciliation import ReconciliationResult
from .schema import connect, initialize
from .security import validate_safe_text


IMPORTANT_KINDS = frozenset(
    {
        EventKind.MILESTONE,
        EventKind.BLOCKED,
        EventKind.FAILED,
        EventKind.RETRYING,
        EventKind.COMPLETED,
        EventKind.STALE,
        EventKind.RECONCILED,
        EventKind.ADAPTER_ERROR,
        EventKind.FINAL_SUMMARY,
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_from_row(row) -> ProgressEvent:
    return ProgressEvent(
        seq=row["seq"],
        event_id=row["event_id"],
        job_id=row["job_id"],
        task_id=row["task_id"],
        parent_id=row["parent_id"],
        role=Role(row["role"]),
        kind=EventKind(row["kind"]),
        status=TaskStatus(row["status"]),
        phase=row["phase"],
        percent=row["percent"],
        message=row["message"],
        artifact_refs=tuple(json.loads(row["artifact_refs"])),
        important=bool(row["important"]),
        created_at=row["created_at"],
    )


class ProgressStore:
    def __init__(self, db_path: Path, heartbeat_seconds: int = 300):
        self.db_path = Path(db_path)
        self.heartbeat_seconds = heartbeat_seconds
        with closing(connect(self.db_path)) as conn:
            initialize(conn)

    def create_job(self, title: str, job_id: str | None = None) -> str:
        safe_title = validate_safe_text(title, "title")
        actual_id = job_id or uuid.uuid4().hex
        now = _format_time(_utc_now())
        with closing(connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO jobs(job_id,title,status,created_at,updated_at) "
                "VALUES(?,?, 'active', ?, ?)",
                (actual_id, safe_title, now, now),
            )
        return actual_id

    def add_task(
        self,
        job_id: str,
        task_id: str,
        role: Role,
        parent_id: str | None = None,
        status: TaskStatus = TaskStatus.QUEUED,
    ) -> None:
        role = Role(role)
        status = TaskStatus(status)
        now = _format_time(_utc_now())
        with closing(connect(self.db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO tasks("
                "task_id,job_id,parent_id,role,status,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,?)",
                (task_id, job_id, parent_id, role.value, status.value, now, now),
            )

    def set_focus(self, job_id: str) -> None:
        now = _format_time(_utc_now())
        with closing(connect(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            found = conn.execute(
                "SELECT 1 FROM jobs WHERE job_id=? AND status!='archived'", (job_id,)
            ).fetchone()
            if not found:
                conn.rollback()
                raise KeyError(f"unknown active job: {job_id}")
            conn.execute(
                "UPDATE jobs SET is_foreground=0,updated_at=? WHERE is_foreground=1",
                (now,),
            )
            conn.execute(
                "UPDATE jobs SET is_foreground=1,updated_at=? WHERE job_id=?",
                (now, job_id),
            )
            conn.commit()

    def archive_job(self, job_id: str) -> None:
        now = _format_time(_utc_now())
        with closing(connect(self.db_path)) as conn, conn:
            changed = conn.execute(
                "UPDATE jobs SET status='archived',is_foreground=0,updated_at=? "
                "WHERE job_id=?",
                (now, job_id),
            ).rowcount
            if not changed:
                raise KeyError(f"unknown job: {job_id}")

    def record(
        self,
        job_id: str,
        task_id: str,
        role: Role,
        kind: EventKind,
        status: TaskStatus,
        phase: str,
        percent: float | None,
        message: str,
        artifact_refs: tuple[str, ...] = (),
        dedupe_key: str | None = None,
    ) -> ProgressEvent:
        role = Role(role)
        kind = EventKind(kind)
        status = TaskStatus(status)
        if percent is not None and not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        safe_phase = validate_safe_text(phase, "phase")
        safe_message = validate_safe_text(message, "message")
        safe_artifacts = tuple(
            validate_safe_text(value, "artifact") for value in artifact_refs
        )
        now_value = _utc_now()
        now = _format_time(now_value)
        heartbeat_due = (
            _format_time(now_value + timedelta(seconds=self.heartbeat_seconds))
            if status in {TaskStatus.RUNNING, TaskStatus.QUEUED}
            else None
        )
        terminal_at = now if status in TERMINAL_STATUSES else None
        event_id = uuid.uuid4().hex
        important = kind in IMPORTANT_KINDS

        with closing(connect(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if dedupe_key:
                existing = conn.execute(
                    "SELECT * FROM events WHERE dedupe_key=?", (dedupe_key,)
                ).fetchone()
                if existing:
                    conn.rollback()
                    return _event_from_row(existing)
            task = conn.execute(
                "SELECT parent_id,role FROM tasks WHERE task_id=? AND job_id=?",
                (task_id, job_id),
            ).fetchone()
            parent_id = task["parent_id"] if task else None
            if task and task["role"] != role.value:
                conn.rollback()
                raise ValueError("event role does not match registered task role")
            synthetic_leader_event = (
                role is Role.LEADER
                and kind in {EventKind.FINAL_SUMMARY, EventKind.ADAPTER_ERROR}
            )
            if not task and not synthetic_leader_event:
                conn.rollback()
                raise KeyError(f"unknown task: {task_id}")
            conn.execute(
                "INSERT INTO events("
                "event_id,dedupe_key,job_id,task_id,parent_id,role,kind,status,"
                "phase,percent,message,artifact_refs,important,created_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    dedupe_key,
                    job_id,
                    task_id,
                    parent_id,
                    role.value,
                    kind.value,
                    status.value,
                    safe_phase,
                    percent,
                    safe_message,
                    json.dumps(safe_artifacts),
                    int(important),
                    now,
                ),
            )
            if task:
                conn.execute(
                    "UPDATE tasks SET status=?,phase=?,percent=?,heartbeat_due=?,"
                    "terminal_at=?,updated_at=? WHERE task_id=?",
                    (
                        status.value,
                        safe_phase,
                        percent,
                        heartbeat_due,
                        terminal_at,
                        now,
                        task_id,
                    ),
                )
            if kind is EventKind.RETRYING:
                conn.execute(
                    "UPDATE jobs SET status='active',finalized_at=NULL,"
                    "generation=generation+1,updated_at=? WHERE job_id=?",
                    (now, job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET updated_at=? WHERE job_id=?", (now, job_id)
                )
            row = conn.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            conn.commit()
            return _event_from_row(row)

    def reconcile_task_snapshot(
        self,
        task_id: str,
        source: str,
        source_status: str,
        canonical_status: TaskStatus,
        role: Role,
        message: str,
        artifact_refs: tuple[str, ...],
        source_version: str,
    ) -> ReconciliationResult:
        source = validate_safe_text(source, "source")
        source_status = validate_safe_text(source_status, "source_status")
        source_version = validate_safe_text(source_version, "source_version")
        role = Role(role)
        canonical_status = TaskStatus(canonical_status)
        with closing(connect(self.db_path)) as conn:
            task = conn.execute(
                "SELECT job_id,role,status,phase,percent FROM tasks "
                "WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if not task:
                raise KeyError(f"unknown task: {task_id}")
        if task["role"] != role.value:
            raise ValueError(
                "source role does not match registered task role"
            )

        job_id = task["job_id"]
        previous = TaskStatus(task["status"])
        dedupe_key = (
            f"reconcile:{source}:{task_id}:"
            f"{canonical_status.value}:{source_version}"
        )
        current_terminal = previous in TERMINAL_STATUSES
        source_terminal = canonical_status in TERMINAL_STATUSES
        conflict = (
            (current_terminal and not source_terminal)
            or (
                current_terminal
                and source_terminal
                and previous is not canonical_status
            )
        )

        if previous is canonical_status:
            return ReconciliationResult(
                job_id,
                task_id,
                previous,
                previous,
                False,
                False,
                None,
            )

        if current_terminal and not source_terminal:
            event = self.record(
                job_id,
                task_id,
                role,
                EventKind.RECONCILED,
                previous,
                "reconcile",
                task["percent"],
                (
                    f"{source} source remained {source_status}; "
                    f"kept terminal progress state {previous.value}"
                ),
                (),
                dedupe_key=dedupe_key,
            )
            return ReconciliationResult(
                job_id,
                task_id,
                previous,
                previous,
                False,
                True,
                event,
            )

        event_kind = {
            TaskStatus.QUEUED: EventKind.QUEUED,
            TaskStatus.RUNNING: EventKind.STARTED,
            TaskStatus.BLOCKED: EventKind.BLOCKED,
            TaskStatus.COMPLETED: EventKind.COMPLETED,
            TaskStatus.FAILED: EventKind.FAILED,
            TaskStatus.STALE: EventKind.STALE,
        }[canonical_status]
        if conflict:
            event_kind = EventKind.RECONCILED
        percent = 100.0 if canonical_status is TaskStatus.COMPLETED else None
        event = self.record(
            job_id,
            task_id,
            role,
            event_kind,
            canonical_status,
            "reconcile",
            percent,
            message,
            artifact_refs,
            dedupe_key=dedupe_key,
        )
        with closing(connect(self.db_path)) as conn:
            current = conn.execute(
                "SELECT status FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if current and current["status"] != canonical_status.value:
            now_value = _utc_now()
            now = _format_time(now_value)
            heartbeat_due = (
                _format_time(
                    now_value + timedelta(seconds=self.heartbeat_seconds)
                )
                if canonical_status
                in {TaskStatus.RUNNING, TaskStatus.QUEUED}
                else None
            )
            terminal_at = (
                now if canonical_status in TERMINAL_STATUSES else None
            )
            with closing(connect(self.db_path)) as conn, conn:
                conn.execute(
                    "UPDATE tasks SET status=?,phase='reconcile',percent=?,"
                    "heartbeat_due=?,terminal_at=?,updated_at=? "
                    "WHERE task_id=?",
                    (
                        canonical_status.value,
                        percent,
                        heartbeat_due,
                        terminal_at,
                        now,
                        task_id,
                    ),
                )
                conn.execute(
                    "UPDATE jobs SET updated_at=? WHERE job_id=?",
                    (now, job_id),
                )
        if source_terminal:
            self.release(task_id)
            self.maybe_finalize(job_id)
        return ReconciliationResult(
            job_id,
            task_id,
            previous,
            canonical_status,
            True,
            conflict,
            event,
        )

    def record_adapter_diagnostic(
        self,
        job_id: str,
        source: str,
        code: str,
        message: str,
        dedupe_key: str,
    ) -> ProgressEvent:
        with closing(connect(self.db_path)) as conn:
            job = conn.execute(
                "SELECT 1 FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if not job:
            raise KeyError(f"unknown job: {job_id}")
        safe_source = validate_safe_text(source, "source")
        safe_code = validate_safe_text(code, "diagnostic code")
        return self.record(
            job_id,
            f"adapter:{safe_source}:{job_id}",
            Role.LEADER,
            EventKind.ADAPTER_ERROR,
            TaskStatus.RUNNING,
            "reconcile",
            None,
            f"{safe_code}: {message}",
            dedupe_key=dedupe_key,
        )

    def read_events(
        self,
        job_id: str | None = None,
        after_seq: int = 0,
        important_only: bool = False,
    ) -> list[ProgressEvent]:
        clauses = ["seq>?"]
        params: list[object] = [after_seq]
        if job_id:
            clauses.append("job_id=?")
            params.append(job_id)
        if important_only:
            clauses.append("important=1")
        sql = "SELECT * FROM events WHERE " + " AND ".join(clauses) + " ORDER BY seq"
        with closing(connect(self.db_path)) as conn:
            return [_event_from_row(row) for row in conn.execute(sql, params)]

    def consume(
        self,
        consumer_name: str,
        job_id: str | None = None,
        important_only: bool = True,
    ) -> list[ProgressEvent]:
        consumer_key = f"{consumer_name}@{job_id or '*'}"
        now = _format_time(_utc_now())
        with closing(connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT last_seq FROM consumers WHERE consumer_name=?",
                (consumer_key,),
            ).fetchone()
            after_seq = row["last_seq"] if row else 0
        events = self.read_events(job_id, after_seq, important_only)
        if events:
            with closing(connect(self.db_path)) as conn, conn:
                conn.execute(
                    "INSERT INTO consumers(consumer_name,last_seq,updated_at) "
                    "VALUES(?,?,?) ON CONFLICT(consumer_name) DO UPDATE SET "
                    "last_seq=excluded.last_seq,updated_at=excluded.updated_at",
                    (consumer_key, events[-1].seq, now),
                )
        return events

    def status(
        self, job_id: str | None, include_archived: bool = False
    ) -> dict[str, list[dict]]:
        clauses = []
        params: list[object] = []
        if job_id:
            clauses.append("job_id=?")
            params.append(job_id)
        if not include_archived:
            clauses.append("status!='archived'")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(connect(self.db_path)) as conn:
            jobs = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM jobs" + where + " ORDER BY created_at", params
                )
            ]
            if job_id:
                tasks = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM tasks WHERE job_id=? ORDER BY created_at",
                        (job_id,),
                    )
                ]
            else:
                visible_ids = [job["job_id"] for job in jobs]
                if not visible_ids:
                    tasks = []
                else:
                    placeholders = ",".join("?" for _ in visible_ids)
                    tasks = [
                        dict(row)
                        for row in conn.execute(
                            f"SELECT * FROM tasks WHERE job_id IN ({placeholders}) "
                            "ORDER BY created_at",
                            visible_ids,
                        )
                    ]
        return {"jobs": jobs, "tasks": tasks}

    def final_status(self, job_id: str) -> dict:
        with closing(connect(self.db_path)) as conn:
            job = conn.execute(
                "SELECT job_id,status FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if not job:
                raise KeyError(f"unknown job: {job_id}")
            final = conn.execute(
                "SELECT 1 FROM events WHERE job_id=? AND kind=? "
                "ORDER BY seq DESC LIMIT 1",
                (job_id, EventKind.FINAL_SUMMARY.value),
            ).fetchone()
            task_rows = list(
                conn.execute(
                    "SELECT status FROM tasks WHERE job_id=? ORDER BY created_at",
                    (job_id,),
                )
            )
            event_rows = list(
                conn.execute(
                    "SELECT task_id,artifact_refs FROM events "
                    "WHERE job_id=? AND kind IN (?,?,?,?,?) ORDER BY seq",
                    (
                        job_id,
                        EventKind.COMPLETED.value,
                        EventKind.FAILED.value,
                        EventKind.BLOCKED.value,
                        EventKind.STALE.value,
                        EventKind.RECONCILED.value,
                    ),
                )
            )
        counts = {
            status.value: sum(
                row["status"] == status.value for row in task_rows
            )
            for status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.STALE,
            )
        }
        latest_artifacts_by_task: dict[str, list[str]] = {}
        for row in event_rows:
            latest_artifacts_by_task[row["task_id"]] = json.loads(
                row["artifact_refs"]
            )
        artifacts: list[str] = []
        for task_artifacts in latest_artifacts_by_task.values():
            for artifact in task_artifacts:
                if artifact not in artifacts:
                    artifacts.append(artifact)
        return {
            "job_id": job["job_id"],
            "job_status": job["status"],
            "has_final_summary": final is not None,
            "completed": counts[TaskStatus.COMPLETED.value],
            "failed": counts[TaskStatus.FAILED.value],
            "blocked": counts[TaskStatus.BLOCKED.value],
            "stale": counts[TaskStatus.STALE.value],
            "artifact_refs": artifacts,
        }

    def acquire(
        self,
        task_id: str,
        resources: tuple[str, ...],
        lease_seconds: int = 360,
    ) -> tuple[bool, str | None]:
        now_value = _utc_now()
        now = _format_time(now_value)
        expires = _format_time(now_value + timedelta(seconds=lease_seconds))
        conflict: str | None = None
        with closing(connect(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            task = conn.execute(
                "SELECT job_id,role FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if not task:
                conn.rollback()
                raise KeyError(f"unknown task: {task_id}")
            safe_resources = {
                validate_safe_text(resource, "resource")
                for resource in resources
                if resource.strip()
            }
            keys = tuple(sorted({f"role:{task['role']}", *safe_resources}))
            conn.execute("DELETE FROM locks WHERE lease_expires_at<=?", (now,))
            for key in keys:
                owner = conn.execute(
                    "SELECT task_id FROM locks "
                    "WHERE lock_key=? AND lease_expires_at>?",
                    (key, now),
                ).fetchone()
                if owner and owner["task_id"] != task_id:
                    conflict = key
                    break
            if conflict is None:
                for key in keys:
                    conn.execute(
                        "INSERT INTO locks(lock_key,job_id,task_id,role,"
                        "lease_expires_at,updated_at) VALUES(?,?,?,?,?,?) "
                        "ON CONFLICT(lock_key) DO UPDATE SET "
                        "job_id=excluded.job_id,task_id=excluded.task_id,"
                        "role=excluded.role,lease_expires_at=excluded.lease_expires_at,"
                        "updated_at=excluded.updated_at",
                        (
                            key,
                            task["job_id"],
                            task_id,
                            task["role"],
                            expires,
                            now,
                        ),
                    )
            conn.commit()
        if conflict is not None:
            self.record(
                task["job_id"],
                task_id,
                Role(task["role"]),
                EventKind.QUEUED,
                TaskStatus.QUEUED,
                "waiting",
                None,
                f"waiting for {conflict}",
                dedupe_key=f"queue:{task_id}:{conflict}",
            )
            return False, conflict
        return True, None

    def renew(self, task_id: str, lease_seconds: int = 360) -> None:
        now_value = _utc_now()
        now = _format_time(now_value)
        lease_expires = _format_time(now_value + timedelta(seconds=lease_seconds))
        heartbeat_due = _format_time(
            now_value + timedelta(seconds=self.heartbeat_seconds)
        )
        with closing(connect(self.db_path)) as conn, conn:
            conn.execute(
                "UPDATE locks SET lease_expires_at=?,updated_at=? WHERE task_id=?",
                (lease_expires, now, task_id),
            )
            conn.execute(
                "UPDATE tasks SET heartbeat_due=?,updated_at=? WHERE task_id=?",
                (heartbeat_due, now, task_id),
            )

    def release(self, task_id: str) -> None:
        with closing(connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM locks WHERE task_id=?", (task_id,))

    def sweep_stale(self, now: datetime | None = None) -> list[str]:
        current = now or _utc_now()
        current_text = _format_time(current)
        with closing(connect(self.db_path)) as conn:
            rows = list(
                conn.execute(
                    "SELECT job_id,task_id,role,phase FROM tasks "
                    "WHERE status IN ('running','queued') "
                    "AND heartbeat_due IS NOT NULL AND heartbeat_due<? "
                    "ORDER BY task_id",
                    (current_text,),
                )
            )
        stale_ids = []
        for row in rows:
            self.record(
                row["job_id"],
                row["task_id"],
                Role(row["role"]),
                EventKind.STALE,
                TaskStatus.STALE,
                row["phase"] or "unknown",
                None,
                "heartbeat overdue",
                dedupe_key=f"stale:{row['task_id']}:{current_text}",
            )
            self.release(row["task_id"])
            self.maybe_finalize(row["job_id"])
            stale_ids.append(row["task_id"])
        return stale_ids

    def maybe_finalize(self, job_id: str) -> ProgressEvent | None:
        with closing(connect(self.db_path)) as conn:
            job = conn.execute(
                "SELECT finalized_at,generation FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if not job:
                raise KeyError(f"unknown job: {job_id}")
            if job["finalized_at"]:
                return None
            rows = list(
                conn.execute("SELECT status FROM tasks WHERE job_id=?", (job_id,))
            )
        if not rows:
            return None
        statuses = [TaskStatus(row["status"]) for row in rows]
        if any(status not in TERMINAL_STATUSES for status in statuses):
            return None
        counts = {
            status: sum(item is status for item in statuses)
            for status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.STALE,
            )
        }
        message = (
            f"job {job_id} terminal: "
            f"completed={counts[TaskStatus.COMPLETED]} "
            f"failed={counts[TaskStatus.FAILED]} "
            f"blocked={counts[TaskStatus.BLOCKED]} "
            f"stale={counts[TaskStatus.STALE]}"
        )
        event = self.record(
            job_id,
            f"finalizer:{job_id}",
            Role.LEADER,
            EventKind.FINAL_SUMMARY,
            TaskStatus.COMPLETED,
            "finalize",
            100,
            message,
            dedupe_key=f"finalizer:{job_id}:{job['generation']}",
        )
        now = _format_time(_utc_now())
        with closing(connect(self.db_path)) as conn, conn:
            conn.execute(
                "UPDATE jobs SET status='terminal',finalized_at=?,updated_at=? "
                "WHERE job_id=?",
                (now, now, job_id),
            )
        return event
