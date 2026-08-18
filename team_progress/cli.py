import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .model import EventKind, ProgressEvent, Role, TaskStatus
from .runner import Identity, run_command, wait_for_lock
from .store import ProgressStore


DEFAULT_DB = Path("/opt/data/team-progress/state/progress.db")


def _db_path(args) -> Path:
    value = args.db or os.environ.get("TEAM_PROGRESS_DB")
    return Path(value) if value else DEFAULT_DB


def _identity(args, parser: argparse.ArgumentParser) -> Identity:
    job_id = getattr(args, "job_id", None) or os.environ.get(
        "TEAM_PROGRESS_JOB_ID"
    )
    task_id = getattr(args, "task_id", None) or os.environ.get(
        "TEAM_PROGRESS_TASK_ID"
    )
    role_value = getattr(args, "role", None) or os.environ.get(
        "TEAM_PROGRESS_ROLE"
    )
    missing = [
        name
        for name, value in (
            ("job-id", job_id),
            ("task-id", task_id),
            ("role", role_value),
        )
        if not value
    ]
    if missing:
        parser.error("missing required identity: " + ", ".join(missing))
    return Identity(
        job_id=job_id,
        task_id=task_id,
        role=Role(role_value),
        phase=getattr(args, "phase", None) or "working",
    )


def _event_dict(event: ProgressEvent) -> dict:
    row = asdict(event)
    row["role"] = event.role.value
    row["kind"] = event.kind.value
    row["status"] = event.status.value
    row["artifact_refs"] = list(event.artifact_refs)
    return row


def _print_rows(rows, as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    else:
        for row in rows:
            print(row)


def format_event(event: ProgressEvent) -> str:
    return (
        f"[JOB {event.job_id[:8]}][{event.role.value}][{event.task_id}] "
        f"{event.kind.value}: {event.message}"
    )


def should_display(event: ProgressEvent, foreground_job_id: str | None) -> bool:
    if event.job_id == foreground_job_id:
        return True
    return event.kind in {
        EventKind.BLOCKED,
        EventKind.FAILED,
        EventKind.COMPLETED,
        EventKind.FINAL_SUMMARY,
    }


def _add_identity_flags(parser):
    parser.add_argument("--job-id")
    parser.add_argument("--task-id")
    parser.add_argument("--role", choices=[role.value for role in Role])
    parser.add_argument("--phase", default="working")


def _add_message_flags(parser):
    _add_identity_flags(parser)
    parser.add_argument("--message", required=True)
    parser.add_argument("--percent", type=float)
    parser.add_argument("--artifact", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="team-progress")
    parser.add_argument("--db")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    job = sub.add_parser("job")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    job_create = job_sub.add_parser("create")
    job_create.add_argument("--job-id")
    job_create.add_argument("--title", required=True)
    job_focus = job_sub.add_parser("focus")
    job_focus.add_argument("job_id")
    job_archive = job_sub.add_parser("archive")
    job_archive.add_argument("job_id")

    task = sub.add_parser("task")
    task_sub = task.add_subparsers(dest="task_command", required=True)
    task_add = task_sub.add_parser("add")
    task_add.add_argument("--job-id", required=True)
    task_add.add_argument("--task-id", required=True)
    task_add.add_argument("--parent-id")
    task_add.add_argument("--role", required=True, choices=[r.value for r in Role])

    start = sub.add_parser("start")
    _add_identity_flags(start)
    start.add_argument("--message", default="task started")
    start.add_argument("--resource", action="append", default=[])
    start.add_argument("--lock-timeout", type=float)
    start.add_argument("--heartbeat-seconds", type=float, default=300)

    for name in ("update", "heartbeat", "block", "fail", "complete", "retry"):
        lifecycle = sub.add_parser(name)
        _add_message_flags(lifecycle)

    status = sub.add_parser("status")
    status.add_argument("job_id", nargs="?")
    status.add_argument("--all", action="store_true")
    status.add_argument("--include-archived", action="store_true")
    status.add_argument("--json", action="store_true")

    events = sub.add_parser("events")
    events.add_argument("--job-id")
    events.add_argument("--after-seq", type=int, default=0)
    events.add_argument("--important-only", action="store_true")
    events.add_argument("--json", action="store_true")

    consume = sub.add_parser("consume")
    consume.add_argument("--consumer", required=True)
    consume.add_argument("--job-id")
    consume.add_argument("--important-only", action="store_true")
    consume.add_argument("--json", action="store_true")

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--json", action="store_true")

    watch = sub.add_parser("watch")
    watch.add_argument("job_id", nargs="?")
    watch.add_argument("--consumer", default="terminal")
    watch.add_argument("--poll-seconds", type=float, default=2)
    watch.add_argument("--sweep-seconds", type=float, default=30)
    watch.add_argument("--once", action="store_true")

    wait_final = sub.add_parser("wait-final")
    wait_final.add_argument("job_id")
    wait_final.add_argument("--poll-seconds", type=float, default=2)
    wait_final.add_argument("--adapter", choices=("hermes",))
    wait_final.add_argument("--source-db", type=Path)
    wait_final.add_argument("--adapter-max-errors", type=int, default=5)

    reconcile = sub.add_parser("reconcile")
    reconcile.add_argument("--adapter", required=True, choices=("hermes",))
    reconcile.add_argument("--source-db", required=True, type=Path)
    reconcile.add_argument("--job-id", required=True)

    hermes = sub.add_parser("hermes")
    hermes_sub = hermes.add_subparsers(dest="hermes_command", required=True)
    hermes_import = hermes_sub.add_parser("import-active")
    hermes_import.add_argument("--input-json", required=True, type=Path)
    hermes_import.add_argument("--job-id", required=True)

    run = sub.add_parser("run")
    _add_identity_flags(run)
    run.add_argument("--resource", action="append", default=[])
    run.add_argument("--heartbeat-seconds", type=float, default=300)
    run.add_argument("--lock-timeout", type=float)
    run.add_argument("external_command", nargs=argparse.REMAINDER)
    return parser


def _record_lifecycle(
    store: ProgressStore, identity: Identity, args, command: str
) -> ProgressEvent:
    mapping = {
        "update": (EventKind.MILESTONE, TaskStatus.RUNNING),
        "heartbeat": (EventKind.HEARTBEAT, TaskStatus.RUNNING),
        "block": (EventKind.BLOCKED, TaskStatus.BLOCKED),
        "fail": (EventKind.FAILED, TaskStatus.FAILED),
        "complete": (EventKind.COMPLETED, TaskStatus.COMPLETED),
        "retry": (EventKind.RETRYING, TaskStatus.QUEUED),
    }
    kind, status = mapping[command]
    if command in {"update", "heartbeat"}:
        store.renew(identity.task_id)
    elif command == "retry":
        store.release(identity.task_id)
    percent = 100.0 if command == "complete" and args.percent is None else args.percent
    event = store.record(
        identity.job_id,
        identity.task_id,
        identity.role,
        kind,
        status,
        identity.phase,
        percent,
        args.message,
        tuple(args.artifact),
    )
    if status in {
        TaskStatus.BLOCKED,
        TaskStatus.FAILED,
        TaskStatus.COMPLETED,
    }:
        store.release(identity.task_id)
        store.maybe_finalize(identity.job_id)
    return event


def _foreground(store: ProgressStore) -> str | None:
    rows = store.status(None, include_archived=False)["jobs"]
    return next((row["job_id"] for row in rows if row["is_foreground"]), None)


def _watch(store: ProgressStore, args) -> int:
    last_sweep = 0.0
    while True:
        now = time.monotonic()
        if now - last_sweep >= args.sweep_seconds:
            store.sweep_stale()
            last_sweep = now
        foreground = args.job_id or _foreground(store)
        events = store.consume(args.consumer, None, important_only=False)
        for event in events:
            if should_display(event, foreground):
                print(format_event(event), flush=True)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


def _format_final(state: dict) -> str:
    artifacts = json.dumps(
        state["artifact_refs"], ensure_ascii=False, separators=(",", ":")
    )
    return (
        f"TEAM_PROGRESS_FINAL job_id={state['job_id']} "
        f"completed={state['completed']} failed={state['failed']} "
        f"blocked={state['blocked']} stale={state['stale']} "
        f"artifacts={artifacts}"
    )


def _wait_final(store: ProgressStore, args) -> int:
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be greater than zero")
    if args.adapter and args.source_db is None:
        raise ValueError("--source-db is required with --adapter")
    if args.adapter_max_errors <= 0:
        raise ValueError("adapter-max-errors must be greater than zero")
    adapter_errors = 0
    while True:
        if args.adapter == "hermes":
            from .adapters.hermes import (
                HermesAdapterError,
                reconcile_hermes_job,
            )

            try:
                reconcile_hermes_job(store, args.source_db, args.job_id)
                adapter_errors = 0
            except HermesAdapterError as exc:
                adapter_errors += 1
                store.record_adapter_diagnostic(
                    args.job_id,
                    "hermes",
                    exc.code,
                    str(exc),
                    f"adapter:hermes:{args.job_id}:{exc.code}",
                )
                if adapter_errors >= args.adapter_max_errors:
                    raise ValueError(
                        "final listener adapter failed after "
                        f"{adapter_errors} consecutive errors: {exc.code}"
                    ) from exc
        state = store.final_status(args.job_id)
        if state["has_final_summary"]:
            print(_format_final(state), flush=True)
            return 0
        if state["job_status"] == "archived":
            raise ValueError(
                f"job {args.job_id} archived without final_summary"
            )
        time.sleep(args.poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        store = ProgressStore(_db_path(args))
        if args.command == "init":
            print(f"initialized={store.db_path}")
        elif args.command == "job":
            if args.job_command == "create":
                print(store.create_job(args.title, args.job_id))
            elif args.job_command == "focus":
                store.set_focus(args.job_id)
                print(f"focused={args.job_id}")
            else:
                store.archive_job(args.job_id)
                print(f"archived={args.job_id}")
        elif args.command == "task":
            store.add_task(
                args.job_id,
                args.task_id,
                Role(args.role),
                args.parent_id,
            )
            print(f"registered={args.task_id}")
        elif args.command == "start":
            identity = _identity(args, parser)
            acquired = wait_for_lock(
                store,
                identity,
                tuple(args.resource),
                args.heartbeat_seconds,
                args.lock_timeout,
            )
            if not acquired:
                return 75
            event = store.record(
                identity.job_id,
                identity.task_id,
                identity.role,
                EventKind.STARTED,
                TaskStatus.RUNNING,
                identity.phase,
                None,
                args.message,
            )
            print(format_event(event))
        elif args.command in {"update", "heartbeat", "block", "fail", "complete", "retry"}:
            print(
                format_event(
                    _record_lifecycle(
                        store, _identity(args, parser), args, args.command
                    )
                )
            )
        elif args.command == "status":
            result = store.status(
                None if args.all else args.job_id,
                include_archived=args.include_archived,
            )
            _print_rows(result if args.json else [repr(result)], args.json)
        elif args.command == "events":
            rows = [
                _event_dict(event)
                for event in store.read_events(
                    args.job_id, args.after_seq, args.important_only
                )
            ]
            _print_rows(rows if args.json else [format_event(e) for e in store.read_events(
                args.job_id, args.after_seq, args.important_only
            )], args.json)
        elif args.command == "consume":
            events = store.consume(
                args.consumer, args.job_id, args.important_only
            )
            _print_rows(
                [_event_dict(event) for event in events]
                if args.json
                else [format_event(event) for event in events],
                args.json,
            )
        elif args.command == "sweep":
            rows = store.sweep_stale()
            _print_rows(rows, args.json)
        elif args.command == "watch":
            return _watch(store, args)
        elif args.command == "wait-final":
            return _wait_final(store, args)
        elif args.command == "reconcile":
            from .adapters.hermes import reconcile_hermes_job

            report = reconcile_hermes_job(
                store, args.source_db, args.job_id
            )
            finalized = str(report.finalized).lower()
            print(
                f"RECONCILED job_id={report.job_id} "
                f"adapter={args.adapter} observed={report.observed} "
                f"changed={report.changed} "
                f"diagnostics={report.diagnostics} "
                f"finalized={finalized}"
            )
        elif args.command == "hermes":
            from .adapters.hermes import register_active_tasks

            if str(args.input_json) == "-":
                raise ValueError("stdin is not accepted; use a mode-600 JSON file")
            with args.input_json.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            count = register_active_tasks(store, payload, args.job_id)
            print(f"registered={count} job_id={args.job_id}")
        elif args.command == "run":
            identity = _identity(args, parser)
            command = list(args.external_command)
            if command and command[0] == "--":
                command.pop(0)
            return run_command(
                store,
                identity,
                command,
                tuple(args.resource),
                args.heartbeat_seconds,
                args.lock_timeout,
            )
        return 0
    except (KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
