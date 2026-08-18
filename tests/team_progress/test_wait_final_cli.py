import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from team_progress.model import EventKind, Role, TaskStatus
from team_progress.store import ProgressStore
from tests.team_progress.hermes_db_fixture import create_hermes_db


class WaitFinalCliTests(unittest.TestCase):
    def run_cli(self, db: Path, *args: str, timeout: float = 5):
        env = os.environ.copy()
        env["TEAM_PROGRESS_DB"] = str(db)
        return subprocess.run(
            [sys.executable, "-m", "team_progress.cli", *args],
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )

    def make_terminal_job(self, db: Path) -> ProgressStore:
        store = ProgressStore(db)
        store.create_job("terminal", "job-final")
        store.add_task("job-final", "task-a", Role.TESTER)
        store.record(
            "job-final",
            "task-a",
            Role.TESTER,
            EventKind.COMPLETED,
            TaskStatus.COMPLETED,
            "done",
            100,
            "finished",
            ("/safe/report.json",),
        )
        store.maybe_finalize("job-final")
        return store

    def test_terminal_job_exits_immediately_with_one_safe_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            self.make_terminal_job(db)

            result = self.run_cli(db, "wait-final", "job-final")

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(1, result.stdout.count("TEAM_PROGRESS_FINAL"))
            self.assertIn("job_id=job-final", result.stdout)
            self.assertIn("completed=1", result.stdout)
            self.assertIn("failed=0", result.stdout)
            self.assertIn("blocked=0", result.stdout)
            self.assertIn("stale=0", result.stdout)
            self.assertIn('artifacts=["/safe/report.json"]', result.stdout)
            self.assertEqual("", result.stderr)

    def test_running_job_ignores_nonfinal_events_until_finalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            store = ProgressStore(db)
            store.create_job("running", "job-final")
            store.add_task("job-final", "task-a", Role.TESTER)
            env = os.environ.copy()
            env["TEAM_PROGRESS_DB"] = str(db)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "team_progress.cli",
                    "wait-final",
                    "job-final",
                    "--poll-seconds",
                    "0.02",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            try:
                store.record(
                    "job-final",
                    "task-a",
                    Role.TESTER,
                    EventKind.HEARTBEAT,
                    TaskStatus.RUNNING,
                    "run",
                    None,
                    "alive",
                )
                store.record(
                    "job-final",
                    "task-a",
                    Role.TESTER,
                    EventKind.MILESTONE,
                    TaskStatus.RUNNING,
                    "run",
                    50,
                    "halfway",
                )
                time.sleep(0.12)
                self.assertIsNone(process.poll())

                store.record(
                    "job-final",
                    "task-a",
                    Role.TESTER,
                    EventKind.COMPLETED,
                    TaskStatus.COMPLETED,
                    "done",
                    100,
                    "finished",
                )
                store.maybe_finalize("job-final")
                stdout, stderr = process.communicate(timeout=3)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)

            self.assertEqual(0, process.returncode, stderr)
            self.assertEqual(1, stdout.count("TEAM_PROGRESS_FINAL"))

    def test_wait_does_not_advance_leader_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            store = self.make_terminal_job(db)

            result = self.run_cli(db, "wait-final", "job-final")
            events = store.consume("leader", "job-final", True)

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(
                ["completed", "final_summary"],
                [event.kind.value for event in events],
            )

    def test_unknown_job_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(
                Path(tmp) / "progress.db",
                "wait-final",
                "missing",
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("unknown job", result.stderr)

    def test_archived_job_without_final_summary_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "progress.db"
            store = ProgressStore(db)
            store.create_job("archived", "job-archived")
            store.archive_job("job-archived")

            result = self.run_cli(db, "wait-final", "job-archived")

            self.assertNotEqual(0, result.returncode)
            self.assertIn("archived without final_summary", result.stderr)

    def test_poll_seconds_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(
                Path(tmp) / "progress.db",
                "wait-final",
                "job-final",
                "--poll-seconds",
                "0",
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("poll-seconds", result.stderr)

    def test_reconcile_command_repairs_kanban_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "progress.db"
            kanban_db = root / "kanban.db"
            store = ProgressStore(db)
            store.create_job("reconcile", "job-final")
            store.add_task("job-final", "task-a", Role.TESTER)
            create_hermes_db(
                kanban_db,
                [
                    {
                        "id": "task-a",
                        "assignee": "tester",
                        "status": "done",
                        "result": '{"artifact_refs":["report.md"]}',
                        "completed_at": "2026-07-27T02:13:00Z",
                        "run_id": 1,
                        "run_status": "completed",
                        "outcome": "completed",
                        "summary": "finished",
                        "ended_at": "2026-07-27T02:13:00Z",
                        "event_id": 2,
                        "event_at": "2026-07-27T02:13:00Z",
                    }
                ],
            )

            result = self.run_cli(
                db,
                "reconcile",
                "--adapter",
                "hermes",
                "--source-db",
                str(kanban_db),
                "--job-id",
                "job-final",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(
                "RECONCILED job_id=job-final adapter=hermes "
                "observed=1 changed=1 diagnostics=0 finalized=true",
                result.stdout.strip(),
            )

    def test_wait_final_reconciles_before_polling_final_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "progress.db"
            kanban_db = root / "kanban.db"
            store = ProgressStore(db)
            store.create_job("reconcile", "job-final")
            store.add_task("job-final", "task-a", Role.TESTER)
            create_hermes_db(
                kanban_db,
                [
                    {
                        "id": "task-a",
                        "assignee": "tester",
                        "status": "done",
                        "result": "report.md",
                        "completed_at": "2026-07-27T02:13:00Z",
                    }
                ],
            )

            result = self.run_cli(
                db,
                "wait-final",
                "job-final",
                "--adapter",
                "hermes",
                "--source-db",
                str(kanban_db),
                "--poll-seconds",
                "0.01",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(1, result.stdout.count("TEAM_PROGRESS_FINAL"))
            self.assertIn("completed=1", result.stdout)

    def test_wait_final_stops_after_adapter_error_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "progress.db"
            store = ProgressStore(db)
            store.create_job("reconcile", "job-final")
            store.add_task("job-final", "task-a", Role.TESTER)

            result = self.run_cli(
                db,
                "wait-final",
                "job-final",
                "--adapter",
                "hermes",
                "--source-db",
                str(root / "missing-kanban.db"),
                "--adapter-max-errors",
                "2",
                "--poll-seconds",
                "0.01",
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("final listener adapter failed", result.stderr)
            self.assertNotIn("worker failed", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
