import sqlite3
import tempfile
import unittest
from pathlib import Path

from team_progress.adapters.hermes import (
    HermesAdapterError,
    canonicalize_hermes_status,
    read_task_snapshots,
    reconcile_hermes_job,
)
from team_progress.model import EventKind, Role, TaskStatus
from team_progress.store import ProgressStore
from tests.team_progress.hermes_db_fixture import create_hermes_db


def done_task(task_id: str, role: str, run_id: int, event_id: int) -> dict:
    return {
        "id": task_id,
        "assignee": role,
        "status": "done",
        "result": '{"artifact_refs":["fallback.md"]}',
        "started_at": "2026-07-27T02:00:00Z",
        "completed_at": "2026-07-27T02:13:00Z",
        "run_id": run_id,
        "run_status": "completed",
        "outcome": "completed",
        "summary": (
            '{"message":"finished","artifact_refs":'
            '["test_report.md","summary.json"]}'
        ),
        "ended_at": "2026-07-27T02:13:00Z",
        "event_id": event_id,
        "event_kind": "completed",
        "payload": {"artifact_refs": ["raw_results.jsonl"]},
        "event_at": "2026-07-27T02:13:00Z",
    }


class HermesSnapshotTests(unittest.TestCase):
    def test_maps_all_supported_lifecycle_states(self):
        cases = {
            ("triage", None): TaskStatus.QUEUED,
            ("todo", None): TaskStatus.QUEUED,
            ("scheduled", None): TaskStatus.QUEUED,
            ("ready", None): TaskStatus.QUEUED,
            ("running", None): TaskStatus.RUNNING,
            ("review", None): TaskStatus.RUNNING,
            ("blocked", None): TaskStatus.BLOCKED,
            ("done", None): TaskStatus.COMPLETED,
            ("archived", "completed"): TaskStatus.COMPLETED,
            ("archived", None): None,
        }
        for arguments, expected in cases.items():
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    expected, canonicalize_hermes_status(*arguments)
                )

    def test_reads_durable_snapshot_and_stable_source_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_db = Path(tmp) / "kanban.db"
            create_hermes_db(
                source_db,
                [done_task("t_test", "tester", 42, 91)],
            )

            snapshots = read_task_snapshots(source_db, ["t_test"])

            snapshot = snapshots["t_test"]
            self.assertEqual(Role.TESTER, snapshot.role)
            self.assertEqual(TaskStatus.COMPLETED, snapshot.canonical_status)
            self.assertEqual(
                "done:42:91:2026-07-27T02:13:00Z",
                snapshot.source_version,
            )
            self.assertEqual("finished", snapshot.message)
            self.assertEqual(
                (
                    "test_report.md",
                    "summary.json",
                    "raw_results.jsonl",
                ),
                snapshot.artifact_refs,
            )

    def test_missing_database_does_not_create_an_empty_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_db = Path(tmp) / "missing.db"

            with self.assertRaises(HermesAdapterError) as caught:
                read_task_snapshots(source_db, ["t_test"])

            self.assertEqual("source_db_missing", caught.exception.code)
            self.assertFalse(source_db.exists())

    def test_incompatible_schema_has_a_stable_error_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_db = Path(tmp) / "kanban.db"
            sqlite3.connect(source_db).close()

            with self.assertRaises(HermesAdapterError) as caught:
                read_task_snapshots(source_db, ["t_test"])

            self.assertEqual("schema_incompatible", caught.exception.code)


class HermesReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.progress_db = root / "progress.db"
        self.kanban_db = root / "kanban.db"
        self.store = ProgressStore(self.progress_db)
        self.store.create_job("regression", "job-regression")
        roles = (
            ("t_deploy", Role.DEPLOYER),
            ("t_test", Role.TESTER),
            ("t_profile", Role.PROFILER),
            ("t_analyse", Role.ANALYST),
        )
        for task_id, role in roles:
            self.store.add_task("job-regression", task_id, role)
        for task_id, role in roles:
            if role is not Role.TESTER:
                self.store.record(
                    "job-regression",
                    task_id,
                    role,
                    EventKind.COMPLETED,
                    TaskStatus.COMPLETED,
                    "done",
                    100,
                    "already completed",
                )

    def tearDown(self):
        self.tmp.cleanup()

    def test_repairs_done_tester_and_finalizes_once(self):
        create_hermes_db(
            self.kanban_db,
            [
                done_task("t_deploy", "deployer", 1, 11),
                done_task("t_test", "tester", 2, 12),
                done_task("t_profile", "profiler", 3, 13),
                done_task("t_analyse", "analyst", 4, 14),
            ],
        )

        first = reconcile_hermes_job(
            self.store, self.kanban_db, "job-regression"
        )
        second = reconcile_hermes_job(
            self.store, self.kanban_db, "job-regression"
        )

        status = self.store.status("job-regression")
        tester = next(
            row for row in status["tasks"] if row["task_id"] == "t_test"
        )
        events = self.store.read_events("job-regression")
        self.assertEqual("completed", tester["status"])
        self.assertEqual("terminal", status["jobs"][0]["status"])
        self.assertEqual(4, first.observed)
        self.assertEqual(1, first.changed)
        self.assertEqual(0, second.changed)
        self.assertEqual(
            1,
            sum(event.kind is EventKind.FINAL_SUMMARY for event in events),
        )

    def test_missing_source_task_records_diagnostic_without_finalizing(self):
        create_hermes_db(
            self.kanban_db,
            [
                done_task("t_deploy", "deployer", 1, 11),
                done_task("t_profile", "profiler", 3, 13),
                done_task("t_analyse", "analyst", 4, 14),
            ],
        )

        report = reconcile_hermes_job(
            self.store, self.kanban_db, "job-regression"
        )

        final = self.store.final_status("job-regression")
        self.assertEqual(1, report.diagnostics)
        self.assertFalse(final["has_final_summary"])
        self.assertEqual("active", final["job_status"])

    def test_same_source_version_restores_a_task_reopened_by_retry(self):
        create_hermes_db(
            self.kanban_db,
            [
                done_task("t_deploy", "deployer", 1, 11),
                done_task("t_test", "tester", 2, 12),
                done_task("t_profile", "profiler", 3, 13),
                done_task("t_analyse", "analyst", 4, 14),
            ],
        )
        reconcile_hermes_job(
            self.store, self.kanban_db, "job-regression"
        )
        self.store.record(
            "job-regression",
            "t_test",
            Role.TESTER,
            EventKind.RETRYING,
            TaskStatus.QUEUED,
            "retry",
            None,
            "manually reopened",
        )

        report = reconcile_hermes_job(
            self.store, self.kanban_db, "job-regression"
        )

        tester = next(
            row
            for row in self.store.status("job-regression")["tasks"]
            if row["task_id"] == "t_test"
        )
        reconciled = [
            event
            for event in self.store.read_events("job-regression")
            if event.task_id == "t_test"
            and event.kind in {EventKind.COMPLETED, EventKind.RECONCILED}
        ]
        self.assertEqual("completed", tester["status"])
        self.assertEqual(1, report.changed)
        self.assertEqual(1, len(reconciled))

    def test_source_role_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "source role does not match registered task role",
        ):
            self.store.reconcile_task_snapshot(
                task_id="t_test",
                source="hermes",
                source_status="done",
                canonical_status=TaskStatus.COMPLETED,
                role=Role.ANALYST,
                message="done",
                artifact_refs=(),
                source_version="done:2:12:2026-07-27T02:13:00Z",
            )


if __name__ == "__main__":
    unittest.main()
