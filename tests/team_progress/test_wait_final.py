import tempfile
import unittest
from pathlib import Path

from team_progress.model import EventKind, Role, TaskStatus
from team_progress.store import ProgressStore


class FinalStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProgressStore(Path(self.tmp.name) / "progress.db")
        self.store.create_job("final wait test", "job-final")

    def tearDown(self):
        self.tmp.cleanup()

    def test_running_job_has_no_final_summary(self):
        self.store.add_task("job-final", "task-a", Role.TESTER)

        state = self.store.final_status("job-final")

        self.assertFalse(state["has_final_summary"])
        self.assertEqual("active", state["job_status"])

    def test_final_snapshot_counts_terminal_types_and_artifacts(self):
        cases = (
            (
                "task-complete",
                Role.DEPLOYER,
                EventKind.COMPLETED,
                TaskStatus.COMPLETED,
            ),
            ("task-fail", Role.TESTER, EventKind.FAILED, TaskStatus.FAILED),
            (
                "task-block",
                Role.PROFILER,
                EventKind.BLOCKED,
                TaskStatus.BLOCKED,
            ),
            ("task-stale", Role.ANALYST, EventKind.STALE, TaskStatus.STALE),
        )
        for task_id, role, kind, status in cases:
            self.store.add_task("job-final", task_id, role)
            self.store.record(
                "job-final",
                task_id,
                role,
                kind,
                status,
                "done",
                100,
                f"{task_id} terminal",
                (f"/safe/{task_id}.json",),
            )
        self.store.maybe_finalize("job-final")

        state = self.store.final_status("job-final")

        self.assertTrue(state["has_final_summary"])
        self.assertEqual("terminal", state["job_status"])
        self.assertEqual(
            (1, 1, 1, 1),
            (
                state["completed"],
                state["failed"],
                state["blocked"],
                state["stale"],
            ),
        )
        self.assertEqual(
            [f"/safe/{task_id}.json" for task_id, *_ in cases],
            state["artifact_refs"],
        )

    def test_artifact_refs_come_from_latest_terminal_event_per_task(self):
        self.store.add_task("job-final", "task-a", Role.TESTER)
        self.store.record(
            "job-final",
            "task-a",
            Role.TESTER,
            EventKind.FAILED,
            TaskStatus.FAILED,
            "done",
            None,
            "failed",
            ("/safe/shared.json", "/safe/unique.json"),
        )
        self.store.record(
            "job-final",
            "task-a",
            Role.TESTER,
            EventKind.FAILED,
            TaskStatus.FAILED,
            "done",
            None,
            "failed again",
            ("/safe/shared.json",),
        )

        state = self.store.final_status("job-final")

        self.assertEqual(
            ["/safe/shared.json"],
            state["artifact_refs"],
        )

    def test_unknown_job_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unknown job"):
            self.store.final_status("missing")


if __name__ == "__main__":
    unittest.main()
