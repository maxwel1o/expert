import tempfile
import unittest
from pathlib import Path

from team_progress.model import EventKind, Role, TaskStatus
from team_progress.store import ProgressStore


class FinalStatusCurrentStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProgressStore(Path(self.tmp.name) / "progress.db")
        self.store.create_job("retry state", "job-retry")
        self.store.add_task("job-retry", "task-a", Role.TESTER)

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_terminal_events_count_one_current_task(self):
        for message in ("failed once", "failed repeated"):
            self.store.record(
                "job-retry",
                "task-a",
                Role.TESTER,
                EventKind.FAILED,
                TaskStatus.FAILED,
                "failed",
                None,
                message,
                ("/safe/failure.json",),
            )

        state = self.store.final_status("job-retry")

        self.assertEqual(1, state["failed"])
        self.assertEqual(0, state["completed"])

    def test_retry_uses_current_task_status_and_latest_terminal_artifacts(self):
        self.store.record(
            "job-retry",
            "task-a",
            Role.TESTER,
            EventKind.FAILED,
            TaskStatus.FAILED,
            "failed",
            None,
            "old failure",
            ("/safe/old-failure.json",),
        )
        self.store.maybe_finalize("job-retry")
        self.store.record(
            "job-retry",
            "task-a",
            Role.TESTER,
            EventKind.RETRYING,
            TaskStatus.QUEUED,
            "retry",
            None,
            "retry requested",
        )
        self.store.record(
            "job-retry",
            "task-a",
            Role.TESTER,
            EventKind.COMPLETED,
            TaskStatus.COMPLETED,
            "done",
            100,
            "new success",
            ("/safe/new-success.json",),
        )
        self.store.maybe_finalize("job-retry")

        state = self.store.final_status("job-retry")

        self.assertEqual(1, state["completed"])
        self.assertEqual(0, state["failed"])
        self.assertEqual(["/safe/new-success.json"], state["artifact_refs"])


if __name__ == "__main__":
    unittest.main()
