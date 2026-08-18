import tempfile
import unittest
from pathlib import Path

from team_progress.model import EventKind, Role, TaskStatus
from team_progress.store import ProgressStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProgressStore(Path(self.tmp.name) / "progress.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_jobs_are_isolated_and_focus_is_unique(self):
        first = self.store.create_job("first", "job-a")
        second = self.store.create_job("second", "job-b")
        self.store.set_focus(first)
        self.store.set_focus(second)
        status = self.store.status(None, include_archived=True)
        foreground = [job for job in status["jobs"] if job["is_foreground"]]
        self.assertEqual(["job-b"], [job["job_id"] for job in foreground])

    def test_record_is_deduplicated(self):
        self.store.create_job("job", "job-a")
        self.store.add_task("job-a", "task-a", Role.TESTER)
        first = self.store.record(
            "job-a",
            "task-a",
            Role.TESTER,
            EventKind.STARTED,
            TaskStatus.RUNNING,
            "prepare",
            None,
            "started",
            dedupe_key="run-1:start",
        )
        second = self.store.record(
            "job-a",
            "task-a",
            Role.TESTER,
            EventKind.STARTED,
            TaskStatus.RUNNING,
            "prepare",
            None,
            "started",
            dedupe_key="run-1:start",
        )
        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(1, len(self.store.read_events("job-a")))

    def test_consumer_advances_only_after_read(self):
        self.store.create_job("job", "job-a")
        self.store.add_task("job-a", "task-a", Role.TESTER)
        self.store.record(
            "job-a",
            "task-a",
            Role.TESTER,
            EventKind.MILESTONE,
            TaskStatus.RUNNING,
            "bench",
            None,
            "input validated",
        )
        self.assertEqual(1, len(self.store.consume("leader", "job-a")))
        self.assertEqual([], self.store.consume("leader", "job-a"))

    def test_rejects_percent_outside_range(self):
        self.store.create_job("job", "job-a")
        self.store.add_task("job-a", "task-a", Role.TESTER)
        with self.assertRaises(ValueError):
            self.store.record(
                "job-a",
                "task-a",
                Role.TESTER,
                EventKind.MILESTONE,
                TaskStatus.RUNNING,
                "bench",
                101,
                "invalid percentage",
            )


if __name__ == "__main__":
    unittest.main()
