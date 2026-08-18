import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from team_progress.model import EventKind, Role, TaskStatus
from team_progress.store import ProgressStore


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProgressStore(Path(self.tmp.name) / "progress.db", 300)
        self.store.create_job("job", "job-a")

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_role_is_serial(self):
        self.store.add_task("job-a", "task-a", Role.DEPLOYER)
        self.store.add_task("job-a", "task-b", Role.DEPLOYER)
        self.assertEqual((True, None), self.store.acquire("task-a", ()))
        acquired, conflict = self.store.acquire("task-b", ())
        self.assertFalse(acquired)
        self.assertEqual("role:deployer", conflict)

    def test_different_roles_share_only_non_conflicting_resources(self):
        self.store.add_task("job-a", "task-a", Role.TESTER)
        self.store.add_task("job-a", "task-b", Role.PROFILER)
        self.assertEqual((True, None), self.store.acquire("task-a", ("npu:0",)))
        acquired, conflict = self.store.acquire("task-b", ("npu:0",))
        self.assertFalse(acquired)
        self.assertEqual("npu:0", conflict)

    def test_release_allows_next_same_role_task(self):
        self.store.add_task("job-a", "task-a", Role.ANALYST)
        self.store.add_task("job-a", "task-b", Role.ANALYST)
        self.assertEqual((True, None), self.store.acquire("task-a", ()))
        self.store.release("task-a")
        self.assertEqual((True, None), self.store.acquire("task-b", ()))

    def test_finalizer_runs_when_all_tasks_terminal_even_with_failure(self):
        self.store.add_task("job-a", "task-a", Role.TESTER)
        self.store.add_task("job-a", "task-b", Role.PROFILER)
        self.store.record(
            "job-a",
            "task-a",
            Role.TESTER,
            EventKind.COMPLETED,
            TaskStatus.COMPLETED,
            "done",
            100,
            "tests complete",
        )
        self.store.record(
            "job-a",
            "task-b",
            Role.PROFILER,
            EventKind.FAILED,
            TaskStatus.FAILED,
            "collect",
            None,
            "collector exited 2",
        )
        final = self.store.maybe_finalize("job-a")
        self.assertEqual(EventKind.FINAL_SUMMARY, final.kind)
        self.assertIn("completed=1", final.message)
        self.assertIn("failed=1", final.message)
        self.assertIsNone(self.store.maybe_finalize("job-a"))

    def test_overdue_task_becomes_stale(self):
        self.store.add_task("job-a", "task-a", Role.ANALYST)
        self.store.record(
            "job-a",
            "task-a",
            Role.ANALYST,
            EventKind.STARTED,
            TaskStatus.RUNNING,
            "analyze",
            None,
            "started",
        )
        future = datetime.now(timezone.utc) + timedelta(seconds=301)
        self.assertEqual(["task-a"], self.store.sweep_stale(future))
        events = self.store.read_events("job-a", important_only=True)
        self.assertEqual(EventKind.STALE, events[-2].kind)
        self.assertEqual(EventKind.FINAL_SUMMARY, events[-1].kind)

    def test_empty_job_does_not_finalize(self):
        self.assertIsNone(self.store.maybe_finalize("job-a"))


if __name__ == "__main__":
    unittest.main()
